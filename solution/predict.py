"""Self-contained inference for the frozen E5 + USER2-D + BM25 CatBoost solution.

All model loading is local. The optional item-vector cache is keyed by ordered IDs,
prepared text, encoder revision, precision, window and pooling. No train labels or
previously generated candidates are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["e5_cosine", "user2_cosine", "e5_rank", "user2_rank", "e5_present", "user2_present",
            "query_category_known", "category_match", "location_match", "neighbor_location",
            "distance_km", "distance_available", "title_word_coverage", "params_word_coverage",
            "query_word_count", "query_char_length", "title_char_length", "params_char_length",
            "description_char_length", "rating", "rating_missing", "log_reviews", "reviews_missing",
            "bm25_title", "bm25_params", "bm25_description"]
ITEM_REQUIRED = ["item_id", "item_category_id", "item_location_id"]
QUERY_REQUIRED = ["query_id", "search_query", "search_category", "search_location_id"]
TEXT_FIELDS = ["item_title_raw", "item_infm_params_text", "item_description_raw"]
TAG = re.compile(r"<[^>]*>")
TERM = re.compile(r"[^\W_]+", re.UNICODE)
WORD = re.compile(r"\w+", re.UNICODE)
SERVICE = ("тип услуги", "вид услуги", "категория услуги")
NEXT = ("Место оказания услуг", "Место оказания", "Тип стоимости", "Гарантия", "Бригада",
        "Опыт работы", "Начальная цена", "Выполняю заказы", "Готов закупать материалы",
        "Время для связи", "График работы")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(values):
    h = hashlib.sha256()
    for value in values:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "little")); h.update(b)
    return h.hexdigest()


def _local_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def _check_encoder_snapshots(cfg):
    """Fail before inference if a pinned local encoder is absent or changed."""
    for kind in ("e5", "user2"):
        entry = cfg["encoders"][kind]
        path = _local_path(entry["local_path"])
        required = ("config.json", "model.safetensors", "tokenizer.json")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Нет файлов модели {kind} в {path}: {', '.join(missing)}. "
                "Запустите python scripts/install_model_archives.py --download"
            )
        if (_sha(path / "model.safetensors") != entry["weights_sha256"] or
                _sha(path / "tokenizer.json") != entry["tokenizer_sha256"]):
            raise ValueError(f"{kind}: локальные веса/tokenizer не совпадают с закреплённым snapshot")


def _ids(series, name):
    if series.isna().any() or not series.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError(f"{name}: ID должны быть непустыми строками; числовое преобразование запрещено")
    if series.duplicated().any():
        raise ValueError(f"{name}: повторяющиеся ID")


def _integer(series, name, minimum=0):
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all() or (values < minimum).any() or (values % 1 != 0).any():
        raise ValueError(f"{name}: нужны целые значения >= {minimum}")
    return values.astype("int64")


def validate_tables(queries, items):
    for frame, required, label in ((queries, QUERY_REQUIRED, "queries"), (items, ITEM_REQUIRED, "items")):
        missing = sorted(set(required) - set(frame.columns))
        if missing: raise ValueError(f"{label}: отсутствуют обязательные колонки {missing}")
    q, it = queries.copy(), items.copy()
    _ids(q.query_id, "query_id"); _ids(it.item_id, "item_id")
    if it.item_id.str.contains(r"\s", regex=True).any():
        raise ValueError("item_id: пробелы несовместимы с форматом answer")
    if q.search_query.isna().any() or not q.search_query.map(lambda x: isinstance(x, str)).all():
        raise ValueError("search_query: нужны строковые значения без NULL")
    for frame, columns in ((q, ["search_category", "search_location_id"]),
                           (it, ["item_category_id", "item_location_id"])):
        for col in columns: frame[col] = _integer(frame[col], col)
    for frame, columns in ((q, ["search_query", "search_infm_params_text"]), (it, TEXT_FIELDS)):
        for col in columns:
            if col not in frame: frame[col] = ""
            frame[col] = frame[col].fillna("").astype(str)
    for col in ("item_latitude", "item_longitude", "item_rating", "item_rating_reviews_count"):
        if col not in it: it[col] = np.nan
    return q.reset_index(drop=True), it.reset_index(drop=True)


def _selected_params(value):
    text = str(value or "")
    kept = []
    boundary = "|".join(re.escape(x) for x in NEXT)
    for key in SERVICE:
        match = re.search(re.escape(key) + r"\s+(.+?)(?=\s+(?:" + boundary + r")\b|$)", text, flags=re.I)
        if match and match.group(1).strip():
            kept.append(f"{key}: {match.group(1).strip(' ,;:')}")
    return "; ".join(kept)


def _clean(value):
    return " ".join(html.unescape(TAG.sub(" ", str(value or ""))).split())


def item_texts(items, kind):
    if kind == "e5":
        return [f"passage: {t[:200]}. {_selected_params(p)[:200]}. {d[:300]}"
                for t, p, d in zip(items.item_title_raw, items.item_infm_params_text, items.item_description_raw)]
    return [f"search_document: TITLE: {_clean(t)}\nPARAMETERS: {_clean(p)}\nDESCRIPTION: {_clean(d)}"
            for t, p, d in zip(items.item_title_raw, items.item_infm_params_text, items.item_description_raw)]


def query_texts(queries, kind):
    prefix = "query: " if kind == "e5" else "search_query: "
    return [prefix + str(q)[:300] + ". " + str(p)[:150]
            for q, p in zip(queries.search_query, queries.search_infm_params_text)]


def _encode(texts, tokenizer, model, kind, window, batch_size, device):
    import torch
    import torch.nn.functional as F
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=window, return_tensors="pt").to(device)
            if kind == "user2":
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = model(**batch).last_hidden_state
                hidden = hidden.float()
            else:
                hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            rows.append(F.normalize(pooled, p=2, dim=1).cpu().numpy().astype("float16"))
            if (start // batch_size) % 100 == 0:
                print(f"{kind} window={window}: {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(rows) if rows else np.empty((0, model.config.hidden_size), np.float16)


def _cached_vectors(items, texts, kind, cfg, cache_dir, model, tokenizer, device, model_path):
    entry = cfg["encoders"][kind]
    signature = {"schema": 1, "kind": kind, "ordered_ids": _digest(items.item_id),
                 "ordered_texts": _digest(texts), "revision": entry["revision"],
                 "window": entry["item_window"], "query_window": entry["query_window"],
                 "pooling": entry["pooling"], "math": entry["math"],
                 "dtype": "float16", "text_recipe": entry["item_text_recipe"],
                 "hidden_size": model.config.hidden_size, "batch_size": entry["item_batch"],
                 "model_weights_sha256": _sha(model_path / "model.safetensors"),
                 "tokenizer_sha256": _sha(model_path / "tokenizer.json")}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{kind}_items.npy"
        manifest = cache_dir / f"{kind}_items.json"
        if path.exists() or manifest.exists():
            if not path.exists() or not manifest.exists() or json.loads(manifest.read_text(encoding="utf8")) != signature:
                raise ValueError(f"Кеш {kind} не соответствует ID, текстам или настройкам: {cache_dir}")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape != (len(items), model.config.hidden_size) or values.dtype != np.float16:
                raise ValueError(f"Некорректная форма или тип кеша {kind}")
            return values
    values = _encode(texts, tokenizer, model, kind, entry["item_window"], entry["item_batch"], device)
    if not np.isfinite(values).all(): raise ValueError(f"{kind}: NaN в embedding")
    if cache_dir is not None:
        tmp = cache_dir / f"{kind}_items.pending.npy"
        np.save(tmp, values)
        os.replace(tmp, path)
        manifest.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf8")
    return values


def _top(scores, indexes, k):
    if len(indexes) == 0: return []
    values = scores[indexes]
    take = np.argpartition(values, -min(k, len(values)))[-min(k, len(values)):]
    take = take[np.argsort(-values[take], kind="stable")]
    return [(int(indexes[j]), float(values[j])) for j in take]


def _haversine(lat1, lon1, lat2, lon2):
    a1, b1, a2, b2 = [np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)]
    d = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((b2-b1)/2)**2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def _geography(items):
    geo = items[["item_location_id", "item_latitude", "item_longitude"]].dropna().copy()
    geo["item_latitude"] = pd.to_numeric(geo.item_latitude, errors="coerce")
    geo["item_longitude"] = pd.to_numeric(geo.item_longitude, errors="coerce")
    geo = geo.dropna()
    geo = geo[geo.item_latitude.between(-90, 90) & geo.item_longitude.between(-180, 180)]
    center = geo.groupby("item_location_id").agg(lat=("item_latitude", "median"),
        lon=("item_longitude", "median"), n=("item_latitude", "size"))
    geo = geo.join(center[["lat", "lon"]], on="item_location_id")
    geo["distance"] = _haversine(geo.lat, geo.lon, geo.item_latitude, geo.item_longitude)
    center["p90_km"] = geo.groupby("item_location_id").distance.quantile(.9)
    center["trusted"] = (center.n >= 10) & (center.p90_km <= 25)
    trusted = center[center.trusted]
    near = {}
    for loc, row in center.iterrows():
        if not row.trusted: near[int(loc)] = []; continue
        distances = _haversine(row.lat, row.lon, trusted.lat.to_numpy(), trusted.lon.to_numpy())
        choices = [(int(other), float(d)) for other, d in zip(trusted.index, distances) if other != loc and d <= 50]
        near[int(loc)] = [x for x, _ in sorted(choices, key=lambda x: (x[1], x[0]))[:5]]
    locrows = {int(k): g.index.to_numpy(dtype=np.int32) for k, g in items.groupby("item_location_id")}
    nearrows = {k: np.concatenate([locrows[n] for n in values]) if values else np.empty(0, np.int32)
                for k, values in near.items()}
    trusted_centers = {int(k): (float(row.lat), float(row.lon)) for k, row in trusted.iterrows()}
    return near, nearrows, trusted_centers


def _round_robin(lists):
    seen, result = set(), []
    for rank in range(max(map(len, lists), default=0)):
        for values in lists:
            if rank < len(values) and values[rank] not in seen:
                seen.add(values[rank]); result.append(values[rank])
    return result


def _source_lists(scores, cat, loc, kind, ids, cats, locs, allrows, catrows, locrows, localrows, nearrows, depths):
    empty = np.empty(0, np.int32)
    category = allrows if cat == 0 else catrows.get(cat, empty)
    local = locrows.get(loc, empty) if cat == 0 else localrows.get((cat, loc), empty)
    glob = _top(scores, allrows, 200)
    cat_hits = _top(scores, category, 100)
    loc_hits = _top(scores, local, 100)
    selected, seen = [], set()
    for hits, quota in ((glob, 100), (cat_hits, 50), (loc_hits, 50), (glob, 200)):
        used = 0
        for j, value in hits:
            if j in seen: continue
            selected.append((j, value)); seen.add(j); used += 1
            if used >= quota or len(selected) >= 200: break
        if len(selected) >= 200: break
    boost = .2 if kind == "e5" else .5
    source = dict(selected)
    neighbor_set = set()
    if kind == "e5":
        nearby = _top(scores, nearrows.get(loc, empty), 50)
        neighbor_set = {j for j, _ in nearby}
        source.update(nearby)
    ordered = sorted(source, key=lambda j: (-source[j] * (1 + .5 * (cat != 0 and cats[j] == cat) + boost * (locs[j] == loc)), ids[j]))
    if kind == "e5":
        reserved = [j for j in ordered if j in neighbor_set][:5]
        reserved_set = set(reserved)
        ordered = reserved + [j for j in ordered if j not in reserved_set]
    base = [ids[j] for j in ordered]
    hits = [_top(scores, allrows, depths["global"]), _top(scores, category, depths["category"]),
            _top(scores, local, depths["local"])]
    if kind == "e5": hits.append(_top(scores, nearrows.get(loc, empty), depths["neighbor_e5"]))
    tail_set = {j for group in hits for j, _ in group}
    return base, tail_set


def _tokenize(value):
    normalized = " ".join(html.unescape(TAG.sub(" ", str(value or ""))).split()).lower().replace("ё", "е")
    return TERM.findall(normalized)


def _bm25_index(items, queries):
    vocab = sorted({term for value in queries.search_query.drop_duplicates() for term in _tokenize(value)})
    lookup = {term: i for i, term in enumerate(vocab)}
    n = len(items)
    built = {}
    for field, name in zip(TEXT_FIELDS, FEATURES[-3:]):
        indptr, indices, data = [0], [], []
        length, df = np.zeros(n, np.int32), np.zeros(len(vocab), np.int32)
        for i, raw in enumerate(items[field]):
            tokens = _tokenize(raw)
            length[i] = len(tokens)
            counts = Counter(lookup[t] for t in tokens if t in lookup)
            for j, tf in sorted(counts.items()):
                indices.append(j); data.append(tf); df[j] += 1
            indptr.append(len(indices))
        matrix = sparse.csr_matrix((np.asarray(data, np.uint32), np.asarray(indices, np.uint32),
                                    np.asarray(indptr, np.int64)), shape=(n, len(vocab)), dtype=np.uint32)
        idf = np.log1p((n - df + .5) / (df + .5)).astype(np.float32)
        built[name] = (matrix, length, idf, float(length.mean()) if n else 0.)
        print(f"BM25 {name}: {n} документов", flush=True)
    return lookup, built


def _bm25_scores(index, qtext, rows, cfg):
    lookup, fields = index
    counted = Counter(lookup[t] for t in _tokenize(qtext) if t in lookup)
    terms = np.asarray(sorted(counted), np.int32)
    qtf = np.asarray([counted[t] for t in terms], np.int32)
    result = []
    for name in FEATURES[-3:]:
        mat, length, idf, avg = fields[name]
        if len(terms) == 0 or avg == 0: result.append(np.zeros(len(rows), np.float32)); continue
        tf = mat[rows][:, terms].toarray().astype(np.float32)
        b, k1 = cfg["bm25"]["b"], cfg["bm25"]["k1"]
        norm = k1 * (1 - b + b * np.asarray(length[rows], np.float32) / avg)
        values = (tf * (k1 + 1)) / (tf + norm[:, None])
        result.append(np.asarray(values @ (np.asarray(idf[terms], np.float32) * np.asarray(qtf, np.float32)), np.float32))
    return result


@lru_cache(maxsize=60000)
def _wordset(value):
    return frozenset(WORD.findall(value.casefold()))


def _features(q, pool, sources, qvectors, ivectors, items, itemcols, geo, index, cfg, idrow):
    near, _, centers = geo
    rows = np.asarray([idrow[x] for x in pool], np.int32)
    cols = itemcols
    cats = cols["item_category_id"]; locs = cols["item_location_id"]
    e5 = np.asarray(ivectors["e5"][rows], np.float32) @ np.asarray(qvectors["e5"], np.float32)
    u2 = np.asarray(ivectors["user2"][rows], np.float32) @ np.asarray(qvectors["user2"], np.float32)
    ranks = sources
    qwords = _wordset(str(q.search_query)); qn = max(1, len(qwords))
    cat, loc = int(q.search_category), int(q.search_location_id)
    nset = set(near.get(loc, []))
    rating = cols["item_rating"]
    reviews = cols["item_rating_reviews_count"]
    bms = _bm25_scores(index, q.search_query, rows, cfg)
    records = []
    for j, ix in enumerate(rows):
        iloc = int(locs[ix]); title = cols["item_title_raw"][ix]
        params = cols["item_infm_params_text"][ix]; desc = cols["item_description_raw"][ix]
        available = loc in centers and iloc in centers
        dist = float(_haversine(*centers[loc], *centers[iloc])) if available else -1.
        r, rv = rating[ix], reviews[ix]
        records.append((float(e5[j]), float(u2[j]), ranks["e5"].get(pool[j], 0),
            ranks["user2"].get(pool[j], 0), int(pool[j] in ranks["e5"]), int(pool[j] in ranks["user2"]),
            int(cat != 0), int(cat != 0 and cat == int(cats[ix])), int(loc == iloc), int(iloc in nset),
            dist, int(available), len(qwords & _wordset(title)) / qn, len(qwords & _wordset(params)) / qn,
            len(qwords), len(str(q.search_query)), len(title), len(params), len(desc),
            0.0 if pd.isna(r) else float(r), int(pd.isna(r)),
            0.0 if pd.isna(rv) else float(np.log1p(max(0, float(rv)))), int(pd.isna(rv)),
            float(bms[0][j]), float(bms[1][j]), float(bms[2][j])))
    frame = pd.DataFrame.from_records(records, columns=FEATURES).astype("float32")
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError(f"{q.query_id}: некорректные признаки")
    return frame


def predict(queries_path, items_path, output_path, config_path, cache_dir=None,
            cache_archive=None, cache_public_url=None, cache_public_path=None):
    """Run frozen inference on new Parquet files; return output Path."""
    # ModernBERT's compile decorators trigger a broken Inductor import on the pinned host.
    import torch
    torch.compile = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn
    from transformers import AutoModel, AutoTokenizer
    from catboost import CatBoostRanker

    cfg = json.loads(Path(config_path).read_text(encoding="utf8"))
    if cfg["features"] != FEATURES or cfg["bm25"] != {"k1": 1.2, "b": 0.25}:
        raise ValueError("Финальный конфиг признаков/BM25 изменён")
    model_path = _local_path(cfg["catboost"]["path"])
    if not model_path.exists() or _sha(model_path) != cfg["catboost"]["sha256"]:
        raise FileNotFoundError(f"Отсутствует или изменена финальная модель CatBoost: {model_path}")
    _check_encoder_snapshots(cfg)
    queries, items = validate_tables(pd.read_parquet(queries_path), pd.read_parquet(items_path))
    print(f"queries={len(queries)} items={len(items)}", flush=True)
    if not torch.cuda.is_available(): raise RuntimeError("Для закреплённого CUDA inference требуется GPU")
    device = "cuda"
    cache = Path(cache_dir) if cache_dir else (_local_path(cfg["cache_dir"]) if cfg.get("cache_dir") else None)
    if (cache_archive or cache_public_url) and cache is None:
        raise ValueError("Для загрузки benchmark-кеша укажите --cache-dir")
    model = CatBoostRanker(); model.load_model(str(model_path))
    if model.tree_count_ != 100 or list(model.feature_names_) != FEATURES:
        raise ValueError("CatBoost: неверные число деревьев или порядок признаков")
    if len(items) == 0:
        output = pd.DataFrame({"query_id": queries.query_id, "answer": [""] * len(queries)})
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False); return output_path
    if cache is not None:
        from solution.benchmark_cache import prepare_cache
        spec_path = _local_path("configs/benchmark_cache.json")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        same_catalog = (len(items) == spec["catalog_items"] and
                        _digest(items.item_id) == spec["ordered_item_ids_sha256"])
        if (cache_archive or cache_public_url) and not same_catalog:
            print("Benchmark cache belongs to another item catalog; recomputing embeddings", flush=True)
        else:
            status = prepare_cache(cache, spec_path,
                archive=Path(cache_archive) if cache_archive else None,
                public_url=cache_public_url, public_path=cache_public_path)
            if status == "missing":
                print("Item cache is absent; computing embeddings", flush=True)
    start = time.perf_counter()
    qvectors, ivectors = {}, {}
    for kind in ("e5", "user2"):
        entry = cfg["encoders"][kind]
        path = _local_path(entry["local_path"])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        encoder = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()
        texts = item_texts(items, kind)
        ivectors[kind] = _cached_vectors(items, texts, kind, cfg, cache, encoder, tokenizer, device, path)
        qvectors[kind] = _encode(query_texts(queries, kind), tokenizer, encoder, kind,
                                 entry["query_window"], entry["query_batch"], device)
        del encoder, tokenizer, texts
        torch.cuda.empty_cache()
    print(f"embeddings: {time.perf_counter()-start:.1f}s", flush=True)
    geo = _geography(items)
    index = _bm25_index(items, queries)
    ids = items.item_id.to_numpy(); cats = items.item_category_id.to_numpy(); locs = items.item_location_id.to_numpy()
    allrows = np.arange(len(items), dtype=np.int32)
    catrows = {int(k): g.index.to_numpy(dtype=np.int32) for k, g in items.groupby("item_category_id")}
    locrows = {int(k): g.index.to_numpy(dtype=np.int32) for k, g in items.groupby("item_location_id")}
    localrows = {(int(c), int(l)): g.index.to_numpy(dtype=np.int32)
                 for (c, l), g in items.groupby(["item_category_id", "item_location_id"])}
    idrow = {x: i for i, x in enumerate(ids)}
    itemcols = {field: items[field].to_numpy() for field in TEXT_FIELDS + ["item_category_id", "item_location_id"]}
    itemcols["item_rating"] = pd.to_numeric(items.item_rating, errors="coerce").to_numpy()
    itemcols["item_rating_reviews_count"] = pd.to_numeric(items.item_rating_reviews_count, errors="coerce").to_numpy()
    depths = cfg["pool"]["channel_depths"]
    # Scores use the same FP32 CUDA matrix multiplication and 64-query blocks as the frozen search.
    matrices = {kind: torch.from_numpy(np.asarray(vec, np.float32)).to(device) for kind, vec in ivectors.items()}
    answers = {}
    for start in range(0, len(queries), 64):
        block = queries.iloc[start:start+64]
        sim = {kind: (torch.from_numpy(np.asarray(qvectors[kind][start:start+len(block)], np.float32)).to(device)
                      @ matrices[kind].T).cpu().numpy() for kind in ("e5", "user2")}
        for offset, q in enumerate(block.itertuples(index=False)):
            cat, loc = int(q.search_category), int(q.search_location_id)
            lists, tails = {}, {}
            for kind in ("e5", "user2"):
                lists[kind], tails[kind] = _source_lists(sim[kind][offset], cat, loc, kind, ids, cats, locs,
                    allrows, catrows, locrows, localrows, geo[1], depths)
            base = _round_robin([lists["e5"], lists["user2"]])[:400]
            excluded = set(base)
            tail_lists = {}
            for kind in ("e5", "user2"):
                scores = sim[kind][offset]; boost = .2 if kind == "e5" else .5
                tail_lists[kind] = [ids[j] for j in sorted((j for j in tails[kind] if ids[j] not in excluded),
                    key=lambda j: (-float(scores[j]) * (1 + .5 * (cat != 0 and cats[j] == cat) + boost * (locs[j] == loc)), ids[j]))]
            pool = (base + [x for x in _round_robin([tail_lists["e5"], tail_lists["user2"]]) if x not in excluded])[:1000]
            if not pool:
                answers[str(q.query_id)] = []; continue
            source_ranks = {}
            for kind in ("e5", "user2"):
                source_ranks[kind] = {v: i for i, v in enumerate(lists[kind], 1)}
                source_ranks[kind].update({v: i for i, v in enumerate(tail_lists[kind], len(lists[kind])+1)})
            features = _features(q, pool, source_ranks, {k: qvectors[k][start+offset] for k in qvectors},
                                 ivectors, items, itemcols, geo, index, cfg, idrow)
            score = model.predict(features[FEATURES])
            ranking = sorted(range(len(pool)), key=lambda j: (-float(score[j]), j))
            answers[str(q.query_id)] = [pool[j] for j in ranking[:50]]
        print(f"ranked {start+len(block)}/{len(queries)}", flush=True)
    output = pd.DataFrame([(qid, " ".join(answers[qid])) for qid in queries.query_id], columns=["query_id", "answer"])
    valid = set(ids)
    if len(output) != len(queries) or output.query_id.duplicated().any() or any(
        len(a.split()) != min(50, len(items)) or len(set(a.split())) != len(a.split()) or not set(a.split()) <= valid
        for a in output.answer):
        raise ValueError("Ответ содержит неверные ID или число кандидатов")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def main():
    p = argparse.ArgumentParser(description="Frozen E5+USER2-D+CatBoost inference")
    p.add_argument("--queries", required=True); p.add_argument("--items", required=True)
    p.add_argument("--output", required=True); p.add_argument("--config", default="configs/final.json")
    p.add_argument("--cache-dir", default=None, help="optional verified item-vector cache")
    p.add_argument("--cache-archive", default=None, help="local verified benchmark cache ZIP")
    p.add_argument("--cache-public-url", default=None,
                   help="optional public Yandex Disk folder or file link for benchmark cache")
    p.add_argument("--cache-public-path", default=None,
                   help="path inside public folder; empty string means direct file link")
    a = p.parse_args()
    print(predict(a.queries, a.items, a.output, a.config, a.cache_dir,
                  a.cache_archive, a.cache_public_url, a.cache_public_path))


if __name__ == "__main__": main()
