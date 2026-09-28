"""Optional verified distribution of the original benchmark item embeddings.

The cache is an optimization for one exact item catalog, not a model input.
Network access occurs only when a public URL is explicitly passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_public(public_url: str, public_path: str | None, archive: Path) -> None:
    params = {"public_key": public_url}
    if public_path is not None:
        params["path"] = public_path
    request = Request(f"{API}?{urlencode(params)}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        href = json.load(response)["href"]
    with urlopen(href, timeout=60) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)


def install_verified_cache(archive: Path, cache_dir: Path, spec: dict) -> None:
    """Validate the complete bundle before publishing any of its four files."""
    if archive.stat().st_size != spec["archive_bytes"] or _sha(archive) != spec["archive_sha256"]:
        raise ValueError(f"Benchmark cache archive SHA-256/size mismatch: {archive}")
    expected = spec["files"]
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-cache-", dir=cache_dir.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            if set(bundle.namelist()) != set(expected):
                raise ValueError("Benchmark cache archive contains unexpected or missing files")
            for name, digest in expected.items():
                target = staging / name
                with bundle.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                if _sha(target) != digest:
                    raise ValueError(f"Benchmark cache member SHA-256 mismatch: {name}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if any((cache_dir / name).exists() for name in expected):
            raise FileExistsError(f"Benchmark cache appeared during installation: {cache_dir}")
        for name in expected:
            os.replace(staging / name, cache_dir / name)


def prepare_cache(cache_dir: Path, spec_path: Path, *, archive: Path | None = None,
                  public_url: str | None = None, public_path: str | None = None) -> str:
    """Return present, installed or missing; a missing cache may be recomputed.

    A partial, mismatched or corrupt cache is an error so that inference never
    silently uses or replaces questionable vectors.
    """
    if archive is not None and public_url is not None:
        raise ValueError("Choose either a local cache archive or a public URL")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    names = tuple(spec["files"])
    existing = [name for name in names if (cache_dir / name).exists()]
    if len(existing) == len(names):
        return "present"
    if existing:
        raise ValueError(f"Incomplete benchmark cache in {cache_dir}: {existing}")
    if archive is not None:
        if not Path(archive).is_file():
            print(f"Benchmark cache archive is absent ({archive}); recomputing item embeddings", flush=True)
            return "missing"
        install_verified_cache(Path(archive), cache_dir, spec)
        return "installed"
    if public_url is None:
        return "missing"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    path = public_path if public_path is not None else "/" + spec["archive_name"]
    with tempfile.TemporaryDirectory(prefix="benchmark-download-", dir=cache_dir.parent) as temporary:
        downloaded = Path(temporary) / spec["archive_name"]
        try:
            _download_public(public_url, path or None, downloaded)
        except (OSError, URLError, TimeoutError, KeyError, ValueError) as error:
            print(f"Benchmark cache download unavailable ({error}); recomputing item embeddings", flush=True)
            return "missing"
        install_verified_cache(downloaded, cache_dir, spec)
    return "installed"
