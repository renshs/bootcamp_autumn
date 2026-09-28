"""Download, verify and install the two frozen encoder snapshots.

Use Yandex Disk first and the pinned Hugging Face revision if it is unavailable.
Inference itself remains offline; installed snapshots are never replaced.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_URL = "https://disk.yandex.ru/d/NwVxda2cRzKFqw"
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
HF_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "sentencepiece.bpe.model",
    "config_sentence_transformers.json", "sentence_bert_config.json",
    "modules.json", "1_Pooling/*",
)
ARCHIVES = {
    "e5": ("multilingual_e5_small.zip", "multilingual-e5-small",
           "eeb8701dd368f714a6a3075769c1d3767b36543fb49111c810a398f7e03fb565"),
    "user2": ("USER2_base.zip", "USER2-base",
              "e35d7a3da09420721dbdd48bc7f702ac1b3eaed6b65b5420b085e0d6f6d887ef"),
}


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def download(public_url, filename, archive, expected_hash):
    """Resolve one member of a public folder and download it to a temporary file."""
    query = urlencode({"public_key": public_url, "path": "/" + filename})
    with urlopen(Request(f"{API}?{query}", headers={"Accept": "application/json"}), timeout=30) as response:
        link = json.load(response)["href"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=filename + ".", suffix=".part",
                                     dir=archive.parent, delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        with pending.open("wb") as tmp:
            with urlopen(link, timeout=60) as response:
                shutil.copyfileobj(response, tmp, length=1024 * 1024)
        with pending.open("rb") as stream:
            actual = digest(stream)
        if actual != expected_hash:
            raise ValueError(f"{filename}: downloaded ZIP SHA-256 mismatch: {actual}")
        pending.replace(archive)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Downloaded: {archive}")


def inspect(archive, folder, expected):
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for name, target_hash in expected.items():
            member = f"{folder}/{name}"
            if member not in names:
                raise ValueError(f"{archive}: missing {member}")
            with zf.open(member) as stream:
                actual = digest(stream)
            if actual != target_hash:
                raise ValueError(f"{archive}: SHA-256 mismatch for {member}")
        for info in zf.infolist():
            path = Path(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != folder:
                raise ValueError(f"{archive}: unsafe entry {info.filename!r}")
    print(f"OK: {archive} ({folder}, weights and tokenizer verified)")


def install(archive, folder, destination):
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; refusing to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-install-", dir=destination.parent) as temporary:
        staging = Path(temporary) / folder
        staging.mkdir()
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename.replace("\\", "/")).parts
                target = staging.joinpath(*parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        staging.rename(destination)
    print(f"Installed: {destination}")


def verify_snapshot(destination, expected):
    for name, target_hash in expected.items():
        with (destination / name).open("rb") as stream:
            actual = digest(stream)
        if actual != target_hash:
            raise ValueError(f"{destination / name}: installed SHA-256 mismatch")
    print(f"Verified model: {destination}")


def download_huggingface(entry, destination, expected, verify_only):
    """Fetch one pinned snapshot into staging and publish only verified files."""
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-hf-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        snapshot_download(repo_id=entry["repo"], revision=entry["revision"],
                          local_dir=staging, allow_patterns=HF_FILES)
        verify_snapshot(staging, expected)
        if not verify_only:
            if destination.exists():
                raise FileExistsError(f"{destination} appeared during download")
            staging.rename(destination)
            print(f"Installed from Hugging Face: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models",
                        help="destination for the encoder directories")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--download", action="store_true",
                        help="fetch missing models from Yandex Disk, falling back to Hugging Face")
    parser.add_argument("--source", choices=("auto", "yandex", "huggingface"),
                        default="auto", help="network source when --download is set")
    parser.add_argument("--public-url", default=PUBLIC_URL)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs" / "final.json").read_text(encoding="utf-8"))
    for key, (filename, folder, archive_hash) in ARCHIVES.items():
        archive = args.archive_dir / filename
        entry = cfg["encoders"][key]
        expected = {"model.safetensors": entry["weights_sha256"],
                    "tokenizer.json": entry["tokenizer_sha256"]}
        destination = args.model_dir / folder
        if destination.exists() and not args.verify_only:
            verify_snapshot(destination, expected)
            continue

        if args.source == "huggingface" and args.download:
            download_huggingface(entry, destination, expected, args.verify_only)
            continue

        if not archive.exists() and args.download:
            try:
                download(args.public_url, filename, archive, archive_hash)
            except Exception as error:
                if args.source == "yandex":
                    raise
                print(f"Yandex Disk unavailable for {filename}: {error}; trying Hugging Face")
                download_huggingface(entry, destination, expected, args.verify_only)
                continue
        if not archive.exists():
            raise FileNotFoundError(f"{archive} is missing; add --download to fetch it")
        try:
            with archive.open("rb") as stream:
                actual = digest(stream)
            if actual != archive_hash:
                raise ValueError(f"{archive}: ZIP SHA-256 mismatch: {actual}")
            inspect(archive, folder, expected)
        except Exception:
            if args.source != "auto" or not args.download:
                raise
            print(f"Invalid Yandex ZIP {archive}; trying Hugging Face")
            download_huggingface(entry, destination, expected, args.verify_only)
            continue
        if not args.verify_only:
            install(archive, folder, destination)


if __name__ == "__main__":
    main()
