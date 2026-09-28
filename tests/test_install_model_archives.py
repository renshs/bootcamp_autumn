"""Offline integration checks for the model archive installer."""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from types import ModuleType
import sys

from scripts import install_model_archives as installer


def sha(data):
    return hashlib.sha256(data).hexdigest()


class InstallerTest(unittest.TestCase):
    def test_download_verify_install_and_reject_bad_data(self):
        folder = "sample-model"
        weights, tokenizer = b"test model", b"test tokenizer"
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as zf:
            zf.writestr(f"{folder}/model.safetensors", weights)
            zf.writestr(f"{folder}/tokenizer.json", tokenizer)
        archive_bytes = archive_buffer.getvalue()
        expected = {"model.safetensors": sha(weights), "tokenizer.json": sha(tokenizer)}

        def fake_urlopen(request, timeout):
            url = request.full_url if hasattr(request, "full_url") else request
            if url.startswith(installer.API):
                query = parse_qs(urlparse(url).query)
                self.assertEqual(query["path"], ["/sample.zip"])
                return io.BytesIO(json.dumps({"href": "https://download.test/sample.zip"}).encode())
            self.assertEqual(url, "https://download.test/sample.zip")
            return io.BytesIO(archive_bytes)

        with tempfile.TemporaryDirectory() as tmp, patch.object(installer, "urlopen", fake_urlopen):
            archive = Path(tmp) / "sample.zip"
            destination = Path(tmp) / folder
            installer.download(installer.PUBLIC_URL, archive.name, archive, sha(archive_bytes))
            self.assertEqual(archive.read_bytes(), archive_bytes)
            installer.inspect(archive, folder, expected)
            installer.install(archive, folder, destination)
            self.assertEqual((destination / "model.safetensors").read_bytes(), weights)
            with self.assertRaises(FileExistsError):
                installer.install(archive, folder, destination)
            archive.unlink()
            with self.assertRaises(ValueError):
                installer.download(installer.PUBLIC_URL, archive.name, archive, sha(b"wrong"))
            self.assertFalse(archive.exists())

    def test_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("sample-model/model.safetensors", b"weights")
                zf.writestr("sample-model/tokenizer.json", b"tokenizer")
                zf.writestr("sample-model/../outside", b"bad")
            with self.assertRaisesRegex(ValueError, "unsafe entry"):
                installer.inspect(archive, "sample-model", {
                    "model.safetensors": sha(b"weights"),
                    "tokenizer.json": sha(b"tokenizer"),
                })

    def test_huggingface_staging_and_hash_check(self):
        weights, tokenizer = b"hf weights", b"hf tokenizer"
        calls = []
        fake_hub = ModuleType("huggingface_hub")

        def snapshot_download(*, repo_id, revision, local_dir, allow_patterns):
            calls.append((repo_id, revision))
            self.assertIn("model.safetensors", allow_patterns)
            local_dir.mkdir(parents=True)
            (local_dir / "model.safetensors").write_bytes(weights)
            (local_dir / "tokenizer.json").write_bytes(tokenizer)

        fake_hub.snapshot_download = snapshot_download
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            destination = Path(tmp) / "model"
            entry = {"repo": "test/model", "revision": "fixed-revision"}
            expected = {"model.safetensors": sha(weights), "tokenizer.json": sha(tokenizer)}
            installer.download_huggingface(entry, destination, expected, False)
            self.assertEqual(calls, [("test/model", "fixed-revision")])
            self.assertEqual((destination / "model.safetensors").read_bytes(), weights)
            with self.assertRaises(FileExistsError):
                installer.download_huggingface(entry, destination, expected, False)
            bad_destination = Path(tmp) / "bad-model"
            with self.assertRaises(ValueError):
                installer.download_huggingface(entry, bad_destination,
                                               {"model.safetensors": sha(b"wrong"),
                                                "tokenizer.json": sha(tokenizer)}, False)
            self.assertFalse(bad_destination.exists())


if __name__ == "__main__":
    unittest.main()
