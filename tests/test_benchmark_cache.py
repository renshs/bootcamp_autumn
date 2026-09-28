"""The benchmark cache is optional, verified and safe to fall back from."""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pandas as pd

from solution import benchmark_cache
from solution import predict as inference


def sha(data):
    return hashlib.sha256(data).hexdigest()


class BenchmarkCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {"e5_items.npy": b"vectors", "e5_items.json": b"manifest",
                      "user2_items.npy": b"vectors2", "user2_items.json": b"manifest2"}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            for name, value in self.files.items():
                bundle.writestr(name, value)
        self.archive_bytes = buffer.getvalue()
        self.archive = self.root / "cache.zip"
        self.archive.write_bytes(self.archive_bytes)
        self.spec = {"archive_name": "cache.zip", "archive_bytes": len(self.archive_bytes),
                     "archive_sha256": sha(self.archive_bytes),
                     "files": {name: sha(value) for name, value in self.files.items()}}
        self.spec_path = self.root / "spec.json"
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.cache = self.root / "cache"

    def test_absent_install_present_and_partial(self):
        self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path), "missing")
        self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path,
                                                       archive=self.archive), "installed")
        self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path), "present")
        for name, value in self.files.items():
            self.assertEqual((self.cache / name).read_bytes(), value)
        (self.cache / "e5_items.npy").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            benchmark_cache.prepare_cache(self.cache, self.spec_path)

    def test_bad_archive_never_installs(self):
        self.archive.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "SHA-256/size mismatch"):
            benchmark_cache.prepare_cache(self.cache, self.spec_path, archive=self.archive)
        self.assertFalse(self.cache.exists())

    def test_missing_archive_recomputes(self):
        self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path,
            archive=self.root / "not-uploaded.zip"), "missing")

    def test_network_failure_recomputes_and_good_download_installs(self):
        with patch.object(benchmark_cache, "_download_public", side_effect=URLError("offline")):
            self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path,
                public_url="https://disk.yandex.ru/d/example"), "missing")
        self.assertFalse(self.cache.exists())

        def fake_download(url, public_path, target):
            self.assertEqual(public_path, "/cache.zip")
            target.write_bytes(self.archive_bytes)

        with patch.object(benchmark_cache, "_download_public", side_effect=fake_download):
            self.assertEqual(benchmark_cache.prepare_cache(self.cache, self.spec_path,
                public_url="https://disk.yandex.ru/d/example"), "installed")

    def test_missing_vectors_are_recomputed_and_then_reused(self):
        model_path = self.root / "model"
        model_path.mkdir()
        (model_path / "model.safetensors").write_bytes(b"model")
        (model_path / "tokenizer.json").write_bytes(b"tokenizer")
        items = pd.DataFrame({"item_id": ["a", "b"]})
        cfg = {"encoders": {"e5": {"revision": "fixed", "item_window": 128,
            "query_window": 128, "pooling": "mean_l2", "math": "fp32",
            "item_text_recipe": "test", "item_batch": 2}}}
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float16)
        with patch.object(inference, "_encode", return_value=vectors) as encode:
            result = inference._cached_vectors(items, ["one", "two"], "e5", cfg,
                self.cache, model, None, "cpu", model_path)
            self.assertTrue(np.array_equal(result, vectors))
            encode.assert_called_once()
        with patch.object(inference, "_encode", side_effect=AssertionError("should use cache")):
            result = inference._cached_vectors(items, ["one", "two"], "e5", cfg,
                self.cache, model, None, "cpu", model_path)
            self.assertTrue(np.array_equal(result, vectors))


if __name__ == "__main__":
    unittest.main()
