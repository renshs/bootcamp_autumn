"""The inference entry point must report missing or changed models early."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from solution import predict as inference


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ModelPreflightTest(unittest.TestCase):
    def test_missing_and_changed_encoder(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(inference, "ROOT", Path(tmp)):
            cfg = {"encoders": {}}
            for kind in ("e5", "user2"):
                cfg["encoders"][kind] = {
                    "local_path": f"models/{kind}",
                    "weights_sha256": sha(kind.encode()),
                    "tokenizer_sha256": sha(b"tokenizer"),
                }
            with self.assertRaisesRegex(FileNotFoundError, "install_model_archives.py --download"):
                inference._check_encoder_snapshots(cfg)
            for kind in ("e5", "user2"):
                folder = Path(tmp) / "models" / kind
                folder.mkdir(parents=True)
                (folder / "config.json").write_text("{}")
                (folder / "model.safetensors").write_bytes(kind.encode())
                (folder / "tokenizer.json").write_bytes(b"tokenizer")
            inference._check_encoder_snapshots(cfg)
            (Path(tmp) / "models" / "user2" / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "не совпадают"):
                inference._check_encoder_snapshots(cfg)


if __name__ == "__main__":
    unittest.main()
