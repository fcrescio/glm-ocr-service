"""Model checksum smoke checks without loading any inference runtime."""
import hashlib
from pathlib import Path
import tempfile

from model_fingerprint import fingerprint

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "openvino_language_model.bin").write_bytes(b"decoder")
    (root / "config.json").write_text("{}")
    cache = root / "model_cache"
    cache.mkdir()
    (cache / "compiled.blob").write_bytes(b"ignored")
    result = fingerprint(root)
    assert result["openvino_language_model.bin"]["sha256"] == hashlib.sha256(b"decoder").hexdigest()
    assert result["openvino_language_model.bin"]["bytes"] == 7
    assert set(result) == {"openvino_language_model.bin", "config.json"}
    (root / "openvino_language_model.bin").unlink()
    try:
        fingerprint(root)
        raise AssertionError("Missing decoder accepted")
    except ValueError:
        pass
print("Model fingerprint checks passed")
