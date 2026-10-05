"""Fault injection: failed inference must not reuse untrusted runtime state."""
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
from glm_ocr_service.model import GLMOCRModel, Prepared, InferenceError, RuntimeUnhealthyError
from glm_ocr_service.server import build_app
from test_service import StubModel


class FailingGeneration:
    def __init__(self):
        self.calls = 0
        self.language_model = SimpleNamespace(forward=lambda *a, **kw: None)

    def get_multimodal_embeddings(self, *a, **kw):
        return None

    def generate(self, **kwargs):
        self.calls += 1
        raise RuntimeError("Injected OpenVINO inference failure")


for streaming in (False, True):
    model = GLMOCRModel.__new__(GLMOCRModel)
    model._lock = threading.Lock()
    model.runtime_error = None
    model.model = FailingGeneration()
    model.tokenizer = SimpleNamespace(decode=lambda *a, **kw: "")
    prepared = Prepared({}, 1, 0)
    try:
        model.infer(prepared, 8, stream_cb=(lambda value: None) if streaming else None)
        raise AssertionError("Inference failure hidden")
    except InferenceError:
        pass
    assert model.runtime_error is not None
    try:
        model.infer(prepared, 8)
        raise AssertionError("Unhealthy runtime reused")
    except RuntimeUnhealthyError:
        pass
    assert model.model.calls == 1

stub = StubModel()
stub.runtime_error = "Injected GPU failure"
client = TestClient(build_app(stub))
for path in ("/health", "/v1/models"):
    assert client.get(path).status_code == 503
assert client.post("/v1/parse", json={}).status_code == 503
assert client.post("/v1/chat/completions", json={"model": "glm-ocr", "messages": [
    {"role": "user", "content": "test"}]}).status_code == 503
print("Unhealthy runtime isolation, streaming failure and API 503 contracts passed")
