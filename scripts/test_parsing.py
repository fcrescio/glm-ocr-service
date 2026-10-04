"""Deterministic parsing API, SDK profile and failure-boundary checks."""
import base64
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from glm_ocr_service.model import InferenceError
from glm_ocr_service.parsing import DocumentParser, LocalRecognitionClient, sdk_config
from glm_ocr_service.server import build_app


def main():
    model = SimpleNamespace(loaded=True, device="GPU", load_s=1, pixel_cap=0)
    parser = SimpleNamespace(parse=lambda url, **kwargs: {"pages": [[{"label": "text", "content": "test"}]]})
    client = TestClient(build_app(model, api_key="test-key", document_parser=parser))
    url = "data:image/png;base64," + base64.b64encode(b"test").decode()
    assert client.post("/v1/parse", json={"document": url}).status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    assert client.post("/v1/parse", headers=headers, json={"document": "file:///secret"}).status_code == 400
    assert client.post("/v1/parse", headers=headers, json={"document": url}).status_code == 200

    def fail(url, **kwargs):
        raise InferenceError("Region OCR truncated at token limit")

    parser.parse = fail
    assert client.post("/v1/parse", headers=headers, json={"document": url}).status_code == 502
    assert TestClient(build_app(model)).post("/v1/parse", json={"document": url}).status_code == 503
    recorder = LocalRecognitionClient(SimpleNamespace(prepare=lambda messages: None,
        loaded=True, infer=lambda *args, **kwargs: SimpleNamespace(finish_reason="length")))
    assert recorder.is_alive()
    with patch("glm_ocr_service.server.normalize_messages", return_value=[]):
        response, status = recorder.process({"messages": [{"role": "user", "content": "Text Recognition:"}],
                                           "max_tokens": 8192, "temperature": 0, "repetition_penalty": 1.1})
    assert status == 500 and "truncated" in response["error"]
    assert recorder.errors
    from glmocr.config import load_config
    config = load_config(mode="selfhosted", layout_device="cpu").pipeline
    assert config.page_loader.pdf_dpi == 200
    assert config.page_loader.max_tokens == 8192
    assert config.page_loader.task_prompt_mapping["table"] == "Table Recognition:"
    assert config.page_loader.repetition_penalty == 1.1
    archive = sdk_config(preserve_marginalia=True)
    assert "header" in archive.layout.label_task_mapping["text"]
    assert "header" not in archive.layout.label_task_mapping["abandon"]
    assert "header" in sdk_config().layout.label_task_mapping["abandon"]
    tuned = sdk_config(preserve_marginalia=True, text_threshold=0.15)
    assert tuned.layout.threshold == 0.3
    assert tuned.layout.threshold_by_class["header"] == 0.15
    assert "table" not in tuned.layout.threshold_by_class
    assert "image" not in tuned.layout.threshold_by_class
    for value in (0, -1, 1.1):
        try:
            sdk_config(text_threshold=value)
            raise AssertionError("Invalid text threshold accepted")
        except ValueError:
            pass
    import threading
    raw = [[{"label": "image", "content": None}]]
    document_parser = DocumentParser.__new__(DocumentParser)
    document_parser._lock = threading.Lock()
    document_parser.recognition = SimpleNamespace(errors=[], usage=[], model=SimpleNamespace(pixel_cap=0))
    document_parser.layout_errors = []
    document_parser.config = archive
    document_parser.preserve_marginalia = True
    result = SimpleNamespace(json_result=raw, raw_json_result=raw, markdown_result="")
    document_parser.pipeline = SimpleNamespace(page_loader=SimpleNamespace(max_tokens=8192), process=lambda *a, **kw: iter([result]))
    assert document_parser.parse(url)["pages"] == raw
    raw[0][0]["label"] = "text"
    try:
        document_parser.parse(url)
        raise AssertionError("Failed textual region was accepted")
    except InferenceError:
        pass
    print("Parsing API, SDK defaults, authentication and truncation checks passed")


if __name__ == "__main__":
    main()
