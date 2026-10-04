"""Official SDK layout/cropping/formatting with local OpenVINO recognition."""
import json
import threading
import time

from .model import InferenceError


class LocalRecognitionClient:
    """SDK OCRClient contract, without a loopback HTTP queue or cloud calls."""

    def __init__(self, model):
        self.model = model
        self.errors = []
        self.usage = []

    def start(self):
        pass

    def stop(self):
        pass

    def is_alive(self):
        return self.model.loaded

    def process(self, payload):
        from .server import ChatCompletionRequest, normalize_messages

        try:
            messages = normalize_messages(ChatCompletionRequest(**payload))
            prepared = self.model.prepare(messages)
            result = self.model.infer(
                prepared, payload["max_tokens"], payload["temperature"],
                payload.get("top_p"), repetition_penalty=payload["repetition_penalty"],
            )
            if result.finish_reason == "length":
                raise InferenceError("Region OCR truncated at token limit")
            self.usage.append({"prompt_tokens": result.prompt_tokens,
                               "completion_tokens": result.new_tokens,
                               "seconds": result.e2e_s})
            return {"choices": [{"message": {"content": result.text}}]}, 200
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return {"error": str(exc)}, 500


class DocumentParser:
    def __init__(self, model):
        from glmocr.config import load_config
        from glmocr.pipeline import Pipeline

        config = load_config(mode="selfhosted", layout_device="cpu").pipeline
        # Keep official task prompts, 200 DPI, 8192 tokens and repetition penalty.
        # The iGPU runtime is single-flight; do not queue 32 concurrent regions.
        config.max_workers = 1
        config.layout.device = "cpu"
        config.layout.cuda_visible_devices = ""
        self.pipeline = Pipeline(config)
        self.recognition = LocalRecognitionClient(model)
        self.pipeline.ocr_client = self.recognition
        self.layout_errors = []
        layout_process = self.pipeline.layout_detector.process

        def checked_layout(*args, **kwargs):
            try:
                return layout_process(*args, **kwargs)
            except Exception as exc:
                self.layout_errors.append(f"{type(exc).__name__}: {exc}")
                raise

        self.pipeline.layout_detector.process = checked_layout
        self.pipeline.start()
        self.config = config
        self._lock = threading.Lock()

    def parse(self, data_url):
        with self._lock:
            self.recognition.errors.clear()
            self.recognition.usage.clear()
            self.layout_errors.clear()
            start = time.monotonic()
            results = list(self.pipeline.process({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}]}, save_layout_visualization=False))
            if self.recognition.errors or self.layout_errors:
                raise InferenceError("; ".join(self.recognition.errors + self.layout_errors))
            if len(results) != 1:
                raise InferenceError("SDK did not return exactly one input result")
            result = results[0]
            pages = result.json_result
            if isinstance(pages, str):
                pages = json.loads(pages)
            if not isinstance(pages, list) or not pages:
                raise InferenceError("SDK returned no pages")
            raw = result.raw_json_result or []
            if any(region.get("task_type") not in ("skip", "abandon") and region.get("content") is None
                   for page in raw for region in page):
                raise InferenceError("SDK returned failed recognition regions")
            return {"pages": pages, "raw_pages": raw,
                    "markdown": result.markdown_result or "",
                    "usage": list(self.recognition.usage),
                    "seconds": round(time.monotonic() - start, 3),
                    "metadata": {"pipeline": "glmocr-sdk", "layout_model": self.config.layout.model_dir,
                                 "layout_device": "cpu", "pdf_dpi": self.config.page_loader.pdf_dpi,
                                 "max_workers": 1, "pixel_cap": self.recognition.model.pixel_cap}}
