"""Official SDK layout/cropping/formatting with local OpenVINO recognition."""
import json
import threading
import time

from .model import InferenceError, RuntimeUnhealthyError


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
            start = time.monotonic()
            messages = normalize_messages(ChatCompletionRequest(**payload))
            prepared = self.model.prepare(messages)
            prepare_s = time.monotonic() - start
            result = self.model.infer(
                prepared, payload["max_tokens"], payload["temperature"],
                payload.get("top_p"), repetition_penalty=payload["repetition_penalty"],
            )
            if result.finish_reason == "length":
                raise InferenceError("Region OCR truncated at token limit")
            self.usage.append({"prompt_tokens": result.prompt_tokens,
                               "completion_tokens": result.new_tokens,
                               "seconds": result.e2e_s,
                               "prepare_s": prepare_s,
                               "image_sizes": prepared.image_sizes,
                               "timings": getattr(result, "timings", {})})
            return {"choices": [{"message": {"content": result.text}}]}, 200
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return {"error": str(exc)}, 500


def sdk_config(preserve_marginalia=False, text_threshold=None):
    from glmocr.config import load_config

    config = load_config(mode="selfhosted", layout_device="cpu").pipeline
    config.max_workers = 1
    config.layout.device = "cpu"
    config.layout.cuda_visible_devices = ""
    if preserve_marginalia:
        labels = {"header", "footer", "number", "footnote", "aside_text", "reference"}
        mapping = config.layout.label_task_mapping
        mapping["text"] = list(dict.fromkeys(mapping["text"] + sorted(labels)))
        mapping["abandon"] = [label for label in mapping["abandon"] if label not in labels]
    if text_threshold is not None:
        if not 0 < text_threshold <= 1:
            raise ValueError("text threshold must be in (0, 1]")
        config.layout.threshold_by_class = {
            **(config.layout.threshold_by_class or {}),
            **{label: text_threshold for label in config.layout.label_task_mapping["text"]},
        }
    return config


class DocumentParser:
    def __init__(self, model, preserve_marginalia=False, text_threshold=None):
        from glmocr.pipeline import Pipeline

        config = sdk_config(preserve_marginalia, text_threshold)
        # Keep official task prompts, 200 DPI, 8192 tokens and repetition penalty.
        # The iGPU runtime is single-flight; do not queue 32 concurrent regions.
        self.pipeline = Pipeline(config)
        self.recognition = LocalRecognitionClient(model)
        self.pipeline.ocr_client = self.recognition
        self.layout_errors = []
        layout_process = self.pipeline.layout_detector.process

        def checked_layout(*args, **kwargs):
            start = time.monotonic()
            try:
                result = layout_process(*args, **kwargs)
                self.layout_regions.extend(result[0])
                return result
            except Exception as exc:
                self.layout_errors.append(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                self.layout_s += time.monotonic() - start

        self.pipeline.layout_detector.process = checked_layout
        self.pipeline.start()
        self.config = config
        self.preserve_marginalia = preserve_marginalia
        self._lock = threading.Lock()

    def parse(self, data_url, max_tokens=8192):
        with self._lock:
            if getattr(self.recognition.model, "runtime_error", None):
                raise RuntimeUnhealthyError(self.recognition.model.runtime_error)
            self.pipeline.page_loader.max_tokens = max_tokens
            self.recognition.errors.clear()
            self.recognition.usage.clear()
            self.layout_errors.clear()
            self.layout_s = 0.0
            self.layout_regions = []
            start = time.monotonic()
            results = list(self.pipeline.process({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}]}, save_layout_visualization=False))
            if getattr(self.recognition.model, "runtime_error", None):
                raise RuntimeUnhealthyError(self.recognition.model.runtime_error)
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
            # SDK raw snapshots omit task_type; classify intentional image skips
            # by the native label mapping, not by a missing snapshot field.
            skip_labels = set(self.config.layout.label_task_mapping.get("skip") or [])
            if any(region.get("label") not in skip_labels and region.get("content") is None
                   for page in raw for region in page):
                raise InferenceError("SDK returned failed recognition regions")
            return {"pages": pages, "raw_pages": raw,
                    "markdown": result.markdown_result or "",
                    "usage": list(self.recognition.usage),
                    "layout_s": self.layout_s,
                    "layout_regions": self.layout_regions,
                    "coverage": [{"page_index": i,
                                  "skipped_regions": [region for region in page
                                                      if region.get("label") in skip_labels]}
                                 for i, page in enumerate(raw)],
                    "seconds": round(time.monotonic() - start, 3),
                    "metadata": {"pipeline": "glmocr-sdk", "layout_model": self.config.layout.model_dir,
                                 "layout_device": "cpu", "pdf_dpi": self.config.page_loader.pdf_dpi,
                                 "max_workers": 1, "pixel_cap": self.recognition.model.pixel_cap,
                                 "region_max_tokens": max_tokens,
                                 "layout_threshold": self.config.layout.threshold,
                                 "layout_threshold_by_class": self.config.layout.threshold_by_class,
                                 "compact_vision_mask": getattr(self.recognition.model, "compact_vision_mask", False),
                                 "last_token_logits": getattr(self.recognition.model, "last_token_logits", False),
                                 "preserve_marginalia": self.preserve_marginalia}}
