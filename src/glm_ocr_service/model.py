"""GLM-OCR OpenVINO session: load once, single-flight inference, token streaming.

Wraps the proven official optimum-intel GLM-OCR export (4-part OV, fp16):
one process, one OpenVINO plugin, all components compiled once. Generation is
serialized with a process-wide lock (the iGPU cannot run two heavy inferences
at once; OV inference is not thread-safe).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from PIL import Image

log = logging.getLogger("glm_ocr_service.model")


class InferenceError(RuntimeError):
    """Raised when preprocessing or OV generation fails."""


@dataclass
class Prepared:
    """Preprocessed prompt ready for generation (thread-safe to pass around)."""

    inputs: dict
    prompt_tokens: int
    n_images: int
    image_sizes: list = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    new_tokens: int
    prompt_tokens: int
    finish_reason: str  # "stop" | "length"
    eos_hit: bool
    e2e_s: float


class _TailStopCriteria:
    """Stop when the decoded tail ends with any of the stop strings.

    Decodes a trailing window (not the whole sequence) each step, which keeps
    per-step cost at O(window) tokens.
    """

    def __init__(self, tokenizer: Any, stops: list, window: int = 96):
        self.tokenizer = tokenizer
        self.stops = [s for s in stops if s]
        self.window = window

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        if input_ids.shape[-1] <= 1:
            return False
        tail = self.tokenizer.decode(
            input_ids[0, -self.window :], skip_special_tokens=True
        )
        return any(tail.endswith(s) for s in self.stops)


class GLMOCRModel:
    """Loaded GLM-OCR OV model with a single-flight generate entry point."""

    def __init__(
        self,
        model_dir: str,
        device: str = "GPU",
        pixel_cap: int = 700_000,
        default_prompt: str = "Text Recognition:",
    ):
        from transformers import AutoProcessor
        from optimum.intel.openvino import OVModelForVisualCausalLM

        self.model_dir = model_dir
        self.device = device
        self.pixel_cap = pixel_cap
        self.default_prompt = default_prompt
        self.loaded = False
        self.load_s = 0.0
        self._lock = threading.Lock()

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = OVModelForVisualCausalLM.from_pretrained(model_dir, device=device)
        self.tokenizer = (
            self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        )
        self.loaded = True
        self.load_s = round(time.time() - t0, 2)
        log.info("GLM-OCR loaded in %.1fs (device=%s, dir=%s)", self.load_s, device, model_dir)

    # ------------------------------------------------------------------ images

    def cap_image(self, img: Image.Image) -> Image.Image:
        """Downscale (never upscale) so w*h <= pixel_cap, dims a multiple of 28."""
        w, h = img.size
        if w * h <= self.pixel_cap:
            return img
        scale = (self.pixel_cap / (w * h)) ** 0.5
        w2 = max(28, int(w * scale) // 28 * 28)
        h2 = max(28, int(h * scale) // 28 * 28)
        return img.resize((w2, h2), Image.LANCZOS)

    def prepare(self, messages: list) -> Prepared:
        """Apply the chat template. `messages` is OpenAI-shaped with image parts
        already resolved to PIL images: {"type":"image","image":PIL} /
        {"type":"text","text":str}. Images are capped to `pixel_cap` here —
        that is what keeps the vision-token count (and prefill cost) at the
        validated level (the GLM processor has no pixel cap of its own).
        """
        capped = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                capped.append(m)
                continue
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image":
                    parts.append({**p, "image": self.cap_image(p["image"])})
                else:
                    parts.append(p)
            capped.append({**m, "content": parts})
        try:
            inputs = self.processor.apply_chat_template(
                capped,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            n_images = sum(
                1
                for m in capped
                for p in (m.get("content") if isinstance(m.get("content"), list) else [])
                if isinstance(p, dict) and p.get("type") == "image"
            )
            return Prepared(
                inputs=dict(inputs),
                prompt_tokens=int(inputs["input_ids"].shape[-1]),
                n_images=n_images,
                image_sizes=[
                    p["image"].size
                    for m in capped
                    for p in (m.get("content") if isinstance(m.get("content"), list) else [])
                    if isinstance(p, dict) and p.get("type") == "image"
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"chat template preparation failed: {exc}") from exc

    # ---------------------------------------------------------------- generate

    @staticmethod
    def _seq_ids(out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "sequences"):
            return out.sequences
        raise InferenceError(f"unexpected generate() output type: {type(out)!r}")

    def _generate(self, prepared: Prepared, kwargs: dict) -> Any:
        return self.model.generate(**prepared.inputs, **kwargs)

    def infer(
        self,
        prepared: Prepared,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        stop: Optional[list] = None,
        stream_cb: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        """Generate under the process-wide lock. With `stream_cb`, decoded text
        chunks are delivered as they are produced; the full sequence is still
        returned for an exact token count.
        """
        with self._lock:
            kwargs: dict = {
                "max_new_tokens": max_tokens,
                "num_beams": 1,
                "do_sample": bool(temperature and temperature > 0),
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = temperature
                if top_p is not None and top_p < 1.0:
                    kwargs["top_p"] = top_p
            if stop:
                from transformers import StoppingCriteriaList

                kwargs["stopping_criteria"] = StoppingCriteriaList([_TailStopCriteria(self.tokenizer, stop)])

            t0 = time.time()
            if stream_cb is None:
                out = self._generate(prepared, kwargs)
                seq = self._seq_ids(out)[0]
                new_ids = seq[prepared.prompt_tokens :]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                n = int(new_ids.shape[-1])
            else:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    decode_kwargs={"skip_special_tokens": True},
                )
                kwargs["streamer"] = streamer
                box: dict = {}

                def _run() -> None:
                    try:
                        box["out"] = self._generate(prepared, kwargs)
                    except Exception as exc:  # noqa: BLE001
                        box["err"] = exc

                th = threading.Thread(target=_run, daemon=True)
                th.start()
                chunks = []
                for chunk in streamer:
                    chunks.append(chunk)
                    stream_cb(chunk)
                th.join()
                if box.get("err") is not None:
                    raise InferenceError(f"generation failed: {box['err']}") from box["err"]
                out = box["out"]
                seq = self._seq_ids(out)[0]
                new_ids = seq[prepared.prompt_tokens :]
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                n = int(new_ids.shape[-1])

            e2e = time.time() - t0
            if n == 0:
                # degenerate: nothing generated (can happen with aggressive stops)
                result = GenerationResult(text, 0, prepared.prompt_tokens, "length", False, e2e)
            else:
                eos_hit = n < max_tokens
                result = GenerationResult(
                    text,
                    n,
                    prepared.prompt_tokens,
                    "stop" if eos_hit else "length",
                    eos_hit,
                    e2e,
                )
            log.info(
                "generate: prompt_tokens=%d new_tokens=%d finish=%s e2e=%.2fs tps=%.2f images=%d",
                result.prompt_tokens,
                result.new_tokens,
                result.finish_reason,
                e2e,
                result.new_tokens / max(e2e, 1e-9),
                prepared.n_images,
            )
            return result
