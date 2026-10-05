#!/usr/bin/env python3
"""Fresh-process GPU reproducer with controls, component phases and exact failed payloads."""
import argparse
import base64
import json
from pathlib import Path
import time

import fitz
from PIL import Image, ImageDraw, ImageFont
import torch

from glm_ocr_service.model import GLMOCRModel
from glm_ocr_service.parsing import DocumentParser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--failed-corpus", type=Path, help="Replay failed INT8 pages from a private corpus experiment")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--payload", type=Path, help="Replay one exact SDK recognition request instead of layout")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--large-allocations", action="store_true")
    ap.add_argument("--compact-mask", action="store_true")
    ap.add_argument("--last-token-logits", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Diagnostic output budget; input and prefill remain unchanged")
    ap.add_argument("--unsafe-after-error", action="store_true",
                    help="Diagnostic only: deliberately reuse a failed runtime for the control")
    ap.add_argument("--no-kv-inspection", action="store_true")
    ap.add_argument("--no-loop-detection", action="store_true")
    ap.add_argument("--region-timeout", type=float, default=0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve() == root or root in args.output.resolve().parents:
        ap.error("Evidence must remain outside the Git checkout")
    if args.output.exists():
        ap.error("Output must be new")
    if sum(bool(value) for value in (args.pdf, args.payload, args.failed_corpus)) != 1:
        ap.error("Choose exactly one of --pdf, --payload or --failed-corpus")
    if args.pdf and not args.pdf.is_file():
        ap.error("PDF does not exist")
    if args.payload and not args.payload.is_file():
        ap.error("Payload does not exist")
    targets = [{"pdf": args.pdf, "page": args.page}]
    if args.failed_corpus:
        targets = []
        for path in sorted(args.failed_corpus.glob("*/pair-*.json")):
            record = json.loads(path.read_text())
            if record.get("int8", {}).get("status") == "failed":
                targets.append({"pdf": path.parent / "normalized.pdf", "page": record["page"],
                                "case": record["ocr_result_id"]})
        if not targets:
            ap.error("No failed INT8 pages found")
    if args.repeat < 1 or args.page < 1 or not 1 <= args.max_tokens <= 8192:
        ap.error("Positive repeat/page and max tokens in 1..8192 required")
    import math
    if not math.isfinite(args.region_timeout) or args.region_timeout < 0:
        ap.error("Region timeout must be finite and nonnegative")
    args.output.mkdir(parents=True, mode=0o700)
    torch.set_num_threads(8)
    model = GLMOCRModel(args.model_dir, ov_config={"GPU_ENABLE_LARGE_ALLOCATIONS": True}
                        if args.large_allocations else None, compact_vision_mask=args.compact_mask,
                        last_token_logits=args.last_token_logits)
    model.detect_loops = not args.no_loop_detection
    model.region_timeout_s = args.region_timeout
    parser = DocumentParser(model, preserve_marginalia=True,
                            failure_dir=args.output / "rejected-regions")
    from optimum.intel.openvino.modeling_base import core

    def state():
        result = {"past_length": model.model.language_model._past_length,
                  "gpu_memory": core.get_property("GPU", "GPU_MEMORY_STATISTICS")}
        if args.no_kv_inspection:
            return result
        try:
            result["kv_shapes"] = [(s.name, list(s.state.shape))
                                   for s in model.model.language_model.request.query_state()][:2]
        except Exception as exc:
            result["query_state_error"] = str(exc)
        return result

    events = []
    failed = False
    def emit(event):
        events.append({"time": time.time(), **event})
        (args.output / "events.json").write_text(json.dumps(events, indent=2, default=str))
        print(json.dumps(event, default=str), flush=True)

    emit({"event": "device", "devices": model.execution_devices(),
          "loop_detection": model.detect_loops, "region_timeout_s": model.region_timeout_s,
          "total_mem": core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE"),
          "max_alloc": core.get_property("GPU", "GPU_DEVICE_MAX_ALLOC_MEM_SIZE"), "state": state()})
    for obj, method, phase in ((model.model, "get_multimodal_embeddings", "multimodal"),
                               (model.model.language_model, "forward", "language")):
        original = getattr(obj, method)
        def measured(*a, _original=original, _phase=phase, **kw):
            try:
                return _original(*a, **kw)
            except Exception as exc:
                emit({"event": "component_error", "phase": _phase, "error": str(exc), "state": state()})
                raise
        setattr(obj, method, measured)

    original_prepare = model.prepare
    def prepare(messages):
        result = original_prepare(messages)
        emit({"event": "prepared", "image_sizes": result.image_sizes,
              "prompt_tokens": result.prompt_tokens,
              "input_shapes": {k: list(v.shape) for k, v in result.inputs.items() if hasattr(v, "shape")}})
        return result
    model.prepare = prepare
    original_infer = model.infer
    def infer(*a, **kw):
        result = original_infer(*a, **kw)
        emit({"event": "generation_result", "tokens": result.new_tokens,
              "finish_reason": result.finish_reason, "seconds": result.e2e_s})
        if result.finish_reason == "length":
            (args.output / f"truncated-{len(events)}.txt").write_text(result.text)
        return result
    model.infer = infer
    process = parser.recognition.process
    counter = 0
    def recognition(payload):
        nonlocal counter
        counter += 1
        result = process(payload)
        if result[1] != 200:
            path = args.output / f"failed-region-{counter:03d}.json"
            path.write_text(json.dumps(payload))
            emit({"event": "region_error", "payload_file": str(path), "error": result[0]})
        return result
    parser.recognition.process = recognition

    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 25), "TOTAL 123.45", font=ImageFont.load_default(size=30), fill="black")
    def control(label):
        nonlocal failed
        if args.unsafe_after_error:
            model.runtime_error = None
        try:
            prepared = model.prepare([{"role": "user", "content": [
                {"type": "text", "text": "Text Recognition:"}, {"type": "image", "image": image}]}])
            result = model.infer(prepared, 64)
            emit({"event": label, "status": "ok", "text": result.text, "seconds": result.e2e_s, "state": state()})
        except Exception as exc:
            failed = True
            emit({"event": label, "status": "failed", "error": str(exc), "state": state()})

    control("control_before")
    for iteration, target in enumerate(targets * args.repeat):
        start = time.monotonic()
        emit({"event": "input_start", "iteration": iteration, **target})
        try:
            if args.payload:
                payload = json.loads(args.payload.read_text())
                payload["max_tokens"] = args.max_tokens
                result, status = parser.recognition.process(payload)
                if status != 200:
                    raise RuntimeError(result)
            else:
                with fitz.open(target["pdf"]) as pdf:
                    png = pdf[target["page"]-1].get_pixmap(matrix=fitz.Matrix(200/72, 200/72), alpha=False).tobytes("png")
                result = parser.parse("data:image/png;base64," + base64.b64encode(png).decode(), args.max_tokens)
                (args.output / f"parse-{iteration}.json").write_text(json.dumps(result))
            emit({"event": "input_result", "iteration": iteration, "status": "ok", "seconds": time.monotonic()-start})
        except Exception as exc:
            failed = True
            emit({"event": "input_result", "iteration": iteration, "status": "failed", "error": str(exc), "seconds": time.monotonic()-start})
        control("control_after")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
