#!/usr/bin/env python
"""Launch the GLM-OCR OpenAI-compatible OCR microservice.

Usage:
    python scripts/serve.py \
        [--model-dir models/glm-ocr-ov] \
        [--device GPU] [--host 0.0.0.0] [--port 8080] \
        [--model-id glm-ocr] [--max-tokens-cap 4096] [--pixel-cap 700000] \
        [--prompt "Text Recognition:"] [--api-key SECRET]

The model is compiled once at startup (GLMOCRModel.__init__ loads all four
OpenVINO components); afterwards every request reuses them. Inference is
single-flight (one iGPU arena at a time; concurrent requests queue).

`--model-dir` is the 4-part OpenVINO model directory produced by
scripts/export_model.py (or a pre-built copy of it). In the container it
defaults to /models/glm-ocr (the mount point).

The GLM_OCR_API_KEY environment variable is picked up as the API key when
--api-key is not given.

Shutdown: SIGTERM/SIGINT force-exit after a 2 s grace period. OpenVINO GPU
plugin threads can keep a naive graceful uvicorn shutdown alive (and
idle-busy), so the service prefers a clean process exit; request arenas are
per-request and there is no state that needs flushing.
"""
import argparse
import logging
import os
import signal
import time

from glm_ocr_service.model import GLMOCRModel
from glm_ocr_service.server import build_app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=os.environ.get("GLM_OCR_MODEL_DIR", "models/glm-ocr-ov"))
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-id", default="glm-ocr")
    ap.add_argument("--max-tokens-cap", type=int, default=8192)
    ap.add_argument("--pixel-cap", type=int, default=0, help="Optional additional image cap; 0 uses the model processor defaults")
    ap.add_argument("--no-layout", action="store_true", help="Run only the low-level OpenAI recognition endpoint")
    ap.add_argument("--preserve-marginalia", action="store_true", help="Recognize textual headers/footers instead of the SDK's default exclusion")
    ap.add_argument("--layout-text-threshold", type=float,
                    help="Experimental textual-class threshold; table/image thresholds unchanged")
    ap.add_argument("--compact-vision-mask", action="store_true",
                    help="Use equivalent broadcast mask for single-image vision attention")
    ap.add_argument("--last-token-logits", action="store_true",
                    help="Generation-only vocabulary head on the final hidden token")
    ap.add_argument("--prompt", default="Text Recognition:")
    ap.add_argument("--no-loop-detection", action="store_true")
    ap.add_argument("--region-timeout", type=float, default=0,
                    help="Cooperative per-region timeout in seconds; 0 disables it")
    ap.add_argument("--failure-dir", help="Private directory for rejected region payloads and partial text")
    ap.add_argument("--api-key", default=os.environ.get("GLM_OCR_API_KEY"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    import math
    if not math.isfinite(args.region_timeout) or args.region_timeout < 0:
        ap.error("region timeout must be nonnegative")
    if args.layout_text_threshold is not None and not 0 < args.layout_text_threshold <= 1:
        ap.error("layout text threshold must be in (0, 1]")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model = GLMOCRModel(
        args.model_dir,
        device=args.device,
        pixel_cap=args.pixel_cap,
        default_prompt=args.prompt,
        compact_vision_mask=args.compact_vision_mask,
        last_token_logits=args.last_token_logits,
        detect_loops=not args.no_loop_detection,
        region_timeout_s=args.region_timeout,
    )
    document_parser = None
    if not args.no_layout:
        from glm_ocr_service.parsing import DocumentParser
        import torch

        torch.set_num_threads(min(8, os.cpu_count() or 1))
        document_parser = DocumentParser(model, preserve_marginalia=args.preserve_marginalia,
                                        text_threshold=args.layout_text_threshold,
                                        failure_dir=args.failure_dir)

    def _force_exit(signum, _frame):
        logging.getLogger("glm_ocr_service").info("signal %d: forcing process exit after grace period", signum)
        time.sleep(2.0)
        os._exit(0)

    signal.signal(signal.SIGTERM, _force_exit)
    signal.signal(signal.SIGINT, _force_exit)

    import uvicorn

    uvicorn.run(
        build_app(model, model_id=args.model_id, api_key=args.api_key, max_tokens_cap=args.max_tokens_cap,
                  document_parser=document_parser),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
