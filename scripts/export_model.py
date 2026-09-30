#!/usr/bin/env python
"""Export GLM-OCR to the 4-part OpenVINO layout this service loads.

Uses the OFFICIAL OpenVINO export path: the optimum-intel "GLM-OCR" branch
(the same package that provides this service's runtime — install it per the
README "Environment" section) adds GLM-OCR support to the standard
`optimum-cli export openvino` tooling.

Prerequisites
    * a python 3.11 venv with the GLM-OCR runtime installed
    * GLM-OCR weights — either the Hugging Face model ID (downloaded on
      demand) or a local HF-format directory:
          huggingface-cli download zai-org/GLM-OCR --local-dir <hf-dir>

Usage
    python scripts/export_model.py \
        [--model zai-org/GLM-OCR | <local-hf-dir>] \
        [--output-dir models/glm-ocr-ov] \
        [--weight-format fp16]

What it runs (the official command from the OpenVINO GLM-OCR notebook):
    optimum-cli export openvino -m <model> --task image-text-to-text \
        --weight-format fp16 --trust-remote-code <output-dir>

Output (~2.1–2.7 GB, 4-part layout)
    openvino_language_model.{xml,bin}          stateful LLM (16 layers, GQA 16q/8kv, head_dim 128)
    openvino_vision_embeddings_model.{xml,bin} patch-embed conv
    openvino_vision_embeddings_merger*.{xml,bin} vision blocks + merger
    openvino_text_embeddings_model.{xml,bin}   token embeddings
    (+ OV tokenizer/detokenizer and processor / chat-template files)
    The 4-part runtime keeps its OpenVINO compile cache in <output-dir>/model_cache/
    — keep the directory writable.

On the reference machine (i7-13700K, 8 physical cores) the export takes ~40 s.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PARTS = {
    "openvino_language_model": ["openvino_language_model"],
    "openvino_vision_embeddings_model": ["openvino_vision_embeddings_model"],
    "vision merger": ["openvino_vision_embeddings_merger", "openvino_vision_embeddings_merger_model"],
    "openvino_text_embeddings_model": ["openvino_text_embeddings_model"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="zai-org/GLM-OCR", help="HF model ID or path to a local HF-format model dir")
    ap.add_argument("--output-dir", default="models/glm-ocr-ov")
    ap.add_argument("--weight-format", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="cpu", help="device for the export-time torch reference run")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "-m", args.model,
        "--task", "image-text-to-text",
        "--weight-format", args.weight_format,
        "--trust-remote-code",
        str(out),
    ]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit("export failed (rc=%d)" % r.returncode)

    # verify the 4-part layout (the merger part name varies with exporter
    # version: with or without the trailing "_model")
    missing = [name for name, alts in PARTS.items()
               if not any((out / (a + ".xml")).is_file() for a in alts)]
    if missing:
        sys.exit("export finished but parts are missing: %s (dir: %s)" % (missing, out))
    total_gb = sum(f.stat().st_size for f in out.rglob("*.bin")) / 1e9
    print("OK — 4-part export complete in %s (%.2f GB .bin)" % (out, total_gb))
    print("Start the service with:  python scripts/serve.py --model-dir %s" % out)


if __name__ == "__main__":
    main()
