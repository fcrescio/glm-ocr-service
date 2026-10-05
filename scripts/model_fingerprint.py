#!/usr/bin/env python3
"""Freeze model artifacts for paired experiments; never include weights in Git."""
import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(directory):
    suffixes = {".xml", ".bin", ".json", ".jinja", ".model", ".txt"}
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in suffixes:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result[path.name] = {"sha256": digest, "bytes": path.stat().st_size}
    if "openvino_language_model.bin" not in result:
        raise ValueError(f"Missing decoder: {directory}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", required=True, type=Path)
    parser.add_argument("--int8", required=True, type=Path)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-vision-mask", action="store_true")
    parser.add_argument("--last-token-logits", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve() == root or root in args.output.resolve().parents:
        parser.error("Evidence must remain outside the Git checkout")
    if args.output.exists():
        parser.error("Output already exists")
    fp16, int8 = fingerprint(args.fp16), fingerprint(args.int8)
    for name in fp16:
        if name.startswith("openvino_language_model."):
            continue
        if fp16[name] != int8.get(name):
            raise ValueError(f"Non-decoder artifact differs: {name}")
    result = {"runtime_image": args.runtime_image, "variants": {"fp16": fp16, "int8": int8},
              "expected_parse_metadata": {"pipeline": "glmocr-sdk", "layout_device": "cpu",
                  "max_workers": 1, "pixel_cap": 0, "preserve_marginalia": True,
                  "layout_threshold": 0.3, "layout_threshold_by_class": None,
                  "compact_vision_mask": args.compact_vision_mask,
                  "last_token_logits": args.last_token_logits}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Model fingerprints saved to {args.output}")


if __name__ == "__main__":
    main()
