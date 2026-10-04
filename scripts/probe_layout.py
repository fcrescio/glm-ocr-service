#!/usr/bin/env python3
"""Inspect CPU layout at selected thresholds without loading GLM or invoking OCR."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from glmocr.layout.layout_detector import PPDocLayoutDetector
from glm_ocr_service.parsing import sdk_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", action="append", type=float)
    parser.add_argument("--text-threshold", type=float,
                        help="Override textual classes only; preserve table/image thresholds")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    thresholds = args.threshold or [0.3, 0.2, 0.1]
    if args.output.exists():
        parser.error("output already exists")
    if any(not 0 < value <= 1 for value in thresholds):
        parser.error("threshold must be in (0, 1]")
    if args.text_threshold is not None and not 0 < args.text_threshold <= 1:
        parser.error("text threshold must be in (0, 1]")
    torch.set_num_threads(8)
    detector = PPDocLayoutDetector(sdk_config(preserve_marginalia=True).layout)
    if args.text_threshold is not None:
        detector.threshold_by_class = {label: args.text_threshold
                                      for label in detector.label_task_mapping["text"]}
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    detector.start()
    try:
        results = []
        for threshold in thresholds:
            detector.threshold = threshold
            pages, _ = detector.process(images)
            results.append({"threshold": threshold, "text_threshold": args.text_threshold,
                            "pages": pages})
        args.output.write_text(json.dumps({"images": args.images, "runs": results}, indent=2))
    finally:
        detector.stop()


if __name__ == "__main__":
    main()
