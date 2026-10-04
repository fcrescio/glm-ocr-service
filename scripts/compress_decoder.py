#!/usr/bin/env python3
"""Create a separate INT8 decoder variant; preserve vision and source weights."""
import argparse
import shutil
from pathlib import Path

import nncf
import openvino as ov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist; source and baseline are never overwritten")
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        parser.error("output must be outside source")
    model = ov.Core().read_model(source / "openvino_language_model.xml")
    compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("model_cache", "*.blob"))
    ov.save_model(compressed, output / "openvino_language_model.xml")
    print(f"INT8 decoder saved to {output}; other components unchanged")


if __name__ == "__main__":
    main()
