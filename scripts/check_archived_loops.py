"""Replay the guard against private completed OCR regions and known loop text."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from glm_ocr_service.generation_guard import GenerationGuard


def stopped_at(ids):
    guard = GenerationGuard(0)
    for end in range(512, len(ids) + 1, 16):
        if guard(torch.tensor([ids[:end]])):
            return end, guard.details
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--known-loop", type=Path, required=True)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    regions = pages = 0
    positives = []
    for path in sorted(args.corpus.glob("*/page-*-int8.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        pages += 1
        structured = record["response"]["pages"]
        if isinstance(structured, str):
            structured = json.loads(structured)
        for page in structured:
            for index, region in enumerate(page):
                content = region.get("content")
                if not isinstance(content, str) or not content:
                    continue
                regions += 1
                hit = stopped_at(tokenizer.encode(content, add_special_tokens=False))
                if hit:
                    positives.append({"page": str(path), "region": index, "hit": hit})
    known = stopped_at(tokenizer.encode(args.known_loop.read_text(), add_special_tokens=False))
    print(json.dumps({"pages": pages, "regions": regions, "successful_region_hits": positives,
                      "known_loop_stop": known}, indent=2))
    if positives or not known:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
