# UHD 770 profiling and isolated experiments

Baseline: OpenVINO FP16, official SDK layout on CPU, archival marginalia enabled,
200 DPI input pages, global layout threshold 0.3. No NVIDIA inference or vLLM
calls. Source scans and results remain private and are not repository fixtures.

## Decoder compression

`scripts/compress_decoder.py` produced an independent INT8 asymmetric decoder:
approximately 1.1 GB to 557 MB. Vision and embedding components remained FP16.
Measured sequentially on the same normalized input pages and generation settings:

| Case | FP16 total | INT8 total | FP16 decode | INT8 decode |
| --- | --- | --- | --- | --- |
| One-page quotation | 30.78 s | 25.21 s | 18.91 s | 12.69 s |
| One-page economic table | 100.95 s | 79.94 s | 54.53 s | 35.33 s |

The economic table full text was identical. The quotation changed whitespace in
a tax identifier and transcription of an already inaccurate handwritten signature.
Only one run per configuration: this is not a statistically controlled benchmark
or proof of accuracy parity. FP16 remains the production default.

The table multimodal phase remained approximately 38 seconds in both variants;
decoder compression cannot remove that cost. Layout detection took less than one
second. Timing fields measure synchronous Optimum host boundaries, not GPU kernels.

## Layout coverage

Two omitted bill fields were correctly recognized when explicitly cropped. Layout
had not detected either region. Lowering the global threshold introduced low-score
almost-full-page tables, changing segmentation radically. Lowering only textual
class thresholds recovered both regions, leaving table/image thresholds unchanged.

On four upright bill pages, `--layout-text-threshold 0.15` preserved all 16 checked
critical fields (amount, due date, invoice number, account code), and included
supplier and postal account on every page: 24/24 presence checks in total.
Runtime: 93.77 seconds versus 91.14 baseline. Other text and false-positive regions
have not been comprehensively annotated. This remains opt-in, not a new default.

## Verification and next experiments

API contracts, SDK parsing and profiling cleanup tests passed. Production health
reports all four recognition components compiled on GPU.0. CPU-only layout probes
can be reproduced with `scripts/probe_layout.py`; no OCR model is loaded by that
diagnostic. Region confidence and coordinates are available in `/v1/parse` output.

Before promotion: broaden annotated samples, repeat warm measurements, inspect
spurious regions and compare critical numeric cells. Then test INT4 or KV precision
independently; do not simultaneously reduce image resolution and quantize weights.
