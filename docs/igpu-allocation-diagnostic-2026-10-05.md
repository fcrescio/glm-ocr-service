# UHD 770 allocation failures: controlled diagnosis

## Scope and evidence

The previous corpus run mixed genuine extraction failures with a dead or unsafe
runtime. Its 31 successful INT8 pages cannot support a broad quality conclusion.
Use INT8 alone for the next comparison, with one resident model, not two decoder
variants plus the production instance. Dots is a reference, not ground truth.

Private PDFs, exact failed SDK requests, generated text and per-step events live
outside Git in `~/megadoc-ocr-benchmarks/2026-10-05-int8-diagnostic/`.
Every experiment below starts a fresh process. Controls read a synthetic image
containing `TOTAL 123.45` before and after the problematic request. All four
OpenVINO components report `GPU.0`. No vLLM or CPU OCR fallback is involved;
the SDK layout detector still runs on CPU.

## Reproduced causes

OpenVINO reports about 41 GB total memory but only **1,073,741,824 bytes per
allocation**. Total memory is not the single-object limit.

One archived page yields an SDK image of 1652 x 2324, 19,588 vision patches and
a 4,909-token prompt. The following experiments replay its exact saved request:

| Experiment | Outcome | Interpretation |
| --- | --- | --- |
| Unmodified INT8, two repeats | Both fail in vision: 1,534,758,976-byte allocation. Controls still succeed. | `19588 * 19588 * 4`: dense FP32 attention mask exceeds the single-object limit, even with one model. Vision failure alone does not necessarily poison the decoder. |
| `GPU_ENABLE_LARGE_ALLOCATIONS=True` | Reaches the decoder; stopped after approximately seven minutes without completion. | The property bypasses the first limit, but this is **not** a verified OCR solution. It is not enabled in the selected runtime. |
| Compact vision mask only | Fails in decoder: 1,166,221,312-byte allocation. Following control crashes the process (exit 139). | `4909 * 59392 * 4`: FP32 logits for all prompt tokens exceed the same limit. Decoder reuse after failed inference is unsafe. |
| Compact mask only, no KV-state inspection | Same decoder allocation failure, followed by exit 139. | Crash reproduces without the diagnostic reading KV tensors. Two independent fresh processes reproduce this sequence. |
| Compact mask and last-token vocabulary head, two repeats | Both complete GPU generation with a deliberately small 32-token budget; following controls succeed. GPU memory after controls is identical between repeats. | Both allocation barriers are removed without changing the input image. SDK rejection for output truncation is expected here: these are **not** complete OCR extractions. |

### Exact transformations

For a single image, the original mask has no cross-image boundary and is entirely
zero. A `(1, 1, 1)` zero tensor broadcasts to the same values without allocating
the quadratic input mask. Multiple image/video segments retain the original
block-diagonal mask. GPU inference accepts the broadcast shape.

Autoregressive generation consumes the last token's logits. Gather the final
hidden token **before** the vocabulary MatMul, rather than producing logits for
every prompt token and slicing afterward. Transformer computation and KV context
remain intact. The transformation validates the expected graph and fails at
startup on an unsupported export; it does not silently switch behavior.

These changes do not reduce image resolution, truncate the input, quantize extra
components, increase the device allocation limit or substitute another OCR model.
CPU tests verify mask semantics and equality of the transformed vocabulary-head
output with the original last-token output for dynamic batch/sequence sizes.

Full SDK regression on an economic table page (not the artificially limited
full-page diagnostic) succeeded twice with the normal 8192-token budget, in
77.95 and 76.49 seconds. Both Markdown outputs are identical to each other and
to the previously successful unmodified INT8 extraction, including all its
amounts. Controls after both runs succeed; measured post-control GPU allocation
is unchanged between runs. This supports output preservation on that fixture,
not a guarantee for every document or the original large failing page.

## Fail closed after runtime errors

A generation exception marks the runtime unhealthy until process restart.
`/health`, `/v1/models`, parsing and new chat requests return 503. Streaming
exceptions terminate the streamer instead of leaving it blocked. The SDK parser
checks health after regional processing and cannot accept a partial page after
a fatal runtime failure. Output-budget truncation alone does not poison runtime.

This is deliberately conservative: although the vision-only failure left the
control usable, continuing after arbitrary GPU exceptions is not demonstrably
safe. No automatic retry or fallback hides the original failure.

## Reproduction

Never use production traffic concurrently with these deliberate fault injections.
Stop the normal OCR container first and restore it afterward. Keep evidence
outside either repository; a saved SDK payload includes the source image.

```bash
docker compose run --rm --no-deps --entrypoint python \
  -w /app -e PYTHONPATH=/app/src -v "$PWD":/app \
  -v "$HOME/megadoc-ocr-benchmarks":/evidence glm-ocr \
  scripts/reproduce_igpu.py --model-dir /app/models/glm-ocr-int8 \
  --payload /evidence/EXPERIMENT/failed-region-001.json \
  --output /evidence/NEW-CASE --no-kv-inspection
```

Add `--compact-mask` to isolate the second allocation, then
`--last-token-logits` to test both fixes. `--unsafe-after-error` deliberately
bypasses the health latch **only in this diagnostic CLI**, to reproduce unsafe
reuse. Do not use it in normal operation. `--max-tokens 32` verifies input/prefill
allocation, not OCR completeness. For full SDK extraction use `--pdf`, `--page`,
the default 8192 output tokens and no unsafe flag. Exit status is nonzero after
any recorded input/control failure; a segfault terminates earlier with exit 139.

Selected single-resident server:

```bash
docker compose -f docker-compose.yml -f compose.int8.yml up -d --build glm-ocr
curl --fail http://localhost:18030/health
```

Health must report both optimizations true and GPU execution for all components.
The optional override mounts INT8 at the existing model target and keeps the
same endpoint. Normal compose without the override retains the original FP16
configuration. Fingerprint benchmark models with the same two optimization flags.

## Unresolved, not explained away

- The old corpus's reshape/broadcast errors have **not** been reproduced exactly.
  Decoder failure followed by a crash is verified; assigning every historical
  shape error to that cause would exceed the evidence.
- The old USM allocation failures are not all proven to be single-object failures.
  Multiple resident models are a confounder, not a demonstrated cause for each.
- The original 8192-token truncation needs its own text/layout diagnosis. Memory
  changes do not establish that repetition or missed stop tokens are resolved.
- Full-page vision remains expensive. These are correctness/memory fixes, not
  evidence that large-page latency is acceptable.
- A corpus comparison must report coverage and isolated extraction errors as well
  as word/numeric concordance. Agreement with dots is not accuracy.

The new Megadoc runner stops on unavailable/unhealthy backends after persisting
the failed page. Healthy isolated 502 extraction failures remain measurable and
do not turn the rest of the corpus into a cascade of invalid observations.
