# OCR generation guard

## Behavior

Generation has a cooperative stopping criterion, checked inside the decoder.
It excludes prompt tokens and examines only newly generated tokens. Every 16
tokens, after at least 512 output tokens, it checks for an exact periodic suffix:
period 1..128 tokens, at least 32 consecutive copies and at least 512 tokens
covered. Only the latest 4096 tokens are inspected. Different numbers or words
within repeated HTML markup break the exact match. The token check does not
decode and compare the growing full output on every generation step.

This is a conservative resource guard, not semantic proof of an infinite loop.
An unusual legitimate document with long identical rows can still trigger it.
Rejected text remains evidence for review, never an authoritative extraction.
Non-periodic hallucinations and periods longer than 128 tokens are not detected.

An independent optional deadline bounds the region at decoder-step boundaries.
The INT8 Compose override sets it to 600 seconds; the generic launcher defaults
to zero (disabled). The clock includes vision and prefill, but the criterion
cannot interrupt a running native GPU call. It is not a hard watchdog for a hung
driver. An HTTP timeout alone likewise does not cancel GPU generation.
The previous successful corpus includes individual regions lasting 315.77 and
323.07 seconds: a blanket 300-second deadline would reject known completed
outputs. The 600-second setting preserves a margin rather than conflating large
valid regions with detected token loops.

## Failure contract

- The normal generation path returns `generation_loop` or `generation_timeout`,
  with token count, elapsed time and repetition details. This is a clean stop,
  not an exception inside the OpenVINO execution request.
- Parsing returns HTTP 502 with the corresponding error code. A page containing
  a rejected region is not accepted, even if other regions succeeded.
- Low-level non-streaming chat also returns 502. Streaming emits an explicit
  error event; previously streamed text must be treated as incomplete/invalid.
- Runtime remains healthy; the next request can proceed normally. Actual GPU
  exceptions still latch the separate unhealthy-runtime protection.
- No retry of the same loop is introduced. Megadoc retains the page number and
  error message and fails the ingestion job rather than silently retrying it.
- Original documents and old corpus results are unchanged.

## Private evidence and settings

The INT8 override enables detection and persists rejected region evidence in
the `ocr-failures` Docker volume, under `/var/lib/glm-ocr-failures/regions`.
Each uniquely named JSON file contains the original SDK recognition request
(including the crop image), partial output, stop reason, token count, dimensions
and elapsed time. Files have mode 0600; the newly created region directory has
mode 0700. These are private data, not source files or Git fixtures. No automatic
purge is performed. Use Docker volume backup/export procedures when preserving
these diagnostics; deleting the volume deletes this evidence.

Generic launcher options:

```bash
--no-loop-detection             # diagnostic comparison only
--region-timeout 600            # seconds; 0 disables the cooperative deadline
--failure-dir /PRIVATE/PATH     # save rejected region evidence
```

Health and successful parse metadata expose detection and timeout settings.
The model fingerprint CLI accepts the same guard switches for future benchmark
manifests. Do not merge results collected with different guard policies.

## Verification

Deterministic tests cover exact repetition, varied row values, prompt exclusion,
disabled detection, a controlled clock deadline, parser rejection, private file
permissions and explicit errors on both normal and streaming API paths.

```bash
PYTHONPATH=src python scripts/test_generation_guard.py
PYTHONPATH=src python scripts/test_parsing.py
PYTHONPATH=src python scripts/test_runtime_failure.py
```

`scripts/check_archived_loops.py` re-tokenizes the stored recognized region text
with the model tokenizer and replays the criterion. On the completed INT8 corpus:
**316 successful pages / 5035 text or table regions / zero detections**. The known
archived loop is detected at 560 re-tokenized tokens. This is an offline regression
on stored text, not a fresh inference of all 316 pages or a guarantee of no false
positives on other archives.

Live replay of the exact known region request stopped twice at 816 output tokens,
in 52.64 and 49.09 seconds instead of the prior 446.70 seconds. Controls following
both rejected generations succeed. Live tokens need not match re-tokenized
archived text; the actual stop evidence is saved separately.

The reproducer also accepts `--failed-corpus /PRIVATE/CORPUS`, replaying every
failed INT8 page via the full SDK, with archived normalization and 200 DPI.
Use a new private output directory. Its nonzero exit status is expected when
guarded pages remain rejected: that must not be reported as successful OCR.

## Five-page live regression (2026-10-05)

All five failed corpus pages were replayed through the full SDK. Each now stops
with `generation_loop`, not `length` or the deadline. Every following synthetic
control succeeds; the runtime remains usable. Ordered as in the reproducer:

| Replay | Previous page seconds | Guarded page seconds | Tokens at stop | Detected period |
| --- | ---: | ---: | ---: | ---: |
| 1 | 529.86 | 134.26 | 1712 | 48 |
| 2 | 446.78 | 74.20 | 976 | 1 |
| 3 | 741.67 | 236.44 | 688 | 11 |
| 4 | 695.80 | 219.00 | 1008 | 28 |
| 5 | 457.68 | 50.07 | 560 | 15 |

Total page request time falls from 2871.79 to 713.97 seconds: **47m52s to 11m54s,
about 75% less time**. Controls and model startup are excluded from both totals.
The two large scans still spend substantial time in vision/prefill before token
generation can be stopped. These are five cheaper failures, not five corrected
or accepted OCR outputs.

Evidence is private under `~/megadoc-ocr-benchmarks/2026-10-05-loop-guard-five-pages/`
and `2026-10-05-loop-guard-known/`: events, exact rejected region requests and
partial text. Original corpus checkpoints and database OCR were not changed.
