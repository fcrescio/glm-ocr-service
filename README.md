# GLM-OCR OCR microservice

An OpenAI-compatible OCR service that runs **GLM-OCR** (Zai, ~1.3B) on the
**Intel iGPU via OpenVINO** (fp16, 4-part stateful export). You send a
document page image, you get Markdown back (text, LaTeX math, HTML tables)
in one request — the same contract as `/v1/chat/completions`, so any OpenAI
client works by swapping the base URL.

## What it is

- **Model**: GLM-OCR (`zai-org/GLM-OCR` on Hugging Face), the
  image-text-to-text variant of the GLM-4V family: one page in (up to 32
  images per request), one structured document out.
- **Runtime**: the official OpenVINO 4-part export (stateful language model
  with device-resident KV cache, vision embeddings, text embeddings, vision
  merger). One process, one OpenVINO plugin, one GPU arena — inference is
  single-flight: concurrent requests queue and each runs to completion.
- **API**: `POST /v1/chat/completions` (non-stream and SSE streaming),
  `GET /v1/models`, `GET /health`; OpenAI-shaped responses and error bodies;
  optional API key.

## Measured performance (reference machine)

Measured on an Intel **UHD 770 iGPU** (Xe-LP, 32 EUs) with an i7-13700K
host, WSL2, greedy decoding, 700k-pixel input cap, `max_tokens` 2048:

| page type (real document pages) | E2E (warm) | decode rate |
|---|---|---|
| Chinese text page (~210 output tokens) | ~18–22 s | ~10–11 tok/s |
| mixed English/math page (~800–1,550 output tokens) | ~50–90 s | ~8–13 tok/s |

- Cold first request adds ~5–9 s (model load + OpenVINO compilation);
  the compile cache (`<model-dir>/model_cache/`) makes subsequent starts
  faster.
- The iGPU decodes at ~8–13 tok/s (Xe-LP hardware limit for a 1.3B fp16
  decoder); a CPU host of the same class runs the same model at ~14–15
  tok/s for short prompts, so on a *weak* iGPU the CPU can be competitive —
  the iGPU remains the default because it leaves the CPU free and scales
  better on stronger discrete Xe GPUs. Choose `--device CPU` if you prefer.
- Idle CPU usage of the running service is ~0% (verified).

These numbers are specific to that hardware class; expect proportionally
different figures elsewhere.

## Repository layout

```
.
├── pyproject.toml              # the distribution (glm-ocr-service)
├── requirements.txt            # pinned base environment (venv or container)
├── src/glm_ocr_service/
│   ├── model.py                # OV session: load once, single-flight generate, streaming
│   └── server.py               # FastAPI app: /v1/chat/completions, /v1/models, /health
├── scripts/
│   ├── export_model.py         # produce the 4-part OV model (official export path)
│   ├── serve.py                # service launcher
│   └── test_service.py         # unit (no model) + E2E (live server) test suite
├── docker/
│   ├── Dockerfile              # container build (repo root is the context)
│   └── run.sh                  # build / run / stop helpers
└── tests/                      # test artifacts (log files; gitignored)
```

## Model acquisition

The model is **not** in this repository (≈2.7 GB). Produce it once:

```bash
# 1. the HF weights (or use a local copy of zai-org/GLM-OCR)
huggingface-cli download zai-org/GLM-OCR --local-dir models/GLM-OCR-hf

# 2. the 4-part OpenVINO export (official path, ~40 s on an 8-core CPU)
python scripts/export_model.py --model models/GLM-OCR-hf --output-dir models/glm-ocr-ov
```

`export_model.py` runs the standard optimum-intel OpenVINO exporter
(`optimum-cli export openvino --task image-text-to-text --weight-format
fp16`) provided by the GLM-OCR branch of optimum-intel (below) and verifies
the 4-part layout in `models/glm-ocr-ov/`:

```
openvino_language_model.{xml,bin}          # stateful LLM (16 layers, GQA 16q/8kv)
openvino_vision_embeddings_model.{xml,bin}
openvino_vision_embeddings_merger.{xml,bin}
openvino_text_embeddings_model.{xml,bin}
+ processor / tokenizer / chat-template files
```

Keep `models/glm-ocr-ov/` writable: the runtime keeps its OpenVINO compile
cache in `models/glm-ocr-ov/model_cache/`.

## Environment (manual venv)

Python 3.11. Install **in this order**:

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# 1. CPU torch + torchvision from the wheel index (never touches CUDA; torch
#    is needed for the processor's return_tensors="pt" and the streamer,
#    torchvision for the GLM-4V video-processor half of the processor family)
pip install torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. the pinned base environment
pip install -r requirements.txt

# 3. the GLM-OCR runtime: official optimum-intel "GLM-OCR" branch.
#    Installed --no-deps because its declared dependency ranges predate the
#    transformers 5.5.4 the GLM processor requires — the pinned set above
#    remains authoritative.
pip install --no-deps \
  "optimum @ git+https://github.com/huggingface/optimum.git@86f0ba219a4e4fd320aab87d60ca9f4773f68078" \
  "optimum-onnx @ git+https://github.com/huggingface/optimum-onnx.git@transformers-v5" \
  "optimum-intel @ git+https://github.com/openvino-dev-samples/optimum-intel.git@ffc9fd3b33b158e07922d839aedb0312a06fb504"

# 4. the service itself
pip install -e .
```

The same sequence, containerized, is `docker/Dockerfile`.

## Running

```bash
python scripts/serve.py --model-dir models/glm-ocr-ov --device GPU --port 8080
```

Options: `--device GPU|CPU`, `--host`, `--port`, `--model-id`,
`--max-tokens-cap` (default 8192), `--pixel-cap` (default 0: no extra downscaling;
the model processor's own limits still apply), `--no-layout` (disable the document
parser explicitly), `--prompt` (default `"Text Recognition:"`), `--api-key`
(or the `GLM_OCR_API_KEY` env var). `SIGTERM`/`SIGINT` force a clean process
exit after a 2 s grace.

## Standard document parsing

The default server loads the official `glmocr` SDK pinned to commit
`cef4d0ea120d1741f5cefe8985eee45f6c8eff1d`. It uses PP-DocLayoutV3 on **CPU**,
region cropping, official text/table/formula prompts, PDF rendering at **200 DPI**,
8192 tokens per region and repetition penalty 1.1. Recognition remains OpenVINO
on **Intel GPU**; CUDA is not used. `max_workers=1` is intentional: the iGPU runtime
is single-flight. This is quality alignment, not a claim to reproduce NVIDIA
benchmark throughput. The SDK's default header/footer filtering is unchanged.

`POST /v1/parse` accepts JSON:

```json
{"model":"glm-ocr","document":"data:application/pdf;base64,...","max_tokens":8192}
```

Images are accepted too. Only inline data URLs are accepted by this endpoint,
with a 40 MB decoded input limit. For long PDFs, callers should split pages to
bound request time. Authentication is the same as the OpenAI endpoint.

The response contains `pages` (one list of SDK regions per page), `raw_pages`,
`markdown`, per-region `usage`, elapsed `seconds`, and configuration `metadata`.
Regions retain labels, reading order and `bbox_2d` coordinates normalized to
0..1000. Table regions contain HTML, including spans. Image-region references
in Markdown are SDK references, not independently served crop files; use their
coordinates and the original document to inspect them.

Recognition calls the local model directly through the SDK client contract;
no loopback HTTP, cloud calls or alternate OCR fallback. Layout exceptions,
recognition errors and token-limit truncation fail the document request rather
than silently returning an incomplete success. `/health` reports whether parsing
is enabled and the compiled OpenVINO devices. Layout weights are cached separately
in the Compose `models/hf-cache` volume.

Verification commands (no model needed):

```bash
docker run --rm --entrypoint python -v "$PWD/scripts:/tests" glm-ocr-service /tests/test_service.py
docker run --rm --entrypoint python -v "$PWD/scripts:/tests" glm-ocr-service /tests/test_parsing.py
```

Historical timings above used the old 700k input cap and page-only prompting;
they must not be presented as timings of the SDK pipeline. Compare source-backed
cells and layout alongside latency. Keep private PDFs and raw OCR outputs out of Git.

## Deploying as a container

### WSL2 Docker deployment verified on 2026-10-04

The root `docker-compose.yml` exposes port 18030, passes `/dev/dxg`, and mounts
both `/usr/lib/wsl/lib` and `/usr/lib/wsl/drivers`. The latter is required for
Intel's Windows user-mode driver: omitting it can crash device enumeration.
The image includes `intel-opencl-icd` and `libze1`. OpenVINO detects the Intel
Raptor Lake iGPU (`0xa780`) inside the container; the NVIDIA GPU is not used.

```bash
docker compose up -d --build
curl http://localhost:18030/health
```

This compose is specific to WSL2. On bare-metal Linux use the render-node
passthrough command below. Model files remain mounted outside the image.

```bash
docker build -t glm-ocr-service .        # from the repo root (or: docker/run.sh build)
```

Run (iGPU — bare-metal Linux; pass through the iGPU render node and the
video group):

```bash
docker run -d -p 8080:8080 \
    --device /dev/dri/renderD128 --group-add 44 \
    -v /path/to/models/glm-ocr-ov:/models/glm-ocr:rw \
    glm-ocr-service --device GPU
```

Run (CPU — no passthrough needed):

```bash
docker run -d -p 8080:8080 \
    -v /path/to/models/glm-ocr-ov:/models/glm-ocr:rw \
    glm-ocr-service --device CPU
```

or simply `./docker/run.sh run [--device GPU|CPU]`.

Operational notes:

- The model is **mounted, not baked** — the image stays portable and
  model updates are a mount change, not a rebuild. The mount must be
  writable (compile cache).
- OpenVINO USM (shared-memory) allocations live in **host RAM**: do not
  set a tight `--memory` cap (leave unset or ≥ 8 GB).
- The image has a `HEALTHCHECK` on `/health`; the first request after start
  pays any residual compilation time.
- **WSL2 caveat (measured on the originating machine):** WSL2 exposes the
  iGPU as a virtualized pool with a fluctuating size (~3.3–4 GB); OpenVINO
  GPU *can* run under it, but GPU pass-through from a WSL2 container is
  *not* the supported path — for WSL2 run the service natively in the
  WSL2 venv (the `--device GPU` venv path is the one that is measured
  here). On bare-metal Linux, `--device /dev/dri/...` is the standard
  iGPU container path.

## API

Base URL: `http://<host>:8080`. Auth (when a key is set):
`Authorization: Bearer <key>` or `x-api-key: <key>`; a missing/wrong key
gets an OpenAI-shaped `401`.

### Non-streaming

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-ocr",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 page.png)"'"}},
      {"type": "text", "text": "Text Recognition:"}
    ]}],
    "temperature": 0,
    "max_tokens": 2048
  }'
```

Returns the standard `chat.completion` object; `choices[0].message.content`
is the Markdown document; `usage` has real token counts.

### Streaming

Same body with `"stream": true` (optionally
`"stream_options": {"include_usage": true}`): `text/event-stream` with
`chat.completion.chunk` objects, first delta carries `role`, final chunk
carries `finish_reason`, optional trailing usage chunk, terminated by
`data: [DONE]`.

### Image inputs

`image_url` accepts: `data:<mime>;base64,…` (primary), `http(s)://` (fetched
server-side), `file://`, or a bare absolute path (server-local). Limits:
40 MB per image, 32 images per request; oversized inputs are downscaled to
the pixel cap (never upscaled). Text-only requests are allowed (plain LLM
use); OCR needs at least one image.

### Errors

OpenAI-shaped: `{"error": {"message", "type", "code"}}` with
`401` authentication, `400` bad request (bad image URL, undecodable image
is `415`, missing file, empty messages), `404` unknown model, `413` image
too large, `500` inference failure.

## Testing

```bash
python scripts/test_service.py                 # unit: contract/auth/errors (no model needed)
python scripts/test_service.py --e2e \
    --model-dir models/glm-ocr-ov --device GPU # live server: OCRs a synthetic page
                                              # (known text), streams, checks parity
                                              # and clean shutdown
```

The E2E suite renders its own test page (English lines always; a CJK line
when a CJK font is installed), asserts the model reads it, checks that
streamed text equals non-streamed text, and verifies SIGTERM shutdown with
no leaked processes. Pass `--image /path/to/real_page.png` to OCR a real
page instead.

## Model & software license

- The service code is provided as-is.
- The GLM-OCR model weights are published by Zai on Hugging Face
  (`zai-org/GLM-OCR`) under **that repository's license — review it before
  commercial use**.
