# GLM-OCR OCR microservice — OpenAI-compatible API, OpenVINO on Intel iGPU/CPU.
#
# The repo root IS the build context (the repo is the distribution):
#     docker build -t glm-ocr-service .
#
# Run (iGPU, bare-metal Linux; the iGPU render node is /dev/dri/renderD128 on
# Intel machines — adapt if yours differs):
#     docker run -d -p 8080:8080 \
#         --device /dev/dri/renderD128 --group-add 44 \
#         -v /path/to/models/glm-ocr-ov:/models/glm-ocr \
#         glm-ocr-service --device GPU
#
# Run (CPU, no passthrough needed):
#     docker run -d -p 8080:8080 \
#         -v /path/to/models/glm-ocr-ov:/models/glm-ocr \
#         glm-ocr-service --device CPU
#
# The model (~2.7 GB: 4-part fp16 OpenVINO export + OV compile cache) is
# MOUNTED, not baked into the image — keeps the image portable and the model
# out of the layer cache. The mount must be WRITABLE: the 4-part runtime
# keeps its OpenVINO compile cache in <model-dir>/model_cache/ (hardcoded in
# optimum-intel's modeling_base.py). USM shared memory lives in host RAM, so
# do not set a tight --memory cap (leave unset or >= 8 GB).
#
# Verification status (2026-09-30): the full build sequence below was
# executed end-to-end in a fresh python 3.11 venv (torch CPU wheel -> pinned
# base -> --no-deps pinned GLM-OCR branch -> pip install .), and the full
# unit + E2E test suite passed from that environment (live iGPU serve,
# generation, streaming, auth, shutdown). `docker build` itself was not run
# in the originating environment (no docker daemon under WSL2); the base
# image (python:3.11-slim-bookworm) and apt step are the only untested layers.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/glm-ocr-service

# torch + torchvision from the CPU wheel index (the default PyPI torch wheel
# drags in multi-GB nvidia-* libraries this service never uses; torch is
# needed for the processor's return_tensors="pt" and the generation
# streamer, torchvision for the GLM-4V video-processor half of the
# AutoProcessor family). Fallback to the PyPI default wheels if the exact
# +cpu tags are ever missing from the index (larger image, still functional).
RUN pip install torch==2.14.0+cpu torchvision==0.29.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    || pip install torch==2.14.0 torchvision==0.29.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# GLM-OCR 4-part OpenVINO runtime = official optimum-intel "GLM-OCR" branch.
# Its declared dep ranges conflict with the transformers 5.5.4 the GLM
# processor needs, so the exact pinned commits go in with --no-deps; the
# requirements file above is the authoritative environment.
RUN pip install --no-deps \
    "optimum @ git+https://github.com/huggingface/optimum.git@86f0ba219a4e4fd320aab87d60ca9f4773f68078" \
    "optimum-onnx @ git+https://github.com/huggingface/optimum-onnx.git@transformers-v5" \
    "optimum-intel @ git+https://github.com/openvino-dev-samples/optimum-intel.git@ffc9fd3b33b158e07922d839aedb0312a06fb504"

# The service itself: one installable distribution (pyproject.toml at the
# root, package in src/). A plain (non-editable) install puts glm_ocr_service
# in site-packages — no PYTHONPATH, no sys.path hacks.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .

# Launcher (imports glm_ocr_service from the installed distribution).
COPY scripts/serve.py /opt/glm-ocr-serve.py

ENV MODEL_DIR=/models/glm-ocr

VOLUME /models/glm-ocr
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --retries=10 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4); sys.exit(0 if b'\"ok\"' in r.read() else 1)"

ENTRYPOINT ["python", "/opt/glm-ocr-serve.py", "--model-dir", "/models/glm-ocr"]
CMD ["--port", "8080", "--device", "GPU"]
