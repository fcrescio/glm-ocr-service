#!/usr/bin/env bash
# Build and run the GLM-OCR service container.
#
#   ./docker/run.sh build                     # docker build -t glm-ocr-service .
#   ./docker/run.sh run [--device GPU|CPU] [extra docker run flags...]
#   ./docker/run.sh run --device GPU --model-dir /path/to/4-part-ov -p 8080:8080
#   ./docker/run.sh stop
#
# The iGPU render node defaults to /dev/dri/renderD128 (typical Intel
# machines); override with RENDER_NODE. --group-add 44 (video group) is
# passed for GPU; override with VIDEO_GID if your box differs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-glm-ocr-service}"
MODEL_DIR="${MODEL_DIR:-/models/glm-ocr}"
RENDER_NODE="${RENDER_NODE:-/dev/dri/renderD128}"
VIDEO_GID="${VIDEO_GID:-44}"
CMD=(build run stop)

case "${1:-}" in
  build)
    docker build -t "$IMAGE" -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"
    ;;
  run)
    shift
    device="GPU"
    if [ "${1:-}" = "--device" ]; then device="${2:-GPU}"; shift 2; fi
    if [ "$device" = "GPU" ]; then
      docker run -d --name glm-ocr-service \
        --device "$RENDER_NODE" --group-add "$VIDEO_GID" \
        -v "$MODEL_DIR:/models/glm-ocr:rw" \
        "$IMAGE" --device "$device" "$@"
    else
      docker run -d --name glm-ocr-service \
        -v "$MODEL_DIR:/models/glm-ocr:rw" \
        "$IMAGE" --device "$device" "$@"
    fi
    docker ps --filter name=glm-ocr-service
    ;;
  stop)
    docker rm -f glm-ocr-service || true
    ;;
  *)
    echo "usage: $0 {build | run [--device GPU|CPU] [flags] | stop}" >&2
    exit 2
    ;;
esac
