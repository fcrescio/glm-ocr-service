"""OpenAI-compatible HTTP API for the GLM-OCR iGPU OCR service.

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (stream=false | stream=true SSE)

Conventions:
    * Images arrive as OpenAI `image_url` parts: `data:<mime>;base64,...`
      (primary), or `http(s)://` / `file://` / existing absolute path.
    * A text-only request is allowed (plain LLM use); OCR needs at least one
      image in a user message.
    * One user turn with N images is the supported multi-page form.
    * Inference is single-flight (one iGPU arena at a time); requests are
      serialized FIFO, each to completion.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .model import GLMOCRModel, InferenceError, Prepared

log = logging.getLogger("glm_ocr_service.server")

MAX_IMAGE_BYTES = 40 * 1024 * 1024  # per-image request body limit
MAX_IMAGES_PER_REQUEST = 32


# --------------------------------------------------------------------- errors

class ApiError(Exception):
    def __init__(self, status_code: int, message: str, code: str, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.err_type = err_type


def _err_response(status_code: int, message: str, code: str, err_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


# --------------------------------------------------------------------- schema

class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Any] = None  # ImageURL or bare string
    image: Optional[str] = None      # lenient form seen in some clients


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = None  # str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = "glm-ocr"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 0.0
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None  # str | list[str]
    stream: bool = False
    stream_options: Optional[dict] = None
    user: Optional[str] = None


# --------------------------------------------------------------------- images

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[a-z0-9.+/;-]+);base64,(?P<b64>.+)$", re.S)


def _decode_data_url(url: str) -> bytes:
    m = _DATA_URL_RE.match(url)
    if not m:
        raise ApiError(400, f"malformed data URL: {url[:64]}", "invalid_image_url")
    try:
        return base64.b64decode(m.group("b64"), validate=False)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, f"invalid base64 in image data URL: {exc}", "invalid_image_url")


def _image_bytes(url: str, timeout: float = 30.0) -> bytes:
    p = urlparse(url)
    if p.scheme in ("", "file"):
        path = Path(p.path if p.scheme == "file" else url)
        if not path.is_file():
            raise ApiError(400, f"image file not found: {path}", "image_not_found")
        data = path.read_bytes()
    elif p.scheme in ("http", "https"):
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
        if r.status_code != 200:
            raise ApiError(502, f"image fetch failed: HTTP {r.status_code}", "image_fetch_failed")
        data = r.content
    elif p.scheme == "data":
        data = _decode_data_url(url)
    else:
        raise ApiError(400, f"unsupported image URL scheme: {p.scheme}", "invalid_image_url")
    if len(data) > MAX_IMAGE_BYTES:
        raise ApiError(413, f"image exceeds {MAX_IMAGE_BYTES} bytes", "image_too_large")
    return data


def _load_image(url: str):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = _image_bytes(url)
    import io
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise ApiError(415, f"could not decode image: {exc}", "invalid_image")


def _part_image_url(part: ContentPart) -> str:
    iu = part.image_url
    if isinstance(iu, dict):
        u = iu.get("url")
    elif isinstance(iu, str):
        u = iu
    else:
        u = part.image
    if not u:
        raise ApiError(400, f"image part without url: {part.model_dump()!r}", "invalid_image_part")
    return u


# ------------------------------------------------------------------ messages

def normalize_messages(req: ChatCompletionRequest) -> list:
    """OpenAI messages -> GLM processor messages (image parts as PIL)."""
    out = []
    n_images = 0
    for m in req.messages:
        if m.role not in ("system", "user", "assistant"):
            raise ApiError(400, f"unsupported role: {m.role}", "invalid_role")
        if m.content is None:
            continue
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": [{"type": "text", "text": m.content}]})
            continue
        parts = []
        for p in m.content:
            if not isinstance(p, ContentPart):
                p = ContentPart(**p) if isinstance(p, dict) else p
            if p.type == "text":
                if p.text:
                    parts.append({"type": "text", "text": p.text})
            elif p.type in ("image", "image_url"):
                img = _load_image(_part_image_url(p))
                n_images += 1
                if n_images > MAX_IMAGES_PER_REQUEST:
                    raise ApiError(400, f"too many images (max {MAX_IMAGES_PER_REQUEST})", "too_many_images")
                parts.append({"type": "image", "image": img})
            else:
                raise ApiError(400, f"unsupported content part type: {p.type}", "invalid_content_part")
        if parts:
            out.append({"role": m.role, "content": parts})
    if not out:
        raise ApiError(400, "no non-empty messages", "empty_messages")
    return out


# ---------------------------------------------------------------------- app

def build_app(
    model: GLMOCRModel,
    *,
    model_id: str = "glm-ocr",
    api_key: Optional[str] = None,
    max_tokens_cap: int = 4096,
) -> FastAPI:
    app = FastAPI(title="GLM-OCR OCR service", version="0.1.0")

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError):
        return _err_response(exc.status_code, exc.message, exc.code, exc.err_type)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x not in ("body",))
        msg = f"invalid request body: {first.get('msg', 'validation error')}" + (f" (field: {loc})" if loc else "")
        return _err_response(400, msg, "invalid_request")

    def _check_auth(request: Request) -> None:
        if not api_key:
            return
        supplied = request.headers.get("x-api-key") or ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth.split(None, 1)[1]
        if supplied != api_key:
            raise ApiError(401, "invalid API key", "invalid_api_key", err_type="authentication_error")

    @app.get("/")
    def root():
        return PlainTextResponse(
            "GLM-OCR OpenAI-compatible OCR service. Try GET /v1/models, POST /v1/chat/completions, GET /health.\n"
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok" if model.loaded else "loading",
            "model": model_id,
            "device": model.device,
            "load_s": model.load_s,
            "execution_devices": model.execution_devices() if hasattr(model, "execution_devices") else {},
        }

    @app.get("/v1/models")
    def models(request: Request):
        _check_auth(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(model.load_s) if model.load_s else 0,
                    "owned_by": "glm-ocr",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, req: ChatCompletionRequest):
        _check_auth(request)
        if req.model not in (model_id, "glm-ocr", ""):
            return _err_response(404, f"model '{req.model}' not found", "model_not_found", err_type="invalid_request_error")

        request_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        stop_list = None
        if req.stop:
            stop_list = [req.stop] if isinstance(req.stop, str) else list(req.stop)

        max_tokens = req.max_tokens if req.max_tokens is not None else 2048
        max_tokens = max(1, min(max_tokens, max_tokens_cap))

        t0 = time.time()
        try:
            messages = normalize_messages(req)
            prepared = model.prepare(messages)
        except ApiError as e:
            return _err_response(e.status_code, e.message, e.code, e.err_type)
        except InferenceError as e:
            return _err_response(400, str(e), "preparation_failed")

        def _usage(res) -> dict:
            return {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.new_tokens,
                "total_tokens": res.prompt_tokens + res.new_tokens,
            }

        if not req.stream:
            import asyncio

            def _sync():
                return model.infer(prepared, max_tokens, req.temperature, req.top_p, stop_list)

            try:
                res = await asyncio.to_thread(_sync)
            except InferenceError as e:
                return _err_response(500, str(e), "inference_failed", err_type="server_error")
            log.info(
                "request %s done: images=%d prompt_tokens=%d new_tokens=%d finish=%s e2e=%.2fs",
                request_id, prepared.n_images, res.prompt_tokens, res.new_tokens, res.finish_reason, time.time() - t0,
            )
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": res.text},
                        "finish_reason": res.finish_reason,
                    }
                ],
                "usage": _usage(res),
            }

        # ------------------------------------------------------------- stream
        import queue
        from concurrent.futures import ThreadPoolExecutor

        q: "queue.Queue" = queue.Queue()
        SENTINEL = object()

        def _stream_cb(chunk: str) -> None:
            q.put(chunk)

        def _stream_worker() -> None:
            try:
                res = model.infer(prepared, max_tokens, req.temperature, req.top_p, stop_list, stream_cb=_stream_cb)
                q.put(res)
            except Exception as e:  # noqa: BLE001
                q.put(e)
            finally:
                q.put(SENTINEL)

        def _gen():
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm-ocr-infer")
            first = True
            try:
                worker = executor.submit(_stream_worker)
                while True:
                    item = q.get()
                    if item is SENTINEL:
                        break
                    if isinstance(item, BaseException):
                        yield "data: " + json.dumps({"error": {"message": str(item), "type": "server_error", "code": "inference_failed"}}) + "\n\n"
                        continue
                    if isinstance(item, str):
                        delta = {"content": item}
                        if first:
                            delta = {"role": "assistant", "content": item}
                            first = False
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                    else:  # GenerationResult -> final chunk (+ optional usage)
                        res = item
                        yield "data: " + json.dumps(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": res.finish_reason}],
                            },
                            ensure_ascii=False,
                        ) + "\n\n"
                        if req.stream_options and req.stream_options.get("include_usage"):
                            yield "data: " + json.dumps(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [],
                                    "usage": _usage(res),
                                }
                            ) + "\n\n"
                yield "data: [DONE]\n\n"
                log.info(
                    "stream %s done: images=%d prompt_tokens=%d e2e=%.2fs",
                    request_id, prepared.n_images, prepared.prompt_tokens, time.time() - t0,
                )
            finally:
                worker.result()  # consume any future state; errors already routed via the queue
                executor.shutdown(wait=False)

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return app
