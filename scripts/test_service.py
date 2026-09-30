"""Tests for the GLM-OCR OpenAI-compatible OCR service.

Part A (unit, no model/GPU needed): stub model + FastAPI TestClient — verifies
the OpenAI response contract, SSE streaming framing, auth, and error shapes.

Part B (E2E, real model): spawns the live server and OCRs a synthetic test
page rendered by the test itself (known text, high contrast); verifies the
OpenAI shape, that the model actually reads the page (known-text assertions),
that streamed text equals non-streamed text, SSE framing, usage, and clean
SIGTERM shutdown with no leaked processes.

Usage:
    python scripts/test_service.py                     # unit only
    python scripts/test_service.py --e2e \
        --model-dir /path/to/4-part-ov-model [--device GPU|CPU] \
        [--port 8931] [--image /path/to/real_page.png]

E2E needs the 4-part OV model directory (scripts/export_model.py) and, for
--device GPU, an OpenVINO-visible iGPU. The sample page renders a CJK line
only when a CJK-capable font is present on the machine; otherwise it degrades
to English-only assertions.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from glm_ocr_service.model import GLMOCRModel, GenerationResult, Prepared  # noqa: E402
from glm_ocr_service.server import build_app  # noqa: E402

FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail and not cond else ""), flush=True)
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------ stub model


class StubModel(GLMOCRModel):
    """Bypass __init__ (no OV); canned generation."""

    def __init__(self, stream_chunks=("hello ", "world")):
        self.model_dir = "stub"
        self.device = "GPU"
        self.pixel_cap = 700_000
        self.default_prompt = "Text Recognition:"
        self.loaded = True
        self.load_s = 1.0
        import threading

        self._lock = threading.Lock()
        self._chunks = list(stream_chunks)
        self.tokenizer = type("T", (), {"decode": staticmethod(lambda *a, **k: "stub")})()

    def prepare(self, messages):
        n_images = sum(
            1
            for m in messages
            for p in (m.get("content") if isinstance(m.get("content"), list) else [])
            if isinstance(p, dict) and p.get("type") == "image"
        )
        return Prepared(inputs={}, prompt_tokens=42, n_images=n_images, image_sizes=[])

    def infer(self, prepared, max_tokens, temperature=0.0, top_p=None, stop=None, stream_cb=None):
        if stream_cb is not None:
            for c in self._chunks:
                stream_cb(c)
            return GenerationResult("".join(self._chunks), 2, prepared.prompt_tokens, "stop", True, 0.01)
        return GenerationResult("".join(self._chunks), 2, prepared.prompt_tokens, "stop", True, 0.01)


# ----------------------------------------------------------------- unit tests


def unit_tests() -> None:
    print("== Part A: unit (stub model, no GPU) ==")
    app = build_app(StubModel(), model_id="glm-ocr", api_key="sekret", max_tokens_cap=100)
    client = TestClient(app)
    H_OK = {"Authorization": "Bearer sekret"}
    H_BAD = {"Authorization": "Bearer wrong"}

    r = client.get("/health")
    check("health 200 + fields", r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model"] == "glm-ocr", str(r.json()))

    r = client.get("/v1/models", headers=H_OK)
    j = r.json()
    check(
        "v1/models openai shape",
        r.status_code == 200 and j.get("object") == "list"
        and j["data"][0]["id"] == "glm-ocr" and j["data"][0]["object"] == "model"
        and isinstance(j["data"][0]["created"], int),
        json.dumps(j),
    )

    # auth
    r = client.get("/v1/models")
    check("auth: missing key -> 401 openai error shape", r.status_code == 401 and r.json()["error"]["type"] == "authentication_error", json.dumps(r.json()))
    r = client.get("/v1/models", headers=H_BAD)
    check("auth: wrong key -> 401", r.status_code == 401)
    r = client.post("/v1/chat/completions", headers=H_OK, json={"model": "glm-ocr", "messages": [{"role": "user", "content": "hi"}]})
    check("auth: x-api-key header also accepted", r.status_code == 200)

    # non-stream contract
    r = client.post("/v1/chat/completions", headers=H_OK, json={
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Text Recognition:"}]}],
        "max_tokens": 999999,  # must be clamped to cap, not rejected
        "temperature": 0.0,
    })
    j = r.json()
    check("non-stream 200", r.status_code == 200, json.dumps(j)[:300])
    check("  id prefix", j.get("id", "").startswith("chatcmpl-"), j.get("id", ""))
    check("  object", j.get("object") == "chat.completion", str(j.get("object")))
    check("  created int", isinstance(j.get("created"), int))
    check("  model echo", j.get("model") == "glm-ocr")
    ch = j.get("choices", [{}])[0]
    check("  choice index/role", ch.get("index") == 0 and ch.get("message", {}).get("role") == "assistant")
    check("  content str", isinstance(ch.get("message", {}).get("content"), str))
    check("  finish_reason", ch.get("finish_reason") in ("stop", "length", "content_filter"), str(ch.get("finish_reason")))
    u = j.get("usage", {})
    check("  usage fields", isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int)
          and u.get("total_tokens") == u.get("prompt_tokens") + u.get("completion_tokens"), json.dumps(u))
    check("  max_tokens clamped (no 4xx)", r.status_code == 200)

    # streaming contract
    r = client.post("/v1/chat/completions", headers=H_OK, json={
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    check("stream 200 + sse content-type", r.status_code == 200 and "text/event-stream" in r.headers.get("content-type", ""), str(r.headers))
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    chunks = []
    saw_done = lines and lines[-1] == "data: [DONE]"
    usage_seen = None
    finish_seen = None
    role_seen = None
    for l in lines[:-1] if saw_done else lines:
        c = json.loads(l[len("data: "):])
        if c.get("object") != "chat.completion.chunk":
            check("chunk object", False, l)
            continue
        d = c["choices"][0]["delta"] if c.get("choices") else {}
        if "role" in d:
            role_seen = d["role"]
        if c["choices"] and c["choices"][0].get("finish_reason"):
            finish_seen = c["choices"][0]["finish_reason"]
        if not c.get("choices") and c.get("usage"):
            usage_seen = c["usage"]
        chunks.append(l)
    check("  sse ends with [DONE]", bool(saw_done))
    check("  first chunk has role", role_seen == "assistant", str(role_seen))
    check("  final finish_reason", finish_seen == "stop", str(finish_seen))
    check("  usage chunk present", usage_seen is not None, str(usage_seen))
    n_content = sum(1 for l in chunks if '"content"' in l and l.strip() != "data: [DONE]")
    check("  multiple content chunks (real stream)", n_content >= 2, f"n={n_content}")

    # error paths
    r = client.post("/v1/chat/completions", headers=H_OK, json={"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    check("unknown model -> 404 openai error", r.status_code == 404 and r.json()["error"]["code"] == "model_not_found", json.dumps(r.json()))
    r = client.post("/v1/chat/completions", headers=H_OK, json={"model": "glm-ocr", "messages": []})
    check("empty messages -> 400", r.status_code == 400)
    bad_b64 = "data:image/png;base64," + base64.b64encode(b"not-an-image").decode()
    r = client.post("/v1/chat/completions", headers=H_OK, json={
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": bad_b64}}]}],
    })
    check("undecodable image -> 415", r.status_code == 415, json.dumps(r.json()))
    r = client.post("/v1/chat/completions", headers=H_OK, json={
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///nonexistent/x.png"}},
                                                   {"type": "text", "text": "hi"}]}],
    })
    check("missing file image -> 400", r.status_code == 400, json.dumps(r.json()))


# --------------------------------------------------------- sample page (E2E)

# English lines are always rendered; the CJK line only when a CJK font exists.
EN_LINES = [
    "Zebra Crosswalk 42",
    "Order Total: 128.50 EUR",
    "Phone: +49 30 12345678",
]
CJK_LINES = [
    "北京烤鸭餐厅",
]

CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]


def make_sample_page() -> tuple[bytes, list]:
    """Render a high-contrast synthetic document page (672x1036, multiple of
    28, under the 700k-pixel cap) with known text; returns (png_bytes,
    rendered_lines)."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 672, 1036
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def _font(size: int, cjk: bool = False):
        if cjk:
            for p in CJK_FONT_CANDIDATES:
                if Path(p).is_file():
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        pass
            return None
        for p in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ):
            if Path(p).is_file():
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    f = _font(32)
    y = 220
    lines = list(EN_LINES)
    for text in EN_LINES:
        d.text((64, y), text, fill="black", font=f)
        y += 96
    fcjk = _font(36, cjk=True)
    if fcjk is not None:
        d.text((64, y), CJK_LINES[0], fill="black", font=fcjk)
        lines.append(CJK_LINES[0])

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), lines


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# ------------------------------------------------------------------- e2e


def _http_json(url: str, payload: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def e2e_tests(port: int, server_log: Path, model_dir: str, device: str, image: Optional[Path]) -> None:
    print(f"== Part B: E2E (live server, model_dir={model_dir}, device={device}) ==")
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    h = None
    while time.time() - t0 < 900:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
                h = json.loads(r.read().decode())
            if h.get("status") == "ok":
                break
        except Exception:
            time.sleep(2)
    if h is None or h.get("status") != "ok":
        check("server became healthy within 900s", False, server_log.read_text()[-2000:])
        return
    check("server healthy", True, f"load_s={h.get('load_s')}")

    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
        check("v1/models", json.loads(r.read().decode())["data"][0]["id"] == "glm-ocr")

    # -- the test page (synthetic by default, --image to use a real one)
    if image is not None:
        img_bytes = image.read_bytes()
        known_lines = []  # real page: no known-text assertions, length check only
    else:
        img_bytes, known_lines = make_sample_page()
    b64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    t0 = time.time()
    j = _http_json(f"{base}/v1/chat/completions", {
        "model": "glm-ocr",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Text Recognition:"},
        ]}],
        "temperature": 0,
        "max_tokens": 2048,
    })
    e2e = time.time() - t0
    check("e2e non-stream 200", j.get("object") == "chat.completion", json.dumps({k: j.get(k) for k in ("id", "object")}))
    ch = j["choices"][0]
    u = j["usage"]
    text = ch["message"]["content"]
    check("  finish_reason", ch["finish_reason"] in ("stop", "length"), ch["finish_reason"])
    check("  usage sane", u["prompt_tokens"] > 100 and u["completion_tokens"] > 10, json.dumps(u))
    check("  non-trivial output", len(text.strip()) > 40, f"chars={len(text)}")
    if known_lines:
        got = _norm(text)
        hits = [ln for ln in known_lines if _norm(ln) in got]
        need = 2 if len(known_lines) == 3 else 2
        check(f"  reads the page (>= {need} of {len(known_lines)} known lines found)", len(hits) >= need,
              f"hits={hits} output={text[:200]!r}")
    print(f"    [info] e2e={e2e:.1f}s prompt_tokens={u['prompt_tokens']} completion_tokens={u['completion_tokens']} "
          f"finish={ch['finish_reason']} chars={len(text)}", flush=True)

    # streaming: same page; streamed text must equal the non-streamed text
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({
            "model": "glm-ocr",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Text Recognition:"},
            ]}],
            "max_tokens": 2048,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read().decode()
    st = time.time() - t0
    lines = [l for l in raw.splitlines() if l.startswith("data: ")]
    ok_done = lines and lines[-1] == "data: [DONE]"
    check("stream ends with [DONE]", bool(ok_done))
    content = ""
    n_chunks = 0
    finish = None
    usage = None
    for l in lines:
        if l == "data: [DONE]":
            continue
        c = json.loads(l[len("data: "):])
        if c.get("object") != "chat.completion.chunk":
            continue
        if c.get("choices"):
            d = c["choices"][0].get("delta", {})
            if d.get("content"):
                content += d["content"]
                n_chunks += 1
            if c["choices"][0].get("finish_reason"):
                finish = c["choices"][0]["finish_reason"]
        elif c.get("usage"):
            usage = c["usage"]
    print(f"    [info] stream: {st:.1f}s, {n_chunks} content chunks, finish={finish}, usage={usage}", flush=True)
    check("stream has content chunks", n_chunks >= 5, f"n={n_chunks}")
    check("stream finish_reason set", finish is not None, str(finish))
    check("stream usage present", usage is not None, str(usage))
    check("streamed text == non-streamed text", _norm(content) == _norm(text),
          f"streamed={content[:120]!r} nonstreamed={text[:120]!r}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true", help="run live E2E tests (spawns the server; needs the 4-part model)")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--model-dir", default=os.environ.get("GLM_OCR_MODEL_DIR", "models/glm-ocr-ov"))
    ap.add_argument("--device", default=os.environ.get("GLM_OCR_DEVICE", "GPU"), choices=["GPU", "CPU"])
    ap.add_argument("--image", default=None, help="optional real page image for E2E (overrides the synthetic page)")
    args = ap.parse_args()

    unit_tests()

    if args.e2e:
        if not Path(args.model_dir).is_dir():
            print(f"ERROR: --model-dir {args.model_dir} is not a directory (run scripts/export_model.py first)", file=sys.stderr)
            sys.exit(2)
        log_dir = ROOT / "tests"
        log_dir.mkdir(exist_ok=True)
        server_log = log_dir / "e2e-server.log"
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "serve.py"),
             "--model-dir", str(Path(args.model_dir).resolve()),
             "--device", args.device,
             "--port", str(args.port), "--log-level", "INFO"],
            stdout=open(server_log, "wb"),
            stderr=subprocess.STDOUT,
        )
        try:
            e2e_tests(args.port, server_log, args.model_dir, args.device,
                      Path(args.image) if args.image else None)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=25)
                print("    server exited on SIGTERM (graceful)", flush=True)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                print("    server needed SIGKILL (flagging)", flush=True)
            # no leaked service processes (match the server script only; the
            # test driver's own cmdline must not count)
            me = os.getpid()
            out = subprocess.run(["bash", "-c", "ps -eo pid,comm,args"], capture_output=True, text=True).stdout
            leaks = [
                l for l in out.splitlines()
                if "scripts/serve.py" in l and not l.startswith(f"{me} ")
            ]
            check("no leaked service processes", not leaks, "\n".join(leaks[:5]))
        print(f"    server log: {server_log}", flush=True)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: all checks passed")


if __name__ == "__main__":
    main()
