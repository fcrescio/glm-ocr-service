"""Deterministic periodicity, prompt exclusion and deadline tests."""
import torch
from glm_ocr_service.generation_guard import GenerationGuard


def check(tokens, prompt=0, **kwargs):
    guard = GenerationGuard(prompt, **kwargs)
    return guard, guard(torch.tensor([tokens]))

guard, stopped = check([1, 2, 3, 4] * 128)
assert stopped and guard.reason == "generation_loop"
assert guard.details["period_tokens"] == 4
assert not check([1, 2, 3, 4] * 127)[1]
assert not check([1] * 512, prompt=512)[1]
assert not check([1] * 512, enabled=False)[1]
# Repeated markup with changing values is not an exact token loop.
values = [token for number in range(128) for token in (1, 2, number + 10, 3)]
assert not check(values)[1]
assert not check(list(range(512)))[1]
clock = [0]
guard = GenerationGuard(0, timeout_s=10, clock=lambda: clock[0])
clock[0] = 11
assert guard(torch.tensor([[1]])) and guard.reason == "generation_timeout"
from fastapi.testclient import TestClient
from glm_ocr_service.model import GenerationResult
from glm_ocr_service.server import build_app
from test_service import StubModel

for reason in ("generation_loop", "generation_timeout"):
    stub = StubModel()
    stub.infer = lambda *a, **kw: GenerationResult("partial", 560, 1, reason, False, 30,
                                                  stop_details={"generated_tokens": 560})
    client = TestClient(build_app(stub))
    payload = {"model": "glm-ocr", "messages": [{"role": "user", "content": "test"}]}
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 502 and response.json()["error"]["code"] == reason
    payload["stream"] = True
    response = client.post("/v1/chat/completions", json=payload)
    assert f'"code": "{reason}"' in response.text
    assert '"finish_reason": "stop"' not in response.text
    assert client.get("/health").status_code == 200
print("Exact-period, varied-row, prompt-exclusion and cooperative deadline tests passed")
