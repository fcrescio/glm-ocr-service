"""Deterministic profiling contract, including exception cleanup."""
from glm_ocr_service.profiling import generation_timings


class Language:
    def forward(self, value):
        return value


class Model:
    def __init__(self):
        self.language_model = Language()

    def get_multimodal_embeddings(self, value):
        return value


model = Model()
with generation_timings(model) as timings:
    assert model.get_multimodal_embeddings(3) == 3
    assert model.language_model.forward(4) == 4
    assert model.language_model.forward(5) == 5
assert timings["language_calls"] == 2
assert all(timings[key] >= 0 for key in ("multimodal_s", "prefill_s", "decode_s"))
assert "forward" not in vars(model.language_model)
try:
    with generation_timings(model):
        raise RuntimeError("test")
except RuntimeError:
    pass
assert "get_multimodal_embeddings" not in vars(model)
print("profiling contracts passed")
