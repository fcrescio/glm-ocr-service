import numpy as np
import openvino as ov
from openvino import opset13 as ops
from glm_ocr_service.generation_graph import last_token_logits

hidden = ops.parameter([-1, -1, 3], dtype=np.float32)
weights = ops.constant(np.arange(15, dtype=np.float32).reshape(5, 3))
head = ops.matmul(hidden, weights, False, True)
head.set_friendly_name("test/lm_head/MatMul")
model = ov.Model([head], [hidden])
reference = ov.Core().compile_model(model, "CPU")
last_token_logits(model)
optimized = ov.Core().compile_model(model, "CPU")
for batch, length in ((1, 1), (1, 68), (2, 37)):
    values = np.arange(batch*length*3, dtype=np.float32).reshape(batch, length, 3)
    expected = reference([values])[0][:, -1:, :]
    actual = optimized([values])[0]
    assert actual.shape == (batch, 1, 5)
    np.testing.assert_allclose(actual, expected)
print("Dynamic last-token logits equivalence passed")
