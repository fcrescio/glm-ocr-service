import torch
from glm_ocr_service.vision import attention_mask

single = attention_mask(torch.tensor([[1, 118, 166]]))
assert single.shape == (1, 1, 1)
assert single.numel() * single.element_size() == 4
assert torch.equal(single.expand(1, 10, 10), torch.zeros(1, 10, 10))
multiple = attention_mask(torch.tensor([[1, 2, 2], [1, 2, 2]]))
assert multiple.shape == (1, 8, 8)
assert (multiple[:, :4, :4] == 0).all()
assert (multiple[:, 4:, 4:] == 0).all()
assert torch.isneginf(multiple[:, :4, 4:]).all()
assert torch.isneginf(multiple[:, 4:, :4]).all()
print("Single-image broadcast and multi-image isolation masks passed")
