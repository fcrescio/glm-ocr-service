"""Equivalent broadcast mask for single-image GLM vision attention."""
import torch


def attention_mask(grid_thw):
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    if len(lengths) == 1:
        # One image has no cross-image boundaries: every attention entry is zero.
        # Broadcasting preserves that meaning without an O(patches**2) buffer.
        return torch.zeros((1, 1, 1), dtype=torch.float32)
    cumulative = torch.nn.functional.pad(lengths.cumsum(0), (1, 0), value=0)
    total = int(cumulative[-1])
    mask = torch.full((1, total, total), float("-inf"), dtype=torch.float32)
    for start, end in zip(cumulative[:-1], cumulative[1:]):
        mask[:, start:end, start:end] = 0
    return mask


def compact_vision_embeddings(model, pixel_values, grid_thw, **kwargs):
    hidden_states = torch.from_numpy(model.vision_embeddings(pixel_values).last_hidden_state)
    rotary = model.rot_pos_emb(grid_thw)
    return model.vision_embeddings_merger(
        hidden_states.numpy(), attention_mask=attention_mask(grid_thw),
        rotary_pos_emb=rotary).last_hidden_state
