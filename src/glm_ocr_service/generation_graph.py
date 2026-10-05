"""Generation-only optimization: apply the vocabulary head to the last token."""
import numpy as np
from openvino import opset13 as ops


def last_token_logits(model):
    results = model.get_results()
    if len(results) != 1:
        raise ValueError("Expected one generation logits output")
    head = results[0].input_value(0).get_node()
    if head.get_type_name() != "MatMul" or "lm_head" not in head.get_friendly_name():
        raise ValueError("Unsupported GLM vocabulary-head graph")
    hidden = head.input_value(0)
    if hidden.get_partial_shape().rank.get_length() != 3:
        raise ValueError("Expected batch/sequence/hidden input")
    selected = ops.gather(hidden, np.array([-1], dtype=np.int64), np.int64(1))
    selected.set_friendly_name("generation_last_hidden_token")
    head.input(0).replace_source_output(selected.output(0))
    model.validate_nodes_and_infer_types()
