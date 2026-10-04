"""Synchronous host-wall timings at Optimum model boundaries."""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def generation_timings(model):
    timings = {"multimodal_s": 0.0, "prefill_s": 0.0,
               "decode_s": 0.0, "language_calls": 0}
    originals = []

    def instrument(obj, name, language=False):
        original = getattr(obj, name)
        had_local = name in vars(obj)

        def measured(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if language:
                    key = "prefill_s" if timings["language_calls"] == 0 else "decode_s"
                    timings["language_calls"] += 1
                else:
                    key = "multimodal_s"
                timings[key] += perf_counter() - start

        originals.append((obj, name, original, had_local))
        setattr(obj, name, measured)

    try:
        instrument(model, "get_multimodal_embeddings")
        instrument(model.language_model, "forward", language=True)
        yield timings
    finally:
        for obj, name, original, had_local in reversed(originals):
            if had_local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)
