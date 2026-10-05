"""Bounded exact-period detection on generated tokens, never on the prompt."""
import math
import time


class GenerationGuard:
    def __init__(self, prompt_tokens, timeout_s=0, enabled=True, clock=time.monotonic):
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("Region timeout must be finite and nonnegative")
        self.prompt_tokens = prompt_tokens
        self.timeout_s = timeout_s
        self.enabled = enabled
        self.clock = clock
        self.started = clock()
        self.reason = None
        self.details = {}

    def __call__(self, input_ids, scores=None, **kwargs):
        if self.reason:
            return True
        elapsed = self.clock() - self.started
        count = input_ids.shape[-1] - self.prompt_tokens
        if self.timeout_s and elapsed >= self.timeout_s:
            self.reason = "generation_timeout"
            self.details = {"seconds": elapsed, "generated_tokens": count}
            return True
        if not self.enabled or count < 512 or count % 16:
            return False
        # Single-flight, single-sequence generation. At most 4096 recent tokens.
        tail = input_ids[0, -min(count, 4096):].tolist()
        for period in range(1, 129):
            repeats = max(32, math.ceil(512 / period))
            length = repeats * period
            if len(tail) < length:
                continue
            block = tail[-period:]
            if tail[-length:] == block * repeats:
                self.reason = "generation_loop"
                self.details = {"period_tokens": period, "repetitions": repeats,
                                "generated_tokens": count, "seconds": elapsed}
                return True
        return False
