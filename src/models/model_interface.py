"""
model_interface.py
==================
Protocol definition for a testable model + four mock implementations
used across the test suite.

Any callable that accepts a single input and returns a prediction
satisfies the ModelPredictor protocol.

Mock models:
  PerfectBinaryModel  — always correct (even=0, odd=1)
  RandomBinaryModel   — non-deterministic, random outputs
  SlowBinaryModel     — correct but adds artificial delay
  CrashingModel       — raises ValueError on every call
  ThresholdModel      — configurable accuracy level
"""

from __future__ import annotations

import time
import random
from typing import Any, Callable, Protocol, runtime_checkable


# ── Protocol ──────────────────────────────────────────────────

@runtime_checkable
class ModelPredictor(Protocol):
    def __call__(self, x: Any) -> Any:
        """Accept one input, return one prediction."""
        ...


# ── Mock Models ───────────────────────────────────────────────

class PerfectBinaryModel:
    """Always predicts correctly: even inputs → 0, odd → 1."""
    def predict(self, x: int) -> int:
        return int(x) % 2

    def __call__(self, x: int) -> int:
        return self.predict(x)


class RandomBinaryModel:
    """Non-deterministic — output changes each call (tests consistency checks)."""
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def predict(self, x: int) -> int:
        return self._rng.randint(0, 1)

    def __call__(self, x: int) -> int:
        return self.predict(x)


class SlowBinaryModel:
    """Correct predictions with configurable sleep (tests latency checks)."""
    def __init__(self, delay_s: float = 0.05):
        self._delay = delay_s

    def predict(self, x: int) -> int:
        time.sleep(self._delay)
        return int(x) % 2

    def __call__(self, x: int) -> int:
        return self.predict(x)


class CrashingModel:
    """Always raises — tests edge case error handling."""
    def predict(self, x: Any) -> Any:
        raise ValueError(f"Model crashed on input: {x!r}")

    def __call__(self, x: Any) -> Any:
        return self.predict(x)


class ThresholdModel:
    """
    Correct on the first `accuracy` fraction of calls,
    wrong on the rest. Useful for accuracy threshold tests.

    accuracy=1.0 → PerfectBinaryModel behaviour
    accuracy=0.0 → always wrong
    """
    def __init__(self, accuracy: float = 0.8):
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        self._accuracy = accuracy
        self._call_n   = 0

    def predict(self, x: int) -> int:
        correct = self._call_n / max(self._call_n + 1, 1) < self._accuracy
        self._call_n += 1
        truth = int(x) % 2
        return truth if correct else 1 - truth

    def __call__(self, x: int) -> int:
        return self.predict(x)

    def reset(self) -> None:
        self._call_n = 0
