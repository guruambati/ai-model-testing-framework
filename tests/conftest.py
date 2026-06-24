"""conftest.py — shared fixtures for ai-model-testing-framework."""

import pytest
from src.models.model_interface import (
    PerfectBinaryModel,
    RandomBinaryModel,
    SlowBinaryModel,
    CrashingModel,
    ThresholdModel,
)
from src.checks.accuracy    import AccuracyCheck
from src.checks.drift       import DriftCheck
from src.checks.latency     import LatencyCheck
from src.checks.consistency import ConsistencyCheck
from src.checks.edge_cases  import EdgeCaseCheck


# ── Model fixtures ────────────────────────────────────────────

@pytest.fixture
def perfect():
    return PerfectBinaryModel()

@pytest.fixture
def random_model():
    # Fixed seed so tests are reproducible
    return RandomBinaryModel(seed=0)

@pytest.fixture
def slow():
    return SlowBinaryModel(delay_s=0.01)

@pytest.fixture
def crashing():
    return CrashingModel()

@pytest.fixture
def threshold_90():
    return ThresholdModel(accuracy=0.90)


# ── Dataset fixtures ──────────────────────────────────────────

@pytest.fixture
def binary_dataset():
    """20 samples: even→0, odd→1."""
    X      = list(range(20))
    y_true = [x % 2 for x in X]
    return X, y_true

@pytest.fixture
def normal_distribution():
    """Two draws from ~N(0,1) via deterministic Gaussian approx."""
    import math
    def gauss_approx(n: int, mean: float = 0, std: float = 1) -> list[float]:
        # Central-limit approx: sum of 12 uniform[0,1] − 6 ≈ N(0,1)
        result = []
        seed   = 42
        for i in range(n):
            s = 0.0
            for j in range(12):
                seed  = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                s    += (seed / 0xFFFFFFFF)
            result.append((s - 6) * std + mean)
        return result
    return gauss_approx


# ── Check fixtures ────────────────────────────────────────────

@pytest.fixture
def acc_check():
    return AccuracyCheck(threshold=0.80)

@pytest.fixture
def drift_check():
    return DriftCheck(threshold=0.10)

@pytest.fixture
def latency_check():
    return LatencyCheck(p99_threshold_ms=500, p95_threshold_ms=400, p50_threshold_ms=300)

@pytest.fixture
def consistency_check():
    return ConsistencyCheck(n_runs=5)

@pytest.fixture
def edge_check():
    return EdgeCaseCheck()
