"""
latency.py
==========
LatencyCheck — profiles model inference latency and gates on P50/P95/P99.

P99 is the primary SLA gate — it catches tail latency that users experience
on the worst 1% of requests.
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from src.checks.accuracy import CheckResult


class LatencyCheck:
    """
    Profiles prediction latency across a sample set and checks
    that P50/P95/P99 are within configurable thresholds.

    Usage:
        check  = LatencyCheck(p99_threshold_ms=200, p95_threshold_ms=150)
        result = check.run(model.predict, X_sample)
    """

    def __init__(self,
                 p99_threshold_ms: float = 200.0,
                 p95_threshold_ms: float = 150.0,
                 p50_threshold_ms: float = 100.0):
        self.p99 = p99_threshold_ms
        self.p95 = p95_threshold_ms
        self.p50 = p50_threshold_ms

    def run(self, predict_fn: Callable[[Any], Any],
            X: list) -> CheckResult:
        """
        Time each prediction and compute percentile latencies.
        Fails on the first percentile that exceeds its threshold.
        """
        if not X:
            return CheckResult(
                "latency_profile", False,
                "Cannot profile: empty input list",
                severity="ERROR",
            )

        latencies_ms = []
        for x in X:
            t0 = time.perf_counter()
            predict_fn(x)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        p50_val = self._percentile(latencies_ms, 50)
        p95_val = self._percentile(latencies_ms, 95)
        p99_val = self._percentile(latencies_ms, 99)

        failures = []
        if p99_val > self.p99:
            failures.append(f"P99 {p99_val:.1f}ms > threshold {self.p99:.1f}ms")
        if p95_val > self.p95:
            failures.append(f"P95 {p95_val:.1f}ms > threshold {self.p95:.1f}ms")
        if p50_val > self.p50:
            failures.append(f"P50 {p50_val:.1f}ms > threshold {self.p50:.1f}ms")

        passed = len(failures) == 0

        return CheckResult(
            check_name = "latency_profile",
            passed     = passed,
            message    = (
                f"P50={p50_val:.1f}ms P95={p95_val:.1f}ms P99={p99_val:.1f}ms"
                + (f" — FAIL: {'; '.join(failures)}" if failures else " — OK")
            ),
            details    = {
                "p50_ms":          round(p50_val, 3),
                "p95_ms":          round(p95_val, 3),
                "p99_ms":          round(p99_val, 3),
                "min_ms":          round(min(latencies_ms), 3),
                "max_ms":          round(max(latencies_ms), 3),
                "mean_ms":         round(statistics.mean(latencies_ms), 3),
                "sample_count":    len(latencies_ms),
                "p99_threshold":   self.p99,
                "p95_threshold":   self.p95,
                "p50_threshold":   self.p50,
                "failures":        failures,
            },
            severity   = "INFO" if passed else "ERROR",
        )

    @staticmethod
    def _percentile(data: list[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = max(0, int(len(sorted_data) * pct / 100) - 1)
        return sorted_data[min(index, len(sorted_data) - 1)]
