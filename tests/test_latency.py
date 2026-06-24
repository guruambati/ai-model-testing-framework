"""
test_latency.py — 10 tests for LatencyCheck.
"""

import pytest
from src.checks.latency import LatencyCheck


class TestLatencyCheck:

    def test_fast_model_passes_all_percentiles(self, perfect, binary_dataset):
        X, _   = binary_dataset
        # Very generous thresholds — pure Python math will be well under 500ms
        result = LatencyCheck(
            p99_threshold_ms=500,
            p95_threshold_ms=400,
            p50_threshold_ms=300,
        ).run(perfect.predict, X)
        assert result.passed

    def test_slow_model_fails_tight_threshold(self, slow, binary_dataset):
        X, _   = binary_dataset
        # slow model sleeps 10ms per call — P99 will exceed 1ms threshold
        result = LatencyCheck(p99_threshold_ms=1).run(slow.predict, X[:5])
        assert not result.passed

    def test_details_contain_all_percentiles(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        for key in ("p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "mean_ms"):
            assert key in result.details

    def test_p99_greater_equal_p95(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        assert result.details["p99_ms"] >= result.details["p95_ms"]

    def test_p95_greater_equal_p50(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        assert result.details["p95_ms"] >= result.details["p50_ms"]

    def test_latencies_non_negative(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        assert result.details["min_ms"] >= 0.0

    def test_sample_count_in_details(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        assert result.details["sample_count"] == len(X)

    def test_empty_input_fails(self, perfect):
        result = LatencyCheck().run(perfect.predict, [])
        assert not result.passed

    def test_single_sample_passes(self, perfect):
        result = LatencyCheck(p99_threshold_ms=500).run(perfect.predict, [2])
        assert result.passed
        assert result.details["sample_count"] == 1

    def test_check_name_correct(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = LatencyCheck().run(perfect.predict, X)
        assert result.check_name == "latency_profile"
