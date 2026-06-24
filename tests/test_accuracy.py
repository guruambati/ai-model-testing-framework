"""
test_accuracy.py — 12 tests for AccuracyCheck.
Covers: threshold gate, regression, A/B comparison, edge cases.
"""

import pytest
from src.checks.accuracy import AccuracyCheck


class TestAccuracyThreshold:

    def test_perfect_model_passes(self, perfect, binary_dataset):
        X, y = binary_dataset
        result = AccuracyCheck(threshold=1.0).run(perfect.predict, X, y)
        assert result.passed
        assert result.details["accuracy"] == 1.0

    def test_bad_model_fails_high_threshold(self, binary_dataset):
        X, y = binary_dataset
        # Model always predicts 0 → 50% accuracy on balanced dataset
        result = AccuracyCheck(threshold=0.9).run(lambda x: 0, X, y)
        assert not result.passed

    def test_low_threshold_always_passes(self, binary_dataset):
        X, y = binary_dataset
        result = AccuracyCheck(threshold=0.01).run(lambda x: 0, X, y)
        assert result.passed

    def test_details_contain_required_keys(self, perfect, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run(perfect.predict, X, y)
        for key in ("accuracy", "threshold", "correct", "total"):
            assert key in result.details

    def test_empty_dataset_fails(self, perfect):
        result = AccuracyCheck().run(perfect.predict, [], [])
        assert not result.passed

    def test_mismatched_lengths_fails(self, perfect):
        result = AccuracyCheck().run(perfect.predict, [1, 2], [0])
        assert not result.passed

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            AccuracyCheck(threshold=1.5)

    def test_result_check_name(self, perfect, binary_dataset):
        X, y = binary_dataset
        result = AccuracyCheck().run(perfect.predict, X, y)
        assert result.check_name == "accuracy_threshold"


class TestAccuracyRegression:

    def test_within_tolerance_passes(self, perfect, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run_regression(
            perfect.predict, X, y, baseline_accuracy=0.98, tolerance=0.05
        )
        assert result.passed

    def test_drop_beyond_tolerance_fails(self, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run_regression(
            lambda x: 0, X, y, baseline_accuracy=0.9, tolerance=0.02
        )
        assert not result.passed

    def test_regression_details_include_floor(self, perfect, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run_regression(
            perfect.predict, X, y, baseline_accuracy=0.95, tolerance=0.03
        )
        assert "floor" in result.details
        assert abs(result.details["floor"] - 0.92) < 1e-9


class TestAccuracyAB:

    def test_better_model_wins(self, perfect, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run_ab(
            perfect.predict, lambda x: 0, X, y, min_improvement=0.0
        )
        assert result.passed
        assert result.details["delta"] > 0

    def test_worse_model_fails(self, perfect, binary_dataset):
        X, y   = binary_dataset
        result = AccuracyCheck().run_ab(
            lambda x: 0, perfect.predict, X, y, min_improvement=0.0
        )
        assert not result.passed
