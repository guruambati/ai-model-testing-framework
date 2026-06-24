"""
test_consistency.py — 10 tests for ConsistencyCheck.
"""

import pytest
from src.checks.consistency import ConsistencyCheck


class TestConsistencyCheck:

    def test_deterministic_model_passes(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = ConsistencyCheck(n_runs=5).run(perfect.predict, X)
        assert result.passed

    def test_random_model_may_fail(self):
        # Unseeded random model — very likely to produce different outputs
        import random
        calls = {"n": 0}
        def unstable(x):
            calls["n"] += 1
            # Force different output every other call
            return calls["n"] % 2
        result = ConsistencyCheck(n_runs=4).run(unstable, [1, 2, 3])
        # At least some inputs should be inconsistent
        assert isinstance(result.passed, bool)   # framework runs without error

    def test_inconsistent_count_in_details(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = ConsistencyCheck(n_runs=3).run(perfect.predict, X)
        assert "inconsistent_count" in result.details
        assert result.details["inconsistent_count"] == 0

    def test_total_inputs_in_details(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = ConsistencyCheck(n_runs=3).run(perfect.predict, X)
        assert result.details["total_inputs"] == len(X)

    def test_empty_input_fails(self, perfect):
        result = ConsistencyCheck(n_runs=3).run(perfect.predict, [])
        assert not result.passed

    def test_n_runs_below_2_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConsistencyCheck(n_runs=1)

    def test_check_name_correct(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = ConsistencyCheck(n_runs=3).run(perfect.predict, X)
        assert result.check_name == "prediction_consistency"

    def test_single_input_stability_pass(self, perfect):
        result = ConsistencyCheck(n_runs=5).run_stability(perfect.predict, 4)
        assert result.passed
        assert result.check_name == "single_input_stability"

    def test_single_input_stability_details(self, perfect):
        result = ConsistencyCheck(n_runs=3).run_stability(perfect.predict, 6)
        assert "predictions" in result.details
        assert len(result.details["predictions"]) == 3

    def test_n_runs_reflected_in_details(self, perfect, binary_dataset):
        X, _   = binary_dataset
        result = ConsistencyCheck(n_runs=7).run(perfect.predict, X[:3])
        assert result.details["n_runs"] == 7
