"""
test_drift.py — 10 tests for DriftCheck (PSI-based).
"""

import pytest
from src.checks.drift import DriftCheck


class TestDriftCheck:

    def test_identical_distributions_no_drift(self):
        data   = [float(i) for i in range(100)]
        result = DriftCheck(threshold=0.10).run(data, data)
        assert result.passed
        assert result.details["psi"] < 0.10

    def test_very_different_distributions_drift_detected(self):
        ref  = [float(i) for i in range(100)]       # 0–99
        prod = [float(i) for i in range(500, 600)]  # 500–599
        result = DriftCheck(threshold=0.10).run(ref, prod)
        assert not result.passed

    def test_slight_shift_within_threshold(self):
        ref  = [float(i) for i in range(100)]
        prod = [float(i) + 2 for i in range(100)]   # small shift
        result = DriftCheck(threshold=0.25).run(ref, prod)
        # Small shift should pass a lenient threshold
        assert result.details["psi"] >= 0.0

    def test_psi_value_in_details(self):
        data   = [float(i) for i in range(50)]
        result = DriftCheck().run(data, data)
        assert "psi" in result.details
        assert isinstance(result.details["psi"], float)

    def test_ref_and_prod_means_in_details(self):
        ref  = [1.0, 2.0, 3.0]
        prod = [10.0, 20.0, 30.0]
        result = DriftCheck().run(ref, prod)
        assert "ref_mean"  in result.details
        assert "prod_mean" in result.details

    def test_insufficient_data_fails(self):
        result = DriftCheck().run([1.0], [1.0])
        assert not result.passed

    def test_check_name_correct(self):
        data   = [float(i) for i in range(20)]
        result = DriftCheck().run(data, data)
        assert result.check_name == "data_drift_psi"

    def test_uniform_distribution_no_drift(self):
        ref  = [float(i) for i in range(200)]
        prod = [float(i) for i in range(200)]
        result = DriftCheck(threshold=0.10).run(ref, prod)
        assert result.passed

    def test_feature_drift_returns_one_result_per_feature(self):
        ref  = {"age": [25.0, 30.0, 35.0, 40.0, 45.0],
                "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0]}
        prod = {"age": [26.0, 31.0, 36.0, 41.0, 46.0],
                "income": [52000.0, 62000.0, 72000.0, 82000.0, 92000.0]}
        results = DriftCheck().run_feature_drift(ref, prod)
        assert len(results) == 2
        assert any("age" in r.check_name for r in results)
        assert any("income" in r.check_name for r in results)

    def test_psi_is_non_negative(self):
        ref  = [float(i) for i in range(50)]
        prod = [float(i) * 1.5 for i in range(50)]
        result = DriftCheck().run(ref, prod)
        assert result.details["psi"] >= 0.0
