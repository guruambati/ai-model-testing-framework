"""
drift.py
========
DriftCheck — detects data distribution shift using Population Stability Index (PSI).

PSI interpretation (industry standard):
  PSI < 0.10   : No significant drift (INFO)
  PSI 0.10–0.25: Moderate drift — monitor closely (WARNING)
  PSI > 0.25   : Significant drift — investigate before deployment (ERROR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.checks.accuracy import CheckResult


class DriftCheck:
    """
    Compares a reference distribution (e.g. training data) against
    a production distribution using PSI.

    Usage:
        check = DriftCheck(threshold=0.10)
        result = check.run(reference=X_train, production=X_live)
    """

    PSI_MODERATE   = 0.10
    PSI_SIGNIFICANT = 0.25

    def __init__(self, threshold: float = 0.10, buckets: int = 10):
        self.threshold = threshold
        self.buckets   = buckets

    def run(self, reference: list[float],
            production: list[float]) -> CheckResult:
        """
        Compute PSI between reference and production distributions.
        Passes when PSI < threshold.
        """
        if len(reference) < 2 or len(production) < 2:
            return CheckResult(
                "data_drift_psi", False,
                "Insufficient data: need at least 2 samples in each distribution",
                severity="ERROR",
            )

        psi_value = self._psi(reference, production)
        passed    = psi_value < self.threshold
        severity  = self._severity(psi_value)

        return CheckResult(
            check_name = "data_drift_psi",
            passed     = passed,
            message    = (
                f"PSI = {psi_value:.4f} "
                f"({'OK' if passed else 'DRIFT DETECTED'} "
                f"— threshold {self.threshold})"
            ),
            details    = {
                "psi":              round(psi_value, 6),
                "threshold":        self.threshold,
                "ref_mean":         round(sum(reference) / len(reference), 4),
                "prod_mean":        round(sum(production) / len(production), 4),
                "ref_n":            len(reference),
                "prod_n":           len(production),
                "interpretation":   severity,
            },
            severity   = severity,
        )

    def run_feature_drift(self,
                           reference_features: dict[str, list[float]],
                           production_features: dict[str, list[float]]
                           ) -> list[CheckResult]:
        """
        Run PSI check for each named feature independently.
        Returns one CheckResult per feature.
        """
        results = []
        for feature in reference_features:
            ref  = reference_features[feature]
            prod = production_features.get(feature, [])
            result = self.run(ref, prod)
            result.check_name = f"data_drift_psi[{feature}]"
            results.append(result)
        return results

    # ── Internal ──────────────────────────────────────────────

    def _psi(self, expected: list[float], actual: list[float]) -> float:
        all_vals  = expected + actual
        min_val   = min(all_vals)
        max_val   = max(all_vals)

        if min_val == max_val:
            return 0.0

        step      = (max_val - min_val) / self.buckets
        eps       = 1e-8

        psi_total = 0.0
        for i in range(self.buckets):
            lo = min_val + i * step
            hi = lo + step if i < self.buckets - 1 else max_val + 1e-10

            exp_count = sum(1 for v in expected  if lo <= v < hi)
            act_count = sum(1 for v in actual    if lo <= v < hi)

            exp_pct = max(exp_count / len(expected), eps)
            act_pct = max(act_count / len(actual),   eps)

            psi_total += (act_pct - exp_pct) * math.log(act_pct / exp_pct)

        return abs(psi_total)

    def _severity(self, psi: float) -> str:
        if psi < self.PSI_MODERATE:
            return "INFO"
        if psi < self.PSI_SIGNIFICANT:
            return "WARNING"
        return "ERROR"
