"""
consistency.py
==============
ConsistencyCheck — verifies that a model produces identical predictions
across N runs for the same input (determinism check).

Non-deterministic models (random seeds not fixed, GPU non-determinism,
dropout at inference time) fail this check.
"""

from __future__ import annotations

from typing import Any, Callable
from src.checks.accuracy import CheckResult


class ConsistencyCheck:
    """
    Runs each input through the model N times and verifies all
    predictions are identical.

    Usage:
        check  = ConsistencyCheck(n_runs=5)
        result = check.run(model.predict, X_sample)
    """

    def __init__(self, n_runs: int = 5):
        if n_runs < 2:
            raise ValueError("n_runs must be at least 2 to check consistency")
        self.n_runs = n_runs

    def run(self, predict_fn: Callable[[Any], Any],
            X: list) -> CheckResult:
        """
        For each input in X, call predict_fn n_runs times.
        Fails if any input produces inconsistent outputs.
        """
        if not X:
            return CheckResult(
                "prediction_consistency", False,
                "Cannot check: empty input list",
                severity="ERROR",
            )

        inconsistent = []
        for x in X:
            preds = [predict_fn(x) for _ in range(self.n_runs)]
            unique = set(str(p) for p in preds)
            if len(unique) > 1:
                inconsistent.append({
                    "input":       str(x),
                    "predictions": [str(p) for p in preds],
                    "unique":      list(unique),
                })

        passed = len(inconsistent) == 0
        return CheckResult(
            check_name = "prediction_consistency",
            passed     = passed,
            message    = (
                f"All {len(X)} inputs produced consistent predictions "
                f"across {self.n_runs} runs"
                if passed else
                f"{len(inconsistent)}/{len(X)} inputs had inconsistent predictions"
            ),
            details    = {
                "total_inputs":      len(X),
                "n_runs":            self.n_runs,
                "inconsistent_count": len(inconsistent),
                "inconsistent":      inconsistent[:5],  # cap at 5 for readability
            },
            severity   = "INFO" if passed else "ERROR",
        )

    def run_stability(self, predict_fn: Callable[[Any], Any],
                      x: Any,
                      n_runs: int | None = None) -> CheckResult:
        """
        Check one specific input for stability.
        Returns the prediction distribution along with pass/fail.
        """
        runs   = n_runs or self.n_runs
        preds  = [predict_fn(x) for _ in range(runs)]
        unique = set(str(p) for p in preds)
        passed = len(unique) == 1

        return CheckResult(
            check_name = "single_input_stability",
            passed     = passed,
            message    = (
                f"Input {x!r} stable across {runs} runs: {preds[0]!r}"
                if passed else
                f"Input {x!r} unstable — saw {len(unique)} different outputs"
            ),
            details    = {
                "input":       str(x),
                "n_runs":      runs,
                "predictions": [str(p) for p in preds],
                "unique":      list(unique),
            },
            severity   = "INFO" if passed else "ERROR",
        )
