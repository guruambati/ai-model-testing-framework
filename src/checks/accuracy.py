"""
accuracy.py
===========
AccuracyCheck — measures model accuracy against labelled test data.

Provides:
  run()             : basic accuracy threshold gate
  run_regression()  : compares against a known baseline with tolerance
  run_ab()          : compares two model versions head-to-head
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CheckResult:
    """Unified result type for all model quality checks."""
    check_name:     str
    passed:         bool
    message:        str
    details:        dict = field(default_factory=dict)
    severity:       str  = "ERROR"   # ERROR | WARNING | INFO

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check_name}: {self.message}"


class AccuracyCheck:
    """
    Gate model deployment on minimum accuracy threshold.

    Usage:
        check = AccuracyCheck(threshold=0.90)
        result = check.run(model.predict, X_test, y_test)
    """

    def __init__(self, threshold: float = 0.80):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        self.threshold = threshold

    def run(self, predict_fn: Callable[[Any], Any],
            X: list, y_true: list) -> CheckResult:
        """
        Compute accuracy and compare against minimum threshold.

        Returns PASS when accuracy >= threshold.
        """
        if not X or not y_true:
            return CheckResult(
                "accuracy_threshold", False,
                "Cannot evaluate: empty dataset",
                severity="ERROR",
            )
        if len(X) != len(y_true):
            return CheckResult(
                "accuracy_threshold", False,
                f"X length {len(X)} != y_true length {len(y_true)}",
                severity="ERROR",
            )

        preds   = [predict_fn(x) for x in X]
        correct = sum(1 for p, t in zip(preds, y_true) if p == t)
        acc     = correct / len(y_true)
        passed  = acc >= self.threshold

        return CheckResult(
            check_name = "accuracy_threshold",
            passed     = passed,
            message    = (
                f"Accuracy {acc:.4f} {'≥' if passed else '<'} "
                f"threshold {self.threshold:.4f}"
            ),
            details    = {
                "accuracy":  round(acc, 6),
                "threshold": self.threshold,
                "correct":   correct,
                "total":     len(y_true),
            },
            severity   = "INFO" if passed else "ERROR",
        )

    def run_regression(self, predict_fn: Callable[[Any], Any],
                       X: list, y_true: list,
                       baseline_accuracy: float,
                       tolerance: float = 0.02) -> CheckResult:
        """
        Ensure current accuracy hasn't dropped more than `tolerance`
        below a known baseline.

        Passes when: current_accuracy >= baseline - tolerance
        """
        preds   = [predict_fn(x) for x in X]
        correct = sum(1 for p, t in zip(preds, y_true) if p == t)
        acc     = correct / len(y_true)
        floor   = baseline_accuracy - tolerance
        passed  = acc >= floor

        return CheckResult(
            check_name = "accuracy_regression",
            passed     = passed,
            message    = (
                f"Current {acc:.4f} {'≥' if passed else '<'} "
                f"floor {floor:.4f} "
                f"(baseline {baseline_accuracy:.4f} − tolerance {tolerance:.4f})"
            ),
            details    = {
                "current_accuracy":  round(acc, 6),
                "baseline_accuracy": baseline_accuracy,
                "tolerance":         tolerance,
                "floor":             round(floor, 6),
            },
            severity   = "INFO" if passed else "ERROR",
        )

    def run_ab(self, predict_fn_a: Callable[[Any], Any],
               predict_fn_b: Callable[[Any], Any],
               X: list, y_true: list,
               min_improvement: float = 0.0) -> CheckResult:
        """
        Compare model A vs model B.
        Passes when A's accuracy >= B's accuracy + min_improvement.
        """
        preds_a = [predict_fn_a(x) for x in X]
        preds_b = [predict_fn_b(x) for x in X]
        acc_a   = sum(p == t for p, t in zip(preds_a, y_true)) / len(y_true)
        acc_b   = sum(p == t for p, t in zip(preds_b, y_true)) / len(y_true)
        delta   = acc_a - acc_b
        passed  = delta >= min_improvement

        return CheckResult(
            check_name = "accuracy_ab_comparison",
            passed     = passed,
            message    = (
                f"Model A {acc_a:.4f} vs Model B {acc_b:.4f} | "
                f"delta {delta:+.4f} "
                f"(min required {min_improvement:+.4f})"
            ),
            details    = {
                "model_a_accuracy": round(acc_a, 6),
                "model_b_accuracy": round(acc_b, 6),
                "delta":            round(delta, 6),
                "min_improvement":  min_improvement,
            },
            severity   = "INFO" if passed else "ERROR",
        )
