"""
model_reporter.py
=================
Aggregates CheckResult objects into a console summary and JSON report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.checks.accuracy import CheckResult


class ModelReporter:
    """
    Usage:
        reporter = ModelReporter(results, model_name="MyClassifier-v2")
        reporter.print_summary()
        reporter.save_json("reports/run.json")
    """

    def __init__(self, results: list[CheckResult],
                 model_name: str = "unknown"):
        self._results    = results
        self._model_name = model_name

    # ── Console ───────────────────────────────────────────────

    def print_summary(self, verbose: bool = False) -> None:
        total  = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"  AI Model Testing Framework — Results")
        print(f"  Model : {self._model_name}")
        print(f"  Checks: {total}  ✓ {passed}  ✗ {failed}")
        print(f"{'='*60}")

        for r in self._results:
            icon = "✓" if r.passed else "✗"
            print(f"  {icon}  [{r.check_name:<28}]  {r.message}")
            if verbose and r.details:
                for k, v in r.details.items():
                    print(f"       {k}: {v}")

        print(f"{'='*60}")
        status = "✅  All checks passed" if failed == 0 \
                 else f"❌  {failed} check(s) failed"
        print(f"  {status}")
        print(f"{'='*60}\n")

    # ── JSON ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        total  = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        return {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "model_name":  self._model_name,
            "summary": {
                "total":     total,
                "passed":    passed,
                "failed":    total - passed,
                "pass_rate": round(passed / total, 4) if total else 0.0,
            },
            "results": [
                {
                    "check_name": r.check_name,
                    "passed":     r.passed,
                    "message":    r.message,
                    "severity":   r.severity,
                    "details":    r.details,
                }
                for r in self._results
            ],
        }

    def save_json(self, path: str = "reports/model_report.json") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    # ── Gate ─────────────────────────────────────────────────

    def assert_all_pass(self) -> None:
        """Raise AssertionError if any ERROR-severity check failed."""
        failures = [r for r in self._results
                    if not r.passed and r.severity == "ERROR"]
        if failures:
            lines = "\n".join(
                f"  ✗ [{r.check_name}] {r.message}" for r in failures
            )
            raise AssertionError(
                f"{len(failures)} model quality check(s) failed:\n{lines}"
            )
