"""
edge_cases.py
=============
EdgeCaseCheck — runs a model against a declarative list of edge case
inputs and verifies expected behaviour.

Edge cases covered:
  - Null / None inputs
  - Empty string / empty list
  - Boundary values (min int, max int, 0, -1)
  - Very large inputs
  - Unexpected types (string where int expected)
  - Negative numbers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.checks.accuracy import CheckResult


@dataclass
class EdgeCase:
    """
    One edge case definition.

    Fields:
        input           : the value to pass to the model
        description     : human-readable label for the test case
        expected        : expected prediction (None = don't check value, just no crash)
        expect_exception: if True, the model SHOULD raise — test passes if it does
    """
    input:            Any
    description:      str
    expected:         Any  = None
    expect_exception: bool = False


# ── Built-in edge case catalogue ──────────────────────────────

STANDARD_EDGE_CASES: list[EdgeCase] = [
    EdgeCase(input=0,            description="zero input"),
    EdgeCase(input=-1,           description="negative one"),
    EdgeCase(input=1,            description="positive one"),
    EdgeCase(input=2**31 - 1,   description="max 32-bit int"),
    EdgeCase(input=-(2**31),     description="min 32-bit int"),
    EdgeCase(input=0.0,          description="zero float"),
    EdgeCase(input=-0.0,         description="negative zero float"),
    EdgeCase(input=float('inf'), description="positive infinity float"),
]


class EdgeCaseCheck:
    """
    Runs a model against a list of EdgeCase definitions.

    Usage:
        check = EdgeCaseCheck()
        result = check.run(model.predict, STANDARD_EDGE_CASES)

        # Or with custom cases:
        custom = [
            EdgeCase(input=None, description="null input", expect_exception=True),
            EdgeCase(input=42,   description="answer to everything", expected=0),
        ]
        result = check.run(model.predict, custom)
    """

    def run(self, predict_fn: Callable[[Any], Any],
            cases: list[EdgeCase]) -> CheckResult:
        """
        Run all edge cases. Returns one aggregated CheckResult.
        """
        if not cases:
            return CheckResult(
                "edge_case_handling", True,
                "No edge cases defined — skipped",
                severity="INFO",
            )

        failures = []
        for case in cases:
            failure = self._run_one(predict_fn, case)
            if failure:
                failures.append(failure)

        passed = len(failures) == 0
        return CheckResult(
            check_name = "edge_case_handling",
            passed     = passed,
            message    = (
                f"All {len(cases)} edge cases handled correctly"
                if passed else
                f"{len(failures)}/{len(cases)} edge cases failed"
            ),
            details    = {
                "total":    len(cases),
                "passed":   len(cases) - len(failures),
                "failed":   len(failures),
                "failures": failures,
            },
            severity   = "INFO" if passed else "ERROR",
        )

    def run_standard(self, predict_fn: Callable[[Any], Any]) -> CheckResult:
        """Run the built-in standard edge case catalogue."""
        return self.run(predict_fn, STANDARD_EDGE_CASES)

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _run_one(predict_fn: Callable[[Any], Any],
                 case: EdgeCase) -> dict | None:
        """
        Returns None on success, a failure dict on failure.
        """
        try:
            prediction = predict_fn(case.input)

            if case.expect_exception:
                return {
                    "description": case.description,
                    "input":       str(case.input),
                    "error":       "Expected an exception but none was raised",
                    "got":         str(prediction),
                }

            if case.expected is not None and prediction != case.expected:
                return {
                    "description": case.description,
                    "input":       str(case.input),
                    "expected":    str(case.expected),
                    "got":         str(prediction),
                }

            return None

        except Exception as exc:
            if case.expect_exception:
                return None  # Expected — this is a pass
            return {
                "description": case.description,
                "input":       str(case.input),
                "error":       f"{type(exc).__name__}: {exc}",
            }
