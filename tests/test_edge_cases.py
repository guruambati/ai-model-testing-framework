"""
test_edge_cases.py — 10 tests for EdgeCaseCheck.
"""

import pytest
from src.checks.edge_cases import EdgeCaseCheck, EdgeCase, STANDARD_EDGE_CASES


class TestEdgeCaseCheck:

    def test_standard_cases_pass_perfect_model(self, perfect, edge_check):
        result = edge_check.run(perfect.predict, STANDARD_EDGE_CASES)
        assert result.passed

    def test_crashing_model_caught(self, crashing, edge_check):
        cases  = [EdgeCase(input=1, description="normal input")]
        result = edge_check.run(crashing.predict, cases)
        assert not result.passed
        assert len(result.details["failures"]) == 1

    def test_expected_exception_passes(self, crashing, edge_check):
        cases  = [EdgeCase(input=99, description="crash expected",
                           expect_exception=True)]
        result = edge_check.run(crashing.predict, cases)
        assert result.passed

    def test_wrong_expected_value_fails(self, perfect, edge_check):
        # perfect model on 4 returns 0, but we expect 1
        cases  = [EdgeCase(input=4, description="wrong expected", expected=1)]
        result = edge_check.run(perfect.predict, cases)
        assert not result.passed

    def test_correct_expected_value_passes(self, perfect, edge_check):
        cases  = [EdgeCase(input=4, description="even", expected=0),
                  EdgeCase(input=7, description="odd",  expected=1)]
        result = edge_check.run(perfect.predict, cases)
        assert result.passed

    def test_empty_cases_returns_skipped(self, perfect, edge_check):
        result = edge_check.run(perfect.predict, [])
        assert result.passed
        assert "skipped" in result.message.lower()

    def test_failure_count_in_details(self, crashing, edge_check):
        cases  = [
            EdgeCase(input=1, description="crash 1"),
            EdgeCase(input=2, description="crash 2"),
        ]
        result = edge_check.run(crashing.predict, cases)
        assert result.details["failed"] == 2
        assert result.details["total"]  == 2

    def test_partial_failure_counted(self, perfect, edge_check):
        cases = [
            EdgeCase(input=2, description="correct", expected=0),  # pass
            EdgeCase(input=2, description="wrong",   expected=1),  # fail
        ]
        result = edge_check.run(perfect.predict, cases)
        assert not result.passed
        assert result.details["failed"]  == 1
        assert result.details["passed"]  == 1

    def test_run_standard_uses_built_in_cases(self, perfect, edge_check):
        result = edge_check.run_standard(perfect.predict)
        assert result.details["total"] == len(STANDARD_EDGE_CASES)

    def test_check_name_correct(self, perfect, edge_check):
        result = edge_check.run(perfect.predict, STANDARD_EDGE_CASES)
        assert result.check_name == "edge_case_handling"
