"""
test_reporter.py — 10 tests for ModelReporter.
"""

import json
import pytest
from src.checks.accuracy    import AccuracyCheck, CheckResult
from src.reporter.model_reporter import ModelReporter


@pytest.fixture
def sample_results(perfect, binary_dataset):
    X, y = binary_dataset
    return [
        AccuracyCheck(threshold=0.90).run(perfect.predict, X, y),
        AccuracyCheck(threshold=0.50).run(perfect.predict, X, y),
    ]

@pytest.fixture
def reporter(sample_results):
    return ModelReporter(sample_results, model_name="TestModel-v1")


class TestModelReporter:

    def test_to_dict_has_required_keys(self, reporter):
        d = reporter.to_dict()
        for key in ("timestamp", "model_name", "summary", "results"):
            assert key in d

    def test_summary_total_correct(self, reporter, sample_results):
        summary = reporter.to_dict()["summary"]
        assert summary["total"] == len(sample_results)

    def test_summary_passed_correct(self, reporter):
        summary = reporter.to_dict()["summary"]
        assert summary["passed"] == 2   # both checks pass for perfect model

    def test_pass_rate_is_1_when_all_pass(self, reporter):
        assert reporter.to_dict()["summary"]["pass_rate"] == 1.0

    def test_model_name_in_output(self, reporter):
        assert reporter.to_dict()["model_name"] == "TestModel-v1"

    def test_save_json_creates_file(self, reporter, tmp_path):
        path = tmp_path / "report.json"
        reporter.save_json(str(path))
        assert path.exists()

    def test_saved_json_is_valid(self, reporter, tmp_path):
        path = tmp_path / "report.json"
        reporter.save_json(str(path))
        data = json.loads(path.read_text())
        assert "summary" in data
        assert "results" in data

    def test_creates_parent_directory(self, reporter, tmp_path):
        path = tmp_path / "nested" / "dir" / "report.json"
        reporter.save_json(str(path))
        assert path.exists()

    def test_assert_all_pass_no_raise_when_all_pass(self, reporter):
        reporter.assert_all_pass()   # should not raise

    def test_assert_all_pass_raises_on_failure(self):
        bad = CheckResult(
            check_name="failing_check",
            passed=False,
            message="Intentional failure",
            severity="ERROR",
        )
        reporter = ModelReporter([bad], model_name="BadModel")
        with pytest.raises(AssertionError, match="failing_check"):
            reporter.assert_all_pass()
