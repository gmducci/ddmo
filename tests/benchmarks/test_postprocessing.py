import json

import pytest

from benchmarks.postprocessing.feature_recovery import collect_records


def _config():
    return {
        "experiment": {
            "methods": ["fixed_library", "adaptive_library"],
            "seeds": [11],
        },
        "truth": {"g1": 2.0, "g2": 1.0},
        "fixed_library": {"g1": 2.0, "g2": 1.0},
    }


def _payload(*, mode="production", test_nrmse=0.02):
    return {
        "status": "complete",
        "mode": mode,
        "job": {
            "job_id": "feature_recovery-006",
            "experiment": "feature_recovery",
            "method": "adaptive_library",
            "data_seed": 11,
        },
        "selected": {
            "variables": {"g1": 2.03, "g2": 1.02},
            "validation": {"rollout_nrmse": 0.04},
            "coefficient_metrics": {
                "relative_l2_error": 0.03,
                "support_accuracy": 1.0,
                "active_coefficients": 5,
            },
        },
        "test": {"rollout_nrmse": test_nrmse},
        "actual_candidate_evaluations": 600,
        "failed_candidate_count": 120,
        "wall_time_seconds": 100.0,
        "provenance": {"git_commit": "abc", "git_dirty": False},
    }


def test_collect_records_computes_parameter_and_failure_metrics(tmp_path):
    (tmp_path / "valid.json").write_text(json.dumps(_payload()))

    records, payloads = collect_records(tmp_path, _config())

    assert len(records) == len(payloads) == 1
    record = records[0]
    assert record.g1_absolute_error == pytest.approx(0.03)
    assert record.g2_absolute_error == pytest.approx(0.02)
    assert record.g1_relative_error == pytest.approx(0.015)
    assert record.g2_relative_error == pytest.approx(0.02)
    assert record.failed_candidate_fraction == 0.2


def test_collect_records_excludes_smoke_and_nonfinite_results(tmp_path):
    (tmp_path / "smoke.json").write_text(json.dumps(_payload(mode="smoke")))
    (tmp_path / "failed.json").write_text(json.dumps(_payload(test_nrmse=None)))

    records, payloads = collect_records(tmp_path, _config())

    assert records == []
    assert payloads == []
