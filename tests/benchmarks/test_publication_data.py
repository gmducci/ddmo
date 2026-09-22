import csv
import json

import numpy as np
import pytest

from benchmarks.common.data import DATA, ROOT, result_data, trajectory_data, validate
from benchmarks.common.goodwin import generate_batch, make_runtime_plan
from benchmarks.common.plan_runs import build_jobs, load_config


@pytest.mark.parametrize(("name", "count"), [("noise", 75), ("joint", 10)])
def test_saved_inputs_are_complete_and_independent_of_local_archive(name, count):
    with result_data(name) as (config, raw):
        paths = list(raw.glob("*.json"))
        assert len(paths) == count
        for path in paths:
            payload = json.loads(path.read_text())
            assert payload["job"]["method"] != "weak_fixed"
            assert len(payload["selected"]["objectives"]) == 3


def test_joint_keeps_the_exact_two_one_two_initial_condition_assignments():
    config_path = ROOT / "benchmarks/joint/config.toml"
    config = load_config(config_path)
    rows = list(csv.DictReader(config_path.with_name("lhs_initial_conditions.csv").open()))
    jobs = build_jobs(config, config_path)
    assert len(jobs) == 10
    for job in jobs:
        plan = make_runtime_plan(config, job, smoke=False)
        assert plan.split_counts == (2, 1, 2)
        batch = generate_batch(config, plan)
        expected = [[float(row[key]) for key in ("x0", "y0", "z0")] for row in rows
                    if int(row["block"]) == job["ic_block"] and row["role"] != "train_3"]
        np.testing.assert_array_equal(batch.initial_conditions, expected)
        assert batch.x_clean.shape == (5, 2001, 3)


def test_data_validation_rejects_missing_mixed_and_wrong_split_results():
    info = json.loads((DATA / "manifest.json").read_text())["datasets"]["joint"]
    with result_data("joint") as (config, raw):
        path = next(raw.glob("*.json"))
        original = path.read_text()
        path.unlink()
        with pytest.raises(ValueError, match="matrix mismatch"):
            validate(config, raw, info["methods"], info["config_sha256"])
        path.write_text(original)
        payload = json.loads(original)
        payload["data"]["split_counts"]["train"] = 3
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Data split mismatch"):
            validate(config, raw, info["methods"], info["config_sha256"])
        path.write_text(original)
        payload = json.loads(original)
        payload["provenance"]["git_commit"] = "different-source"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Mixed or unexpected"):
            validate(config, raw, info["methods"], info["config_sha256"])


@pytest.mark.parametrize("name", ["noise", "joint"])
def test_diagnostic_trajectories_match_saved_test_scores_and_run_matrix(name):
    with result_data(name) as (_, raw), trajectory_data(name) as trajectories:
        payloads = {p.stem: json.loads(p.read_text()) for p in raw.glob("*.json")}
        assert trajectories.keys() == payloads.keys()
        failed = []
        paired = {}
        for job_id, payload in payloads.items():
            arrays = trajectories[job_id]
            clean, prediction = arrays["test_clean"], arrays["test_prediction"]
            score = payload["test"]["rollout_nrmse"]
            if score is None:
                assert np.all(np.isnan(prediction))
                failed.append(payload["job"])
            else:
                rmse = np.sqrt(np.mean((prediction - clean) ** 2, axis=(0, 1)))
                scale = np.std(clean, axis=(0, 1))
                np.testing.assert_allclose(np.mean(rmse / scale), score, rtol=1e-12)
            if name == "noise":
                job = payload["job"]
                key = (job["noise_level"], job["data_seed"])
                reference = paired.setdefault(key, arrays)
                for field in ("time", "test_clean", "test_noisy"):
                    np.testing.assert_array_equal(arrays[field], reference[field])
            else:
                assert arrays["train_prediction"].shape == (2, 2001, 3)
                assert arrays["validation_prediction"].shape == (1, 2001, 3)
                assert np.all(np.isfinite(arrays["train_prediction"]))
                assert np.all(np.isfinite(arrays["validation_prediction"]))
        if name == "noise":
            assert {(j["method"], j["noise_level"], j["data_seed"]) for j in failed} == {
                ("strong", 0.2, 11), ("strong", 0.2, 44)
            }
        else:
            assert failed == []


def test_library_overview_preserves_example_and_conditions_on_active_families():
    from benchmarks.diagnostics import library_summary
    from benchmarks.joint.postprocess import _canonical_feature

    with result_data("joint") as (config, raw):
        payloads = [json.loads(p.read_text()) for p in raw.glob("*.json")]
    representative, parameters = library_summary(payloads, config)
    assert representative["job"]["ic_block"] == 4
    assert len(parameters["gamma_1"]) == 6
    assert len(parameters["gamma_2"]) == 4
    np.testing.assert_allclose(np.median(parameters["gamma_1"]), 9.39, atol=.005)
    np.testing.assert_allclose(np.median(parameters["gamma_2"]), 5.44, atol=.005)
    # Inactive search variables must not affect conditional shape summaries.
    for p in payloads:
        selected = p["selected"]
        for family, variable in (("Hill(", "gamma_1"), ("exp(", "gamma_2")):
            active = any(_canonical_feature(n).startswith(family) and np.any(np.abs(row) > 1e-6)
                         for n, row in zip(selected["feature_names"], np.asarray(selected["coefficients"])))
            if not active:
                selected["variables"][variable] = 9999
    rerun, changed = library_summary(list(reversed(payloads)), config)
    assert rerun["job"]["job_id"] == representative["job"]["job_id"]
    for variable in parameters:
        np.testing.assert_array_equal(changed[variable], parameters[variable])
