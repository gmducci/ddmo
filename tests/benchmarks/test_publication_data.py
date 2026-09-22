import csv
import json

import numpy as np
import pytest

from benchmarks.common.data import DATA, ROOT, result_data, validate
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
