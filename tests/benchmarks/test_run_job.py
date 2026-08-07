from pathlib import Path

from benchmarks.plan_runs import build_jobs, load_config
from benchmarks.run_job import (
    SMOKE_GENERATIONS,
    SMOKE_POPULATION,
    make_runtime_plan,
)

CONFIG = Path(__file__).parents[2] / "benchmarks" / "configs" / "feature_recovery.toml"


def test_smoke_plan_is_small_and_does_not_mutate_job():
    config = load_config(CONFIG)
    job = build_jobs(config, CONFIG)[0]
    original_population = job["population"]

    plan = make_runtime_plan(config, job, smoke=True)

    assert plan.population == SMOKE_POPULATION
    assert plan.generations == SMOKE_GENERATIONS
    assert plan.split_counts == (1, 1, 1)
    assert plan.t_end == 2.0
    assert plan.evaluation_budget == SMOKE_POPULATION * SMOKE_GENERATIONS
    assert job["population"] == original_population


def test_production_plan_preserves_committed_budget_and_splits():
    config = load_config(CONFIG)
    job = build_jobs(config, CONFIG)[0]

    plan = make_runtime_plan(config, job, smoke=False)

    assert plan.population == 40
    assert plan.generations == 15
    assert plan.split_counts == (3, 1, 2)
    assert plan.t_end == 25.0


def test_fixed_library_is_the_oracle_comparison():
    config = load_config(CONFIG)

    assert config["fixed_library"]["g1"] == config["truth"]["g1"]
    assert config["fixed_library"]["g2"] == config["truth"]["g2"]
