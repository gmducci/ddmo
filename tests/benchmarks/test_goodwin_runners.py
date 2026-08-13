import csv
from pathlib import Path

import numpy as np
from pysindy.feature_library import CustomLibrary
from scipy.stats import qmc

from benchmarks.common.goodwin import (
    build_identification_settings,
    build_library,
    generate_batch,
    make_runtime_plan,
    split_slices,
    true_coefficient_matrix,
)
from benchmarks.common.plan_runs import build_jobs, load_config

ROOT = Path(__file__).parents[2]
FIGURE_2_CONFIG = ROOT / "benchmarks/figures/figure_2/config.toml"
FIGURE_3_CONFIG = ROOT / "benchmarks/figures/figure_3/config.toml"
FIGURE_3_LHS = ROOT / "benchmarks/figures/figure_3/lhs_initial_conditions.csv"


def _job_for_method(config, config_path, method):
    return next(
        job for job in build_jobs(config, config_path) if job["method"] == method
    )


def test_figure_2_smoke_plan_and_frozen_split():
    config = load_config(FIGURE_2_CONFIG)
    job = _job_for_method(config, FIGURE_2_CONFIG, "strong")

    plan = make_runtime_plan(config, job, smoke=True)
    batch = generate_batch(config, plan)

    assert plan.population == 12
    assert plan.generations == 3
    assert plan.split_counts == (1, 1, 1)
    assert plan.t_end == 20.0
    assert batch.x_clean.shape == (3, 201, 3)
    assert np.allclose(batch.initial_conditions[0], [0.5, 1.0, 1.0])
    assert np.allclose(batch.initial_conditions[1], [0.2, 1.4, 0.8])
    assert np.allclose(batch.initial_conditions[2], [0.8, 0.6, 1.5])


def test_figure_2_uses_stable_semantic_ids_through_twenty_percent_noise():
    config = load_config(FIGURE_2_CONFIG)
    jobs = build_jobs(config, FIGURE_2_CONFIG)

    assert len(jobs) == 100
    assert config["experiment"]["noise_levels"] == [0.0, 0.05, 0.10, 0.15, 0.20]
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert any(
        job["job_id"] == "weak_noise_poly2-weak_optimized-noise020-seed055"
        and job["noise_level"] == 0.20
        for job in jobs
    )


def test_figure_2_methods_optimize_only_their_configured_parameters():
    config = load_config(FIGURE_2_CONFIG)
    for method, expected_names in {
        "strong": ["threshold", "alpha"],
        "weak_fixed": ["threshold", "alpha"],
        "weak_optimized": ["degree", "window_length", "stride", "threshold", "alpha"],
        "weak_optimized_sg": [
            "degree",
            "window_length",
            "stride",
            "threshold",
            "alpha",
        ],
    }.items():
        job = _job_for_method(config, FIGURE_2_CONFIG, method)
        plan = make_runtime_plan(config, job, smoke=True)
        batch = generate_batch(config, plan)
        train_slice, _, _ = split_slices(plan.split_counts)
        settings = build_identification_settings(
            config,
            plan,
            batch,
            train_slice,
            adaptive_exponent=False,
        )
        optimizer_settings = settings[2]
        assert list(optimizer_settings.all_bounds) == expected_names


def test_strong_preprocessing_preserves_positive_hill_domain_at_high_noise():
    config = load_config(FIGURE_2_CONFIG)
    job = next(
        job
        for job in build_jobs(config, FIGURE_2_CONFIG)
        if job["method"] == "strong" and job["noise_level"] == 0.20
    )
    plan = make_runtime_plan(config, job, smoke=True)
    batch = generate_batch(config, plan)
    train_slice, _, _ = split_slices(plan.split_counts)

    settings = build_identification_settings(
        config,
        plan,
        batch,
        train_slice,
        adaptive_exponent=False,
    )
    data_instance = settings[0]

    assert np.all(data_instance.x_train > 0)


def test_sg_preprocessed_weak_uses_filtered_states_for_fit_and_objective():
    config = load_config(FIGURE_2_CONFIG)
    job = next(
        job
        for job in build_jobs(config, FIGURE_2_CONFIG)
        if job["method"] == "weak_optimized_sg" and job["noise_level"] == 0.20
    )
    plan = make_runtime_plan(config, job, smoke=True)
    batch = generate_batch(config, plan)
    train_slice, _, _ = split_slices(plan.split_counts)
    data_instance = build_identification_settings(
        config,
        plan,
        batch,
        train_slice,
        adaptive_exponent=False,
    )[0]

    assert np.all(data_instance.x_train > 0)
    assert np.array_equal(data_instance.x_train, data_instance.x_raw)
    assert not np.array_equal(data_instance.x_train, batch.x_noisy[train_slice])


def test_goodwin_hill_libraries_are_finite_outside_positive_domain():
    figure_2 = load_config(FIGURE_2_CONFIG)
    fixed_library, _ = build_library(figure_2, adaptive_exponent=False)
    fixed_hill = fixed_library.build({})[-1][0]

    figure_3 = load_config(FIGURE_3_CONFIG)
    adaptive_library, _ = build_library(figure_3, adaptive_exponent=True)
    adaptive_hill = adaptive_library.build({"gamma_1": 9.5, "gamma_2": 2.0})[-2][0]

    values = np.asarray([-0.1, 0.0, 0.1])
    assert np.all(np.isfinite(fixed_hill(values)))
    assert np.all(np.isfinite(adaptive_hill(values)))


def test_figure_2_adds_degree_two_polynomials_and_interactions():
    config = load_config(FIGURE_2_CONFIG)
    library, _ = build_library(config, adaptive_exponent=False)
    functions = library.build({})

    assert len(functions) == 4
    assert [function.__code__.co_argcount for function, _ in functions] == [1, 1, 2, 1]
    assert functions[0][1]("x") == "x"
    assert functions[1][1]("x") == "x^2"
    assert functions[2][1]("x", "y") == "x*y"
    assert functions[3][1]("z").startswith("1/(1+z^9.5")
    feature_library = CustomLibrary(
        [function for function, _ in functions],
        [name for _, name in functions],
    ).fit(np.ones((4, 3)))
    assert feature_library.get_feature_names(["x", "y", "z"]) == [
        "x",
        "y",
        "z",
        "x^2",
        "y^2",
        "z^2",
        "x*y",
        "x*z",
        "y*z",
        "1/(1+x^9.5)",
        "1/(1+y^9.5)",
        "1/(1+z^9.5)",
    ]


def test_figure_3_joint_parameter_set_and_truth_matrix():
    config = load_config(FIGURE_3_CONFIG)
    assert config["optimizer"]["objectives"] == [
        "trajectory_rmse",
        "durbin_watson_deviation",
        "active_coefficients",
    ]
    job = _job_for_method(config, FIGURE_3_CONFIG, "weak_joint_sg")
    plan = make_runtime_plan(config, job, smoke=True)
    batch = generate_batch(config, plan)
    train_slice, _, _ = split_slices(plan.split_counts)

    settings = build_identification_settings(
        config,
        plan,
        batch,
        train_slice,
        adaptive_exponent=True,
    )
    optimizer_settings = settings[2]

    assert list(optimizer_settings.all_bounds) == [
        "gamma_1",
        "gamma_2",
        "degree",
        "window_length",
        "stride",
        "threshold",
        "alpha",
    ]
    truth = true_coefficient_matrix(
        ["x", "y", "z", "1/(1+x^9.5)", "1/(1+y^9.5)", "1/(1+z^9.5)"],
        config,
    )
    assert np.count_nonzero(truth) == 6
    assert np.allclose(truth[0], [-0.2, 1.0, 0.0])
    assert np.allclose(truth[-1], [1.0, 0.0, 0.0])


def test_figure_3_paired_lhs_job_matrix_and_independent_seeds():
    config = load_config(FIGURE_3_CONFIG)
    jobs = build_jobs(config, FIGURE_3_CONFIG)

    assert len(jobs) == 30
    assert {job["training_trajectories"] for job in jobs} == {1, 2, 3}
    assert {job["ic_block"] for job in jobs} == set(range(1, 11))
    assert all(job["data_seed"] != job["optimizer_seed"] for job in jobs)
    for block in range(1, 11):
        block_jobs = [job for job in jobs if job["ic_block"] == block]
        assert len({job["data_seed"] for job in block_jobs}) == 1
        assert len({job["optimizer_seed"] for job in block_jobs}) == 1


def test_figure_3_lhs_design_is_frozen_and_reproducible():
    with FIGURE_3_LHS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    stored = np.asarray(
        [[float(row[name]) for name in ("x0", "y0", "z0")] for row in rows]
    )
    unit_design = qmc.LatinHypercube(d=3, seed=20260813).random(60)
    regenerated = qmc.scale(
        unit_design,
        [0.2, 0.6, 0.6],
        [1.0, 1.5, 1.5],
    )

    assert stored.shape == (60, 3)
    assert np.allclose(stored, regenerated, atol=5e-13, rtol=0.0)
    assert [row["role"] for row in rows[:6]] == [
        "train_1",
        "train_2",
        "train_3",
        "validation",
        "test_1",
        "test_2",
    ]


def test_figure_3_lhs_training_sizes_are_nested_and_paired():
    config = load_config(FIGURE_3_CONFIG)
    jobs = [job for job in build_jobs(config, FIGURE_3_CONFIG) if job["ic_block"] == 1]
    batches = {}
    for job in jobs:
        plan = make_runtime_plan(config, job, smoke=True)
        batches[job["training_trajectories"]] = generate_batch(config, plan)

    one, two, three = (batches[size] for size in (1, 2, 3))
    assert one.x_clean.shape == (4, 201, 3)
    assert two.x_clean.shape == (5, 201, 3)
    assert three.x_clean.shape == (6, 201, 3)
    assert np.array_equal(one.x_noisy[0], three.x_noisy[0])
    assert np.array_equal(two.x_noisy[:2], three.x_noisy[:2])
    assert np.array_equal(one.x_noisy[1:], three.x_noisy[3:])
    assert np.array_equal(two.x_noisy[2:], three.x_noisy[3:])


def test_figure_3_uses_sg_and_full_parametric_candidate_library():
    config = load_config(FIGURE_3_CONFIG)
    job = _job_for_method(config, FIGURE_3_CONFIG, "weak_joint_sg")
    plan = make_runtime_plan(config, job, smoke=True)
    batch = generate_batch(config, plan)
    train_slice, _, _ = split_slices(plan.split_counts)
    data_instance, library, *_ = build_identification_settings(
        config,
        plan,
        batch,
        train_slice,
        adaptive_exponent=True,
    )
    functions = library.build({"gamma_1": 9.5, "gamma_2": 2.0})

    assert not np.array_equal(data_instance.x_train, batch.x_noisy[train_slice])
    assert len(functions) == 5
    assert [function.__code__.co_argcount for function, _ in functions] == [
        1,
        1,
        2,
        1,
        1,
    ]
    feature_library = CustomLibrary(
        [function for function, _ in functions],
        [name for _, name in functions],
    ).fit(np.ones((4, 3)))
    assert feature_library.get_feature_names(["x", "y", "z"]) == [
        "x",
        "y",
        "z",
        "x^2",
        "y^2",
        "z^2",
        "x*y",
        "x*z",
        "y*z",
        "1/(1+x^9.5)",
        "1/(1+y^9.5)",
        "1/(1+z^9.5)",
        "exp(-2.00*x)",
        "exp(-2.00*y)",
        "exp(-2.00*z)",
    ]
