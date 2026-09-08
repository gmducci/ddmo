"""Generate the reviewer-facing computational workflow for Figure 3."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")


COLORS = {
    "ink": "#17202A",
    "muted": "#52606D",
    "line": "#9AA5B1",
    "design": "#E8F1F8",
    "design_edge": "#4C78A8",
    "data": "#E9F4EF",
    "data_edge": "#2A7F62",
    "identify": "#FFF2DF",
    "identify_edge": "#D8871E",
    "report": "#F1ECF8",
    "report_edge": "#7656A5",
    "train": "#2F6FB0",
    "validation": "#C26B18",
    "test": "#7656A5",
    "white": "#FFFFFF",
}


def _box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    science: str,
    code: str,
    edge: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.05",
        facecolor=COLORS["white"],
        edgecolor=edge,
        linewidth=1.05,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(
        x + 0.13,
        y + height - 0.18,
        title,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    axis.text(
        x + 0.13,
        y + height - 0.50,
        science,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["ink"],
        linespacing=1.25,
        zorder=4,
    )
    axis.text(
        x + 0.13,
        y + 0.14,
        code,
        ha="left",
        va="bottom",
        fontsize=5.9,
        family="monospace",
        color=COLORS["muted"],
        linespacing=1.22,
        zorder=4,
    )


def _arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    linestyle: str = "-",
    label: str | None = None,
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> None:
    arrow_color = color or COLORS["line"]
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.15,
            linestyle=linestyle,
            color=arrow_color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.10,
            label,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=arrow_color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
            zorder=5,
        )


def build_workflow(output_stem: Path) -> None:
    """Render the full study workflow to SVG, PDF, and PNG."""
    fig, axis = plt.subplots(figsize=(14.4, 8.1))
    fig.patch.set_facecolor("white")
    axis.set(xlim=(0, 14.4), ylim=(0, 8.1))
    axis.axis("off")

    axis.text(
        0.32,
        7.86,
        "Goodwin case study: reproducible computational workflow",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        0.32,
        7.48,
        "Paired Latin-hypercube design · SG-preprocessed weak SINDy · joint library, formulation, and solver optimization",
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
    )

    stages = [
        (0.30, 3.18, "1  EXPERIMENTAL DESIGN", "design", "design_edge"),
        (3.74, 3.18, "2  DATA GENERATION", "data", "data_edge"),
        (7.18, 3.18, "3  MULTI-OBJECTIVE IDENTIFICATION", "identify", "identify_edge"),
        (10.62, 3.48, "4  VALIDATION, TEST & REPORTING", "report", "report_edge"),
    ]
    for x, width, title, fill_name, edge_name in stages:
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.42),
                width,
                6.78,
                boxstyle="round,pad=0.018,rounding_size=0.07",
                facecolor=COLORS[fill_name],
                edgecolor=COLORS[edge_name],
                linewidth=0.9,
                zorder=0,
            )
        )
        axis.text(
            x + 0.16,
            6.95,
            title,
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=COLORS[edge_name],
        )

    _box(
        axis,
        0.52,
        5.26,
        2.74,
        1.40,
        title="Frozen inputs",
        science="Truth, time grid, 10% noise, bounds and\noptimizer budget; 60 frozen LHS initial states",
        code="figure_3/config.toml\nfigure_3/lhs_initial_conditions.csv\nload_config() · build_jobs()",
        edge=COLORS["design_edge"],
    )
    _box(
        axis,
        0.52,
        3.42,
        2.74,
        1.40,
        title="Paired job matrix",
        science="10 IC blocks × Ntrain ∈ {1,2,3}\n30 jobs; fixed data and common optimizer\nrandom numbers within each paired block",
        code="common/plan_runs.py\ncommon/run_campaign.py\nGoodwinRuntimePlan · make_runtime_plan()",
        edge=COLORS["design_edge"],
    )
    _box(
        axis,
        0.52,
        1.02,
        2.74,
        1.82,
        title="Reproducibility guardrails",
        science="Clean Git commit, config SHA-256, package\nversions, data/optimizer seeds and\nresumable jobs",
        code="common/run_campaign.py\nrequire_clean_commit() · is_complete()\ncampaign_manifest.json",
        edge=COLORS["design_edge"],
    )

    _box(
        axis,
        3.96,
        5.26,
        2.74,
        1.40,
        title="Simulate ground truth",
        science="Goodwin ODE; t = 0…200, Δt = 0.1\nSix trajectories generated once per block",
        code="common/goodwin.py::generate_batch()\ngoodwin_eqs.py::GoodwinModel.dynamics()\nODEDataGenerator.generate()",
        edge=COLORS["data_edge"],
    )
    _box(
        axis,
        3.96,
        3.42,
        2.74,
        1.40,
        title="Noise, preprocessing & split",
        science="State-wise Gaussian noise; SG window 21,\norder 3; nested training sets and fixed\nvalidation/test sets",
        code="common/goodwin.py\nODEDataGenerator._add_noise() · _savgol_states()\nsplit_slices() · _weak_training_data()",
        edge=COLORS["data_edge"],
    )
    _box(
        axis,
        3.96,
        1.02,
        2.74,
        1.82,
        title="Leakage-controlled data roles",
        science="TRAIN fits each candidate\nVALIDATION selects one Pareto model\nTEST is opened only after selection",
        code="ODEDataBatch\ntrain_* · validation_* · test_* arrays\nJSON metadata + NPZ trajectories",
        edge=COLORS["data_edge"],
    )

    _box(
        axis,
        7.40,
        5.26,
        2.74,
        1.40,
        title="Parameterized candidate library",
        science="Polynomial degree ≤2 + interactions;\nHill(γ₁) + nuisance exponential(γ₂)",
        code="common/goodwin.py::build_library()\ncomposer.py::LibraryBuilder\n.add_polynomial() · .add_custom()\n.add_exponential()",
        edge=COLORS["identify_edge"],
    )
    _box(
        axis,
        7.40,
        3.16,
        2.74,
        1.66,
        title="Candidate evaluation loop",
        science="Weak integrals → feature matrix → scaled\nSTLSQ fit → rollout/residual diagnostics",
        code="src/ddmo/main.py::identify()\nOptimizer._compute_metrics()\nWeakFormulation.get_lhs()/get_rhs()\nPysindy_Solver.solve()",
        edge=COLORS["identify_edge"],
    )
    _box(
        axis,
        7.40,
        1.02,
        2.74,
        1.60,
        title="NSGA-II search: 80 × 30",
        science="7 variables: γ₁, γ₂, degree, window, stride,\nthreshold and α; 2,400 evaluations per job",
        code="OptimizerSettings · Optimizer._evaluate()\ntrajectory_rmse() · durbin_watson_deviation()\nmake_active_coefficient_count()",
        edge=COLORS["identify_edge"],
    )

    _box(
        axis,
        10.84,
        5.26,
        3.04,
        1.40,
        title="Pareto front & validation selection",
        science="Keep valid nondominated models; select by\nminimum validation rollout NRMSE only",
        code="execute_goodwin_job()\nrollout()\nIdentificationResult",
        edge=COLORS["report_edge"],
    )
    _box(
        axis,
        10.84,
        3.42,
        3.04,
        1.40,
        title="Untouched held-out test",
        science="Two test trajectories; rollout NRMSE,\ncoefficient/support metrics and equations",
        code="rollout() · coefficient_metrics()\nIdentificationResult.symbolic_equations()\n_ManualSINDyModel.simulate()",
        edge=COLORS["report_edge"],
    )
    _box(
        axis,
        10.84,
        1.02,
        3.04,
        1.82,
        title="Reviewer-facing outputs",
        science="30 JSON/NPZ job pairs → CSV summaries,\nquality report and publication PNG/PDF/SVG",
        code="write_outputs() · postprocess.main()\nresults/raw/ · results/aggregated/\nresults/figures/",
        edge=COLORS["report_edge"],
    )

    # Main scientific execution path.
    _arrow(axis, (1.89, 5.26), (1.89, 4.82), color=COLORS["design_edge"])
    _arrow(axis, (3.26, 4.12), (3.96, 5.90), color=COLORS["line"])
    _arrow(axis, (5.33, 5.26), (5.33, 4.82), color=COLORS["data_edge"])
    _arrow(axis, (5.33, 3.42), (5.33, 2.84), color=COLORS["data_edge"])
    _arrow(
        axis,
        (6.70, 4.12),
        (7.40, 5.90),
        color=COLORS["train"],
        label="training data",
    )
    _arrow(axis, (8.77, 5.26), (8.77, 4.82), color=COLORS["identify_edge"])
    _arrow(axis, (8.77, 3.16), (8.77, 2.62), color=COLORS["identify_edge"])
    _arrow(
        axis,
        (10.14, 4.12),
        (10.84, 5.68),
        color=COLORS["identify_edge"],
        label="Pareto models",
        connectionstyle="arc3,rad=-0.12",
    )
    _arrow(axis, (12.36, 5.26), (12.36, 4.82), color=COLORS["validation"])
    _arrow(axis, (12.36, 3.42), (12.36, 2.84), color=COLORS["test"])

    # Leakage-control routes run below the identification boxes, then rise in
    # the gutter before the reporting stage so they never cross model fitting.
    validation_route = [(6.70, 2.12), (6.96, 0.80), (10.45, 0.80), (10.45, 5.50)]
    test_route = [(6.70, 1.56), (7.08, 0.60), (10.56, 0.60), (10.56, 3.96)]
    axis.plot(
        *zip(*validation_route),
        color=COLORS["validation"],
        linewidth=1.15,
        linestyle="--",
        zorder=1,
    )
    axis.plot(
        *zip(*test_route),
        color=COLORS["test"],
        linewidth=1.15,
        linestyle=":",
        zorder=1,
    )
    _arrow(
        axis,
        validation_route[-1],
        (10.84, 5.50),
        color=COLORS["validation"],
        linestyle="--",
        zorder=1,
    )
    _arrow(
        axis,
        test_route[-1],
        (10.84, 3.96),
        color=COLORS["test"],
        linestyle=":",
        zorder=1,
    )
    axis.text(
        7.22,
        0.86,
        "validation only",
        fontsize=6.2,
        color=COLORS["validation"],
        ha="left",
        va="bottom",
    )
    axis.text(
        8.75,
        0.64,
        "held-out test after selection",
        fontsize=6.2,
        color=COLORS["test"],
        ha="left",
        va="bottom",
    )

    # Optimization feedback loop.
    _arrow(
        axis,
        (9.94, 2.62),
        (9.94, 3.16),
        color=COLORS["identify_edge"],
        connectionstyle="arc3,rad=-0.55",
        label="next candidate",
    )

    axis.text(
        0.32,
        0.22,
        "Code roots: figure_3/ → benchmarks/figures/figure_3/  ·  common/ → benchmarks/common/  ·  ddmo/ → src/ddmo/",
        ha="left",
        va="bottom",
        fontsize=6.7,
        family="monospace",
        color=COLORS["muted"],
    )
    axis.text(
        0.32,
        0.06,
        "Entry points: run.py::main()  ·  run_campaign.py::main()  ·  postprocess.py::main()",
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=COLORS["muted"],
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Output stem without extension")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_stem = args.output or (
        repo_root
        / "benchmarks/results/figures/figure_3_computational_workflow"
    )
    build_workflow(output_stem.resolve())
    print(f"wrote {output_stem.with_suffix('.svg')}")
    print(f"wrote {output_stem.with_suffix('.pdf')}")
    print(f"wrote {output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
