"""Inspect archived benchmark data or render historical figures locally."""

import argparse
import importlib
import json
from pathlib import Path

from benchmarks.common.data import ROOT, digest, result_data, trajectory_data


def check():
    for name in ("noise", "joint"):
        with result_data(name):
            pass
        with trajectory_data(name):
            pass
    reference = json.loads(
        (ROOT / "benchmarks/data/figure2_tutorial_block04.json").read_text()
    )
    if reference["provenance"]["weak_source_sha256"] != digest(
        ROOT / "src/ddmo/formulation/weak.py"
    ):
        raise ValueError("Figure 2 tutorial reference uses different weak-form weights")
    print("Verified 85 archived production runs and trajectories, plus the corrected Figure 2 tutorial source hash.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("all", "noise", "joint"), default="all")
    parser.add_argument("--output-directory", type=Path, default=ROOT / ".local/benchmark_plots/historical")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--historical", action="store_true", help="render from the archived, pre-correction campaigns")
    parser.add_argument("--diagnostics", action="store_true", help="Render only supplementary trajectory diagnostics")
    parser.add_argument("--library-overview", action="store_true", help="Render only S4: joint trajectories and nonlinear library shapes")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if not args.historical:
        parser.error("The bundled ten-run campaigns use older weak-form weights; pass --historical to render them")
    if args.library_overview:
        if args.experiment == "noise":
            parser.error("--library-overview applies to joint only")
        from benchmarks.diagnostics import render_library_overview
        with result_data("joint") as (config, raw):
            render_library_overview(raw, config, args.output_directory)
        return
    for name in (("noise", "joint") if args.experiment == "all" else (args.experiment,)):
        with result_data(name) as (config, raw):
            if not args.diagnostics:
                module = importlib.import_module(f"benchmarks.{name}.postprocess")
                module.render(raw, config, args.output_directory)
            from benchmarks.diagnostics import render
            render(name, raw, config, args.output_directory)
            if name == "noise":
                from benchmarks.noise.equation_mismatch import render as render_mismatch

                render_mismatch(config, raw, args.output_directory)
    print(f"Rendered historical figures in {args.output_directory}")


if __name__ == "__main__":
    main()
