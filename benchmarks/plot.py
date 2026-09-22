"""Regenerate publication figures from saved numerical data; no optimization."""

import argparse
import importlib
import json
from pathlib import Path

from benchmarks.common.data import ROOT, digest, result_data, trajectory_data

FIGURES = ROOT / "benchmarks/figures"


def check():
    manifest = json.loads((FIGURES / "manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Missing or changed figure/input: {path}")
    for name in ("noise", "joint"):
        with result_data(name):
            pass
        with trajectory_data(name):
            pass
    expected = {Path(item["path"]).name for item in manifest["files"]
                if Path(item["path"]).parent == Path("benchmarks/figures")}
    actual = {p.name for p in FIGURES.iterdir() if p.suffix in (".pdf", ".png")}
    if actual != expected:
        raise ValueError("Unexpected publication artwork")
    print(f"Verified {len(manifest['figures'])} figures and all 85 saved production runs and trajectories; no archive dependency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("all", "noise", "joint"), default="all")
    parser.add_argument("--output-directory", type=Path, default=ROOT / ".local/benchmark_plots")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Render only supplementary trajectory diagnostics")
    parser.add_argument("--library-overview", action="store_true", help="Render only S4: joint trajectories and nonlinear library shapes")
    args = parser.parse_args()
    if args.check:
        check()
        return
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
    print(f"Regenerated figures in {args.output_directory}")


if __name__ == "__main__":
    main()
