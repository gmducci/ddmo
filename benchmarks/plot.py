"""Regenerate publication figures from saved numerical data; no optimization."""

import argparse
import importlib
import json
from pathlib import Path

from benchmarks.common.data import ROOT, digest, result_data

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
    expected = {Path(item["path"]).name for item in manifest["files"]
                if Path(item["path"]).parent == Path("benchmarks/figures")}
    actual = {p.name for p in FIGURES.iterdir() if p.suffix in (".pdf", ".png")}
    if actual != expected:
        raise ValueError("Unexpected publication artwork")
    print("Verified three figures and 85 saved production runs; no archive dependency.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("all", "noise", "joint"), default="all")
    parser.add_argument("--output-directory", type=Path, default=ROOT / ".local/benchmark_plots")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    for name in (("noise", "joint") if args.experiment == "all" else (args.experiment,)):
        module = importlib.import_module(f"benchmarks.{name}.postprocess")
        with result_data(name) as (config, raw):
            module.render(raw, config, args.output_directory)
    print(f"Regenerated figures in {args.output_directory}")


if __name__ == "__main__":
    main()
