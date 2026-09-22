"""Run or resume a full production benchmark and generate its figures."""

import argparse
from pathlib import Path

from benchmarks.common.run_campaign import run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("noise", "joint"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run_campaign(config_path=Path(__file__).parent / args.experiment / "config.toml",
                 runner_module=f"benchmarks.{args.experiment}.run",
                 postprocess_module=f"benchmarks.{args.experiment}.postprocess",
                 workers=args.workers, label=args.experiment)


if __name__ == "__main__":
    main()
