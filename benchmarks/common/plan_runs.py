"""Validate a benchmark TOML file and enumerate its deterministic job matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import tomllib

REQUIRED_EXPERIMENT_KEYS = {
    "name",
    "system",
    "description",
    "methods",
    "noise_levels",
    "seeds",
}


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    experiment = config.get("experiment", {})
    missing = REQUIRED_EXPERIMENT_KEYS - experiment.keys()
    if missing:
        raise ValueError(f"missing experiment keys: {sorted(missing)}")

    optimizer = config.get("optimizer", {})
    population = int(optimizer.get("population", 0))
    generations = int(optimizer.get("generations", 0))
    if population <= 0 or generations <= 0:
        raise ValueError("optimizer population and generations must be positive")

    if not experiment["methods"] or not experiment["seeds"]:
        raise ValueError("methods and seeds must not be empty")
    if any(float(noise) < 0 for noise in experiment["noise_levels"]):
        raise ValueError("noise levels must be non-negative")

    return config


def build_jobs(config: dict, source: Path) -> list[dict]:
    experiment = config["experiment"]
    optimizer = config["optimizer"]
    evaluation_budget = int(optimizer["population"]) * int(optimizer["generations"])

    jobs = []
    combinations = itertools.product(
        experiment["methods"],
        experiment["noise_levels"],
        experiment["seeds"],
    )
    for index, (method, noise, seed) in enumerate(combinations, start=1):
        jobs.append(
            {
                "job_id": f"{experiment['name']}-{index:03d}",
                "experiment": experiment["name"],
                "system": experiment["system"],
                "method": method,
                "noise_level": float(noise),
                "data_seed": int(seed),
                "optimizer_seed": int(seed),
                "population": int(optimizer["population"]),
                "generations": int(optimizer["generations"]),
                "evaluation_budget": evaluation_budget,
                "config": str(source),
            }
        )
    return jobs


def write_jobs(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    jobs = build_jobs(config, config_path)

    print(
        f"{config['experiment']['name']}: {len(jobs)} jobs, "
        f"{jobs[0]['evaluation_budget']} candidate evaluations per job"
    )
    for job in jobs:
        print(
            f"{job['job_id']}  method={job['method']:<14} "
            f"noise={job['noise_level']:.2f} seed={job['data_seed']}"
        )

    if args.output:
        write_jobs(args.output, jobs)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
