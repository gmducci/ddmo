"""Run one noise-robustness benchmark job."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common.goodwin import (
    default_output_paths,
    execute_goodwin_job,
    make_runtime_plan,
    resolve_job,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--job-id", required=True, help="Job name or zero-based index")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="JSON output path")
    parser.add_argument("--artifact-output", type=Path, help="NPZ output path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        args.config.resolve()
        if args.config
        else repo_root / "benchmarks/noise/config.toml"
    )
    try:
        config, job = resolve_job(config_path, args.job_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if config["experiment"]["system"] != "goodwin":
        raise SystemExit("Noise benchmark requires a Goodwin noise config")
    plan = make_runtime_plan(config, job, smoke=args.smoke)
    default_json, default_artifact = default_output_paths(repo_root, plan)
    json_path = args.output.resolve() if args.output else default_json
    artifact_path = (
        args.artifact_output.resolve() if args.artifact_output else default_artifact
    )
    existing = [path for path in (json_path, artifact_path) if path.exists()]
    if existing and not args.force:
        rendered = ", ".join(str(path) for path in existing)
        raise SystemExit(f"Output already exists: {rendered} (use --force to replace)")

    print(
        f"running {job['job_id']} ({job['method']}, noise={job['noise_level']:.2f}) "
        f"in {plan.mode} mode: {plan.population} x {plan.generations} = "
        f"{plan.evaluation_budget} evaluations"
    )
    payload, artifacts = execute_goodwin_job(
        config,
        plan,
        config_path=config_path,
        repo_root=repo_root,
        adaptive_exponent=False,
    )
    write_outputs(json_path, artifact_path, payload, artifacts)
    print(
        f"completed in {payload['wall_time_seconds']:.2f}s; "
        f"selected validation NRMSE={payload['selected']['validation']['rollout_nrmse']:.6g}; "
        f"test NRMSE={payload['test']['rollout_nrmse']:.6g}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {artifact_path}")


if __name__ == "__main__":
    main()
