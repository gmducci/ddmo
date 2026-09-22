"""Strict, resumable execution for publication benchmark campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.common.plan_runs import build_jobs, load_config


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_commit(repo_root: Path) -> str:
    """Return HEAD, refusing to run if tracked or untracked files are dirty."""
    status = git_output(repo_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise SystemExit(
            "Refusing publication campaign from a dirty worktree. Commit or "
            f"remove all changes first:\n{preview}"
        )
    return git_output(repo_root, "rev-parse", "HEAD")


def is_complete(
    path: Path,
    *,
    config_hash: str,
    git_commit: str,
    job: dict[str, Any],
) -> bool:
    """Validate one cached result against the exact current campaign."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    recorded_job = payload.get("job", {})
    return (
        payload.get("status") == "complete"
        and payload.get("mode") == "production"
        and provenance.get("config_sha256") == config_hash
        and provenance.get("git_commit") == git_commit
        and provenance.get("git_dirty") is False
        and recorded_job.get("job_id") == job["job_id"]
        and int(recorded_job.get("evaluation_budget", -1))
        == int(job["evaluation_budget"])
        and int(payload.get("actual_candidate_evaluations", -1))
        == int(job["evaluation_budget"])
        and path.with_suffix(".npz").exists()
    )


def _run_job(
    repo_root: Path,
    config_path: Path,
    runner_module: str,
    raw_directory: Path,
    job_id: str,
    *,
    force: bool,
) -> int:
    command = [
        sys.executable,
        "-m",
        runner_module,
        str(config_path),
        "--job-id",
        job_id,
    ]
    if force:
        command.append("--force")
    log_directory = raw_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    with (log_directory / f"{job_id}.log").open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=repo_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def run_campaign(
    *,
    config_path: Path,
    runner_module: str,
    postprocess_module: str,
    workers: int,
    label: str,
) -> None:
    if workers < 1:
        raise SystemExit("--workers must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    git_commit = require_clean_commit(repo_root)
    config = load_config(config_path)
    jobs = build_jobs(config, config_path)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    raw_directory = (
        repo_root / ".local/benchmark_runs" / str(config["experiment"]["name"]) / "raw"
    )

    pending: list[tuple[dict[str, Any], bool]] = []
    for job in jobs:
        output = raw_directory / f"{job['job_id']}.json"
        if not is_complete(
            output,
            config_hash=config_hash,
            git_commit=git_commit,
            job=job,
        ):
            pending.append((job, output.exists() or output.with_suffix(".npz").exists()))
    print(
        f"{label}: commit={git_commit[:12]}, config={config_hash[:12]}, "
        f"{len(jobs) - len(pending)}/{len(jobs)} verified; "
        f"running {len(pending)} with {workers} workers",
        flush=True,
    )

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_job,
                repo_root,
                config_path,
                runner_module,
                raw_directory,
                str(job["job_id"]),
                force=force,
            ): str(job["job_id"])
            for job, force in pending
        }
        base = len(jobs) - len(pending)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            job_id = futures[future]
            if future.result() != 0:
                failed.append(job_id)
            print(
                f"{label} progress: {base + completed_count}/{len(jobs)} "
                f"({job_id})",
                flush=True,
            )

    if failed:
        print(f"{label}: retrying {len(failed)} failed jobs once", flush=True)
        failed = [
            job_id
            for job_id in failed
            if _run_job(
                repo_root,
                config_path,
                runner_module,
                raw_directory,
                job_id,
                force=True,
            )
            != 0
        ]
    incomplete = [
        str(job["job_id"])
        for job in jobs
        if not is_complete(
            raw_directory / f"{job['job_id']}.json",
            config_hash=config_hash,
            git_commit=git_commit,
            job=job,
        )
    ]
    if failed or incomplete:
        raise SystemExit(
            f"{label} incomplete; failed={sorted(set(failed))}, "
            f"missing_or_stale={incomplete}. See {raw_directory / 'logs'}."
        )

    subprocess.run(
        [sys.executable, "-m", postprocess_module, "--raw-directory", str(raw_directory),
         "--output-directory", str(raw_directory.parent / "figures")], cwd=repo_root, check=True
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": config["experiment"]["name"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": False,
        "config_path": str(config_path.relative_to(repo_root)),
        "config_sha256": config_hash,
        "python": sys.version.split()[0],
        "workers": workers,
        "job_count": len(jobs),
        "evaluations_per_job": int(jobs[0]["evaluation_budget"]),
        "total_candidate_evaluations": sum(
            int(job["evaluation_budget"]) for job in jobs
        ),
        "job_ids": [job["job_id"] for job in jobs],
    }
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{label}: verified, postprocessed, and complete", flush=True)


def campaign_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", type=Path)
    return parser
