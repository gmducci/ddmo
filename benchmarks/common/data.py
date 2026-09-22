"""Portable, checksummed numerical inputs for the publication figures."""

import hashlib
import json
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tomllib

from benchmarks.common.plan_runs import build_jobs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/data"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def trajectory_data(experiment):
    """Load the deduplicated, exact saved trajectories for the SI diagnostics."""
    info = json.loads((DATA / "manifest.json").read_text())["datasets"][experiment]["trajectories"]
    path = DATA / info["file"]
    if digest(path) != info["sha256"]:
        raise ValueError(f"Saved trajectory data changed: {path}")
    with np.load(path, allow_pickle=False) as archive:
        runs = {}
        for job, mapping in info["arrays"].items():
            arrays = {name: archive[key] for name, key in mapping.items()}
            splits = {"test": 2} if experiment == "noise" else {"train": 2, "validation": 1, "test": 2}
            for split, count in splits.items():
                for kind in ("clean", "noisy", "prediction"):
                    if arrays[f"{split}_{kind}"].shape != (count, len(arrays["time"]), 3):
                        raise ValueError(f"Trajectory shape mismatch: {job}, {split}, {kind}")
            runs[job] = arrays
        yield runs


def validate(config, raw, methods, config_hash, commits=None):
    expected = {job["job_id"]: job for job in build_jobs(config, Path("config.toml"))
                if job["method"] in methods}
    paths = {p.stem: p for p in raw.glob("*.json") if p.name != "campaign_manifest.json"}
    if paths.keys() != expected.keys():
        raise ValueError(f"Run matrix mismatch: missing={expected.keys()-paths.keys()}, extra={paths.keys()-expected.keys()}")
    observed_commits = set()
    for job_id, path in paths.items():
        p = json.loads(path.read_text())
        job = expected[job_id]
        if p["status"] != "complete" or p["mode"] != "production":
            raise ValueError(f"Not a completed production run: {job_id}")
        for key, value in job.items():
            if key != "config" and p["job"].get(key) != value:
                raise ValueError(f"Job setting {key} mismatch: {job_id}")
        provenance = p["provenance"]
        if provenance["config_sha256"] != config_hash or provenance["git_dirty"]:
            raise ValueError(f"Source provenance mismatch: {job_id}")
        observed_commits.add(provenance["git_commit"])
        if p["actual_candidate_evaluations"] != job["evaluation_budget"]:
            raise ValueError(f"Incomplete evaluation budget: {job_id}")
        counts = {"train": job.get("training_trajectories", config["data"]["train_trajectories"]),
                  "validation": config["data"]["validation_trajectories"],
                  "test": config["data"]["test_trajectories"]}
        if p["data"]["split_counts"] != counts:
            raise ValueError(f"Data split mismatch: {job_id}")
        if set(p["selected"]["objectives"]) != set(config["optimizer"]["objectives"]):
            raise ValueError(f"Objective mismatch: {job_id}")
        if not np.isfinite(p["selected"]["validation"]["rollout_nrmse"]):
            raise ValueError(f"Invalid validation selection: {job_id}")
        if not np.all(np.isfinite(p["selected"]["coefficients"])):
            raise ValueError(f"Non-finite coefficients: {job_id}")
    if len(observed_commits) != 1 or (commits is not None and observed_commits != set(commits)):
        raise ValueError("Mixed or unexpected production source commits")


@contextmanager
def result_data(experiment, raw_directory=None):
    if raw_directory is not None:
        config_path = ROOT / "benchmarks" / experiment / "config.toml"
        config = tomllib.loads(config_path.read_text())
        raw = raw_directory.resolve()
        validate(config, raw, config["experiment"]["methods"], digest(config_path))
        # Only job JSON files are passed to the renderer, excluding supervisor metadata.
        with tempfile.TemporaryDirectory(prefix="ddmo-results-") as temp:
            folder = Path(temp)
            for path in raw.glob("*.json"):
                if path.name != "campaign_manifest.json":
                    (folder / path.name).write_bytes(path.read_bytes())
            yield config, folder
        return
    manifest = json.loads((DATA / "manifest.json").read_text())["datasets"][experiment]
    archive = DATA / manifest["file"]
    if digest(archive) != manifest["sha256"]:
        raise ValueError(f"Saved numerical data changed: {archive}")
    with tempfile.TemporaryDirectory(prefix="ddmo-figure-data-") as temp:
        raw = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            config_bytes = bundle.read("config.toml")
            if hashlib.sha256(config_bytes).hexdigest() != manifest["config_sha256"]:
                raise ValueError("Saved configuration hash mismatch")
            config = tomllib.loads(config_bytes.decode())
            for filename, checksum in manifest["source_files"].items():
                content = bundle.read("runs/" + filename)
                if Path(filename).name != filename or hashlib.sha256(content).hexdigest() != checksum:
                    raise ValueError(f"Saved result hash mismatch: {filename}")
                (raw / filename).write_bytes(content)
        validate(config, raw, manifest["methods"], manifest["config_sha256"], manifest["production_commits"])
        yield config, raw
