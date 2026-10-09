"""Check the frozen corrected paper results and their file hashes.

Run ``python -m benchmarks.freeze`` after cloning the repository.
Use ``--write-manifest`` only when intentionally replacing paper artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import tomllib

from benchmarks.common.plan_runs import build_jobs

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "benchmarks/data/current"
FIGURES = ROOT / "benchmarks/figures"
MANIFEST = CURRENT / "manifest.json"
METHODS = {"noise": {"strong", "weak_optimized", "weak_optimized_sg"},
           "joint": {"weak_joint_sg"}}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory() -> dict[str, str]:
    paths = sorted(
        path for folder in (CURRENT, FIGURES) for path in folder.rglob("*")
        if path.is_file() and path != MANIFEST
    )
    return {str(path.relative_to(ROOT)): _hash(path) for path in paths}


def _load_payloads(name: str) -> tuple[dict, dict[str, dict]]:
    folder = CURRENT / name
    config_path = folder / "config.toml"
    config = tomllib.loads(config_path.read_text())
    source = json.loads((folder / "source.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if report["status"] != "complete_local_preview":
        raise ValueError(f"Incomplete {name} campaign")
    if source["config_sha256"] != _hash(config_path):
        raise ValueError(f"Config hash differs for {name}")
    if source["weak_py_sha256"] != _hash(ROOT / "src/ddmo/formulation/weak.py"):
        raise ValueError(f"Weak formulation differs from the {name} runs")
    expected = {job["job_id"] for job in build_jobs(config, config_path)
                if job["method"] in METHODS[name]}
    payloads = {}
    for path in (folder / "raw").glob("*.json"):
        payload = json.loads(path.read_text())
        job_id = payload["job"]["job_id"]
        if job_id != path.stem or job_id in payloads:
            raise ValueError(f"Duplicate or misnamed {name} run: {path}")
        payloads[job_id] = payload
    if set(payloads) != expected:
        raise ValueError(f"{name} run matrix differs from its config")
    for job_id, payload in payloads.items():
        if payload["status"] != "complete" or payload["mode"] != "production":
            raise ValueError(f"Incomplete run: {job_id}")
        if payload["provenance"]["config_sha256"] != source["config_sha256"]:
            raise ValueError(f"Config provenance differs: {job_id}")
        if payload["actual_candidate_evaluations"] != payload["job"]["evaluation_budget"]:
            raise ValueError(f"Incomplete evaluation budget: {job_id}")
        with np.load(folder / "raw" / (job_id + ".npz"), allow_pickle=False) as archive:
            clean = archive["test_clean"]
            prediction = archive["test_prediction"]
            score = payload["test"]["rollout_nrmse"]
            if score is None:
                if not np.all(np.isnan(prediction)):
                    raise ValueError(f"Failed rollout has finite values: {job_id}")
            else:
                rmse = np.sqrt(np.mean((prediction - clean) ** 2, axis=(0, 1)))
                scale = np.std(clean, axis=(0, 1))
                measured = float(np.mean(rmse / scale))
                if not math.isclose(measured, score, rel_tol=1e-11, abs_tol=1e-12):
                    raise ValueError(f"Saved trajectory score differs: {job_id}")
    figure = "figure_1_noise_robustness" if name == "noise" else "figure_2_library_recovery"
    number = "1" if name == "noise" else "2"
    for suffix in ("png", "pdf"):
        key = f"figure_{number}_{suffix}_sha256"
        if report[key] != _hash(FIGURES / f"{figure}.{suffix}"):
            raise ValueError(f"Main figure hash differs: {figure}.{suffix}")
    return config, payloads


def _check_paper_pdf(stem: str, count: int, jobs: set[str]) -> None:
    metadata = json.loads((FIGURES / f"{stem}.json").read_text())
    pages = metadata["pages"]
    listed = {job for page in pages
              for job in page.get("job_ids", [page.get("job_id")])}
    if len(pages) != count or listed != jobs:
        raise ValueError(f"Figure page manifest differs: {stem}")
    pdf = (FIGURES / f"{stem}.pdf").read_bytes()
    page_counts = [int(value) for value in re.findall(rb"/Count\s+(\d+)", pdf)]
    sizes = [(float(w), float(h)) for w, h in
             re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)]
    if count not in page_counts or not sizes:
        raise ValueError(f"PDF page count or dimensions missing: {stem}")
    if any(abs(w - 595.276) > .5 or abs(h - 841.89) > .5 for w, h in sizes):
        raise ValueError(f"PDF is not A4 portrait: {stem}")


def check() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema_version"] != 1 or manifest["files"] != _inventory():
        raise ValueError("Frozen file inventory or checksum differs")
    for path, digest in manifest["source_hashes"].items():
        if _hash(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    _, noise = _load_payloads("noise")
    _, joint = _load_payloads("joint")
    _check_paper_pdf("figure_s1_noise_all_models_a4", 25, set(noise))
    _check_paper_pdf("figure_s2_joint_all_models_a4", 10, set(joint))
    print("Verified 75 noise and 10 joint selections, trajectories, A4 pages, and frozen checksums.")


def write_manifest() -> None:
    manifest = {
        "schema_version": 1,
        "source_hashes": {
            "src/ddmo/formulation/weak.py": _hash(ROOT / "src/ddmo/formulation/weak.py"),
            "benchmarks/si_paper.py": _hash(ROOT / "benchmarks/si_paper.py"),
        },
        "files": _inventory(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    check()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    else:
        check()


if __name__ == "__main__":
    main()
