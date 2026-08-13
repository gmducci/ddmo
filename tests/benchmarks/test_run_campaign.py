import json

import pytest

from benchmarks.common import run_campaign


def _job():
    return {"job_id": "campaign-seed011", "evaluation_budget": 2400}


def _payload():
    return {
        "status": "complete",
        "mode": "production",
        "job": {"job_id": "campaign-seed011", "evaluation_budget": 2400},
        "provenance": {
            "config_sha256": "config-hash",
            "git_commit": "commit-hash",
            "git_dirty": False,
        },
        "actual_candidate_evaluations": 2400,
    }


def test_complete_cache_requires_exact_clean_provenance(tmp_path):
    output = tmp_path / "campaign-seed011.json"
    output.write_text(json.dumps(_payload()), encoding="utf-8")
    output.with_suffix(".npz").write_bytes(b"artifact")

    assert run_campaign.is_complete(
        output,
        config_hash="config-hash",
        git_commit="commit-hash",
        job=_job(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_sha256", "stale-config"),
        ("git_commit", "stale-commit"),
        ("git_dirty", True),
    ],
)
def test_complete_cache_rejects_stale_or_dirty_provenance(tmp_path, field, value):
    payload = _payload()
    payload["provenance"][field] = value
    output = tmp_path / "campaign-seed011.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    output.with_suffix(".npz").write_bytes(b"artifact")

    assert not run_campaign.is_complete(
        output,
        config_hash="config-hash",
        git_commit="commit-hash",
        job=_job(),
    )


def test_complete_cache_rejects_wrong_evaluation_count(tmp_path):
    payload = _payload()
    payload["actual_candidate_evaluations"] = 2399
    output = tmp_path / "campaign-seed011.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    output.with_suffix(".npz").write_bytes(b"artifact")

    assert not run_campaign.is_complete(
        output,
        config_hash="config-hash",
        git_commit="commit-hash",
        job=_job(),
    )


def test_clean_commit_gate_rejects_any_worktree_change(monkeypatch, tmp_path):
    def fake_git_output(repo_root, *args):
        del repo_root
        return " M benchmark.py\n?? new-file.py" if args[0] == "status" else "commit"

    monkeypatch.setattr(run_campaign, "git_output", fake_git_output)

    with pytest.raises(SystemExit, match="dirty worktree"):
        run_campaign.require_clean_commit(tmp_path)


def test_clean_commit_gate_returns_head(monkeypatch, tmp_path):
    def fake_git_output(repo_root, *args):
        del repo_root
        return "" if args[0] == "status" else "commit-hash"

    monkeypatch.setattr(run_campaign, "git_output", fake_git_output)

    assert run_campaign.require_clean_commit(tmp_path) == "commit-hash"
