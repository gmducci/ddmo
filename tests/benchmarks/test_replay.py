"""A selected benchmark model must independently reproduce saved rollouts."""

import numpy as np
import pytest

from benchmarks.replay import replay


@pytest.mark.parametrize(
    ("experiment", "method", "noise", "seed", "split", "index"),
    [
        ("noise", "weak_optimized", 0.10, 11, "test", 0),
        ("noise", "strong", 0.10, 11, "test", 1),
        ("joint", None, 0.10, 104, "test", 1),
    ],
)
def test_replay_matches_frozen_trajectory(experiment, method, noise, seed, split, index):
    job_id, time, trajectory, max_error = replay(
        experiment, method=method, noise=noise, seed=seed, split=split, index=index
    )
    assert job_id
    assert trajectory.shape == (len(time), 3)
    assert np.all(np.isfinite(trajectory))
    assert max_error < 1e-10


def test_noise_selection_requires_method_and_rejects_wrong_seed():
    with pytest.raises(ValueError, match="specify --method"):
        replay("noise", noise=0.10, seed=11)
    with pytest.raises(ValueError, match="No frozen"):
        replay("joint", noise=0.10, seed=11)


def test_replay_rejects_trajectory_outside_split():
    with pytest.raises(ValueError, match="test index"):
        replay("joint", noise=0.10, seed=104, index=2)
