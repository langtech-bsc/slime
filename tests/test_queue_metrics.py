from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from slime.rollout.queue_metrics import (
    QueueDepthSnapshot,
    TrainingOutputQueue,
    compute_queue_depth,
)
from slime.utils.types import Sample


@pytest.mark.unit
def test_compute_queue_depth_counts_reward_and_training_samples():
    pending_generation = deque([(1, 2, Sample(index=2))])
    active_generation = {object(): (1, 0)}

    snapshot = compute_queue_depth(
        groups={},
        pending_generation=pending_generation,
        active_generation=active_generation,
        reward_backlog_samples=2,
        training_queue_samples=6,
    )

    assert snapshot.reward_samples == 4
    assert snapshot.training_samples == 6


@pytest.mark.unit
def test_queue_depth_snapshot_to_wandb_dict_has_no_limit_key():
    snapshot = QueueDepthSnapshot(reward_samples=4, training_samples=8)
    payload = snapshot.to_wandb_dict(12.5)

    assert payload == {
        "queues/time": 12.5,
        "queues/reward_samples": 4,
        "queues/training_samples": 8,
    }
    assert "queues/backlog_limit_samples" not in payload


@pytest.mark.unit
def test_training_output_queue_tracks_sample_count_on_put_and_drain():
    output_queue = TrainingOutputQueue()
    group_a = [Sample(index=1), Sample(index=2)]
    group_b = [Sample(index=3)]

    output_queue.put(1, group_a)
    output_queue.put(2, group_b)
    assert output_queue.sample_count == 3
    assert output_queue.group_count() == 2

    drained = output_queue.drain()
    assert len(drained) == 2
    assert output_queue.sample_count == 0
    assert output_queue.group_count() == 0
