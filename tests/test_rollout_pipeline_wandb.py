from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slime.rollout.queue_metrics import QueueDepthSnapshot
from slime.rollout.reward_computation_metrics import RewardComputationSnapshot
from slime.utils.rollout_pipeline_wandb import (
    RolloutPipelineWandbMonitor,
    merge_rollout_pipeline_wandb_dict,
)


@pytest.mark.unit
def test_merge_rollout_pipeline_wandb_dict():
    queue = QueueDepthSnapshot(reward_samples=10, training_samples=3)
    reward = RewardComputationSnapshot(
        concurrency_utilization=0.5,
        oldest_wait_seconds=4.0,
        rm_latency_mean=0.25,
    )
    payload = merge_rollout_pipeline_wandb_dict(queue, reward, elapsed_s=30.0)

    assert payload == {
        "queues/time": 30.0,
        "queues/reward_samples": 10,
        "queues/training_samples": 3,
        "reward_computation/time": 30.0,
        "reward_computation/concurrency_utilization": 0.5,
        "reward_computation/oldest_wait_seconds": 4.0,
        "reward_computation/rm_latency_mean": 0.25,
    }
    assert "reward_computation/pending_samples" not in payload
    assert "reward_computation/completed_samples" not in payload


@pytest.mark.unit
def test_rollout_pipeline_wandb_monitor_maybe_log():
    args = SimpleNamespace(use_wandb=True)
    monitor = RolloutPipelineWandbMonitor(args)
    monitor.mark_started()

    queue = QueueDepthSnapshot(reward_samples=1, training_samples=2)
    reward = RewardComputationSnapshot(
        concurrency_utilization=1.0,
        oldest_wait_seconds=0.0,
        rm_latency_mean=None,
    )

    with patch("slime.utils.logging_utils.log") as mock_log:
        monitor.maybe_log(queue, reward)

    mock_log.assert_called_once()
    logged_args, logged_payload, kwargs = mock_log.call_args[0][0], mock_log.call_args[0][1], mock_log.call_args[1]
    assert logged_args is args
    assert kwargs["step_key"] == "queues/time"
    assert logged_payload["queues/reward_samples"] == 1
    assert logged_payload["reward_computation/concurrency_utilization"] == 1.0
    assert "reward_computation/rm_latency_mean" not in logged_payload
