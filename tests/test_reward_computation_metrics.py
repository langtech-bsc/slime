from __future__ import annotations

import pytest

from slime.rollout.reward_computation_metrics import (
    RewardComputationSnapshot,
    RewardComputationTracker,
    build_reward_computation_snapshot,
)


@pytest.mark.unit
def test_reward_computation_snapshot_to_wandb_dict():
    snapshot = RewardComputationSnapshot(
        concurrency_utilization=0.75,
        oldest_wait_seconds=12.5,
        rm_latency_mean=0.42,
    )
    payload = snapshot.to_wandb_dict()
    assert payload == {
        "reward_computation/concurrency_utilization": 0.75,
        "reward_computation/oldest_wait_seconds": 12.5,
        "reward_computation/rm_latency_mean": 0.42,
    }


@pytest.mark.unit
def test_reward_computation_snapshot_omits_rm_latency_when_none():
    snapshot = RewardComputationSnapshot(
        concurrency_utilization=0.0,
        oldest_wait_seconds=0.0,
        rm_latency_mean=None,
    )
    assert "reward_computation/rm_latency_mean" not in snapshot.to_wandb_dict()


@pytest.mark.unit
def test_reward_computation_tracker_measure_rm_records_latency():
    tracker = RewardComputationTracker()
    tracker.record_rm_latency(0.2)
    tracker.record_rm_latency(0.4)
    mean = tracker.consume_window_mean_latency()
    assert mean == pytest.approx(0.3)


@pytest.mark.unit
def test_reward_computation_tracker_window_resets_after_consume():
    tracker = RewardComputationTracker()
    tracker.record_rm_latency(1.0)
    tracker.record_rm_latency(3.0)
    assert tracker.consume_window_mean_latency() == 2.0
    assert tracker.consume_window_mean_latency() is None


@pytest.mark.unit
def test_build_reward_computation_snapshot():
    tracker = RewardComputationTracker()
    tracker.record_rm_latency(0.2)
    tracker.record_rm_latency(0.4)

    snapshot = build_reward_computation_snapshot(
        active_reward={object(): (1, None)},
        reward_concurrency=4,
        oldest_wait_seconds=7.0,
        tracker=tracker,
    )

    assert snapshot.concurrency_utilization == 0.25
    assert snapshot.oldest_wait_seconds == 7.0
    assert snapshot.rm_latency_mean == pytest.approx(0.3)
