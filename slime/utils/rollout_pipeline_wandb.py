"""Unified W&B logging for rollout queue depth and reward-computation KPIs."""

from __future__ import annotations

import time

from slime.rollout.queue_metrics import QueueDepthSnapshot
from slime.rollout.reward_computation_metrics import RewardComputationSnapshot
from slime.utils import logging_utils

__all__ = [
    "RolloutPipelineWandbMonitor",
    "merge_rollout_pipeline_wandb_dict",
]


def merge_rollout_pipeline_wandb_dict(
    queue_snapshot: QueueDepthSnapshot,
    reward_snapshot: RewardComputationSnapshot,
    elapsed_s: float,
) -> dict[str, float | int]:
    payload = queue_snapshot.to_wandb_dict(elapsed_s)
    payload["reward_computation/time"] = elapsed_s
    payload.update(reward_snapshot.to_wandb_dict())
    return payload


class RolloutPipelineWandbMonitor:
    """Emit queue depth and reward-computation snapshots on a wall-clock cadence."""

    def __init__(self, args) -> None:
        self.args = args
        self._started_at: float | None = None

    def mark_started(self) -> None:
        self._started_at = time.monotonic()

    def maybe_log(
        self,
        queue_snapshot: QueueDepthSnapshot,
        reward_snapshot: RewardComputationSnapshot,
    ) -> None:
        if self._started_at is None or not getattr(self.args, "use_wandb", False):
            return
        elapsed_s = time.monotonic() - self._started_at
        payload = merge_rollout_pipeline_wandb_dict(queue_snapshot, reward_snapshot, elapsed_s)
        logging_utils.log(self.args, payload, step_key="queues/time")
