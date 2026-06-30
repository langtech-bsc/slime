"""Unified W&B logging for rollout queue depth and reward-computation KPIs."""

from __future__ import annotations

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
) -> dict[str, float | int]:
    payload = queue_snapshot.to_wandb_dict()
    payload.update(reward_snapshot.to_wandb_dict())
    return payload


class RolloutPipelineWandbMonitor:
    """Emit queue depth and reward-computation snapshots on a wall-clock cadence."""

    def __init__(self, args) -> None:
        self.args = args
        self._snapshot_index = 0

    def maybe_log(
        self,
        queue_snapshot: QueueDepthSnapshot,
        reward_snapshot: RewardComputationSnapshot,
    ) -> None:
        if not getattr(self.args, "use_wandb", False):
            return
        self._snapshot_index += 1
        payload = merge_rollout_pipeline_wandb_dict(queue_snapshot, reward_snapshot)
        logging_utils.log(self.args, payload, step=self._snapshot_index)
