"""W&B time-series logging for rollout/training queue depths."""

from __future__ import annotations

import wandb

from slime.rollout.queue_metrics import QueueDepthSnapshot

__all__ = [
    "QueueWandbMonitor",
    "log_queue_depth",
    "queues_backlog_limit_samples",
    "update_queues_wandb_config",
]


def queues_backlog_limit_samples(args) -> int | None:
    max_staleness = getattr(args, "max_rollout_weight_staleness", None)
    if max_staleness is None:
        return None
    return int(args.global_batch_size * max_staleness)


def update_queues_wandb_config(args) -> None:
    if not args.use_wandb:
        return
    limit = queues_backlog_limit_samples(args)
    if limit is None or wandb.run is None:
        return
    wandb.config.update({"queues/backlog_limit_samples": limit})


def log_queue_depth(args, snapshot: QueueDepthSnapshot, *, step: int) -> None:
    if not getattr(args, "use_wandb", False):
        return
    from slime.utils import logging_utils

    logging_utils.log(args, snapshot.to_wandb_dict(), step=step)


class QueueWandbMonitor:
    """Emit queue depth snapshots on a wall-clock cadence."""

    def __init__(self, args) -> None:
        self.args = args
        self._snapshot_index = 0

    def maybe_log(self, snapshot: QueueDepthSnapshot) -> None:
        self._snapshot_index += 1
        log_queue_depth(self.args, snapshot, step=self._snapshot_index)
