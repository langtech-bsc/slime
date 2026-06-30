"""W&B time-series logging for rollout/training queue depths."""

from __future__ import annotations

import time

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


def log_queue_depth(args, snapshot: QueueDepthSnapshot, elapsed_s: float) -> None:
    if not getattr(args, "use_wandb", False):
        return
    from slime.utils import logging_utils

    logging_utils.log(args, snapshot.to_wandb_dict(elapsed_s), step_key="queues/time")


class QueueWandbMonitor:
    """Emit queue depth snapshots on a wall-clock cadence."""

    def __init__(self, args) -> None:
        self.args = args
        self._started_at: float | None = None

    def mark_started(self) -> None:
        self._started_at = time.monotonic()

    def maybe_log(self, snapshot: QueueDepthSnapshot) -> None:
        if self._started_at is None:
            return
        elapsed_s = time.monotonic() - self._started_at
        log_queue_depth(self.args, snapshot, elapsed_s)
