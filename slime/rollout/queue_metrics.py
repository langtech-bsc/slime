"""Queue depth metrics for fully-async rollout workers."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import queue

from slime.utils.types import Sample

__all__ = [
    "QueueDepthSnapshot",
    "TrainingOutputQueue",
    "compute_queue_depth",
]


@dataclass(frozen=True)
class QueueDepthSnapshot:
    reward_samples: int
    training_samples: int

    def to_wandb_dict(self) -> dict[str, int]:
        return {
            "queues/reward_samples": self.reward_samples,
            "queues/training_samples": self.training_samples,
        }


def compute_queue_depth(
    *,
    groups: object,
    pending_generation: deque[tuple[int, int, Sample]],
    active_generation: dict[asyncio.Task, tuple[int, int]],
    reward_backlog_samples: int,
    training_queue_samples: int,
) -> QueueDepthSnapshot:
    del groups
    reward_samples = len(pending_generation) + len(active_generation) + reward_backlog_samples
    return QueueDepthSnapshot(
        reward_samples=reward_samples,
        training_samples=training_queue_samples,
    )


class TrainingOutputQueue:
    """Completed rollout groups waiting to be collected for training."""

    def __init__(self, *, maxsize: int = 1000) -> None:
        self._queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=maxsize)
        self._sample_count = 0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def group_count(self) -> int:
        return self._queue.qsize()

    def put(self, gid: int, group: list[Sample]) -> None:
        self._queue.put((gid, group))
        self._sample_count += len(group)

    def drain(self) -> list[tuple[int, list[Sample]]]:
        drained: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                gid, group = self._queue.get_nowait()
            except queue.Empty:
                break
            self._sample_count -= len(group)
            drained.append((gid, group))
        return drained
