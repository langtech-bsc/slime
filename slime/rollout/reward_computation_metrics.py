"""Reward-pipeline KPIs for fully-async rollout (W&B ``reward_computation/*``)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from statistics import fmean, pstdev
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slime.utils.types import Sample

__all__ = [
    "RewardComputationSnapshot",
    "RewardComputationTracker",
    "build_reward_computation_snapshot",
]


@dataclass(frozen=True)
class RewardComputationSnapshot:
    concurrency_utilization: float
    oldest_wait_seconds: float
    rm_latency_mean: float | None
    group_reward_std_mean: float | None = None

    def to_wandb_dict(self) -> dict[str, float]:
        payload = {
            "reward_computation/concurrency_utilization": self.concurrency_utilization,
            "reward_computation/oldest_wait_seconds": self.oldest_wait_seconds,
        }
        if self.rm_latency_mean is not None:
            payload["reward_computation/rm_latency_mean"] = self.rm_latency_mean
        if self.group_reward_std_mean is not None:
            payload["reward_computation/group_reward_std_mean"] = self.group_reward_std_mean
        return payload


class RewardComputationTracker:
    """Accumulates RM wall times between periodic W&B snapshots."""

    def __init__(self) -> None:
        self._rm_latency_total_s = 0.0
        self._rm_latency_count = 0
        self._group_reward_std_values: list[float] = []

    def record_rm_latency(self, duration_s: float) -> None:
        self._rm_latency_total_s += duration_s
        self._rm_latency_count += 1

    def consume_window_mean_latency(self) -> float | None:
        if self._rm_latency_count == 0:
            return None
        mean = self._rm_latency_total_s / self._rm_latency_count
        self._rm_latency_total_s = 0.0
        self._rm_latency_count = 0
        return mean

    def record_group_rewards(self, rewards: Iterable[float]) -> None:
        """Record one completed group's population reward standard deviation."""
        values = [float(reward) for reward in rewards]
        if len(values) < 2:
            return
        self._group_reward_std_values.append(pstdev(values))

    def consume_window_mean_group_reward_std(self) -> float | None:
        if not self._group_reward_std_values:
            return None
        mean = fmean(self._group_reward_std_values)
        self._group_reward_std_values.clear()
        return mean

    @asynccontextmanager
    async def measure_rm(self, sample_or_group: Sample | list[Sample]):
        started = time.monotonic()
        try:
            yield
        finally:
            self.record_rm_latency(time.monotonic() - started)


def build_reward_computation_snapshot(
    *,
    active_reward: dict[asyncio.Task, tuple[int, int | None]],
    reward_concurrency: int,
    oldest_wait_seconds: float,
    tracker: RewardComputationTracker,
) -> RewardComputationSnapshot:
    utilization = len(active_reward) / reward_concurrency if reward_concurrency > 0 else 0.0
    return RewardComputationSnapshot(
        concurrency_utilization=utilization,
        oldest_wait_seconds=oldest_wait_seconds,
        rm_latency_mean=tracker.consume_window_mean_latency(),
        group_reward_std_mean=tracker.consume_window_mean_group_reward_std(),
    )
