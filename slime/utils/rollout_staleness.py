"""Rollout weight-version staleness helpers for async training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from slime.utils import logging_utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutWeightStalenessStats:
    mean: float | None
    median: float | None
    p95: float | None


@dataclass(frozen=True)
class RolloutStalenessStats:
    discarded: int = 0
    eligible: int = 0
    unknown_version: int = 0

    @property
    def kept(self) -> int:
        return self.eligible - self.discarded

    @property
    def discard_ratio(self) -> float:
        if self.eligible == 0:
            return 0.0
        return self.discarded / self.eligible


def min_rollout_weight_version(weight_versions: list[str] | None) -> int | None:
    """Return the oldest rollout weight version recorded on a sample."""
    if not weight_versions:
        return None
    parsed: list[int] = []
    for version in weight_versions:
        try:
            parsed.append(int(version))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None
    return min(parsed)


def rollout_weight_staleness(trainer_weight_version: int, weight_versions: list[str] | None) -> int | None:
    """Trainer minus rollout weight version, or None when rollout version is unknown."""
    rollout_version = min_rollout_weight_version(weight_versions)
    if rollout_version is None:
        return None
    return trainer_weight_version - rollout_version


def discard_stale_rollout_samples(
    rollout_data: dict[str, Any],
    trainer_weight_version: int,
    max_staleness: int,
) -> RolloutStalenessStats:
    """Zero loss masks for samples whose rollout/trainer version gap exceeds ``max_staleness``."""
    weight_versions = rollout_data.get("weight_versions")
    if not weight_versions:
        return RolloutStalenessStats(unknown_version=len(rollout_data.get("loss_masks", [])))

    loss_masks = rollout_data["loss_masks"]
    discarded = 0
    unknown_version = 0
    for index, versions in enumerate(weight_versions):
        staleness = rollout_weight_staleness(trainer_weight_version, versions)
        if staleness is None:
            unknown_version += 1
            continue
        if staleness <= max_staleness:
            continue

        mask = loss_masks[index]
        if torch.is_tensor(mask):
            loss_masks[index] = torch.zeros_like(mask)
        else:
            loss_masks[index] = [0] * len(mask)
        discarded += 1

    eligible = len(weight_versions) - unknown_version
    stats = RolloutStalenessStats(
        discarded=discarded,
        eligible=eligible,
        unknown_version=unknown_version,
    )
    if stats.discarded:
        _recompute_rollout_mask_sums(rollout_data)
        logger.info(
            "discarded %d/%d rollout samples with weight staleness > %d (trainer_version=%d)",
            stats.discarded,
            stats.eligible,
            max_staleness,
            trainer_weight_version,
        )

    return stats


def _loss_mask_is_active(mask: Any) -> bool:
    if torch.is_tensor(mask):
        return int(mask.sum().item()) > 0
    return sum(mask) > 0


def rollout_weight_staleness_stats_for_training(
    rollout_data: dict[str, Any],
    trainer_weight_version: int,
) -> RolloutWeightStalenessStats:
    """Mean/median/p95 trainer-minus-rollout gap for samples kept for training."""
    weight_versions = rollout_data.get("weight_versions")
    loss_masks = rollout_data.get("loss_masks")
    if not weight_versions or not loss_masks:
        return RolloutWeightStalenessStats(mean=None, median=None, p95=None)

    gaps: list[float] = []
    for versions, mask in zip(weight_versions, loss_masks, strict=True):
        if not _loss_mask_is_active(mask):
            continue
        gap = rollout_weight_staleness(trainer_weight_version, versions)
        if gap is None:
            continue
        gaps.append(float(gap))

    if not gaps:
        return RolloutWeightStalenessStats(mean=None, median=None, p95=None)

    arr = np.asarray(gaps, dtype=np.float64)
    return RolloutWeightStalenessStats(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
    )


def log_rollout_weight_staleness_metrics(
    rollout_id: int,
    args: Any,
    stats: RolloutStalenessStats | None,
    *,
    trainer_weight_version: int,
    max_staleness: int,
    rollout_data: dict[str, Any] | None = None,
) -> None:
    """Log per-rollout staleness discard counts to stdout and tracking backends."""
    if stats is None:
        return

    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_rank() != 0 or not mpu.is_pipeline_last_stage():
        return

    from slime.backends.megatron_utils.data import gather_log_data
    from slime.utils.metric_utils import compute_rollout_step

    log_dict = {
        "discarded": (stats.discarded, 1),
        "eligible": (stats.eligible, 1),
        "kept": (stats.kept, 1),
        "unknown_version": (stats.unknown_version, 1),
        "discard_ratio": (stats.discarded, max(stats.eligible, 1)),
        "trainer_weight_version": (trainer_weight_version, 1),
        "max_staleness": (max_staleness, 1),
    }
    gather_log_data("rollout_weight_staleness", args, rollout_id, log_dict)

    step = compute_rollout_step(args, rollout_id)
    wandb_payload: dict[str, float | int] = {
        "train/rollout_weight_staleness_discarded": stats.discarded,
        "train/rollout_weight_staleness_discard_ratio": stats.discard_ratio,
        "train/step": step,
    }
    if rollout_data is not None:
        weight_stats = rollout_weight_staleness_stats_for_training(rollout_data, trainer_weight_version)
        if weight_stats.mean is not None:
            wandb_payload["train/mean_rollout_weight_staleness"] = weight_stats.mean
        if weight_stats.median is not None:
            wandb_payload["train/median_rollout_weight_staleness"] = weight_stats.median
        if weight_stats.p95 is not None:
            wandb_payload["train/p95_rollout_weight_staleness"] = weight_stats.p95

    logging_utils.log(
        args,
        wandb_payload,
        step_key="train/step",
    )


def _recompute_rollout_mask_sums(rollout_data: dict[str, Any]) -> None:
    if "rollout_mask_sums" not in rollout_data:
        return

    loss_masks = rollout_data["loss_masks"]
    rollout_ids = rollout_data["rollout_ids"]
    mask_sums_per_sample = [int(m.sum()) if torch.is_tensor(m) else sum(m) for m in loss_masks]
    rollout_total_mask: dict[int, int] = {}
    for rollout_id, mask_sum in zip(rollout_ids, mask_sums_per_sample, strict=True):
        rollout_total_mask[rollout_id] = rollout_total_mask.get(rollout_id, 0) + mask_sum
    recomputed = [rollout_total_mask[rollout_id] for rollout_id in rollout_ids]
    original = rollout_data["rollout_mask_sums"]
    if torch.is_tensor(original):
        rollout_data["rollout_mask_sums"] = original.new_tensor(recomputed, dtype=torch.float32)
    else:
        rollout_data["rollout_mask_sums"] = torch.tensor(recomputed, dtype=torch.float32)
