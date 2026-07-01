"""Tests for rollout weight-version staleness filtering."""

import torch
import pytest

from slime.utils.rollout_staleness import (
    RolloutStalenessStats,
    RolloutWeightStalenessStats,
    discard_stale_rollout_samples,
    min_rollout_weight_version,
    raise_on_stale_rollout_samples,
    resolve_effective_max_staleness,
    rollout_weight_staleness,
    rollout_weight_staleness_stats_for_training,
)


def test_min_rollout_weight_version_uses_oldest_version():
    assert min_rollout_weight_version(["7", "5", "6"]) == 5
    assert min_rollout_weight_version([]) is None


def test_rollout_weight_staleness():
    assert rollout_weight_staleness(10, ["7"]) == 3
    assert rollout_weight_staleness(10, ["7", "9"]) == 3


def test_resolve_effective_max_staleness_relaxes_when_batch_is_uniformly_fresh():
    assert resolve_effective_max_staleness(3, [2, 2, 2, 4]) == 5


def test_resolve_effective_max_staleness_stays_strict_when_batch_max_is_high():
    assert resolve_effective_max_staleness(3, [2, 2, 2, 7]) == 3


def test_resolve_effective_max_staleness_stays_strict_for_pathological_gap():
    assert resolve_effective_max_staleness(3, [2, 2, 40]) == 3


def test_resolve_effective_max_staleness_without_gaps_returns_base():
    assert resolve_effective_max_staleness(3, []) == 3
    assert resolve_effective_max_staleness(None, [1, 2]) is None


def test_discard_stale_rollout_samples_zeros_masks_and_recomputes_rollout_totals():
    rollout_data = {
        "weight_versions": [["4"], ["8"]],
        "loss_masks": [torch.ones(3, dtype=torch.int32), torch.ones(2, dtype=torch.int32)],
        "rollout_ids": [0, 0],
        "rollout_mask_sums": [5, 5],
    }

    stats = discard_stale_rollout_samples(rollout_data, trainer_weight_version=10, max_staleness=3)

    assert stats == RolloutStalenessStats(discarded=1, eligible=2, unknown_version=0)
    assert rollout_data["loss_masks"][0].sum().item() == 0
    assert rollout_data["loss_masks"][1].sum().item() == 2
    assert rollout_data["rollout_mask_sums"].tolist() == [2, 2]


def test_rollout_weight_staleness_stats_for_training_mean_median_p95():
    rollout_data = {
        "weight_versions": [["7"], ["8"], ["9"], ["10"]],
        "loss_masks": [
            torch.ones(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
        ],
    }

    stats = rollout_weight_staleness_stats_for_training(rollout_data, trainer_weight_version=10)

    assert stats.mean == pytest.approx(5 / 3)
    assert stats.median == pytest.approx(2.0)
    assert stats.p95 == pytest.approx(2.9)


def test_rollout_weight_staleness_stats_for_training_skips_unknown_versions():
    rollout_data = {
        "weight_versions": [["bad"], ["9"]],
        "loss_masks": [torch.ones(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)],
    }

    stats = rollout_weight_staleness_stats_for_training(rollout_data, trainer_weight_version=10)

    assert stats == RolloutWeightStalenessStats(mean=1.0, median=1.0, p95=1.0)


def test_rollout_weight_staleness_stats_for_training_empty_when_no_kept_samples():
    rollout_data = {
        "weight_versions": [["7"]],
        "loss_masks": [torch.zeros(1, dtype=torch.int32)],
    }

    stats = rollout_weight_staleness_stats_for_training(rollout_data, trainer_weight_version=10)

    assert stats == RolloutWeightStalenessStats(mean=None, median=None, p95=None)


def test_raise_on_stale_rollout_samples_rejects_stale_in_actor_guard():
    rollout_data = {
        "weight_versions": [["4"], ["8"]],
        "loss_masks": [torch.ones(3, dtype=torch.int32), torch.ones(2, dtype=torch.int32)],
    }

    with pytest.raises(ValueError, match="Stale rollout samples reached actor training"):
        raise_on_stale_rollout_samples(rollout_data, trainer_weight_version=10, max_staleness=3)
