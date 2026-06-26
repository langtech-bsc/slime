"""Tests for rollout weight-version staleness filtering."""

import torch

from slime.utils.rollout_staleness import (
    RolloutStalenessStats,
    discard_stale_rollout_samples,
    min_rollout_weight_version,
    rollout_weight_staleness,
)


def test_min_rollout_weight_version_uses_oldest_version():
    assert min_rollout_weight_version(["7", "5", "6"]) == 5
    assert min_rollout_weight_version([]) is None


def test_rollout_weight_staleness():
    assert rollout_weight_staleness(10, ["7"]) == 3
    assert rollout_weight_staleness(10, ["7", "9"]) == 3


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
