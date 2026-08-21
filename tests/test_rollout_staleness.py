"""Tests for rollout weight-version staleness filtering."""

import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from slime.utils.rollout_staleness import (
    RolloutStalenessStats,
    RolloutWeightStalenessStats,
    discard_stale_rollout_samples,
    log_rollout_weight_staleness_metrics,
    min_rollout_weight_version,
    raise_on_stale_rollout_samples,
    resolve_effective_max_staleness,
    rollout_weight_staleness_gaps,
    rollout_weight_staleness,
    rollout_weight_staleness_stats_for_training,
)


def _run_staleness_logger_cpu_worker(rank: int, world_size: int, init_method: str) -> None:
    """Exercise the production logger on CPU DP source and non-source ranks."""
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)

    from megatron.core import mpu

    args = SimpleNamespace(
        use_wandb=True,
        use_tensorboard=False,
        wandb_always_use_train_step=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        global_batch_size=2,
    )
    stats = RolloutStalenessStats(discarded=1, eligible=2, unknown_version=0)
    rollout_data = {
        "weight_versions": [["7"], ["9"]],
        "loss_masks": [torch.ones(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)],
    }

    # Deliberately initialize W&B only where production code does: the DP
    # source rank. A non-source wandb.log() would fail this worker.
    if rank == 0:
        wandb.init(mode="disabled", project="cpu-staleness-e2e", name="source")

    try:
        with (
            patch.object(mpu, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(mpu, "is_pipeline_last_stage", return_value=True),
            patch.object(mpu, "get_data_parallel_world_size", return_value=world_size),
            patch.object(mpu, "get_data_parallel_src_rank", return_value=0),
            patch.object(mpu, "get_data_parallel_group_gloo", return_value=dist.group.WORLD),
        ):
            log_rollout_weight_staleness_metrics(
                rollout_id=1,
                args=args,
                stats=stats,
                trainer_weight_version=10,
                max_staleness=6,
                num_steps_per_rollout=1,
                rollout_data=rollout_data,
            )
    finally:
        if rank == 0:
            wandb.finish()
        dist.destroy_process_group()


@pytest.mark.integration
def test_staleness_logger_cpu_e2e_skips_non_source_wandb_logging():
    """The full staleness logger must work with W&B only on the DP source."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(
        _run_staleness_logger_cpu_worker,
        args=(2, f"tcp://127.0.0.1:{port}"),
        nprocs=2,
        join=True,
    )


def test_min_rollout_weight_version_uses_oldest_version():
    assert min_rollout_weight_version(["7", "5", "6"]) == 5
    assert min_rollout_weight_version([]) is None


def test_rollout_weight_staleness():
    assert rollout_weight_staleness(10, ["7"]) == 3
    assert rollout_weight_staleness(10, ["7", "9"]) == 3


def test_rollout_weight_staleness_gaps_tracks_min_mean_max():
    gaps = rollout_weight_staleness_gaps(10, ["6", "9"])

    assert gaps is not None
    assert gaps.min == 1
    assert gaps.mean == pytest.approx(2.5)
    assert gaps.max == 4


def test_resolve_effective_max_staleness_uses_policy_max_gap():
    assert resolve_effective_max_staleness(3, [2, 2, 2, 4]) == 6


def test_resolve_effective_max_staleness_without_cap_returns_none():
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
