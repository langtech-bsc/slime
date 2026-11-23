"""Divisor of the per-rollout-mean loss for ``--loss-normalizer valid_rollouts``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu


def valid_rollout_counts_per_step(
    rollout_data: dict[str, Any],
    num_microbatches: Sequence[int],
    group: Any = None,
) -> list[int]:
    """Number of rollouts with at least one loss token in each training step, summed over DP.

    Uses the microbatch schedule (``micro_batch_indices`` flattened over steps,
    ``num_microbatches[s]`` per step) and ``rollout_ids`` to count each rollout
    once even when it spans several samples.
    """
    loss_masks = rollout_data["loss_masks"]
    rollout_ids = rollout_data.get("rollout_ids") or list(range(len(loss_masks)))
    mb_indices = rollout_data["micro_batch_indices"]
    counts = []
    offset = 0
    for n_mb in num_microbatches:
        valid: set[int] = set()
        for mb in mb_indices[offset : offset + n_mb]:
            for i in mb:
                mask = loss_masks[i]
                if (mask.sum() if torch.is_tensor(mask) else sum(mask)) > 0:
                    valid.add(int(rollout_ids[i]))
        offset += n_mb
        counts.append(len(valid))
    if dist.is_initialized():
        if group is None:
            group = mpu.get_data_parallel_group()
        t = torch.tensor(counts, dtype=torch.int64, device=torch.cuda.current_device())
        dist.all_reduce(t, group=group)
        counts = [int(x) for x in t.tolist()]
    # A step where nothing is valid has zero loss; keep the divisor at 1.
    return [max(c, 1) for c in counts]
