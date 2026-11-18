"""Training-side behaviours of NeMo-RL GRPO that slime needs for a matched run.

* ``mask_high_logprob_error_sequences``: NeMo's ``seq_logprob_error_threshold``.
  A sequence whose mean ``exp(|train_logprob - rollout_logprob|)`` over its
  loss tokens exceeds the threshold is taken out of the loss (all loss-mask
  tokens set to 0) before the advantages are whitened, so it is also out of
  the whitening statistics. It still counted in its group baseline, which the
  rollout side computed earlier.
* ``valid_rollout_counts_per_step``: NeMo divides the sequence-mean loss by the
  number of sequences that still have a loss token (``sum(sample_mask)``,
  summed over DP), not by the nominal batch size. The LR scheduler keeps the
  nominal count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.utils.rollout_staleness import _recompute_rollout_mask_sums

from .cp_utils import all_gather_with_cp


def sequence_logprob_errors(
    log_probs: Sequence[torch.Tensor],
    rollout_log_probs: Sequence[torch.Tensor],
    loss_masks: Sequence[torch.Tensor],
) -> list[float]:
    """Per sequence ``mean_t exp(|lp - rollout_lp|)`` over loss tokens (full response tensors).

    A sequence with no loss token gets 0.0 (nothing to mask).
    """
    errors = []
    for lp, rlp, mask in zip(log_probs, rollout_log_probs, loss_masks, strict=True):
        mask = mask.to(device=lp.device, dtype=torch.float32)
        n = mask.sum()
        if n <= 0:
            errors.append(0.0)
            continue
        diff = torch.exp((lp.float() - rlp.to(lp.device).float()).abs())
        errors.append(float((diff * mask).sum() / n))
    return errors


def mask_high_logprob_error_sequences(args: Any, rollout_data: dict[str, Any]) -> dict[str, float]:
    """Zero the loss mask of sequences above ``args.seq_logprob_error_threshold``.

    Must run after the trainer log-prob pass and before
    ``compute_advantages_and_returns``. Works on every CP rank: the local CP
    chunks are gathered to full responses, so all CP ranks of a DP rank take
    the same decision. Returns local metrics (counts on this DP rank).
    """
    threshold = getattr(args, "seq_logprob_error_threshold", None)
    if threshold is None or not mpu.is_pipeline_last_stage():
        return {}
    log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    if not log_probs or not rollout_log_probs:
        raise RuntimeError("--seq-logprob-error-threshold needs trainer and rollout log probs")
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    full_lp = [all_gather_with_cp(x, t, r) for x, t, r in zip(log_probs, total_lengths, response_lengths, strict=True)]
    full_rlp = [
        all_gather_with_cp(x, t, r) for x, t, r in zip(rollout_log_probs, total_lengths, response_lengths, strict=True)
    ]
    loss_masks = rollout_data["loss_masks"]
    errors = sequence_logprob_errors(full_lp, full_rlp, loss_masks)
    masked = 0
    for i, err in enumerate(errors):
        if err > threshold:
            loss_masks[i] = torch.zeros_like(loss_masks[i])
            masked += 1
    if masked:
        _recompute_rollout_mask_sums(rollout_data)
    return {
        "nemo_parity/seq_logprob_error_masked": float(masked),
        "nemo_parity/seq_logprob_error_max": max(errors, default=0.0),
    }


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
