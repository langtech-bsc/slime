# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = None
    tl = None


OpdKLDirection = Literal["forward", "reverse"]
BACKWARD_CHUNK_ROWS = 64


def _dist_ready(group: dist.ProcessGroup | None) -> bool:
    return (
        group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=group) > 1
    )


def _all_reduce_no_grad(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not _dist_ready(group):
        return tensor
    result = tensor.detach().clone()
    dist.all_reduce(result, op=op, group=group)
    return result


def _all_reduce_sum_autograd(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if not _dist_ready(group):
        return tensor
    from torch.distributed.nn import functional as dist_nn

    return dist_nn.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def _global_vocab_size(weight: torch.Tensor, group: dist.ProcessGroup | None) -> int:
    world_size = dist.get_world_size(group=group) if _dist_ready(group) else 1
    return int(weight.size(0)) * world_size


def _merge_topk(
    values: torch.Tensor | None,
    indices: torch.Tensor | None,
    new_values: torch.Tensor,
    new_indices: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values is None:
        return new_values, new_indices
    candidates = torch.cat((values, new_values), dim=-1)
    candidate_indices = torch.cat((indices, new_indices), dim=-1)
    values, positions = torch.topk(candidates, k=min(top_k, candidates.size(-1)), dim=-1)
    return values, torch.gather(candidate_indices, dim=-1, index=positions)


@torch.no_grad()
def teacher_topk_from_hidden(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    vocab_chunk_size: int,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project teacher hidden states without retaining a full-vocabulary tensor."""
    if temperature <= 0:
        raise ValueError("OPD temperature must be positive")
    if top_k <= 0 or top_k > _global_vocab_size(weight, process_group):
        raise ValueError(f"Invalid OPD top_k={top_k}")
    vocab_chunk_size = max(1, int(vocab_chunk_size))
    hidden = hidden.to(device=weight.device, dtype=weight.dtype)
    local_vocab = int(weight.size(0))
    tp_rank = dist.get_rank(group=process_group) if _dist_ready(process_group) else 0
    global_offset = tp_rank * local_vocab

    row_max = torch.full((hidden.size(0), 1), -math.inf, device=hidden.device, dtype=torch.float32)
    for start in range(0, local_vocab, vocab_chunk_size):
        logits = torch.matmul(hidden, weight[start : start + vocab_chunk_size].t()).float()
        row_max = torch.maximum(row_max, logits.div(temperature).amax(dim=-1, keepdim=True))
    row_max = _all_reduce_no_grad(row_max, dist.ReduceOp.MAX, process_group)

    row_sum = torch.zeros_like(row_max)
    for start in range(0, local_vocab, vocab_chunk_size):
        logits = torch.matmul(hidden, weight[start : start + vocab_chunk_size].t()).float()
        row_sum.add_(torch.exp(logits.div(temperature).sub(row_max)).sum(dim=-1, keepdim=True))
    row_sum = _all_reduce_no_grad(row_sum, dist.ReduceOp.SUM, process_group)
    log_denom = row_max + row_sum.log()

    values = None
    indices = None
    for start in range(0, local_vocab, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, local_vocab)
        logits = torch.matmul(hidden, weight[start:stop].t()).float().div_(temperature)
        logps = logits.sub_(log_denom)
        chunk_k = min(top_k, stop - start)
        chunk_values, chunk_indices = torch.topk(logps, k=chunk_k, dim=-1)
        chunk_indices.add_(global_offset + start)
        values, indices = _merge_topk(values, indices, chunk_values, chunk_indices, top_k)

    if _dist_ready(process_group):
        world_size = dist.get_world_size(group=process_group)
        gathered_values = [torch.empty_like(values) for _ in range(world_size)]
        gathered_indices = [torch.empty_like(indices) for _ in range(world_size)]
        dist.all_gather(gathered_values, values, group=process_group)
        dist.all_gather(gathered_indices, indices, group=process_group)
        values = torch.cat(gathered_values, dim=-1)
        indices = torch.cat(gathered_indices, dim=-1)
        values, positions = torch.topk(values, k=top_k, dim=-1)
        indices = torch.gather(torch.cat(gathered_indices, dim=-1), dim=-1, index=positions)
    return values, indices


if HAVE_TRITON:

    @triton.jit
    def _linear_max_kernel(
        hidden,
        weight,
        output,
        n_rows: tl.constexpr,
        hidden_size: tl.constexpr,
        vocab_size: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols_base = tl.arange(0, BLOCK_N)
        h_base = tl.arange(0, BLOCK_H)
        valid_rows = rows < n_rows
        maxima = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        for col_start in range(0, vocab_size, BLOCK_N):
            cols = col_start + cols_base
            accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for h_start in range(0, hidden_size, BLOCK_H):
                hs = h_start + h_base
                h = tl.load(
                    hidden + rows[:, None] * hidden_size + hs[None, :],
                    mask=valid_rows[:, None] & (hs[None, :] < hidden_size),
                    other=0.0,
                )
                w = tl.load(
                    weight + cols[:, None] * hidden_size + hs[None, :],
                    mask=(cols[:, None] < vocab_size) & (hs[None, :] < hidden_size),
                    other=0.0,
                )
                accum += tl.dot(h, tl.trans(w))
            logits = tl.where(
                valid_rows[:, None] & (cols[None, :] < vocab_size),
                accum * inv_temperature,
                -float("inf"),
            )
            maxima = tl.maximum(maxima, tl.max(logits, axis=1))
        tl.store(output + rows, maxima, mask=valid_rows)


    @triton.jit
    def _linear_sum_kernel(
        hidden,
        weight,
        maxima,
        output,
        n_rows: tl.constexpr,
        hidden_size: tl.constexpr,
        vocab_size: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        cols_base = tl.arange(0, BLOCK_N)
        h_base = tl.arange(0, BLOCK_H)
        valid_rows = rows < n_rows
        row_max = tl.load(maxima + rows, mask=valid_rows, other=-float("inf"))
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        for col_start in range(0, vocab_size, BLOCK_N):
            cols = col_start + cols_base
            accum = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for h_start in range(0, hidden_size, BLOCK_H):
                hs = h_start + h_base
                h = tl.load(
                    hidden + rows[:, None] * hidden_size + hs[None, :],
                    mask=valid_rows[:, None] & (hs[None, :] < hidden_size),
                    other=0.0,
                )
                w = tl.load(
                    weight + cols[:, None] * hidden_size + hs[None, :],
                    mask=(cols[:, None] < vocab_size) & (hs[None, :] < hidden_size),
                    other=0.0,
                )
                accum += tl.dot(h, tl.trans(w))
            valid = valid_rows[:, None] & (cols[None, :] < vocab_size)
            logits = tl.where(valid, accum * inv_temperature, -float("inf"))
            row_sum += tl.sum(tl.exp(logits - row_max[:, None]), axis=1)
        tl.store(output + rows, row_sum, mask=valid_rows)


    @triton.jit
    def _selected_logps_kernel(
        hidden,
        weight,
        token_ids,
        valid_mask,
        log_denom,
        output,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        vocab_start: tl.constexpr,
        vocab_end: tl.constexpr,
        inv_temperature: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        k_block = tl.program_id(1)
        ks = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        hs_base = tl.arange(0, BLOCK_H)
        ids = tl.load(token_ids + row * top_k + ks, mask=ks < top_k, other=0)
        valid = (ks < top_k) & (tl.load(valid_mask + row * top_k + ks, mask=ks < top_k, other=0) != 0)
        local = valid & (ids >= vocab_start) & (ids < vocab_end)
        local_ids = ids - vocab_start
        accum = tl.zeros((BLOCK_K,), tl.float32)
        for h_start in range(0, hidden_size, BLOCK_H):
            hs = h_start + hs_base
            h = tl.load(hidden + row * hidden_size + hs, mask=hs < hidden_size, other=0.0)
            w = tl.load(
                weight + local_ids[:, None] * hidden_size + hs[None, :],
                mask=local[:, None] & (hs[None, :] < hidden_size),
                other=0.0,
            )
            accum += tl.sum(w * h[None, :], axis=1)
        result = tl.where(local, accum * inv_temperature - tl.load(log_denom + row), 0.0)
        tl.store(output + row * top_k + ks, result, mask=ks < top_k)


class _SelectedLogProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, token_ids, valid_mask, temperature, process_group):
        if not HAVE_TRITON or not hidden.is_cuda:
            raise RuntimeError("Asynchronous OPD selected-logprob projection requires CUDA Triton")
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        token_ids = token_ids.to(device=hidden.device, dtype=torch.long).contiguous()
        valid_mask = valid_mask.to(device=hidden.device, dtype=torch.bool).contiguous()
        n_rows, hidden_size = hidden.shape
        top_k = token_ids.size(1)
        local_vocab = weight.size(0)
        tp_rank = dist.get_rank(group=process_group) if _dist_ready(process_group) else 0
        vocab_start = tp_rank * local_vocab
        grid = (triton.cdiv(n_rows, 16),)

        row_max = torch.empty(n_rows, dtype=torch.float32, device=hidden.device)
        _linear_max_kernel[grid](
            hidden,
            weight,
            row_max,
            n_rows,
            hidden_size,
            local_vocab,
            1.0 / float(temperature),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_H=64,
            num_warps=4,
        )
        row_max = _all_reduce_no_grad(row_max, dist.ReduceOp.MAX, process_group)
        row_sum = torch.empty_like(row_max)
        _linear_sum_kernel[grid](
            hidden,
            weight,
            row_max,
            row_sum,
            n_rows,
            hidden_size,
            local_vocab,
            1.0 / float(temperature),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_H=64,
            num_warps=4,
        )
        row_sum = _all_reduce_no_grad(row_sum, dist.ReduceOp.SUM, process_group)
        log_denom = row_max + row_sum.log()

        block_k = min(64, triton.next_power_of_2(top_k))
        output = torch.empty((n_rows, top_k), dtype=torch.float32, device=hidden.device)
        _selected_logps_kernel[(n_rows, triton.cdiv(top_k, block_k))](
            hidden,
            weight,
            token_ids,
            valid_mask,
            log_denom,
            output,
            hidden_size,
            top_k,
            vocab_start,
            vocab_start + local_vocab,
            1.0 / float(temperature),
            BLOCK_K=block_k,
            BLOCK_H=64,
            num_warps=4,
        )
        output = _all_reduce_sum_autograd(output, process_group)
        ctx.save_for_backward(hidden, weight, token_ids, valid_mask, log_denom)
        ctx.temperature = float(temperature)
        ctx.process_group = process_group
        ctx.vocab_start = vocab_start
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, token_ids, valid_mask, log_denom = ctx.saved_tensors
        local_vocab = weight.size(0)
        local_ids = token_ids - ctx.vocab_start
        local = valid_mask & (local_ids >= 0) & (local_ids < local_vocab)
        safe_ids = local_ids.clamp(0, local_vocab - 1)
        grad_topk = grad_output.float().masked_fill(~valid_mask, 0.0)
        local_grad = grad_topk.masked_fill(~local, 0.0)
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        inv_temperature = 1.0 / ctx.temperature

        for start in range(0, hidden.size(0), BACKWARD_CHUNK_ROWS):
            stop = min(start + BACKWARD_CHUNK_ROWS, hidden.size(0))
            h = hidden[start:stop].to(weight.dtype)
            logits = torch.matmul(h, weight.t()).float().mul_(inv_temperature)
            probabilities = logits.sub_(log_denom[start:stop, None]).exp_()
            probabilities.mul_(grad_topk[start:stop].sum(dim=-1, keepdim=True)).neg_()
            probabilities.scatter_add_(1, safe_ids[start:stop], local_grad[start:stop])
            grad_logits = probabilities.mul_(inv_temperature)
            grad_hidden[start:stop] = torch.matmul(grad_logits.to(weight.dtype), weight).to(hidden.dtype)
            grad_weight.add_(torch.matmul(grad_logits.t().to(h.dtype), h).to(weight.dtype))
        if _dist_ready(ctx.process_group):
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.process_group)
        return grad_hidden, grad_weight, None, None, None, None


def _dense_selected_logps(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.matmul(hidden.to(weight.dtype), weight.t()).float().div(temperature)
    logps = torch.log_softmax(logits, dim=-1)
    safe_ids = token_ids.clamp(0, weight.size(0) - 1)
    return torch.gather(logps, dim=-1, index=safe_ids).masked_fill(~valid_mask, 0.0)


def topk_kl(
    student_logps: torch.Tensor,
    teacher_logps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    direction: OpdKLDirection,
    normalize_teacher: bool,
    pointwise_clip: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = valid_mask.bool() & torch.isfinite(teacher_logps)
    safe_teacher = teacher_logps.masked_fill(~valid, -math.inf)
    if normalize_teacher or direction == "reverse":
        teacher_norm = safe_teacher - torch.logsumexp(safe_teacher, dim=-1, keepdim=True)
    else:
        teacher_norm = safe_teacher
    teacher_prob = teacher_norm.exp().masked_fill(~valid, 0.0)

    if direction == "forward":
        pointwise = teacher_prob * (teacher_norm - student_logps)
    elif direction == "reverse":
        student_norm = student_logps.masked_fill(~valid, -math.inf)
        student_norm = student_norm - torch.logsumexp(student_norm, dim=-1, keepdim=True)
        student_prob = student_norm.exp().masked_fill(~valid, 0.0)
        pointwise = student_prob * (student_norm - teacher_norm)
    else:
        raise ValueError(f"Unsupported OPD KL direction: {direction}")

    pointwise = pointwise.masked_fill(~valid, 0.0)
    unclipped = pointwise
    if pointwise_clip > 0:
        pointwise = pointwise.clamp(max=pointwise_clip)
    valid_count = valid.float().sum().clamp_min(1)
    return pointwise.sum(dim=-1), {
        "clip_frac": ((unclipped > pointwise) & valid).float().sum() / valid_count,
        "topk_mass": teacher_logps.exp().masked_fill(~valid, 0.0).sum(dim=-1),
    }


def hidden_topk_kl(
    student_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_weight: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    direction: OpdKLDirection,
    normalize_teacher: bool,
    pointwise_clip: float,
    token_chunk_size: int,
    vocab_chunk_size: int,
    process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | str]]:
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(f"OPD hidden-state shape mismatch: {student_hidden.shape} != {teacher_hidden.shape}")
    if student_weight.shape != teacher_weight.shape:
        raise ValueError(f"OPD output-head shape mismatch: {student_weight.shape} != {teacher_weight.shape}")
    if valid_token_mask.shape != student_hidden.shape[:-1]:
        raise ValueError("OPD valid-token mask does not match hidden-state rows")

    shape = student_hidden.shape[:-1]
    student_flat = student_hidden.reshape(-1, student_hidden.size(-1))
    teacher_flat = teacher_hidden.detach().reshape_as(student_flat)
    valid_flat = valid_token_mask.reshape(-1).bool()
    active_rows = torch.nonzero(valid_flat, as_tuple=False).flatten()
    losses = torch.zeros(student_flat.size(0), dtype=torch.float32, device=student_flat.device)
    masses = torch.zeros_like(losses)
    clip_fracs = []
    start_time = time.time()
    token_chunk_size = max(1, int(token_chunk_size))

    for start in range(0, active_rows.numel(), token_chunk_size):
        rows = active_rows[start : start + token_chunk_size]
        teacher_logps, teacher_ids = teacher_topk_from_hidden(
            teacher_flat.index_select(0, rows),
            teacher_weight,
            top_k=top_k,
            temperature=temperature,
            vocab_chunk_size=vocab_chunk_size,
            process_group=process_group,
        )
        selected_student = student_flat.index_select(0, rows)
        topk_valid = torch.ones_like(teacher_ids, dtype=torch.bool)
        if selected_student.is_cuda:
            student_logps = _SelectedLogProbs.apply(
                selected_student,
                student_weight,
                teacher_ids,
                topk_valid,
                temperature,
                process_group,
            )
            backend = "triton_selected_logps"
        else:
            student_logps = _dense_selected_logps(
                selected_student,
                student_weight,
                teacher_ids,
                topk_valid,
                temperature,
            )
            backend = "cpu_dense_reference"
        chunk_loss, stats = topk_kl(
            student_logps,
            teacher_logps,
            topk_valid,
            direction=direction,
            normalize_teacher=normalize_teacher,
            pointwise_clip=pointwise_clip,
        )
        losses.index_copy_(0, rows, chunk_loss)
        masses.index_copy_(0, rows, stats["topk_mass"])
        clip_fracs.append(stats["clip_frac"])

    return losses.view(shape), {
        "topk_mass": masses.view(shape),
        "clip_frac": torch.stack(clip_fracs).mean()
        if clip_fracs
        else torch.tensor(0.0, device=student_hidden.device),
        "tokens": float(active_rows.numel()),
        "elapsed_s": time.time() - start_time,
        "backend": backend if active_rows.numel() else "empty",
    }


@dataclass
class OpdHiddenStatePayload:
    request_id: str
    hidden_states: torch.Tensor
    valid_mask: torch.Tensor
    created_at: float

    @property
    def nbytes(self) -> int:
        return self.hidden_states.nbytes + self.valid_mask.nbytes


class OpdHiddenStateBuffer:
    def __init__(self, max_pending_batches: int):
        if max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        self.max_pending_batches = int(max_pending_batches)
        self._payloads: dict[str, OpdHiddenStatePayload] = {}
        self._lock = threading.Lock()
        self.total_puts = 0
        self.total_pops = 0
        self.overflow_count = 0

    def put(self, payload: OpdHiddenStatePayload) -> None:
        with self._lock:
            if payload.request_id in self._payloads:
                raise ValueError(f"Duplicate OPD request id: {payload.request_id}")
            if len(self._payloads) >= self.max_pending_batches:
                self.overflow_count += 1
                raise BufferError("OPD hidden-state cache is full")
            self._payloads[payload.request_id] = payload
            self.total_puts += 1

    def pop(self, request_id: str) -> OpdHiddenStatePayload:
        with self._lock:
            try:
                payload = self._payloads.pop(request_id)
            except KeyError as exc:
                raise KeyError(f"Unknown OPD request id: {request_id}") from exc
            self.total_pops += 1
            return payload

    def discard(self, request_id: str) -> bool:
        with self._lock:
            return self._payloads.pop(request_id, None) is not None

    def metrics(self) -> dict[str, float]:
        with self._lock:
            now = time.time()
            oldest = max((now - payload.created_at for payload in self._payloads.values()), default=0.0)
            return {
                "opd/cache_pending": float(len(self._payloads)),
                "opd/cache_bytes": float(sum(payload.nbytes for payload in self._payloads.values())),
                "opd/cache_total_puts": float(self.total_puts),
                "opd/cache_total_pops": float(self.total_pops),
                "opd/cache_overflow_count": float(self.overflow_count),
                "opd/cache_oldest_age_seconds": float(oldest),
            }
