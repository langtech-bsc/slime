"""Chunked LM head + vocab-parallel log-probability that never materialises the full logits.

Why: with a 248,320-token vocabulary the fp32 logits ``[tokens, V/TP]`` (plus the clone, the saved
softmax and the gradient made by the fused cross-entropy) cost ~0.75 MB per token per GPU at TP4,
which caps the tokens per GPU near 32k and blocks TP < 4. Here the output projection and the
log-probability are computed ``chunk`` tokens at a time and the chunk logits are recomputed in the
backward pass, so only ``[chunk, V/TP]`` is ever alive.

Numerics follow the existing path: the logits of a chunk are a bf16 GEMM whose output is cast to
fp32 (Megatron's Float16Module does that cast for the last stage), the softmax is fp32 and
vocab-parallel (max / sum-exp / target logit all-reduced over the TP group), and the logit gradient
is cast to bf16 before the two backward GEMMs, as autograd does through the ``.float()``.

Enable with the environment variable THC_CHUNKED_LM_HEAD=<tokens per chunk> (e.g. 4096). Not supported
(raises): entropy, rollout top-p replay, critic heads.
"""
from __future__ import annotations

import os
import types

import torch
import torch.distributed as dist

_STATE = {"weight": None}


def chunk_size() -> int:
    return int(os.environ.get("THC_CHUNKED_LM_HEAD", "0") or 0)


def enabled() -> bool:
    return chunk_size() > 0


def register_lm_head_weight(weight: torch.nn.Parameter) -> None:
    _STATE["weight"] = weight


def lm_head_weight() -> torch.nn.Parameter:
    if _STATE["weight"] is None:
        raise RuntimeError("chunked LM head enabled but no output-layer weight was registered")
    return _STATE["weight"]


def patch_output_layer(gpt_model) -> bool:
    """Make ``gpt_model.output_layer`` return the (sequence-gathered) hidden states instead of logits.

    The method is replaced on the instance, so parameter names, checkpoints and weight sync are
    unchanged. Returns True if a post-process output layer was patched.
    """
    layer = getattr(gpt_model, "output_layer", None)
    if layer is None or getattr(layer, "weight", None) is None or not getattr(gpt_model, "post_process", False):
        return False
    from megatron.core import tensor_parallel

    if not layer.sequence_parallel and dist.get_world_size(layer.tp_group) > 1:
        raise RuntimeError("chunked LM head needs sequence_parallel when tensor parallelism is used")

    def hidden_forward(self, input_, weight=None, runtime_gather_output=None, **kwargs):
        if kwargs.get("output_cross_entropy_loss"):
            raise RuntimeError("fused linear cross-entropy is not supported with the chunked LM head")
        if self.sequence_parallel:
            # same gather ColumnParallelLinear does for its input; backward reduce-scatters the partial dgrads
            input_ = tensor_parallel.gather_from_sequence_parallel_region(
                input_, tensor_parallel_output_grad=True, group=self.tp_group
            )
        return input_, None

    layer.forward = types.MethodType(hidden_forward, layer)
    register_lm_head_weight(layer.weight)
    return True


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with bf16/fp16 inputs and an fp32 result (single GEMM, fp32 accumulation, no bf16 rounding)."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(a.float(), b.float())


class _ChunkedLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, tp_group, chunk, temperature):
        T, Vl = hidden.size(0), weight.size(0)
        vstart = dist.get_rank(tp_group) * Vl if dist.get_world_size(tp_group) > 1 else 0
        out = torch.empty(T, dtype=torch.float32, device=hidden.device)
        lse_all = torch.empty_like(out)
        tgt_local = targets - vstart
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            logits = torch.matmul(hidden[s:e], weight.t()).float()  # bf16 GEMM -> fp32, as Float16Module does
            if temperature != 1.0:
                logits = logits / temperature
            m = logits.amax(dim=-1)
            idx = tgt_local[s:e]
            ok = (idx >= 0) & (idx < Vl)
            tl = torch.zeros(e - s, dtype=torch.float32, device=hidden.device)
            rows = ok.nonzero(as_tuple=True)[0]
            tl[rows] = logits[rows, idx[rows]]
            if dist.get_world_size(tp_group) > 1:
                dist.all_reduce(m, op=dist.ReduceOp.MAX, group=tp_group)
            se = (logits - m[:, None]).exp().sum(dim=-1)
            if dist.get_world_size(tp_group) > 1:
                dist.all_reduce(se, group=tp_group)
                dist.all_reduce(tl, group=tp_group)
            lse = m + se.log()
            out[s:e] = tl - lse
            lse_all[s:e] = lse
        ctx.save_for_backward(hidden, weight, targets, lse_all)
        ctx.tp_group, ctx.chunk, ctx.temperature = tp_group, chunk, temperature
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets, lse_all = ctx.saved_tensors
        tp_group, chunk, temperature = ctx.tp_group, ctx.chunk, ctx.temperature
        T, Vl = hidden.size(0), weight.size(0)
        vstart = dist.get_rank(tp_group) * Vl if dist.get_world_size(tp_group) > 1 else 0
        tgt_local = targets - vstart
        dh = torch.empty_like(hidden)
        # Megatron contract: fold the wgrad straight into the fp32 main_grad only when the DDP hook can be told
        # (flag present); otherwise hand back an ordinary grad and let the hook add it.
        use_main_grad = hasattr(weight, "main_grad") and hasattr(weight, "grad_added_to_main_grad")
        dW = None if use_main_grad else torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            logits = torch.matmul(hidden[s:e], weight.t()).float()
            if temperature != 1.0:
                logits = logits / temperature
            p = (logits - lse_all[s:e, None]).exp_()  # softmax over the full vocab, local shard
            d = p.neg_()
            idx = tgt_local[s:e]
            ok = (idx >= 0) & (idx < Vl)
            rows = ok.nonzero(as_tuple=True)[0]
            d[rows, idx[rows]] += 1.0  # d logp / d logits = onehot - softmax
            d.mul_(grad_out[s:e, None])
            if temperature != 1.0:
                d.div_(temperature)
            d_b = d.to(hidden.dtype)  # bf16 for the backward GEMMs, as autograd does through the fp32 cast
            dh[s:e] = torch.matmul(d_b, weight)  # partial over the local vocab shard (summed by the SP gather)
            g = _mm_fp32(d_b.t(), hidden[s:e])
            if use_main_grad:
                weight.main_grad.add_(g)
            else:
                dW += g
        if use_main_grad:
            # DDP hook contract (as Megatron's linear does): grad already folded into main_grad
            weight.grad_added_to_main_grad = True
            dweight = torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)
        else:
            dweight = dW.to(weight.dtype)
        return dh, dweight, None, None, None, None


def chunked_log_probs(hidden, weight, targets, tp_group, temperature=1.0):
    """Per-token log p(target) for hidden ``[T, H]``, TP-sharded ``weight`` ``[V/TP, H]`` and global ids ``[T]``."""
    hidden = hidden.to(weight.dtype)  # Float16Module hands the last stage fp32; bf16 -> fp32 -> bf16 is exact
    return _ChunkedLogProb.apply(hidden, weight, targets, tp_group, chunk_size(), float(temperature))
