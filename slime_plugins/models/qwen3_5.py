import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers.activations import ACT2FN

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
except ImportError:
    pass

from .hf_attention import HuggingfaceAttention, _load_hf_config
from .qwen_gdn_backend import get_chunk_gated_delta_rule


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


# Adapted from Qwen3NextGatedDeltaNet but with separate in_proj_qkv and in_proj_z
class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet with varlen support.
    Unlike Qwen3Next which uses a combined in_proj_qkvz, Qwen3.5 uses
    separate in_proj_qkv (for Q,K,V) and in_proj_z (for Z).
    """

    def __init__(self, config, layer_idx: int, args=None):
        super().__init__()
        self.gdn_backend = getattr(args, "qwen_gdn_backend", "fla")
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule(self.gdn_backend)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # Separate projections for QKV and Z (unlike Qwen3Next which combines QKVZ)
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        projection_size_z = self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, projection_size_qkv, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, projection_size_z, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=torch.cuda.current_device(),
            dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward_tp_sharded(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tp_rank: int,
        tp_size: int,
    ):
        """Head-sharded forward: this TP rank computes only its key/value heads on the full sequence.

        Same parameters and maths as ``forward`` (rows/columns of the replicated weights are sliced per
        rank), but returns a PARTIAL ``out_proj`` result that the caller must sum over the TP group
        (a reduce-scatter in the sequence-parallel layout). Key head ``k`` serves value heads
        ``k * rep .. (k + 1) * rep - 1``, so a rank's key and value heads line up. Parameter gradients
        are likewise partial and are summed over TP through Megatron's ``sequence_parallel`` param flag.
        """
        from einops import rearrange
        from fla.modules.conv.causal_conv1d import causal_conv1d

        batch_size, seq_len, _ = hidden_states.shape
        nk, nv = self.num_k_heads // tp_size, self.num_v_heads // tp_size
        kd, vd = self.head_k_dim, self.head_v_dim
        q0, v0 = tp_rank * nk * kd, tp_rank * nv * vd

        def qkv_rows(w):  # rows/channels of the flat [Q_all, K_all, V_all] layout owned by this rank
            return torch.cat(
                [w[q0 : q0 + nk * kd], w[self.key_dim + q0 : self.key_dim + q0 + nk * kd],
                 w[2 * self.key_dim + v0 : 2 * self.key_dim + v0 + nv * vd]],
                dim=0,
            )

        mixed_qkv = F.linear(hidden_states, qkv_rows(self.in_proj_qkv.weight))
        z = F.linear(hidden_states, self.in_proj_z.weight[v0 : v0 + nv * vd]).reshape(batch_size, seq_len, nv, vd)
        b = F.linear(hidden_states, self.in_proj_b.weight[tp_rank * nv : (tp_rank + 1) * nv])
        a = F.linear(hidden_states, self.in_proj_a.weight[tp_rank * nv : (tp_rank + 1) * nv])

        mixed_qkv, _ = causal_conv1d(
            x=mixed_qkv,
            weight=rearrange(qkv_rows(self.conv1d.weight), "d 1 w -> d w"),
            bias=None,
            residual=None,
            initial_state=None,
            output_final_state=False,
            activation=self.conv1d.activation,
            backend=self.conv1d.backend,
            cu_seqlens=cu_seqlens,
        )
        query, key, value = torch.split(mixed_qkv, [nk * kd, nk * kd, nv * vd], dim=-1)
        query = query.reshape(batch_size, seq_len, nk, kd)
        key = key.reshape(batch_size, seq_len, nk, kd)
        value = value.reshape(batch_size, seq_len, nv, vd)

        beta = b.sigmoid()
        sl = slice(tp_rank * nv, (tp_rank + 1) * nv)
        g = -self.A_log[sl].float().exp() * F.softplus(a.float() + self.dt_bias[sl])
        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)
        if self.gdn_backend == "flashqla":
            query, key, value, g, beta = (t.contiguous() for t in (query, key, value, g, beta))

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        )
        z_shape_og = z.shape
        core_attn_out = self.norm(core_attn_out.reshape(-1, core_attn_out.shape[-1]), z.reshape(-1, z.shape[-1]))
        core_attn_out = core_attn_out.reshape(z_shape_og).reshape(batch_size, seq_len, -1)
        return F.linear(core_attn_out, self.out_proj.weight[:, v0 : v0 + nv * vd])  # partial over TP

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # Projections (flat layout: [Q_all, K_all, V_all])
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # Convolution on the flat QKV
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )

        # Split into Q, K, V (flat split, matching HF layout)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if self.gdn_backend == "flashqla":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            g = g.contiguous()
            beta = beta.contiguous()

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        # Qwen3.5 is a VLM model with nested text_config
        self.hf_config = _get_text_config(self.hf_config)
        self.hf_config._attn_implementation = "flash_attention_2"

        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx, args=args)

        # Use a simple RMSNorm
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

        # Opt-in head sharding of the GDN over the TP group (THC_GDN_TP_SHARD=1): each TP rank computes
        # 1/TP of the heads on the full sequence instead of all heads redundantly. Parameters stay replicated
        # (same checkpoint); their gradients become per-rank partials, so flag them like Megatron's
        # sequence-parallel params (grads all-reduced over TP).
        from megatron.core import mpu

        self.gdn_tp_sharded = False
        tp = mpu.get_tensor_model_parallel_world_size()
        if os.environ.get("THC_GDN_TP_SHARD", "0") == "1" and tp > 1:
            assert args.sequence_parallel, "THC_GDN_TP_SHARD needs --sequence-parallel"
            cfg = self.linear_attn
            assert cfg.num_k_heads % tp == 0 and cfg.num_v_heads % tp == 0, "GDN heads must divide by TP"
            self.gdn_tp_sharded = True
            for p in self.parameters():
                p.sequence_parallel = True

    def hf_forward(self, hidden_states, packed_seq_params):
        if self.gdn_tp_sharded:
            from megatron.core import mpu

            # input_layernorm already ran on the local sequence shard in HuggingfaceAttention.forward
            return self.linear_attn.forward_tp_sharded(
                hidden_states,
                packed_seq_params.cu_seqlens_q,
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_tensor_model_parallel_world_size(),
            )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
