# Prelude-9B (BSC tool-use model): Qwen3 dense shape with a 262k vocabulary and
# YaRN RoPE (HF rope_parameters: factor 11.9 over 16384, theta 813000,
# beta_fast 1448, beta_slow 32.44, truncate false). YaRN needs --no-rope-fusion
# because the fused packed-sequence RoPE kernel drops the YaRN mscale.
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base 813000
   --max-position-embeddings 163840
   --yarn-rotary-scaling-factor 11.9
   --yarn-original-max-position-embeddings 16384
   --yarn-beta-fast 1448
   --yarn-beta-slow 32.44
   --no-yarn-truncate
   --no-rope-fusion
   --vocab-size 262272
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)
