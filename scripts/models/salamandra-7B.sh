MODEL_ARGS=(
   --swiglu
   --num-layers 32
   --hidden-size 4096
   --ffn-hidden-size 11008
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --max-position-embeddings 163840
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base 10000
   --vocab-size 256000
   --kv-channels 128
   --use-rope-scaling
   --rotary-scaling-factor 20.0
   --untie-embeddings-and-output-weights
)
