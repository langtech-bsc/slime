# Salamandra-7B CISPO (fully-async)

Disaggregated CISPO training for Salamandra-7B, migrated from VERL `config/cispo.yaml`. Uses `train_async.py` with fully-async SGLang rollout, Megatron TP=4 actor, and SGLang TP=2 rollout engines.

**Before your first run:** see [prerequisites and downloads](salamandra-7b-cispo-downloads.md) (datasets, torch_dist conversion, slime install). Nothing is downloaded automatically by the scripts.

## Repository layout

Salamandra launch scripts live in the parent [rl-training](https://github.com/langtech-bsc/rl-training) repo; slime is a submodule there:

```text
rl-training/
├── slime/          # this fork (train_async.py, docs, …)
└── scripts/        # Salamandra launchers (run from rl-training root)
```

## Environment Setup

Inside the slime Docker image (or equivalent Megatron + SGLang stack):

```bash
cd /path/to/rl-training/slime
pip install -e . --no-deps
```

## Model and Data

**HF checkpoint** (default in the launcher):

```text
/gpfs/projects/bsc88/text/models/BSC-NeMo-RL_prod_2026-03-20/results/sft/yolo-M4/hf-safetensors/Salamandra-7b_pre-1.4_sft-5.0_lr2e-5_bs256_warmup20
```

Override with `MODEL_DIR` if needed.

**Sample training data** (repo convention from [quick_start](../get_started/quick_start.md)):

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

Set `DATA_PATH=/root/dapo-math-17k/dapo-math-17k.jsonl` (default in the script).

## Convert HF → Megatron torch_dist

One-time conversion (4 GPUs minimum for TP=4 layout during convert):

```bash
MODEL_DIR=/gpfs/projects/bsc88/text/models/BSC-NeMo-RL_prod_2026-03-20/results/sft/yolo-M4/hf-safetensors/Salamandra-7b_pre-1.4_sft-5.0_lr2e-5_bs256_warmup20

cd /path/to/rl-training/slime
source ../scripts/models/salamandra-7B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 4 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${MODEL_DIR}_torch_dist"
```

Model args match the HF `config.json`: 32 layers, 4096 hidden, 11008 FFN, vocab 256000, Llama3 rope scaling factor 20.

## Run Training

Full run (16 GPUs: 4 actor + 12 rollout):

```bash
cd /path/to/rl-training
bash scripts/run-salamandra-7B-cispo-async.sh
```

**Quick smoke test** (fewer rollouts / smaller batch):

```bash
cd /path/to/rl-training
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=4 GLOBAL_BATCH_SIZE=16 SAVE_INTERVAL=9999 \
  bash scripts/run-salamandra-7B-cispo-async.sh
```

### Multi-node Ray (VERL-style)

On MN5, from rl-training root: `sbatch scripts/mn5/sbatch_salamandra-7B-cispo.sh`. It starts Ray head/workers with `srun` + Singularity (same pattern as `Salamandra-rl/quick_start/sbatch_cispo.sh`), then launches training with **`python3 train_async.py` directly** — not `ray job submit`. Set `RAY_ADDRESS=${head_ip}:6379` and `SLIME_SCRIPT_EXTERNAL_RAY=1`; `train_async.py` calls `ray.init(address=RAY_ADDRESS)` like VERL's `run_ppo()`.

Manual cluster (1 actor node × 4 GPU + 3 rollout nodes × 4 GPU):

- Node 0 (head): `ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4`
- Worker nodes: `ray start --address=${MASTER_ADDR}:6379 --num-gpus 4`

Then on the head node:

```bash
export RAY_ADDRESS=${MASTER_ADDR}:6379
export SLIME_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=<head-ip>
export NUM_GPUS=16
export ACTOR_GPUS=4
export ROLLOUT_GPUS=12
cd /path/to/rl-training && bash scripts/run-salamandra-7B-cispo-async.sh
```

For checkpoint conversion on multi-node, see `slime/utils/external_utils/command_utils.py` (`ExecuteTrainConfig`, `convert_checkpoint` with `multinode=True`).

## VERL → slime mapping

| VERL `cispo.yaml` | slime |
|-------------------|-------|
| `policy_loss.loss_mode: cispo` | `--advantage-estimator cispo` |
| `clip_ratio_low: 10.0`, `clip_ratio_high: 0.2` | `--eps-clip 10.0 --eps-clip-high 0.2` |
| `adv_estimator: grpo`, `norm_adv_by_std_in_grpo: True` | default GRPO std normalization (no flag) |
| `use_rollout_log_probs` + `rollout_correction.bypass_mode` | `--use-rollout-logprobs` |
| `use_kl_loss: False` | omit `--use-kl-loss` |
| actor TP=4, `sequence_parallel` | `--tensor-model-parallel-size 4 --sequence-parallel` |
| `use_dynamic_bsz`, `ppo_max_token_len_per_gpu: 16384` | `--use-dynamic-batch-size --max-tokens-per-gpu 16384` |
| `log_prob_max_token_len_per_gpu: 16384` | `--log-probs-max-tokens-per-gpu 16384` |
| rollout TP=2, `gpu_memory_utilization: 0.75` | `--rollout-num-gpus-per-engine 2 --sglang-mem-fraction-static 0.75` |
| `enforce_eager`, `enable_chunked_prefill`, FA3 | `--sglang-enforce-eager --sglang-chunked-prefill-size 4096 --sglang-attention-backend fa3` |
| `n: 16`, lengths 4096 / 12288 (16k context) | `--n-samples-per-prompt 16`, `--rollout-max-prompt-len 4096`, `--rollout-max-response-len 12288`, `--rollout-max-context-len 16384` |
| `reward_manager.name: dapo` | `--rm-type dapo` |
| `hybrid_engine: False`, async | `train_async.py`, no `--colocate` |
| `partial_rollout` | `fully_async_rollout` (aborted samples recycled) |

Launcher: [rl-training/scripts/run-salamandra-7B-cispo-async.sh](https://github.com/langtech-bsc/rl-training/blob/main/scripts/run-salamandra-7B-cispo-async.sh)

Rollout prompt/response dumps are **always on** (async, under `{SAVE_DIR}/rollout_dumps/`). See [rl-training/docs/rollout_examples.md](../../../../docs/rollout_examples.md).

## Explicitly excluded from VERL config

| VERL feature | Reason |
|--------------|--------|
| GEM datasets / live envs / custom reward | Out of scope; use `dapo-math-17k` + `--rm-type dapo` |
| `allow_stale_kv_cache_after_weight_sync` | Stale KV not needed |
| Vision / VL chat template, `freeze_vision_tower` | Text-only Salamandra-7B |
| Multi-turn tools / agent loops | Not migrated |
| `early_truncation`, `forced_answer` | SGLang-only VERL hooks |
| VERL `checkpoint_engine` / Hydra | slime weight sync via `--update-weights-interval` (default 1) |

## Known gaps vs VERL

- **DAPO overlong buffer** (`overlong_buffer_cfg`): slime `dapo` reward does not apply the length penalty; truncated responses may score differently.
- **Adam epsilon `1e-15`**: not set in the launcher; Megatron default is used.
- **Eval**: fully-async path skips eval during training (same as VERL `test_freq: 0` in the source config). Run eval separately if needed.

## Topology

```text
Actor (4 GPU, Megatron TP=4)  ←── train_async.py
Rollout (12 GPU, 6 × SGLang TP=2)  ←── fully_async_rollout worker
Weight sync every rollout step (`update_weights_interval=1`). Samples with
`trainer_weight_version - rollout_weight_version > 3` are discarded at train
time via `--max-rollout-weight-staleness 3` (override with `MAX_ROLLOUT_WEIGHT_STALENESS`).
Logged metrics: `rollout_weight_staleness/discarded`, `rollout_weight_staleness/discard_ratio`,
`train/rollout_weight_staleness_discarded`.
```
