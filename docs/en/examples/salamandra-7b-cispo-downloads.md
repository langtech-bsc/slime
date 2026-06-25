# Salamandra-7B CISPO — prerequisites and downloads

Checklist for running [Salamandra-7B CISPO async training](salamandra-7b-cispo.md) on MN5 with the shared VERL Singularity runtime. **Nothing in this file is downloaded automatically**; use it to prepare paths before `sbatch` or an interactive `salloc`.

## Already on GPFS (no internet download)

These are expected to exist on MareNostrum 5 project storage. Verify paths before submitting a job.

| Item | Path | Notes |
|------|------|--------|
| Singularity image | `/gpfs/projects/bsc88/singularity-images/verlai.sif` | Frozen VERL/Megatron/SGLang stack |
| VERL / SGLang env | `/gpfs/projects/bsc88/text/environments/verl_mn5_python3.12_20260220` | `VERL_ENV` in `runtime_env_common.sh` |
| Extra Python packages | `.../vision_rl_mn5_python3.12_20260206/lib/python3.12/site-packages` | Bind-mounted as `/extra_site_packages` |
| FlashAttention overlay | `.../vision_rl_mn5_python3.12_20260206/lib/python3.12/flash_attn_site_packages` | Bind-mounted as `/flash_attn_site_packages` |
| Runtime helpers | `/gpfs/projects/bsc88/text/models/vision/salamandra-rl-parent/Salamandra-rl/quick_start/runtime_env_common.sh` | Shared bind/env wiring |
| Salamandra-7B HF weights | `/gpfs/projects/bsc88/text/models/BSC-NeMo-RL_prod_2026-03-20/results/sft/yolo-M4/hf-safetensors/Salamandra-7b_pre-1.4_sft-5.0_lr2e-5_bs256_warmup20` | Test model for this migration |
| slime repo checkout | e.g. `/home/bsc/bsc474046/repositories/rl-training` (with `slime/` submodule) | Bind `RL_TRAINING_ROOT` and `SLIME_ROOT` at job launch |
| Hugging Face hub cache (optional) | `/gpfs/scratch/bsc88/${USER}/.cache/huggingface` | Speeds up tokenizer/config loads |

**Not used for this workflow:** GEM integration (`gem-llm` / `/gem_site_packages`). Do not install or bind GEM for the text-only CISPO run.

**Inside the container (no separate download):** Megatron-LM is provided at `/root/Megatron-LM/` in `verlai.sif`.

---

## Download from the internet

### 1. Training dataset — required

slime uses the same sample RL data as upstream slime quick start: **dapo-math-17k** (`prompt` / `label` JSONL).

```bash
# From a login node or inside the container (needs outbound HF access)
export HF_HOME=/gpfs/scratch/bsc88/${USER}/.cache/huggingface

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /gpfs/scratch/bsc88/${USER}/datasets/dapo-math-17k
```

Then point the launcher at:

```bash
export DATA_PATH=/gpfs/scratch/bsc88/${USER}/datasets/dapo-math-17k/dapo-math-17k.jsonl
```

Alternative if you prefer the Docker-style path inside the container:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
# DATA_PATH=/root/dapo-math-17k/dapo-math-17k.jsonl
```

**HF dataset:** [zhuzilin/dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)

### 2. Evaluation dataset — optional

Not required for the initial CISPO smoke run (eval is disabled in the async launcher). For later eval:

```bash
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /gpfs/scratch/bsc88/${USER}/datasets/aime-2024
```

**HF dataset:** [zhuzilin/aime-2024](https://huggingface.co/datasets/zhuzilin/aime-2024)

### 3. slime Python package — required once per runtime

The container does not ship this repo. Install editable from your checkout **inside** `verlai.sif` (no PyPI download if you use `--no-deps`):

```bash
cd /path/to/rl-training/slime   # slime submodule
python -m pip install -e . --no-deps --user
python -c "import slime; print(slime.__file__)"
```

Use `--user` so packages land in `${CONTAINER_HOME}/.local/...` when you launch with `singularity exec --home "${CONTAINER_HOME}"`. That avoids the PEP 668 `externally-managed-environment` error on the container's system Python.

If `--user` is still rejected on your image, use either:

```bash
python -m pip install -e . --no-deps --user --break-system-packages
# or
PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install -e . --no-deps --user
```

If `pip install -e .` fails on missing dependencies, compare against the slime Docker image / `requirements.txt` and install only what is missing from GPFS overlays first.

### 4. Weights & Biases — optional

Only if you uncomment `WANDB_ARGS` in [rl-training/scripts/run-salamandra-7B-cispo-async.sh](https://github.com/langtech-bsc/rl-training/blob/main/scripts/run-salamandra-7B-cispo-async.sh):

- Create a W&B account and API key: [https://wandb.ai](https://wandb.ai)
- `export WANDB_API_KEY=...` (or use `WANDB_MODE=offline` on air-gapped runs)

---

## Generate locally (not a download)

### Megatron `torch_dist` checkpoint — required before training

The HF safetensors checkpoint on GPFS is **not** Megatron format. Convert once on 4 GPUs (matches actor TP=4):

```bash
MODEL_DIR=/gpfs/projects/bsc88/text/models/BSC-NeMo-RL_prod_2026-03-20/results/sft/yolo-M4/hf-safetensors/Salamandra-7b_pre-1.4_sft-5.0_lr2e-5_bs256_warmup20

cd /path/to/rl-training/slime
source ../scripts/models/salamandra-7B.sh

# Inside verlai.sif on an ACC node with 4 GPUs:
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 4 \
  tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_DIR}" \
  --save "${MODEL_DIR}_torch_dist"
```

**Status to check:** `${MODEL_DIR}_torch_dist/` must exist before `bash scripts/run-salamandra-7B-cispo-async.sh` from rl-training root.

### Training checkpoints — produced by the job

Default save directory:

```text
${MODEL_DIR}_slime_cispo/
```

Override with `SAVE_DIR=...` if you want checkpoints on scratch instead of next to the HF tree.

### Scratch directories — create empty dirs

```bash
export SCRATCH_ROOT=/gpfs/scratch/bsc88/${USER}
mkdir -p "${SCRATCH_ROOT}/singularity_homes/verl"
mkdir -p "${SCRATCH_ROOT}/datasets"
mkdir -p "${SCRATCH_ROOT}/vision/logs/slime"   # Slurm stdout/stderr
mkdir -p /tmp/${USER}/r                         # Ray temp (or set RAY_TMP_ROOT on scratch)
```

---

## Pre-flight checklist

| Step | Action | Required? |
|------|--------|-----------|
| 1 | Confirm `verlai.sif` and `VERL_ENV` paths exist | Yes |
| 2 | Confirm Salamandra-7B HF directory has all `model-*-of-*.safetensors` shards | Yes |
| 3 | Download **dapo-math-17k** and set `DATA_PATH` | Yes |
| 4 | Run **HF → torch_dist** conversion | Yes |
| 5 | `pip install -e . --no-deps` for slime in container | Yes |
| 6 | Slurm allocation: 4 nodes × 4 GPU (1 actor + 3 rollout) or 16 GPUs on one node | Yes |
| 7 | Download aime-2024 | No (eval later) |
| 8 | W&B API key | No |

---

## Quick path verification (no downloads)

Run on a login or GPU node:

```bash
MODEL_DIR=/gpfs/projects/bsc88/text/models/BSC-NeMo-RL_prod_2026-03-20/results/sft/yolo-M4/hf-safetensors/Salamandra-7b_pre-1.4_sft-5.0_lr2e-5_bs256_warmup20

for p in \
  /gpfs/projects/bsc88/singularity-images/verlai.sif \
  /gpfs/projects/bsc88/text/environments/verl_mn5_python3.12_20260220 \
  "${MODEL_DIR}/config.json" \
  "${MODEL_DIR}_torch_dist" \
  "${DATA_PATH:-/gpfs/scratch/bsc88/${USER}/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
do
  if [[ -e "$p" ]]; then echo "OK  $p"; else echo "MISS $p"; fi
done
```

---

## Related docs

- [Salamandra-7B CISPO training guide](salamandra-7b-cispo.md)
- [slime quick start — model and dataset download](../get_started/quick_start.md#model-and-dataset-download)
- MN5 runtime layout: `~/.codex/skills/mn5-singularity-runtime/references/runtime_layout.md`
