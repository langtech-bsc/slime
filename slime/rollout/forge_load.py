"""Load a forged rollout dump from disk so memory-test runs can keep
sglang alive while bypassing real generation.

Plug in by setting:
  --rollout-function-path slime.rollout.forge_load.generate_rollout
  --load-forge-rollout-data <path>

The path follows the same {rollout_id} format convention as
--load-debug-rollout-data:
  - Literal path (recommended for memory tests):
      --load-forge-rollout-data /path/to/forged_dump/rollout_data/rollout_0
    Every rollout reuses the same dump (rollout_id is left untouched so
    the framework's per-rollout bookkeeping still works). Legacy single
    `.pt` files are also supported.
  - Template path (matches rollout dump directory layout):
      --load-forge-rollout-data /path/to/dumps/rollout_{rollout_id}
    Each rollout loads its own directory (chunked parts + manifest.json).
    Legacy templates ending in `{rollout_id}.pt` are still accepted.

Unlike --load-debug-rollout-data, this path does NOT set
skip_sglang=True / debug_train_only=True (see
slime/utils/arguments.py: skip_sglang computation in _pre_parse_mode and
the debug_train_only flip when load_debug_rollout_data is set), so
sglang servers, router, weight_update and the full colocate
offload/onload dance still run. That is exactly what we want when
measuring real GPU memory.
"""

import logging
from pathlib import Path

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.rollout_dump_utils import load_rollout_dump, resolve_rollout_dump_load_path, rollout_dump_exists
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _template_variants(template: str) -> list[str]:
    if "{rollout_id}" not in template:
        return [template]
    variants = [template]
    if "{rollout_id}.pt" in template:
        variants.append(template.replace("{rollout_id}.pt", "rollout_{rollout_id}"))
    elif not template.endswith("rollout_{rollout_id}"):
        variants.append(template.rstrip("/") + "/rollout_{rollout_id}")
    return variants


def _resolve_path(args, rollout_id: int, evaluation: bool) -> Path | None:
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError(
            "--load-forge-rollout-data not set. Pass the dump path, "
            "e.g. /path/to/rollout_data/rollout_0 (literal) or "
            "/path/to/rollout_data/rollout_{rollout_id} (template)."
        )
    if evaluation and "{rollout_id}" not in tpl:
        return None

    for variant in _template_variants(tpl):
        path = resolve_rollout_dump_load_path(variant, rollout_id, evaluation=evaluation)
        if rollout_dump_exists(path):
            return path

    if not evaluation:
        for variant in _template_variants(tpl):
            path = resolve_rollout_dump_load_path(variant, 0, evaluation=False)
            if rollout_dump_exists(path):
                logger.info("forge_load: rollout_id=%s missing, falling back to %s", rollout_id, path)
                return path
    return None


def _load_samples(path: Path) -> list[Sample]:
    return [Sample.from_dict(sample) for sample in load_rollout_dump(path)]


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:
        if path is None:
            logger.info("forge_load: no eval dump found; returning empty eval result")
            return RolloutFnEvalOutput(data={})
        logger.info("forge_load: loading eval samples from %s", path)
        samples = _load_samples(path)
        reward_key = args.eval_reward_key or args.reward_key
        rewards = [s.reward if (not reward_key or s.reward is None) else s.reward[reward_key] for s in samples]
        return RolloutFnEvalOutput(
            data={
                "forge_eval": {
                    "rewards": [r if r is not None else 0.0 for r in rewards],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
                    "samples": samples,
                }
            }
        )

    if path is None:
        raise RuntimeError(
            f"forge_load: no dump found for rollout_id={rollout_id} "
            f"(--load-forge-rollout-data={args.load_forge_rollout_data!r})"
        )

    logger.info("forge_load: loading samples from %s", path)
    samples = _load_samples(path)
    logger.info(
        "forge_load: loaded %d samples for rollout_id=%d from %s",
        len(samples),
        rollout_id,
        path.name,
    )
    return RolloutFnTrainOutput(samples=samples)
