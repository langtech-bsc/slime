"""Keep Ray runtime objects off Megatron common.pt args.

Offline W&B metrics use ``ray.util.queue.Queue``, which serializes as an
``ActorHandle``. Megatron pickles ``args`` into ``common.pt``. Do not wrap
``torch.load`` with a custom pickle module: PyTorch weight shards use
``weights_only`` and reject an explicit ``pickle_module``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

_RAY_RUNTIME_ARG_ATTRS = (
    "_wandb_metric_queue",
    "_fully_async_trainer_perf_queue",
)


def _is_ray_runtime_value(value: Any) -> bool:
    if value is None:
        return False
    module = getattr(type(value), "__module__", "") or ""
    return module.startswith("ray.")


def snapshot_ray_runtime_args(args: Any) -> dict[str, Any]:
    """Copy live Ray objects off ``args`` so checkpoint load cannot replace them."""

    snapshot: dict[str, Any] = {}
    for name in _RAY_RUNTIME_ARG_ATTRS:
        if hasattr(args, name):
            snapshot[name] = getattr(args, name)
    return snapshot


def restore_ray_runtime_args(args: Any, snapshot: Mapping[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(args, name, value)


def strip_ray_runtime_args(state_dict: Any) -> Any:
    """Return a state dict whose pickled ``args`` have no Ray actor handles."""

    if not isinstance(state_dict, dict) or "args" not in state_dict:
        return state_dict
    ckpt_args = state_dict["args"]
    if ckpt_args is None:
        return state_dict
    stripped = copy.copy(ckpt_args)
    for name, value in vars(stripped).items():
        if _is_ray_runtime_value(value):
            setattr(stripped, name, None)
    updated = dict(state_dict)
    updated["args"] = stripped
    return updated
