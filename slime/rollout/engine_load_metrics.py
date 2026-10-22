"""Per-GPU in-flight request counts for the rollout SGLang engines."""

from __future__ import annotations

import asyncio
import logging

from slime.backends.sglang_utils.server_control import num_requests_from_load
from slime.utils.http_utils import get

logger = logging.getLogger(__name__)

__all__ = [
    "build_gpu_load_metrics",
    "collect_engine_load_metrics",
]

POLL_TIMEOUT_SECONDS = 3.0


def _count(load, *keys: str) -> int:
    if isinstance(load, list):
        return sum(_count(item, *keys) for item in load)
    if not isinstance(load, dict):
        return 0
    if "loads" in load:
        return _count(load["loads"], *keys)
    for key in keys:
        if isinstance(load.get(key), int):
            return load[key]
    return 0


def build_gpu_load_metrics(
    engine_loads: list[tuple[int, int]],
    *,
    gpus_per_engine: int,
) -> dict[str, int]:
    """Expand per-engine (running, waiting) pairs into one entry per GPU.

    GPUs of a tensor-parallel engine all serve the same requests, so each GPU of an
    engine reports that engine's count.
    """
    metrics: dict[str, int] = {}
    total_running = 0
    total_waiting = 0
    for engine_idx, (running, waiting) in enumerate(engine_loads):
        total_running += running
        total_waiting += waiting
        for local_gpu in range(gpus_per_engine):
            gpu = engine_idx * gpus_per_engine + local_gpu
            metrics[f"rollout_gpu/{gpu}/running_reqs"] = running
            metrics[f"rollout_gpu/{gpu}/waiting_reqs"] = waiting
    metrics["rollout_gpu/total_running_reqs"] = total_running
    metrics["rollout_gpu/total_waiting_reqs"] = total_waiting
    return metrics


async def _engine_load(url: str) -> tuple[int, int]:
    load = await asyncio.wait_for(get(f"{url}/v1/loads?include=core"), POLL_TIMEOUT_SECONDS)
    running = _count(load, "num_running_reqs", "total_running_reqs")
    waiting = _count(load, "num_waiting_reqs", "total_waiting_reqs")
    if running == 0 and waiting == 0:
        # Payloads exposing only a combined count.
        running = num_requests_from_load(load)
    return running, waiting


async def collect_engine_load_metrics(args) -> dict[str, int]:
    """Best-effort snapshot; returns {} when the router or engines are unreachable."""
    gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None) or 1
    try:
        response = await asyncio.wait_for(
            get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"),
            POLL_TIMEOUT_SECONDS,
        )
        urls = sorted(worker["url"] for worker in response["workers"])
        loads = await asyncio.gather(*(_engine_load(url) for url in urls))
    except Exception as e:
        logger.warning("Failed to collect rollout engine load: %s", e)
        return {}
    return build_gpu_load_metrics(list(loads), gpus_per_engine=gpus_per_engine)
