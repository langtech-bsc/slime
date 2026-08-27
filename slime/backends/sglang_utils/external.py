"""Helpers for pre-launched external SGLang engines."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ExternalEngineInfo:
    url: str
    host: str
    port: int
    worker_type: str
    num_gpus: int
    disaggregation_bootstrap_port: int | None = None
    server_info: dict = dataclasses.field(default_factory=dict)

    @property
    def is_pd_worker(self) -> bool:
        return self.worker_type in ("prefill", "decode")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def normalize_external_engine_addr(addr: str) -> str:
    """Normalize ``host:port`` or ``http://host:port`` to an HTTP base URL."""
    if "://" not in addr:
        addr = f"http://{addr}"
    addr = addr.rstrip("/")
    parsed = urlparse(addr)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Invalid external SGLang engine address {addr!r}. "
            "Use host:port or http://host:port (IPv6 must be bracketed)."
        )
    return addr


def external_engine_init_kwargs(info: ExternalEngineInfo) -> dict:
    init_kwargs = {
        "dist_init_addr": f"{info.host}:{info.port}",
        "nccl_port": None,
        "host": info.host,
        "port": info.port,
    }
    if info.worker_type == "prefill":
        init_kwargs["disaggregation_bootstrap_port"] = info.disaggregation_bootstrap_port
    return init_kwargs


def get_server_info(url: str, timeout: float = 30.0) -> dict:
    errors = []
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            response = requests.get(f"{url}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError(f"Failed to fetch SGLang server info from {url}: {'; '.join(errors)}")


def _infer_worker_type(server_info: dict) -> str:
    if server_info.get("encoder_only"):
        return "encoder"
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"


def discover_external_engines(addrs: list[str], timeout: float = 30.0) -> list[ExternalEngineInfo]:
    infos = []
    for addr in addrs:
        url = normalize_external_engine_addr(addr)
        parsed = urlparse(url)
        assert parsed.hostname is not None and parsed.port is not None
        server_info = get_server_info(url, timeout=timeout)

        pp_size = int(server_info.get("pp_size") or server_info.get("pipeline_parallel_size") or 1)
        tp_size = int(server_info.get("tp_size") or server_info.get("tensor_parallel_size") or 1)
        num_gpus = int(server_info.get("num_gpus") or server_info.get("num_gpus_per_engine") or tp_size * pp_size)
        bootstrap_port = server_info.get("disaggregation_bootstrap_port")
        bootstrap_port = int(bootstrap_port) if bootstrap_port is not None else None

        infos.append(
            ExternalEngineInfo(
                url=url,
                host=parsed.hostname,
                port=parsed.port,
                worker_type=_infer_worker_type(server_info),
                num_gpus=num_gpus,
                disaggregation_bootstrap_port=bootstrap_port,
                server_info=server_info,
            )
        )
    return infos


def _external_engine_infos_to_args(args, infos: list[ExternalEngineInfo]) -> None:
    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.num_gpus for info in infos)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def discover_external_engines_with_retry(
    addrs: list[str],
    *,
    logger: logging.Logger | None = None,
    request_timeout: float | None = None,
    overall_timeout: float | None = None,
    retry_interval: float | None = None,
) -> list[ExternalEngineInfo]:
    """Discover pre-launched engines while they finish asynchronous startup.

    The launcher starts the trainer and external engines concurrently.  During
    that window ``/server_info`` can legitimately refuse connections, so a
    single discovery attempt is not an actionable configuration failure.
    """
    request_timeout = request_timeout or _env_float(
        "EXTERNAL_ENGINE_DISCOVERY_REQUEST_TIMEOUT_S", 30.0
    )
    overall_timeout = overall_timeout or _env_float(
        "EXTERNAL_ENGINE_DISCOVERY_TIMEOUT_S", 1800.0
    )
    retry_interval = retry_interval or _env_float(
        "EXTERNAL_ENGINE_DISCOVERY_RETRY_INTERVAL_S", 5.0
    )
    deadline = time.monotonic() + overall_timeout
    attempt = 0
    while True:
        attempt += 1
        try:
            infos = discover_external_engines(addrs, timeout=request_timeout)
            if not infos:
                raise RuntimeError("discovery returned no external engines")
            if logger is not None and attempt > 1:
                logger.info(
                    "External SGLang discovery succeeded on attempt %d after %.1fs",
                    attempt,
                    overall_timeout - max(0.0, deadline - time.monotonic()),
                )
            return infos
        except Exception as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "External SGLang engines did not become discoverable within "
                    f"{overall_timeout:g}s; last error: {exc}"
                ) from exc
            if logger is not None:
                logger.warning(
                    "External SGLang discovery attempt %d failed (%.1fs remain): %s; "
                    "retrying in %.1fs",
                    attempt,
                    remaining,
                    exc,
                    min(retry_interval, remaining),
                )
            time.sleep(min(retry_interval, remaining))


def apply_external_engine_info_to_args(args, logger=None) -> None:
    """Detect external engines and store the derived topology on ``args``."""
    addrs = args.rollout_external_engine_addrs
    if not addrs:
        raise ValueError("apply_external_engine_info_to_args requires --rollout-external-engine-addrs.")

    if os.environ.get("SLIME_DEFER_EXTERNAL_ENGINE_DISCOVERY") == "1":
        # The MN5 launcher knows the external topology before the servers are
        # healthy.  Preserve the known topology rather than requiring the
        # SGLang metadata endpoint, which can remain blocked while the server's
        # regular health endpoint is already responding.
        count = int(os.environ.get("ROLLOUT_EXTERNAL_ENGINE_COUNT", "0"))
        gpus_per_engine = int(
            os.environ.get("ROLLOUT_EXTERNAL_ENGINE_GPUS_PER_ENGINE", "0")
        )
        if count <= 0 or gpus_per_engine <= 0:
            raise ValueError(
                "deferred external discovery requires positive "
                "ROLLOUT_EXTERNAL_ENGINE_COUNT and "
                "ROLLOUT_EXTERNAL_ENGINE_GPUS_PER_ENGINE"
            )
        if len(addrs) != count:
            raise ValueError(
                "deferred external discovery address count does not match "
                f"ROLLOUT_EXTERNAL_ENGINE_COUNT: {len(addrs)} != {count}"
            )
        infos = []
        for addr in addrs:
            url = normalize_external_engine_addr(addr)
            parsed = urlparse(url)
            assert parsed.hostname is not None and parsed.port is not None
            infos.append(
                ExternalEngineInfo(
                    url=url,
                    host=parsed.hostname,
                    port=parsed.port,
                    worker_type="regular",
                    num_gpus=gpus_per_engine,
                    server_info={
                        "tp_size": gpus_per_engine,
                        "pp_size": 1,
                        "num_gpus": gpus_per_engine,
                        "disaggregation_mode": "null",
                    },
                )
            )
        _external_engine_infos_to_args(args, infos)
        if logger is not None:
            logger.info(
                "Using launcher-provided external SGLang topology; "
                "metadata discovery deferred/skipped (%d engines, %d GPUs)",
                count,
                args.rollout_num_gpus,
            )
        return

    infos = discover_external_engines_with_retry(addrs, logger=logger)
    _external_engine_infos_to_args(args, infos)

    if logger is not None:
        summary = [
            {
                "url": info.url,
                "worker_type": info.worker_type,
                "num_gpus": info.num_gpus,
                "disaggregation_bootstrap_port": info.disaggregation_bootstrap_port,
            }
            for info in infos
        ]
        logger.info(f"Detected external SGLang engines: {summary}")


@dataclasses.dataclass
class ExternalRolloutServer:
    """Rollout server backed by pre-launched external SGLang engines."""

    engines: list
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    num_new_engines: int = 0
    server_groups: list = dataclasses.field(default_factory=list)

    @property
    def all_engines(self):
        return self.engines

    def recover(self):
        logger.warning("Fault tolerance is not supported for external rollout engines; skip recover.")

    def offload(self):
        return []

    def onload(self, tags: list[str] | None = None):
        return []

    def onload_weights(self):
        return []

    def onload_kv(self):
        return []


def external_engine_infos_from_args(args) -> list[ExternalEngineInfo]:
    raw_infos = getattr(args, "rollout_external_engine_infos", None)
    if raw_infos is None:
        raise RuntimeError(
            "External rollout engine info is missing. "
            "apply_external_engine_info_to_args must run before starting external rollout servers."
        )
    return [ExternalEngineInfo(**info) if isinstance(info, dict) else info for info in raw_infos]


def start_external_rollout_servers(args, *, start_router) -> tuple[dict[str, ExternalRolloutServer], list]:
    import ray

    from slime.backends.sglang_utils.sglang_engine import SGLangEngine
    from slime.ray.utils import add_default_ray_env_vars

    if getattr(args, "rollout_external_engine_infos", None) is None:
        infos = discover_external_engines_with_retry(
            args.rollout_external_engine_addrs,
            logger=logger,
        )
        _external_engine_infos_to_args(args, infos)
    else:
        infos = external_engine_infos_from_args(args)
    router_ip, router_port = start_router(args, has_pd_disaggregation=any(info.is_pd_worker for info in infos))
    args.sglang_router_ip = router_ip
    args.sglang_router_port = router_port

    engines = []
    engine_gpu_counts = []
    engine_gpu_offsets = []
    init_handles = []
    RolloutRayActor = ray.remote(SGLangEngine)
    gpu_offset = 0
    for rank, info in enumerate(infos):
        rollout_engine = RolloutRayActor.options(
            num_cpus=0.2,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote(
            args=args,
            rank=rank,
            worker_type=info.worker_type,
            base_gpu_id=0,
            num_gpus_per_engine=info.num_gpus,
        )
        engines.append(rollout_engine)
        engine_gpu_counts.append(info.num_gpus)
        engine_gpu_offsets.append(gpu_offset)
        gpu_offset += info.num_gpus
        init_handles.append(
            rollout_engine.init.remote(
                **external_engine_init_kwargs(info),
                router_ip=router_ip,
                router_port=router_port,
            )
        )

    args.sglang_model_routers = {"default": (router_ip, router_port)}
    servers = {
        "default": ExternalRolloutServer(
            engines=engines,
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
            router_ip=router_ip,
            router_port=router_port,
            model_name="default",
            update_weights=True,
            num_new_engines=len(engines),
        )
    }
    return servers, init_handles
