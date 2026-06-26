"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
from collections import deque
from dataclasses import dataclass, field
import logging
import queue
import threading
import time
import uuid

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.rollout.sglang_rollout import GenerateState, generate_sample_only
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.trace_utils import trace_span
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            default_concurrency = args.sglang_server_concurrency * get_rollout_num_engines(args)
            generation_concurrency = getattr(args, "fully_async_generation_concurrency", None) or default_concurrency
            reward_concurrency = getattr(args, "fully_async_reward_concurrency", None) or generation_concurrency
            max_reward_backlog_groups = (
                getattr(args, "fully_async_max_reward_backlog_groups", None) or 4 * args.rollout_batch_size
            )
            reward_frontier_groups = (
                getattr(args, "fully_async_reward_frontier_groups", None)
                or max_reward_backlog_groups
            )
            if generation_concurrency < 1 or reward_concurrency < 1 or max_reward_backlog_groups < 1:
                raise ValueError(
                    "fully async rollout requires positive generation/reward concurrency and reward backlog limits; "
                    f"got generation_concurrency={generation_concurrency}, "
                    f"reward_concurrency={reward_concurrency}, "
                    f"max_reward_backlog_groups={max_reward_backlog_groups}"
                )
            if reward_frontier_groups < 1:
                raise ValueError(
                    "fully async rollout requires positive reward frontier groups; "
                    f"got reward_frontier_groups={reward_frontier_groups}"
                )
            _global_worker = AsyncRolloutWorker(
                args,
                data_buffer,
                generation_concurrency=generation_concurrency,
                reward_concurrency=reward_concurrency,
                max_reward_backlog_groups=max_reward_backlog_groups,
                reward_frontier_groups=reward_frontier_groups,
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop with separate generation and RM pools."""

    def __init__(
        self,
        args,
        data_buffer,
        generation_concurrency: int,
        reward_concurrency: int,
        max_reward_backlog_groups: int,
        reward_frontier_groups: int | None = None,
    ):
        self.args = args
        self.data_buffer = data_buffer
        self.generation_concurrency = generation_concurrency
        self.reward_concurrency = reward_concurrency
        self.max_reward_backlog_groups = max_reward_backlog_groups
        self.reward_frontier_groups = reward_frontier_groups or max_reward_backlog_groups
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self.metrics = FullyAsyncMetrics()

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def collect_metrics(self) -> dict[str, float | int]:
        return self.metrics.snapshot()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        gid_counter = 0
        active_generation: dict[asyncio.Task, tuple[int, int]] = {}
        active_reward: dict[asyncio.Task, tuple[int, int | None]] = {}
        pending_generation: deque[tuple[int, int, Sample]] = deque()
        pending_reward: deque[tuple[int, int | None, Sample | list[Sample]]] = deque()
        groups: dict[int, GroupState] = {}
        warned_high_backlog = False
        last_reward_queue_log = 0.0

        while self.running:
            try:
                for task in [task for task in active_generation if task.done()]:
                    gid, sample_idx = active_generation.pop(task)
                    await self._handle_generation_done(task, gid, sample_idx, groups, pending_reward)

                for task in [task for task in active_reward if task.done()]:
                    gid, sample_idx = active_reward.pop(task)
                    self._handle_reward_done(task, gid, sample_idx, groups)

                self._publish_ready_groups(groups)
                self._refresh_backlog_metrics(groups)
                warned_high_backlog = self._maybe_log_backlog_warning(groups, warned_high_backlog)
                now = time.time()
                if now - last_reward_queue_log >= 30.0:
                    logger.info(
                        "fully-async reward queues reason=periodic %s",
                        self._reward_queue_summary(
                            groups=groups,
                            pending_reward=pending_reward,
                            active_reward=active_reward,
                            active_generation=active_generation,
                            pending_generation=pending_generation,
                        ),
                    )
                    last_reward_queue_log = now
                self._drop_over_backlog_groups(
                    groups,
                    pending_reward=pending_reward,
                    active_reward=active_reward,
                    active_generation=active_generation,
                    pending_generation=pending_generation,
                )

                while len(active_reward) < self.reward_concurrency and pending_reward:
                    item = self._pop_next_pending_reward(pending_reward, groups)
                    if item is None:
                        break
                    gid, sample_idx, sample_or_group = item
                    group_state = groups[gid]
                    group_state.active_reward_count += len(sample_or_group) if isinstance(sample_or_group, list) else 1
                    task = asyncio.create_task(self._reward_sample_or_group(sample_or_group))
                    active_reward[task] = (gid, sample_idx)

                while len(active_generation) < self.generation_concurrency and self.running:
                    while not pending_generation:
                        fetched = self.data_buffer.get_samples(1)
                        if not fetched:
                            break
                        group = fetched[0]
                        gid = gid_counter
                        gid_counter += 1
                        groups[gid] = GroupState(gid=gid, samples=group)
                        for sample_idx, sample in enumerate(group):
                            if sample.session_id is None:
                                sample.session_id = str(uuid.uuid4())
                            pending_generation.append((gid, sample_idx, sample))
                    if not pending_generation:
                        break

                    gid, sample_idx, sample = pending_generation.popleft()
                    group_state = groups.get(gid)
                    if group_state is None or group_state.dropped:
                        continue
                    task = asyncio.create_task(self._generate_sample(sample, sample_idx))
                    active_generation[task] = (gid, sample_idx)

                await asyncio.sleep(0.01)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        active_tasks = set(active_generation) | set(active_reward)
        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    async def _generate_sample(self, sample: Sample, sample_idx: int) -> Sample | list[Sample]:
        if sample.status in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
            return sample
        sampling_params = self.state.sampling_params.copy()
        if getattr(self.args, "sglang_enable_deterministic_inference", False):
            sampling_params["sampling_seed"] = self.state.group_sampling_seeds[sample_idx]
        return await generate_sample_only(self.args, sample, sampling_params=sampling_params, evaluation=False)

    async def _handle_generation_done(
        self,
        task: asyncio.Task,
        gid: int,
        sample_idx: int,
        groups: dict[int, "GroupState"],
        pending_reward: deque[tuple[int, int | None, Sample | list[Sample]]],
    ) -> None:
        group = groups.get(gid)
        if group is None or group.dropped:
            return
        try:
            generated = task.result()
        except Exception:  # noqa: BLE001
            logger.exception("fully-async: generation task failed for gid=%s sample_idx=%s", gid, sample_idx)
            group.dropped = True
            groups.pop(gid, None)
            self.metrics.generation_failed_groups += 1
            return

        group.generated_units[sample_idx] = generated
        group.generated_count += 1
        if _contains_aborted(generated):
            group.dropped = True
            try:
                self.data_buffer.add_samples([group.samples])
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: failed to requeue aborted group")
            groups.pop(gid, None)
            return

        if self.args.group_rm:
            if group.generated_count == len(group.samples):
                pending_reward.append((gid, None, list(group.generated_units)))
                group.reward_scheduled_count = len(group.samples)
        else:
            pending_reward.append((gid, sample_idx, generated))
            group.reward_scheduled_count += 1

    def _handle_reward_done(
        self,
        task: asyncio.Task,
        gid: int,
        sample_idx: int | None,
        groups: dict[int, "GroupState"],
    ) -> None:
        group = groups.get(gid)
        if group is None or group.dropped:
            self.metrics.late_reward_results += 1
            logger.warning("fully-async: ignoring late reward result for dropped gid=%s sample_idx=%s", gid, sample_idx)
            return
        group.active_reward_count = max(0, group.active_reward_count - (len(group.samples) if sample_idx is None else 1))
        try:
            rewarded = task.result()
        except Exception:  # noqa: BLE001
            logger.exception("fully-async: reward task failed for gid=%s sample_idx=%s", gid, sample_idx)
            group.dropped = True
            groups.pop(gid, None)
            self.metrics.reward_failed_groups += 1
            return

        if sample_idx is None:
            group.generated_units = list(rewarded)
            group.rewarded_count = len(group.samples)
        else:
            group.generated_units[sample_idx] = rewarded
            group.rewarded_count += 1

    def _pop_next_pending_reward(
        self,
        pending_reward: deque[tuple[int, int | None, Sample | list[Sample]]],
        groups: dict[int, "GroupState"],
    ) -> tuple[int, int | None, Sample | list[Sample]] | None:
        if not pending_reward:
            return None

        frontier = {
            group.gid
            for group in sorted(
                (group for group in groups.values() if group.is_waiting_for_reward),
                key=lambda group: (group.created_at, group.gid),
            )[: self.reward_frontier_groups]
        }
        best_index: int | None = None
        best_key: tuple[int, int, float, int, int] | None = None
        stale_indices: list[int] = []

        for index, (gid, sample_idx, _sample_or_group) in enumerate(pending_reward):
            group = groups.get(gid)
            if group is None or group.dropped:
                stale_indices.append(index)
                continue
            in_frontier = 0 if gid in frontier else 1
            remaining = len(group.samples) - group.rewarded_count - group.active_reward_count
            sample_order = sample_idx if sample_idx is not None else -1
            key = (in_frontier, max(remaining, 0), group.created_at, group.gid, sample_order)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index

        for index in reversed(stale_indices):
            del pending_reward[index]

        if best_index is None:
            return None
        stale_before_best = sum(1 for index in stale_indices if index < best_index)
        adjusted_index = best_index - stale_before_best
        item = pending_reward[adjusted_index]
        del pending_reward[adjusted_index]
        return item

    async def _reward_sample_or_group(self, sample_or_group: Sample | list[Sample]) -> Sample | list[Sample]:
        if isinstance(sample_or_group, list):
            samples_need_reward = [sample for sample in sample_or_group if sample.reward is None]
            if samples_need_reward:
                with trace_span(samples_need_reward, "reward_model"):
                    rewards = await batched_async_rm(self.args, samples_need_reward)
                for sample, reward in zip(samples_need_reward, rewards, strict=False):
                    sample.reward = reward
            return sample_or_group

        sample = sample_or_group
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(self.args, sample)
        return sample

    def _publish_ready_groups(self, groups: dict[int, "GroupState"]) -> None:
        ready_gids = [
            gid
            for gid, group in groups.items()
            if not group.dropped and group.rewarded_count == len(group.samples)
        ]
        for gid in ready_gids:
            group = groups.pop(gid)
            result = list(group.generated_units)
            if any(item is None for item in result):
                logger.error("fully-async: refusing to publish incomplete group gid=%s", gid)
                continue
            self.output_queue.put((gid, result))

    def _refresh_backlog_metrics(self, groups: dict[int, "GroupState"]) -> None:
        backlogged = [group for group in groups.values() if group.is_waiting_for_reward]
        self.metrics.reward_backlog_groups = len(backlogged)
        self.metrics.reward_backlog_samples = sum(group.generated_count - group.rewarded_count for group in backlogged)
        self.metrics.reward_oldest_pending_seconds = max(
            (time.time() - group.created_at for group in backlogged),
            default=0.0,
        )

    def _reward_queue_summary(
        self,
        *,
        groups: dict[int, "GroupState"],
        pending_reward: deque[tuple[int, int | None, Sample | list[Sample]]],
        active_reward: dict[asyncio.Task, tuple[int, int | None]],
        active_generation: dict[asyncio.Task, tuple[int, int]],
        pending_generation: deque[tuple[int, int, Sample]],
    ) -> str:
        waiting = [group for group in groups.values() if group.is_waiting_for_reward]
        pending_reward_samples = sum(
            len(sample_or_group) if isinstance(sample_or_group, list) else 1
            for _gid, _sample_idx, sample_or_group in pending_reward
        )
        active_reward_groups = len({gid for _task, (gid, _sample_idx) in active_reward.items()})
        active_reward_samples = sum(
            len(groups[gid].samples) if sample_idx is None and gid in groups else 1
            for _task, (gid, sample_idx) in active_reward.items()
        )
        unscheduled_samples = sum(
            max(group.generated_count - group.rewarded_count - group.active_reward_count, 0)
            for group in waiting
        )
        active_or_scheduled_samples = sum(group.active_reward_count for group in waiting)
        response_lengths = [
            length
            for group in waiting
            for length in self._generated_response_lengths(group)
        ]
        response_token_summary = self._response_token_summary(response_lengths)

        def group_digest(group: GroupState) -> tuple[int, int, int, int, int]:
            return (
                group.gid,
                group.generated_count,
                group.reward_scheduled_count,
                group.active_reward_count,
                group.rewarded_count,
            )

        oldest = [group_digest(group) for group in sorted(waiting, key=lambda g: (g.created_at, g.gid))[:3]]
        newest = [
            group_digest(group)
            for group in sorted(waiting, key=lambda g: (g.created_at, g.gid), reverse=True)[:3]
        ]
        return (
            f"groups_total={len(groups)} waiting_groups={len(waiting)} "
            f"waiting_samples={self.metrics.reward_backlog_samples} "
            f"pending_reward_items={len(pending_reward)} pending_reward_samples={pending_reward_samples} "
            f"active_reward_tasks={len(active_reward)} active_reward_groups={active_reward_groups} "
            f"active_reward_samples={active_reward_samples} unscheduled_samples={unscheduled_samples} "
            f"active_or_scheduled_samples={active_or_scheduled_samples} "
            f"response_tokens={response_token_summary} "
            f"output_queue_groups={self.output_queue.qsize()} active_generation={len(active_generation)} "
            f"pending_generation_samples={len(pending_generation)} limit_groups={self.max_reward_backlog_groups} "
            f"oldest(gid,gen,scheduled,active,rewarded)={oldest} "
            f"newest(gid,gen,scheduled,active,rewarded)={newest}"
        )

    @staticmethod
    def _generated_response_lengths(group: "GroupState") -> list[int]:
        lengths: list[int] = []
        for item in group.generated_units:
            samples = item if isinstance(item, list) else [item]
            for sample in samples:
                if not isinstance(sample, Sample):
                    continue
                response_length = getattr(sample, "response_length", None)
                if response_length is not None:
                    lengths.append(int(response_length))
        return lengths

    @staticmethod
    def _response_token_summary(lengths: list[int]) -> str:
        if not lengths:
            return "count=0,total=0,avg=0,p50=0,p90=0,max=0"

        ordered = sorted(lengths)

        def percentile(percent: float) -> int:
            index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percent)))
            return ordered[index]

        total = sum(ordered)
        avg = total / len(ordered)
        return (
            f"count={len(ordered)},total={total},avg={avg:.1f},"
            f"p50={percentile(0.50)},p90={percentile(0.90)},max={ordered[-1]}"
        )

    def _maybe_log_backlog_warning(self, groups: dict[int, "GroupState"], already_warned: bool) -> bool:
        backlog = self.metrics.reward_backlog_groups
        threshold = max(1, self.max_reward_backlog_groups // 2)
        if backlog < threshold:
            return False
        if already_warned:
            return True
        logger.warning(
            "fully-async reward backlog high: groups=%s samples=%s oldest_pending=%.1fs "
            "limit_groups=%s generation_concurrency=%s reward_concurrency=%s",
            backlog,
            self.metrics.reward_backlog_samples,
            self.metrics.reward_oldest_pending_seconds,
            self.max_reward_backlog_groups,
            self.generation_concurrency,
            self.reward_concurrency,
        )
        return True

    def _drop_over_backlog_groups(
        self,
        groups: dict[int, "GroupState"],
        *,
        pending_reward: deque[tuple[int, int | None, Sample | list[Sample]]] | None = None,
        active_reward: dict[asyncio.Task, tuple[int, int | None]] | None = None,
        active_generation: dict[asyncio.Task, tuple[int, int]] | None = None,
        pending_generation: deque[tuple[int, int, Sample]] | None = None,
    ) -> None:
        while self.metrics.reward_backlog_groups > self.max_reward_backlog_groups:
            candidates = [
                group
                for group in groups.values()
                if group.is_waiting_for_reward and group.active_reward_count == 0
            ]
            if not candidates:
                if pending_reward is not None and active_reward is not None:
                    logger.error(
                        "fully-async reward backlog over limit but all waiting groups have active rewards: %s",
                        self._reward_queue_summary(
                            groups=groups,
                            pending_reward=pending_reward,
                            active_reward=active_reward,
                            active_generation=active_generation or {},
                            pending_generation=pending_generation or deque(),
                        ),
                    )
                return
            victim = max(candidates, key=lambda group: (group.created_at, group.gid))
            victim.dropped = True
            groups.pop(victim.gid, None)
            self.metrics.reward_dropped_groups += 1
            sample_indices = [getattr(sample, "index", None) for sample in victim.samples]
            logger.error(
                "\n"
                "================ FULLY ASYNC REWARD BACKLOG DROP ================\n"
                "Dropping unrewarded rollout group because reward backlog exceeded the configured limit.\n"
                "gid=%s sample_indices=%s backlog_groups=%s backlog_samples=%s oldest_pending=%.1fs\n"
                "generated=%s rewarded=%s generation_concurrency=%s reward_concurrency=%s limit_groups=%s\n"
                "reward_queues=%s\n"
                "=================================================================\n",
                victim.gid,
                sample_indices,
                self.metrics.reward_backlog_groups,
                self.metrics.reward_backlog_samples,
                self.metrics.reward_oldest_pending_seconds,
                victim.generated_count,
                victim.rewarded_count,
                self.generation_concurrency,
                self.reward_concurrency,
                self.max_reward_backlog_groups,
                self._reward_queue_summary(
                    groups=groups,
                    pending_reward=pending_reward or deque(),
                    active_reward=active_reward or {},
                    active_generation=active_generation or {},
                    pending_generation=pending_generation or deque(),
                ),
            )
            self._refresh_backlog_metrics(groups)


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=out, metrics=worker.collect_metrics())


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))


@dataclass
class GroupState:
    gid: int
    samples: list[Sample]
    created_at: float = field(default_factory=time.time)
    generated_units: list[Sample | list[Sample] | None] = field(init=False)
    generated_count: int = 0
    reward_scheduled_count: int = 0
    active_reward_count: int = 0
    rewarded_count: int = 0
    dropped: bool = False

    def __post_init__(self) -> None:
        self.generated_units = [None] * len(self.samples)

    @property
    def is_waiting_for_reward(self) -> bool:
        return not self.dropped and self.generated_count > self.rewarded_count


@dataclass
class FullyAsyncMetrics:
    reward_backlog_groups: int = 0
    reward_backlog_samples: int = 0
    reward_dropped_groups: int = 0
    reward_oldest_pending_seconds: float = 0.0
    late_reward_results: int = 0
    generation_failed_groups: int = 0
    reward_failed_groups: int = 0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "fully_async/reward_backlog_groups": self.reward_backlog_groups,
            "fully_async/reward_backlog_samples": self.reward_backlog_samples,
            "fully_async/reward_dropped_groups": self.reward_dropped_groups,
            "fully_async/reward_oldest_pending_seconds": self.reward_oldest_pending_seconds,
            "fully_async/late_reward_results": self.late_reward_results,
            "fully_async/generation_failed_groups": self.generation_failed_groups,
            "fully_async/reward_failed_groups": self.reward_failed_groups,
        }


def _contains_aborted(sample_or_group: Sample | list[Sample]) -> bool:
    if isinstance(sample_or_group, list):
        return any(getattr(sample, "status", None) == Sample.Status.ABORTED for sample in sample_or_group)
    return getattr(sample_or_group, "status", None) == Sample.Status.ABORTED
