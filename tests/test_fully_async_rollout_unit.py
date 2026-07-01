from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from slime.rollout.fully_async_rollout import AsyncRolloutWorker, GroupState
from slime.utils.types import Sample


NUM_GPUS = 0


class _FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.sampling_params = {}
        self.group_sampling_seeds = [args.rollout_seed + i for i in range(args.n_samples_per_prompt)]
        self.aborted = False

    @contextmanager
    def dp_rank_context(self):
        yield 0


class _FiniteDataSource:
    def __init__(self, *, groups: int, group_size: int):
        self._groups = []
        index = 0
        for _ in range(groups):
            group = []
            for _ in range(group_size):
                group.append(Sample(index=index, prompt=f"prompt-{index}"))
                index += 1
            self._groups.append(group)
        self.requeued = []

    def get_samples(self, num_samples: int):
        selected = self._groups[:num_samples]
        self._groups = self._groups[num_samples:]
        return selected

    def add_samples(self, samples):
        self.requeued.extend(samples)


class _StaticDataSource:
    def __init__(self, groups):
        self._groups = list(groups)

    def get_samples(self, num_samples: int):
        selected = self._groups[:num_samples]
        self._groups = self._groups[num_samples:]
        return selected

    def add_samples(self, samples):
        self._groups.extend(samples)


def _args(**overrides):
    values = dict(
        rollout_seed=7,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        sglang_enable_deterministic_inference=False,
        group_rm=False,
        custom_rm_path=None,
        rm_type="random",
        reward_key=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


async def _wait_for(predicate, *, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


@pytest.mark.unit
def test_fully_async_generation_refills_while_rewards_are_blocked(monkeypatch):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    generated = []
    reward_gate = asyncio.Event()

    async def fake_generate(args, sample, sampling_params, evaluation=False):
        generated.append(sample.index)
        sample.response = f"response-{sample.index}"
        sample.response_length = 1
        sample.tokens = [sample.index]
        sample.status = Sample.Status.COMPLETED
        return sample

    async def fake_rm(args, sample):
        await reward_gate.wait()
        return 1.0

    monkeypatch.setattr(fully_async_rollout, "generate_sample_only", fake_generate)
    monkeypatch.setattr(fully_async_rollout, "async_rm", fake_rm)

    async def run_case():
        worker = AsyncRolloutWorker(
            _args(),
            _FiniteDataSource(groups=4, group_size=2),
            generation_concurrency=3,
            reward_concurrency=1,
            max_reward_backlog_groups=99,
            max_inference_groups=99,
            max_reward_groups=99,
        )
        task = asyncio.create_task(worker._loop())
        try:
            await _wait_for(lambda: len(generated) >= 6)
            assert worker.queue_size() == 0
            reward_gate.set()
            await _wait_for(lambda: worker.queue_size() >= 2)
        finally:
            worker.running = False
            await task

    asyncio.run(run_case())


@pytest.mark.unit
def test_fully_async_rewards_already_completed_unrewarded_sample(monkeypatch):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    sample = Sample(index=1, prompt="prompt-1")
    sample.response = "done"
    sample.response_length = 1
    sample.tokens = [1]
    sample.status = Sample.Status.COMPLETED
    sample.reward = None

    async def fake_generate(*args, **kwargs):
        raise AssertionError("completed samples should not be generated again")

    async def fake_rm(args, sample):
        return 1.0

    monkeypatch.setattr(fully_async_rollout, "generate_sample_only", fake_generate)
    monkeypatch.setattr(fully_async_rollout, "async_rm", fake_rm)

    async def run_case():
        worker = AsyncRolloutWorker(
            _args(n_samples_per_prompt=1, rollout_batch_size=1),
            _StaticDataSource([[sample]]),
            generation_concurrency=1,
            reward_concurrency=1,
            max_reward_backlog_groups=8,
            max_inference_groups=99,
            max_reward_groups=99,
        )
        task = asyncio.create_task(worker._loop())
        try:
            await _wait_for(lambda: worker.queue_size() >= 1)
        finally:
            worker.running = False
            await task
        assert sample.reward == 1.0

    asyncio.run(run_case())


@pytest.mark.unit
def test_fully_async_requeued_aborted_group_resets_generation_state(monkeypatch):
    from collections import deque

    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    data_source = _FiniteDataSource(groups=0, group_size=1)
    worker = AsyncRolloutWorker(
        _args(rollout_max_response_len=10, rollout_max_context_len=16),
        data_source,
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=8,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    sample = Sample(index=1, prompt="prompt")
    sample.tokens = [1, 2, 3, 4]
    sample.response = "partial"
    sample.response_length = 2
    sample.loss_mask = [1, 1]
    sample.rollout_log_probs = [-1.0, -1.0]
    sample.weight_versions = ["13"]
    sample.status = Sample.Status.ABORTED
    group = GroupState(gid=1, samples=[sample])
    groups = {1: group}

    async def done():
        return sample

    async def run_case():
        task = asyncio.create_task(done())
        await task
        await worker._handle_generation_done(task, 1, 0, groups, deque())

    asyncio.run(run_case())

    assert groups == {}
    assert data_source.requeued == [[sample]]
    assert sample.tokens == [1, 2]
    assert sample.response_length == 0
    assert sample.loss_mask is None
    assert sample.rollout_log_probs is None
    assert sample.weight_versions == []
    assert sample.status == Sample.Status.PENDING


@pytest.mark.unit
def test_fully_async_drops_overlength_generated_sample_before_reward(monkeypatch):
    from collections import deque

    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(rollout_max_response_len=10, rollout_max_context_len=16),
        _FiniteDataSource(groups=0, group_size=1),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=8,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    sample = Sample(index=1, prompt="prompt")
    sample.tokens = list(range(15))
    sample.response_length = 11
    sample.status = Sample.Status.COMPLETED
    group = GroupState(gid=1, samples=[sample])
    groups = {1: group}
    pending_reward = deque()

    async def done():
        return sample

    async def run_case():
        task = asyncio.create_task(done())
        await task
        await worker._handle_generation_done(task, 1, 0, groups, pending_reward)

    asyncio.run(run_case())

    assert groups == {}
    assert not pending_reward
    assert worker.metrics.generation_invalid_groups == 1


@pytest.mark.unit
def test_fully_async_generation_concurrency_is_counted_per_sample(monkeypatch):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    active = 0
    max_active = 0

    async def fake_generate(args, sample, sampling_params, evaluation=False):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.03)
        sample.response = "done"
        sample.response_length = 1
        sample.tokens = [sample.index]
        sample.status = Sample.Status.COMPLETED
        active -= 1
        return sample

    async def fake_rm(args, sample):
        return 1.0

    monkeypatch.setattr(fully_async_rollout, "generate_sample_only", fake_generate)
    monkeypatch.setattr(fully_async_rollout, "async_rm", fake_rm)

    async def run_case():
        worker = AsyncRolloutWorker(
            _args(rollout_batch_size=1),
            _FiniteDataSource(groups=2, group_size=4),
            generation_concurrency=3,
            reward_concurrency=4,
            max_reward_backlog_groups=99,
            max_inference_groups=99,
            max_reward_groups=99,
        )
        task = asyncio.create_task(worker._loop())
        try:
            await _wait_for(lambda: worker.queue_size() >= 1)
        finally:
            worker.running = False
            await task
        assert max_active == 3

    asyncio.run(run_case())


@pytest.mark.unit
def test_fully_async_drops_newest_inactive_group_and_logs_big_warning(monkeypatch, caplog):
    from slime.rollout import fully_async_rollout
    from slime.rollout.fully_async_rollout import FullyAsyncMetrics

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(),
        _FiniteDataSource(groups=0, group_size=2),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=1,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    now = 1000.0
    groups = {
        1: GroupState(gid=1, samples=[Sample(index=1), Sample(index=2)]),
        2: GroupState(gid=2, samples=[Sample(index=3), Sample(index=4)]),
    }
    for offset, group in enumerate(groups.values()):
        group.created_at = now + offset
        group.generated_count = 1
        group.rewarded_count = 0

    worker.metrics = FullyAsyncMetrics(reward_backlog_groups=2, reward_backlog_samples=2)

    with caplog.at_level("ERROR", logger="slime.rollout.fully_async"):
        worker._drop_over_backlog_groups(groups)

    assert 1 in groups
    assert 2 not in groups
    assert worker.metrics.reward_dropped_groups == 1
    assert "FULLY ASYNC REWARD BACKLOG DROP" in caplog.text
    assert "gid=2" in caplog.text


@pytest.mark.unit
def test_fully_async_does_not_drop_group_with_active_reward(monkeypatch):
    from slime.rollout import fully_async_rollout
    from slime.rollout.fully_async_rollout import FullyAsyncMetrics

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(),
        _FiniteDataSource(groups=0, group_size=2),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=1,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    groups = {
        1: GroupState(gid=1, samples=[Sample(index=1), Sample(index=2)]),
        2: GroupState(gid=2, samples=[Sample(index=3), Sample(index=4)]),
    }
    groups[1].created_at = 1000.0
    groups[1].generated_count = 1
    groups[1].active_reward_count = 1
    groups[2].created_at = 1001.0
    groups[2].generated_count = 1
    worker.metrics = FullyAsyncMetrics(reward_backlog_groups=2, reward_backlog_samples=2)

    worker._drop_over_backlog_groups(groups)

    assert 1 in groups
    assert 2 not in groups


@pytest.mark.unit
def test_fully_async_reward_scheduler_prioritizes_earliest_near_completion(monkeypatch):
    from collections import deque

    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(),
        _FiniteDataSource(groups=0, group_size=4),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=8,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    groups = {
        1: GroupState(gid=1, samples=[Sample(index=10), Sample(index=11), Sample(index=12), Sample(index=13)]),
        2: GroupState(gid=2, samples=[Sample(index=20), Sample(index=21), Sample(index=22), Sample(index=23)]),
        3: GroupState(gid=3, samples=[Sample(index=30), Sample(index=31), Sample(index=32), Sample(index=33)]),
    }
    for gid, group in groups.items():
        group.created_at = 1000.0 + gid
        group.generated_count = 4
    groups[1].rewarded_count = 1
    groups[2].rewarded_count = 3
    groups[3].rewarded_count = 3

    pending = deque(
        [
            (3, 0, groups[3].samples[0]),
            (1, 1, groups[1].samples[1]),
            (2, 1, groups[2].samples[1]),
        ]
    )

    item = worker._pop_next_pending_reward(pending, groups)

    assert item is not None
    assert item[0] == 2
    assert [entry[0] for entry in pending] == [3, 1]


@pytest.mark.unit
def test_fully_async_reward_queue_summary_includes_response_tokens(monkeypatch):
    from collections import deque

    from slime.rollout import fully_async_rollout
    from slime.rollout.fully_async_rollout import FullyAsyncMetrics

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(),
        _FiniteDataSource(groups=0, group_size=3),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=8,
        max_inference_groups=99,
        max_reward_groups=99,
    )
    samples = [Sample(index=1), Sample(index=2), Sample(index=3)]
    for sample, response_length in zip(samples, [10, 20, 90]):
        sample.response_length = response_length
        sample.status = Sample.Status.COMPLETED

    group = GroupState(gid=1, samples=samples)
    group.generated_units = list(samples)
    group.generated_count = len(samples)
    groups = {1: group}
    worker.metrics = FullyAsyncMetrics(reward_backlog_groups=1, reward_backlog_samples=3)

    summary = worker._reward_queue_summary(
        groups=groups,
        pending_reward=deque((1, index, sample) for index, sample in enumerate(samples)),
        active_reward={},
        active_generation={},
        pending_generation=deque(),
    )

    assert "response_tokens=count=3,total=120,avg=40.0,p50=20,p90=90,max=90" in summary


@pytest.mark.unit
def test_fully_async_ignores_late_reward_for_dropped_group(monkeypatch, caplog):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    worker = AsyncRolloutWorker(
        _args(),
        _FiniteDataSource(groups=0, group_size=2),
        generation_concurrency=1,
        reward_concurrency=1,
        max_reward_backlog_groups=1,
        max_inference_groups=99,
        max_reward_groups=99,
    )

    async def done():
        return Sample(index=1, reward=1.0)

    async def run_case():
        task = asyncio.create_task(done())
        await task
        with caplog.at_level("WARNING", logger="slime.rollout.fully_async"):
            worker._handle_reward_done(task, gid=99, sample_idx=0, groups={})

    asyncio.run(run_case())

    assert worker.metrics.late_reward_results == 1
    assert "ignoring late reward result" in caplog.text


@pytest.mark.unit
def test_fully_async_inference_group_cap_blocks_new_fetch(monkeypatch):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    max_inference_groups_seen = 0

    async def fake_generate(args, sample, sampling_params, evaluation=False):
        await asyncio.sleep(0.05)
        sample.response = "done"
        sample.response_length = 1
        sample.tokens = [sample.index]
        sample.status = Sample.Status.COMPLETED
        return sample

    async def fake_rm(args, sample):
        return 1.0

    monkeypatch.setattr(fully_async_rollout, "generate_sample_only", fake_generate)
    monkeypatch.setattr(fully_async_rollout, "async_rm", fake_rm)

    async def run_case():
        nonlocal max_inference_groups_seen
        worker = AsyncRolloutWorker(
            _args(rollout_batch_size=1),
            _FiniteDataSource(groups=4, group_size=2),
            generation_concurrency=4,
            reward_concurrency=4,
            max_reward_backlog_groups=99,
            max_inference_groups=2,
            max_reward_groups=99,
        )
        task = asyncio.create_task(worker._loop())
        try:
            await _wait_for(lambda: worker.metrics.inference_active_groups > 0)
            deadline = asyncio.get_running_loop().time() + 1.0
            while asyncio.get_running_loop().time() < deadline:
                max_inference_groups_seen = max(
                    max_inference_groups_seen,
                    worker.metrics.inference_active_groups,
                )
                await asyncio.sleep(0.01)
        finally:
            worker.running = False
            await task

    asyncio.run(run_case())
    assert max_inference_groups_seen <= 2


@pytest.mark.unit
def test_fully_async_reward_group_cap_blocks_new_group(monkeypatch):
    from slime.rollout import fully_async_rollout

    monkeypatch.setattr(fully_async_rollout, "GenerateState", _FakeGenerateState)

    max_reward_groups_seen = 0

    async def fake_generate(args, sample, sampling_params, evaluation=False):
        sample.response = "done"
        sample.response_length = 1
        sample.tokens = [sample.index]
        sample.status = Sample.Status.COMPLETED
        return sample

    async def fake_rm(args, sample):
        await asyncio.sleep(0.05)
        return 1.0

    monkeypatch.setattr(fully_async_rollout, "generate_sample_only", fake_generate)
    monkeypatch.setattr(fully_async_rollout, "async_rm", fake_rm)

    async def run_case():
        nonlocal max_reward_groups_seen
        worker = AsyncRolloutWorker(
            _args(rollout_batch_size=1),
            _FiniteDataSource(groups=4, group_size=2),
            generation_concurrency=4,
            reward_concurrency=4,
            max_reward_backlog_groups=99,
            max_inference_groups=99,
            max_reward_groups=2,
        )
        task = asyncio.create_task(worker._loop())
        try:
            await _wait_for(lambda: worker.metrics.reward_active_groups > 0)
            deadline = asyncio.get_running_loop().time() + 1.0
            while asyncio.get_running_loop().time() < deadline:
                max_reward_groups_seen = max(
                    max_reward_groups_seen,
                    worker.metrics.reward_active_groups,
                )
                await asyncio.sleep(0.01)
        finally:
            worker.running = False
            await task

    asyncio.run(run_case())
    assert max_reward_groups_seen <= 2
