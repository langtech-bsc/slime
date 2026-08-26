from __future__ import annotations

from queue import Queue
from types import SimpleNamespace

import pytest

import train_async
from train_async import _prepare_train_data_with_recovery, _primary_trainer_perf, _publish_trainer_perf


def test_primary_trainer_perf_selects_primary_actor_result():
    results = [None, {"perf/step_time": 20.0, "perf/effective_global_batch_size": 64, "other": 1}]

    assert _primary_trainer_perf(results) == {
        "perf/step_time": 20.0,
        "perf/effective_global_batch_size": 64,
    }


def test_publish_trainer_perf_keeps_latest_measurement_when_queue_is_full():
    perf_queue = Queue(maxsize=1)
    perf_queue.put_nowait({"perf/step_time": 100.0, "perf/effective_global_batch_size": 1})

    _publish_trainer_perf(
        perf_queue,
        [{"perf/step_time": 10.0, "perf/effective_global_batch_size": 32}],
    )

    assert perf_queue.get_nowait() == {
        "perf/step_time": 10.0,
        "perf/effective_global_batch_size": 32,
    }


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self, prepare_results):
        self._prepare_results = iter(prepare_results)
        self.prepare_calls = []
        self.collect_calls = []
        self.prepare_train_data = _FakeRemoteMethod(self._prepare)
        self.collect_rollout_samples = _FakeRemoteMethod(self._collect)

    def _prepare(self, rollout_id, payload, *, trainer_weight_version):
        self.prepare_calls.append((rollout_id, payload, trainer_weight_version))
        result = next(self._prepare_results)
        if isinstance(result, Exception):
            return result
        return ("prepared", result)

    def _collect(self, rollout_id):
        self.collect_calls.append(rollout_id)
        return ("payload", rollout_id)


@pytest.fixture
def fake_ray_get(monkeypatch):
    def get(ref):
        if isinstance(ref, Exception):
            raise ref
        return ref

    monkeypatch.setattr(train_async.ray, "get", get)


def test_prepare_train_data_recovers_from_all_groups_dropped(fake_ray_get):
    manager = _FakeRolloutManager(
        [ValueError("No rollout groups survived pre-training filtering; metrics={}"), "replacement"]
    )
    args = SimpleNamespace(
        rollout_function_path=train_async._FULLY_ASYNC_ROLLOUT_PATH,
        num_rollout=1,
    )

    def request_rollout():
        return manager.collect_rollout_samples.remote(1)

    result, next_future = _prepare_train_data_with_recovery(
        args,
        manager,
        rollout_id=0,
        rollout_payload=("discarded-payload", 0),
        trainer_weight_version=7,
        rollout_data_next_future=None,
        request_rollout=request_rollout,
    )

    assert result == ("prepared", "replacement")
    assert next_future is None
    assert manager.collect_calls == [1]
    assert manager.prepare_calls == [
        (0, ("discarded-payload", 0), 7),
        (0, ("payload", 1), 7),
    ]


def test_prepare_train_data_normal_path_keeps_prefetched_batch(fake_ray_get):
    manager = _FakeRolloutManager(["current"])
    args = SimpleNamespace(
        rollout_function_path=train_async._FULLY_ASYNC_ROLLOUT_PATH,
        num_rollout=2,
    )
    next_future = ("prefetched-payload", 1)

    result, returned_next_future = _prepare_train_data_with_recovery(
        args,
        manager,
        rollout_id=0,
        rollout_payload=("current-payload", 0),
        trainer_weight_version=7,
        rollout_data_next_future=next_future,
        request_rollout=lambda: pytest.fail("normal path must not request a replacement"),
    )

    assert result == ("prepared", "current")
    assert returned_next_future is next_future
    assert manager.collect_calls == []
