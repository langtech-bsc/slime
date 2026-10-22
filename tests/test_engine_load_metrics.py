from __future__ import annotations

import pytest

from slime.rollout.engine_load_metrics import build_gpu_load_metrics


@pytest.mark.unit
def test_build_gpu_load_metrics_expands_engines_to_gpus():
    metrics = build_gpu_load_metrics([(5, 1), (7, 0)], gpus_per_engine=2)
    assert metrics["rollout_gpu/0/running_reqs"] == 5
    assert metrics["rollout_gpu/1/running_reqs"] == 5
    assert metrics["rollout_gpu/2/running_reqs"] == 7
    assert metrics["rollout_gpu/3/waiting_reqs"] == 0
    assert metrics["rollout_gpu/total_running_reqs"] == 12
    assert metrics["rollout_gpu/total_waiting_reqs"] == 1
