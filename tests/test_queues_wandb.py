from __future__ import annotations

from types import SimpleNamespace

import pytest

from slime.utils.queues_wandb import queues_backlog_limit_samples


@pytest.mark.unit
def test_queues_backlog_limit_samples_uses_global_batch_and_staleness():
    args = SimpleNamespace(global_batch_size=512, max_rollout_weight_staleness=3)
    assert queues_backlog_limit_samples(args) == 1536

    args_no_staleness = SimpleNamespace(global_batch_size=512, max_rollout_weight_staleness=None)
    assert queues_backlog_limit_samples(args_no_staleness) is None
