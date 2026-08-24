from __future__ import annotations

from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from slime.utils import logging_utils


@pytest.mark.unit
def test_log_wandb_uses_step_key_for_wandb_step():
    args = SimpleNamespace(use_wandb=True, use_tensorboard=False)
    metrics = {"train/loss": 0.5, "train/step": 47}

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        logging_utils.log(args, metrics, step_key="train/step")

    mock_wandb_log.assert_called_once_with(metrics, step=47)


@pytest.mark.unit
def test_log_wandb_does_not_use_bare_step_kwarg():
    args = SimpleNamespace(use_wandb=True, use_tensorboard=False)
    metrics = {"queues/reward_samples": 10}

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        logging_utils.log(args, metrics, step=15)

    mock_wandb_log.assert_called_once_with(metrics)


@pytest.mark.unit
def test_log_offline_worker_queues_metrics_for_driver():
    args = SimpleNamespace(use_wandb=True, use_tensorboard=False, _wandb_metric_queue=Queue())
    metrics = {"rollout/loss": 0.5, "rollout/step": 12}

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        logging_utils.log(args, metrics, step_key="rollout/step")

    mock_wandb_log.assert_not_called()
    assert args._wandb_metric_queue.get_nowait() == (metrics, "rollout/step")


@pytest.mark.unit
def test_drain_offline_wandb_queue_uses_monotonic_history_step():
    args = SimpleNamespace(use_wandb=True, _wandb_metric_queue=Queue())
    metrics = {"train/loss": 0.5, "train/step": 12}
    args._wandb_metric_queue.put((metrics, "train/step"))

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        drained = logging_utils.drain_offline_wandb_queue(args)

    assert drained == 1
    mock_wandb_log.assert_called_once_with(metrics)


@pytest.mark.unit
def test_log_skips_wandb_when_disabled():
    args = SimpleNamespace(use_wandb=False, use_tensorboard=False)
    metrics = {"train/loss": 0.5, "train/step": 47}

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        logging_utils.log(args, metrics, step_key="train/step")

    mock_wandb_log.assert_not_called()
