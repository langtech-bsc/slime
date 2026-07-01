from __future__ import annotations

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
def test_log_skips_wandb_when_disabled():
    args = SimpleNamespace(use_wandb=False, use_tensorboard=False)
    metrics = {"train/loss": 0.5, "train/step": 47}

    with patch("slime.utils.logging_utils.wandb.log") as mock_wandb_log:
        logging_utils.log(args, metrics, step_key="train/step")

    mock_wandb_log.assert_not_called()
