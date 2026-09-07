from __future__ import annotations

import argparse

from slime.utils.checkpoint_runtime_args import (
    prepare_common_state_dict_for_save,
    restore_ray_runtime_args,
    snapshot_ray_runtime_args,
    strip_ray_runtime_args,
)


class _FakeRayQueue:
    __module__ = "ray.util.queue"


def test_strip_ray_runtime_args_nulls_queue_without_mutating_live_args():
    live_queue = _FakeRayQueue()
    args = argparse.Namespace(load="/ckpt", _wandb_metric_queue=live_queue)
    state = {"args": args, "iteration": 59}

    stripped = strip_ray_runtime_args(state)

    assert stripped["args"] is not args
    assert stripped["args"]._wandb_metric_queue is None
    assert stripped["args"].load == "/ckpt"
    assert args._wandb_metric_queue is live_queue
    assert stripped["iteration"] == 59


def test_snapshot_restores_live_ray_queue_after_checkpoint_args_overwrite():
    live_queue = _FakeRayQueue()
    args = argparse.Namespace(_wandb_metric_queue=live_queue)
    snapshot = snapshot_ray_runtime_args(args)
    args._wandb_metric_queue = None
    restore_ray_runtime_args(args, snapshot)
    assert args._wandb_metric_queue is live_queue


def test_prepare_common_state_dict_replaces_saved_args_but_preserves_live_queue():
    live_queue = _FakeRayQueue()
    args = argparse.Namespace(load="/ckpt", _wandb_metric_queue=live_queue)
    state = {"args": args, "iteration": 9}

    prepared = prepare_common_state_dict_for_save(state)

    assert prepared is state
    assert state["args"] is not args
    assert state["args"]._wandb_metric_queue is None
    assert args._wandb_metric_queue is live_queue


def test_prepare_common_state_dict_applies_preprocessor_to_saved_values():
    args = argparse.Namespace(load="/ckpt")
    state = {"args": args, "iteration": 9}

    prepared = prepare_common_state_dict_for_save(
        state,
        preprocess_common_state_dict_fn=lambda value: {
            **value,
            "iteration": value["iteration"] + 1,
        },
    )

    assert prepared is state
    assert state["iteration"] == 10
