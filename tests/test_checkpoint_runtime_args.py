from __future__ import annotations

import argparse

from slime.utils.checkpoint_runtime_args import (
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
