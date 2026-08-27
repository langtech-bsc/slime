import logging
from queue import Empty

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


def drain_offline_wandb_queue(args) -> int:
    """Forward queued worker metrics through the driver's offline W&B run."""
    queue = getattr(args, "_wandb_metric_queue", None)
    if queue is None or not args.use_wandb:
        return 0

    drained = 0
    while True:
        try:
            metrics, _step_key = queue.get_nowait()
        except Empty:
            break
        # Worker streams have independent train/rollout steps, while queue
        # snapshots use wall-clock cadence. Use one monotonic W&B history
        # step and let define_metric() map the custom step axes.
        wandb.log(metrics)
        drained += 1
    return drained


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str | None = None, *, step: int | float | None = None):
    if args.use_wandb:
        queue = getattr(args, "_wandb_metric_queue", None)
        if queue is not None:
            queue.put((dict(metrics), step_key))
        elif step_key is not None and step_key in metrics:
            wandb.log(metrics, step=int(metrics[step_key]))
        else:
            wandb.log(metrics)

    if args.use_tensorboard:
        if step_key is not None and step_key in metrics:
            tb_step = metrics[step_key]
            metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        elif step is not None:
            tb_step = step
            metrics_except_step = metrics
        else:
            return
        _TensorboardAdapter(args).log(data=metrics_except_step, step=tb_step)
