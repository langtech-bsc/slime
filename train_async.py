import logging
import os
from queue import Empty, Full

import ray
from ray.util.queue import Queue

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.ray.utils import add_default_ray_env_vars
from slime.utils.arguments import parse_args
from slime.utils import logging_utils
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


_FULLY_ASYNC_ROLLOUT_PATH = "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
logger = logging.getLogger(__name__)


def _is_empty_filtered_batch_error(error: Exception) -> bool:
    return "No rollout groups survived pre-training filtering" in str(error)


def _prepare_train_data_with_recovery(
    args,
    rollout_manager,
    rollout_id,
    rollout_payload,
    trainer_weight_version,
    rollout_data_next_future,
    request_rollout,
):
    """Prepare a batch, replacing fully filtered batches in fully-async mode.

    ``rollout_data_next_future`` is the normal async lookahead. Reusing it as
    the replacement avoids advancing the data source twice when the current
    batch is discarded. A replacement is requested directly only when there
    is no lookahead (notably for the final training step).
    """
    while True:
        try:
            rollout_data_ref = ray.get(
                rollout_manager.prepare_train_data.remote(
                    rollout_id,
                    rollout_payload,
                    trainer_weight_version=trainer_weight_version,
                )
            )
        except Exception as error:
            if (
                getattr(args, "rollout_function_path", None) != _FULLY_ASYNC_ROLLOUT_PATH
                or not _is_empty_filtered_batch_error(error)
            ):
                raise

            logger.warning(
                "Fully-async rollout %s was discarded because pre-training filtering removed every group; "
                "collecting a replacement batch.",
                rollout_id,
            )
            if rollout_data_next_future is None:
                rollout_data_next_future = request_rollout()
            rollout_payload = ray.get(rollout_data_next_future)
            rollout_data_next_future = None

            # Keep the normal one-batch lookahead for the next training step.
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = request_rollout()
            continue

        return rollout_data_ref, rollout_data_next_future


def _primary_trainer_perf(results):
    for result in results or []:
        if isinstance(result, dict) and "perf/step_time" in result:
            return {
                "perf/step_time": result["perf/step_time"],
                "perf/effective_global_batch_size": result.get("perf/effective_global_batch_size"),
            }
    return None


def _publish_trainer_perf(perf_queue, results) -> None:
    if perf_queue is None or (perf := _primary_trainer_perf(results)) is None:
        return
    try:
        perf_queue.put_nowait(perf)
    except Full:
        try:
            perf_queue.get_nowait()
        except Empty:
            pass
        perf_queue.put_nowait(perf)


def _init_ray_for_driver():
    """Connect to an existing cluster when RAY_ADDRESS is set (VERL sbatch pattern)."""
    if ray.is_initialized():
        return
    ray_address = os.environ.get("RAY_ADDRESS")
    if ray_address:
        ray.init(
            address=ray_address,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        )


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
    if args.use_wandb and (
        args.wandb_mode == "offline" or os.environ.get("WANDB_MODE") == "offline"
    ):
        # Ray workers cannot append to the same offline W&B file. Give them a
        # shared queue so the driver remains the only offline W&B writer.
        args._wandb_metric_queue = Queue(maxsize=10000)

    # Start actor/critic initialization before waiting for external SGLang
    # discovery.  The MN5 launcher starts those servers concurrently; the
    # rollout manager performs discover_external_engines_with_retry after the
    # actor model has begun loading.  create_training_models resolves
    # --num-epoch into --num-rollout first, because Megatron's LR scheduler
    # needs that value during actor init.
    actor_model, critic_model = create_training_models(
        args,
        pgs,
        rollout_manager=None,
        attach_rollout_manager=False,
    )

    trainer_perf_queue = None
    if args.rollout_function_path == _FULLY_ASYNC_ROLLOUT_PATH:
        trainer_perf_queue = Queue(maxsize=16)
        args._fully_async_trainer_perf_queue = trainer_perf_queue

    # Create the rollout manager after actor initialization.  For external
    # engines this is the synchronization point at which server_info is
    # discovered and the Ray-side engine adapters are initialized.
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model.set_rollout_manager(rollout_manager)
    if critic_model is not None:
        critic_model.set_rollout_manager(rollout_manager)
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    next_rollout_id = args.start_rollout_id

    def request_rollout():
        nonlocal next_rollout_id
        rollout_data_future = rollout_manager.collect_rollout_samples.remote(next_rollout_id)
        next_rollout_id += 1
        return rollout_data_future

    rollout_data_next_future = request_rollout()
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            rollout_payload_curr = ray.get(rollout_data_next_future)
            rollout_data_next_future = None
            logging_utils.drain_offline_wandb_queue(args)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = request_rollout()

        trainer_weight_version = actor_model.get_weight_version()
        rollout_data_curr_ref, rollout_data_next_future = _prepare_train_data_with_recovery(
            args,
            rollout_manager,
            rollout_id,
            rollout_payload_curr,
            trainer_weight_version,
            rollout_data_next_future,
            request_rollout,
        )

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                actor_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs)
                )
                _publish_trainer_perf(trainer_perf_queue, actor_results)
            else:
                ray.get(value_refs)
        else:
            actor_results = ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            _publish_trainer_perf(trainer_perf_queue, actor_results)
        logging_utils.drain_offline_wandb_queue(args)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_payload_curr = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))
            logging_utils.drain_offline_wandb_queue(args)

    ray.get(rollout_manager.dispose.remote())
    logging_utils.drain_offline_wandb_queue(args)
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    _init_ray_for_driver()
    train(args)
