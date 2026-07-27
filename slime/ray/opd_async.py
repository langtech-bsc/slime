from __future__ import annotations

import uuid
from dataclasses import dataclass

import ray

from slime.utils.misc import should_run_periodic_action


@dataclass
class _OpdSlot:
    rollout_id: int
    request_id: str
    rollout_data_refs: list
    teacher_cache_refs: list


def _prepare_slot(
    args,
    rollout_manager,
    teacher_model,
    rollout_id,
    rollout_payload,
    trainer_weight_version,
) -> _OpdSlot:
    rollout_data_refs = ray.get(
        rollout_manager.prepare_train_data.remote(
            rollout_id,
            rollout_payload,
            trainer_weight_version=trainer_weight_version,
        )
    )
    request_id = f"opd-{rollout_id}-{uuid.uuid4().hex}"
    cache_refs = teacher_model.async_cache_opd_hidden(rollout_id, rollout_data_refs, request_id)
    return _OpdSlot(
        rollout_id=rollout_id,
        request_id=request_id,
        rollout_data_refs=rollout_data_refs,
        teacher_cache_refs=cache_refs,
    )


def train_async_opd(args, rollout_manager, actor_model, teacher_model, num_rollout_per_epoch):
    """Run a bounded one-batch-lookahead teacher/actor pipeline."""
    timeout = float(args.opd_stage_timeout_seconds)
    actor_model.set_opd_teacher_heads(teacher_model.get_opd_teacher_heads())
    teacher_model.init_opd_nccl_peer(actor_model)

    trainer_weight_version = actor_model.get_weight_version()
    rollout_future = rollout_manager.collect_rollout_samples.remote(args.start_rollout_id)
    current_slot = None
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if current_slot is None:
            rollout_payload = ray.get(rollout_future)
            rollout_future = (
                rollout_manager.collect_rollout_samples.remote(rollout_id + 1)
                if rollout_id + 1 < args.num_rollout
                else None
            )
            current_slot = _prepare_slot(
                args,
                rollout_manager,
                teacher_model,
                rollout_id,
                rollout_payload,
                trainer_weight_version,
            )

        next_slot = None
        try:
            cache_results = ray.get(current_slot.teacher_cache_refs, timeout=timeout)
            metadata = cache_results[0]
            if "shape" not in metadata:
                raise RuntimeError("OPD teacher rank 0 did not return hidden-state metadata")
            send_refs = teacher_model.async_send_opd_hidden(
                current_slot.request_id,
                args.opd_hidden_transfer_chunk_rows,
            )
            actor_refs = actor_model.async_train(
                current_slot.rollout_id,
                current_slot.rollout_data_refs,
                external_data=metadata,
            )

            update_due = (rollout_id + 1) % args.update_weights_interval == 0
            if rollout_future is not None and args.opd_pipeline_depth > 1:
                next_payload = ray.get(rollout_future)
                rollout_future = None
                next_slot = _prepare_slot(
                    args,
                    rollout_manager,
                    teacher_model,
                    rollout_id + 1,
                    next_payload,
                    trainer_weight_version,
                )
                if not update_due and rollout_id + 2 < args.num_rollout:
                    rollout_future = rollout_manager.collect_rollout_samples.remote(rollout_id + 2)

            ray.get([*send_refs, *actor_refs], timeout=timeout)
        except Exception:
            teacher_model.discard_opd_hidden(current_slot.request_id)
            if next_slot is not None:
                teacher_model.discard_opd_hidden(next_slot.request_id)
            raise

        if should_run_periodic_action(
            rollout_id,
            args.save_interval,
            num_rollout_per_epoch,
            args.num_rollout,
        ):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if update_due:
            actor_model.update_weights()
            trainer_weight_version = actor_model.get_weight_version()
            if rollout_future is None and rollout_id + 2 < args.num_rollout:
                rollout_future = rollout_manager.collect_rollout_samples.remote(rollout_id + 2)

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))
        current_slot = next_slot
