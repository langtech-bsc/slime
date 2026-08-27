import copy
import logging
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .utils import add_default_ray_env_vars

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    # Wait for the placement group to be scheduled. Poll rather than a bare
    # ray.get(pg.ready()) so the wait is observable: when it can't be placed yet
    # (a node's GPUs haven't registered with the GCS, or an autoscaler is still
    # bringing nodes up) log the GPU counts periodically instead of hanging with no
    # output. The wait stays unbounded, so autoscaling clusters — where a pending
    # placement group is what drives scale-up — are unaffected.
    ready_ref = pg.ready()
    elapsed = 0
    log_interval = 30
    while not ray.wait([ready_ref], timeout=log_interval)[0]:
        elapsed += log_interval
        total = ray.cluster_resources().get("GPU", 0)
        available = ray.available_resources().get("GPU", 0)
        logger.info(
            f"Waiting for placement group of {num_gpus} GPUs (elapsed {elapsed}s): "
            f"{total:g} GPUs registered with Ray, {available:g} available."
        )

    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        logger.info(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0

    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus

    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0

    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0

    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)
    teacher_offset = num_gpus
    teacher_gpus = 0
    if args.use_opd and args.opd_type == "megatron_async":
        teacher_gpus = args.opd_teacher_num_nodes * args.opd_teacher_num_gpus_per_node
        num_gpus += teacher_gpus

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    actor_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    actor_pg_bundle_indices = actor_pg_reordered_bundle_indices[:actor_gpus]
    actor_pg_gpu_ids = actor_pg_reordered_gpu_ids[:actor_gpus]
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:teacher_offset]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:teacher_offset]

    result = {
        "actor": (pg, actor_pg_bundle_indices, actor_pg_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }
    result["opd_teacher"] = (
        (
            pg,
            actor_pg_reordered_bundle_indices[teacher_offset : teacher_offset + teacher_gpus],
            actor_pg_reordered_gpu_ids[teacher_offset : teacher_offset + teacher_gpus],
        )
        if teacher_gpus
        else None
    )

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def create_opd_teacher_model(args, pgs):
    if not (args.use_opd and args.opd_type == "megatron_async"):
        return None
    teacher_args = copy.deepcopy(args)
    if args.opd_teacher_megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        teacher_args = parse_megatron_role_args(
            args,
            args.opd_teacher_megatron_config_path,
            role="opd_teacher",
        )
    else:
        teacher_args.load = args.opd_teacher_load
        teacher_args.use_opd = False
        teacher_args.use_critic = False
        teacher_args.kl_coef = 0
        teacher_args.use_kl_loss = False
        teacher_args.entropy_coef = 0
        teacher_args.no_load_optim = True
        teacher_args.no_load_rng = True

    actor_world = args.actor_num_nodes * args.actor_num_gpus_per_node
    teacher_world = args.opd_teacher_num_nodes * args.opd_teacher_num_gpus_per_node
    if actor_world != teacher_world:
        raise ValueError(
            "Asynchronous OPD currently requires rank-aligned actor and teacher groups: "
            f"actor_world={actor_world}, teacher_world={teacher_world}"
        )
    expected_world = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    if actor_world != expected_world or args.pipeline_model_parallel_size != 1:
        raise ValueError(
            "Asynchronous OPD currently requires one data-parallel replica and pipeline parallel size 1: "
            f"actor_world={actor_world}, tp*pp={expected_world}, pp={args.pipeline_model_parallel_size}"
        )
    for field in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ):
        if getattr(args, field) != getattr(teacher_args, field):
            raise ValueError(
                f"Asynchronous OPD requires matching actor/teacher {field}: "
                f"{getattr(args, field)} != {getattr(teacher_args, field)}"
            )

    teacher_model = allocate_train_group(
        args=teacher_args,
        num_nodes=args.opd_teacher_num_nodes,
        num_gpus_per_node=args.opd_teacher_num_gpus_per_node,
        pg=pgs["opd_teacher"],
        role="opd_teacher",
    )
    ray.get(teacher_model.async_init(teacher_args, role="opd_teacher"))
    return teacher_model


def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
    )


def resolve_num_rollout(args):
    """Fill ``args.num_rollout`` from ``num_epoch * dataset_size`` when needed.

    Sync ``train.py`` can wait until ``RolloutManager`` exists, because that
    path creates the manager before Megatron. Async ``train_async.py`` starts
    actor init first so it can overlap model load with external SGLang
    discovery, but ``get_optimizer_param_scheduler`` needs ``num_rollout`` to
    size ``train_iters``. Instantiating the configured data source on the
    driver is enough to compute the same value ``RolloutManager`` would.

    Returns the number of rollouts per epoch when computed from the dataset,
    otherwise ``None``.
    """
    if getattr(args, "num_rollout", None) is not None:
        return None
    if getattr(args, "num_epoch", None) is None:
        return None

    assert args.rollout_global_dataset, (
        "num_epoch is set, but rollout_global_dataset is not set, "
        "please remove --disable-rollout-global-dataset to use num_epoch"
    )

    from slime.utils.misc import load_function

    data_source_cls = load_function(args.data_source_path)
    data_source = data_source_cls(args)
    dataset_size = len(data_source)
    num_rollout_per_epoch = dataset_size // args.rollout_batch_size
    args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0, (
        f"num_rollout computed as {args.num_rollout} from num_epoch={args.num_epoch}, "
        f"dataset_size={dataset_size}, rollout_batch_size={args.rollout_batch_size}"
    )
    logger.info(
        "Resolved num_rollout=%s from num_epoch=%s, dataset_size=%s, "
        "rollout_batch_size=%s (%s rollouts/epoch)",
        args.num_rollout,
        args.num_epoch,
        dataset_size,
        args.rollout_batch_size,
        num_rollout_per_epoch,
    )
    return num_rollout_per_epoch


def create_training_models(args, pgs, rollout_manager=None, *, attach_rollout_manager=True):
    # Async training initializes Megatron before RolloutManager can compute
    # num_rollout from --num-epoch. Resolve it here so actor/critic args
    # already carry the value used to size the LR schedule.
    resolve_num_rollout(args)

    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )

    critic_model = None
    if args.use_critic:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if attach_rollout_manager:
        if rollout_manager is None:
            raise ValueError("rollout_manager is required when attaching it during model creation")
        actor_model.set_rollout_manager(rollout_manager)
        if args.use_critic:
            critic_model.set_rollout_manager(rollout_manager)

        if args.rollout_global_dataset:
            ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    from .rollout import RolloutManager

    rollout_manager_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": add_default_ray_env_vars()},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        rollout_manager_options["enable_tensor_transport"] = True
    rollout_manager = RolloutManager.options(**rollout_manager_options).remote(args, pg)

    # calculate num_rollout from num_epoch. When async training already
    # resolved num_rollout for Megatron init, still fetch the per-epoch
    # count so save/eval can fire on epoch boundaries.
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0
    elif args.num_epoch is not None and args.rollout_global_dataset:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
