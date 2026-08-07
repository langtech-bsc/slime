import gc
import json
import os
import shutil
from types import SimpleNamespace
from glob import glob

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import print_memory


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def _qwen3_5_config_from_checkpoint(hf_model_path: str):
    """Load the config fields consumed by the Qwen3.5 mbridge plugin.

    Qwen3.5 checkpoints predate support in the Transformers version bundled in
    the frozen training image.  The mbridge plugin only needs the serialized
    config attributes (especially ``text_config``), so avoid making conversion
    depend on an unavailable Transformers model class.
    """

    def as_namespace(value):
        if isinstance(value, dict):
            return SimpleNamespace(**{key: as_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [as_namespace(item) for item in value]
        return value

    with open(os.path.join(hf_model_path, "config.json"), encoding="utf-8") as config_file:
        config = as_namespace(json.load(config_file))
    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError(f"Expected a qwen3_5 checkpoint at {hf_model_path}")
    return config


def _load_bridge(hf_model_path: str):
    try:
        return AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    except ValueError as exc:
        # Transformers raises this exact architecture error before mbridge gets
        # a chance to select slime_plugins.mbridge.Qwen3_5Bridge.
        if "model type `qwen3_5`" not in str(exc):
            raise
        bridge = AutoBridge.from_config(_qwen3_5_config_from_checkpoint(hf_model_path))
        # SafeTensorIO independently loads AutoConfig only to decide whether a
        # tied lm_head should be omitted.  Qwen3.5-9B is untied, so construct
        # the same index directly and keep the conversion independent from a
        # Transformers architecture registration.
        from mbridge.core.safetensor_io import SafeTensorIO

        def qwen3_5_safetensor_io(weights_path: str):
            safetensor_io = SafeTensorIO.__new__(SafeTensorIO)
            index_file = os.path.join(weights_path, "model.safetensors.index.json")
            safetensor_io.index = {}
            safetensor_io.origin_index = {}
            if os.path.exists(index_file):
                with open(index_file, encoding="utf-8") as index_handle:
                    safetensor_io.origin_index = json.load(index_handle)
                safetensor_io.index = safetensor_io.origin_index["weight_map"]
            else:
                from safetensors import safe_open

                for filename in glob(os.path.join(weights_path, "*.safetensors")):
                    with safe_open(filename, framework="pt", device="cpu") as handle:
                        safetensor_io.index.update({key: os.path.basename(filename) for key in handle.keys()})
            safetensor_io.hf_dir = weights_path
            return safetensor_io

        bridge._get_safetensor_io = lambda weights_path: qwen3_5_safetensor_io(bridge._get_actual_hf_path(weights_path))
        return bridge


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = _load_bridge(hf_model_path)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
