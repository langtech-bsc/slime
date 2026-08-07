# the file to manage all sglang deps in the megatron actor
try:
    from sglang.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from sglang.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    try:
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
    except ImportError:
        # External engines own SGLang. Full NCCL update does not need this
        # local-server optimization, so preserve a callable no-op for the
        # trainer-only runtime.
        def monkey_patch_torch_reductions():
            return None


try:
    from sglang.srt.managers.io_struct import DeltaEncoding, DeltaParam, DeltaSpec
except ImportError:
    # Older sglang images don't have delta-sync io_struct. Only --update-weight-mode=delta
    # needs these; the default full-sync path runs without them.
    DeltaEncoding = None
    DeltaParam = None
    DeltaSpec = None

class _MissingSGLangFeature:
    """Delay an optional SGLang-only failure until the feature is selected."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("This weight-update mode requires SGLang in the trainer runtime.")

    @staticmethod
    def serialize(*args, **kwargs):
        raise RuntimeError("Tensor weight transport requires SGLang in the trainer runtime.")


try:
    from sglang.srt.utils import MultiprocessingSerializer
except ImportError:
    MultiprocessingSerializer = _MissingSGLangFeature


try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    try:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]
    except ImportError:
        FlattenedTensorBucket = _MissingSGLangFeature

__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
    "DeltaEncoding",
    "DeltaParam",
    "DeltaSpec",
]
