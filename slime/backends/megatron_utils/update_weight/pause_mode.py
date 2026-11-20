"""How weight updates pause the SGLang engines (``--sglang-pause-generation-mode``)."""


def flush_cache_after_pause(args) -> bool:
    """``in_place`` keeps running requests (and their KV cache) in the engine across the update;
    SGLang refuses to flush the cache with requests in the running batch, so it is skipped."""
    return getattr(args, "sglang_pause_generation_mode", "abort") != "in_place"
