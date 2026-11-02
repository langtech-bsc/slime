"""Streaming sglang rollout (example).

Drop-in alternative to :func:`slime.rollout.sglang_rollout.generate` that
consumes sglang's SSE stream incrementally instead of awaiting one final JSON
response. The win is on **abort**: every chunk we receive lands directly on
``sample`` (tokens, response text, log-probs), so when a partial-rollout
recycling or weight-update abort fires mid-generation, the partial state is
already on the sample — we don't depend on ``/abort_request`` returning the
collected text.

Wire it in as the per-sample generate function::

    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \\
    --custom-generate-function-path slime.rollout.sglang_streaming_rollout.generate_streaming

The outer rollout loop (semaphore, dp_rank balancing, abort orchestration,
partial-rollout buffer hand-off) is still owned by ``sglang_rollout``; this
file only replaces the inner HTTP call.

Chunk formats. In SGLang 0.5.9 ``text`` and ``meta_info.output_token_logprobs``
are cumulative in every chunk, with or without ``--stream-output`` (which only
makes ``output_ids`` disjoint). The server also coalesces queued chunks by
sending only the newest one, so disjoint ``output_ids`` deltas can be dropped,
and an abort of a retracted request re-sends ``[last_id]``. Content is
therefore always taken from the logprob entries ``(logprob, token_id, ...)``,
never from ``output_ids``:

* chunks marked ``meta_info["rl_delta_stream"]`` (the server patch in
  ``runtime_patches/sglang_rollout``) carry only the new text and logprobs;
* any other chunk is treated as cumulative and the new tokens are the entries
  past the ones already applied, which survives dropped and repeated chunks.
"""

import json
import logging
from argparse import Namespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, _prepare_prompt_ids, clamp_sampling_params_for_sample
from slime.utils import http_utils
from slime.utils.degeneration import detect_degenerate_response, mark_degenerate_response
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_span
from slime.utils.types import Sample

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover - orjson ships with the runtime images
    _loads = json.loads

__all__ = ["DELTA_STREAM_MARKER", "generate_streaming"]

logger = logging.getLogger(__name__)

DELTA_STREAM_MARKER = "rl_delta_stream"
STREAM_TOKEN_MISMATCH_KEY = "stream_token_mismatch"


async def generate_streaming(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Streaming counterpart to :func:`slime.rollout.sglang_rollout.generate`.

    Writes the cumulative state from each SSE chunk onto ``sample`` so an
    abort that cuts the stream still leaves a coherent partial sample behind.
    """
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status in (
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
    sampling_params = clamp_sampling_params_for_sample(
        args,
        sample,
        sampling_params,
        prompt_token_count=len(prompt_ids),
    )

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    payload: dict[str, Any] = {
        "sampling_params": sampling_params,
        "return_logprob": True,
        "stream": True,
    }
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        headers = {"X-SMG-Routing-Key": sample.session_id}

    # Text is rebuilt from this base only when a cumulative chunk no longer
    # extends the text seen so far (a detokenizer correction).
    base_response = sample.response or ""

    last_meta_info: dict[str, Any] = {}
    applied_tokens = 0
    call_text: str = ""

    client = http_utils.get_stream_client()
    assert client is not None, "http client not initialized; call init_http_client first"

    with trace_span(
        sample, "sglang_generate_stream", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}
    ) as span:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:") :].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = _loads(data_str)
                except ValueError:
                    logger.warning("sglang_streaming: skipping non-JSON chunk: %r", data_str[:120])
                    continue

                meta = chunk.get("meta_info") or {}
                last_meta_info = meta
                chunk_text = chunk.get("text", "")
                logprob_items = meta.get("output_token_logprobs") or []

                if meta.get(DELTA_STREAM_MARKER):
                    new_items = logprob_items
                    text_delta = chunk_text
                else:
                    new_items = logprob_items[applied_tokens:]
                    if chunk_text.startswith(call_text):
                        text_delta = chunk_text[len(call_text) :]
                    else:
                        text_delta = None
                    call_text = chunk_text

                finished = bool(meta.get("finish_reason"))
                sample.append_response_tokens(
                    args,
                    tokens=[item[1] for item in new_items],
                    log_probs=[item[0] for item in new_items],
                    trainable=True,
                    meta_info=meta,
                    text=text_delta,
                    update_terminal_info=finished,
                )
                if text_delta is None:
                    sample.response = base_response + chunk_text
                applied_tokens += len(new_items)

                completion_tokens = meta.get("completion_tokens")
                if finished and completion_tokens is not None and completion_tokens != applied_tokens:
                    sample.metadata[STREAM_TOKEN_MISMATCH_KEY] = {
                        "applied": applied_tokens,
                        "completion_tokens": completion_tokens,
                    }
                    logger.debug(
                        "sglang_streaming: applied %d tokens but the server reports %d",
                        applied_tokens,
                        completion_tokens,
                    )

                if getattr(args, "rollout_degeneration_detection_enable", False):
                    signal = detect_degenerate_response(sample.response)
                    if signal.detected:
                        mark_degenerate_response(sample.metadata, signal)
                        if getattr(args, "rollout_degeneration_stop_enable", False):
                            sample.status = Sample.Status.TRUNCATED
                            break

                if state.aborted:
                    break

        if last_meta_info.get("finish_reason"):
            span.update(build_sglang_meta_trace_attrs(last_meta_info))

    if state.aborted and not last_meta_info.get("finish_reason"):
        sample.status = Sample.Status.ABORTED

    return sample
