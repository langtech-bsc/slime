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

sglang's default streaming output is cumulative — server-side
``state.output_token_logprobs`` accumulates and every chunk references the
full list-so-far (see ``tokenizer_manager.py``). Production servers should use
``--stream-output`` together with slime's ``--sglang-stream-output`` so each
chunk is a disjoint segment. This avoids quadratic JSON traffic and client
work for long generations.
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

__all__ = ["generate_streaming"]

logger = logging.getLogger(__name__)


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

    # Snapshot pre-call sample state. sglang's SSE chunks are cumulative
    # *within this call*; on each chunk we rebuild the post-call view of the
    # sample = prior state + chunk delta. That way a mid-stream break leaves
    # the sample exactly at the boundary of the last chunk we observed.
    base_tokens = list(sample.tokens)
    base_response = sample.response or ""
    base_response_length = sample.response_length
    base_log_probs = None if sample.rollout_log_probs is None else list(sample.rollout_log_probs)
    base_top_p_token_ids = sample.rollout_top_p_token_ids
    base_top_p_token_offsets = sample.rollout_top_p_token_offsets
    base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

    last_meta_info: dict[str, Any] = {}
    call_tokens: list[int] = []
    call_log_probs: list[float] = []
    call_text: str = ""
    stream_output = bool(getattr(args, "sglang_stream_output", False))

    client = http_utils._http_client
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
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("sglang_streaming: skipping non-JSON chunk: %r", data_str[:120])
                    continue

                meta = chunk.get("meta_info") or {}
                last_meta_info = meta

                chunk_text = chunk.get("text", "")
                if "output_token_logprobs" in meta:
                    cumulative_logprobs = meta["output_token_logprobs"]
                else:
                    cumulative_logprobs = []

                if stream_output:
                    # In SGLang 0.5.9, --stream-output makes output_ids
                    # disjoint while text and output_token_logprobs remain
                    # cumulative. Use the disjoint ids and the corresponding
                    # logprob tail, then derive the text suffix.
                    chunk_tokens = list(chunk.get("output_ids") or [])
                    if not chunk_tokens and len(cumulative_logprobs) > len(call_tokens):
                        chunk_tokens = [item[1] for item in cumulative_logprobs[len(call_tokens) :]]
                    chunk_log_probs = [item[0] for item in cumulative_logprobs[-len(chunk_tokens) :]] if chunk_tokens else []

                    if chunk_text.startswith(call_text):
                        text_delta = chunk_text[len(call_text) :]
                    else:
                        # Defensive fallback for a tokenizer correction: keep
                        # token metadata incremental and replace only text.
                        text_delta = None

                    sample.append_response_tokens(
                        args,
                        tokens=chunk_tokens,
                        log_probs=chunk_log_probs,
                        trainable=True,
                        meta_info=meta,
                        text=text_delta,
                        update_terminal_info=bool(meta.get("finish_reason")),
                    )
                    if text_delta is None:
                        sample.response = base_response + chunk_text
                    call_tokens.extend(chunk_tokens)
                    call_log_probs.extend(chunk_log_probs)
                    call_text = chunk_text
                else:
                    # Compatibility path for servers using cumulative SSE
                    # output. This necessarily recopies cumulative state and
                    # should not be used for long production generations.
                    call_tokens = [item[1] for item in cumulative_logprobs]
                    call_log_probs = [item[0] for item in cumulative_logprobs]
                    call_text = chunk_text

                    # Surface partial state on the sample immediately. If the
                    # outer abort path cuts us, whatever we've written so far
                    # is what survives — no /abort_request round-trip needed.
                    sample.tokens = list(base_tokens)
                    sample.response = base_response
                    sample.response_length = base_response_length
                    sample.rollout_log_probs = None if base_log_probs is None else list(base_log_probs)
                    sample.rollout_top_p_token_ids = base_top_p_token_ids
                    sample.rollout_top_p_token_offsets = base_top_p_token_offsets
                    sample.loss_mask = None if base_loss_mask is None else list(base_loss_mask)
                    sample.append_response_tokens(
                        args,
                        tokens=call_tokens,
                        log_probs=call_log_probs,
                        trainable=True,
                        meta_info=meta,
                        text=call_text,
                        update_terminal_info=bool(meta.get("finish_reason")),
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
