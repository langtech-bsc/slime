import itertools
import json
import logging
import os
import random
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)

_MISSING = object()


def _get_data_field(data: Mapping, key: str | None):
    """Read a flat or dotted field from one dataset row.

    Flat keys retain precedence so existing datasets containing literal dots in
    a field name continue to work. Dotted paths are used by the new RL data
    format, for example ``specific_metadata.label``.
    """

    if key is None:
        return _MISSING
    if key in data:
        return data[key]

    value = data
    for part in str(key).split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _iter_jsonl_files(path: str) -> list[Path]:
    """Return JSONL files for a file or recursively configured directory."""

    candidate = Path(path)
    if candidate.is_dir():
        files = sorted(
            (item for item in candidate.rglob("*.jsonl") if item.is_file()),
            key=lambda item: item.as_posix(),
        )
        if not files:
            raise ValueError(f"Prompt dataset directory '{path}' contains no JSONL files.")
        return files
    return [candidate]


def _iter_rows_from_file(path: Path) -> Iterator[tuple[dict, str]]:
    """Read one supported data file and include a source location per row."""

    path_text = os.fspath(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                location = f"{path_text}:{line_number}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    logger.warning("Skipping invalid JSON row at %s: %s", location, error)
                    continue
                if not isinstance(data, dict):
                    logger.warning(
                        "Skipping row at %s: expected a JSON object, got %s",
                        location,
                        type(data).__name__,
                    )
                    continue
                yield data, location
        return

    if path.suffix.lower() == ".parquet":
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")
        parquet_file = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet_file.iter_batches():
            for data in batch.to_pylist():
                location = f"{path_text}:row={row_number}"
                row_number += 1
                if not isinstance(data, dict):
                    logger.warning(
                        "Skipping row at %s: expected a mapping, got %s",
                        location,
                        type(data).__name__,
                    )
                    continue
                yield data, location
        return

    raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")


def _iter_rows_with_source(path: str) -> Iterator[tuple[dict, str]]:
    path, row_slice = _parse_generalized_path(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    files = _iter_jsonl_files(path) if os.path.isdir(path) else [Path(path)]
    reader: Iterator[tuple[dict, str]] = (
        row for file_path in files for row in _iter_rows_from_file(file_path)
    )
    if row_slice is not None:
        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)
    yield from reader


def read_file(path):
    """Yield rows from one file or all sorted JSONL shards in a directory."""

    for data, _location in _iter_rows_with_source(os.fspath(path)):
        yield data


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from slime.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(sample.prompt, processor)
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = _get_data_field(data, prompt_key)
    if prompt is _MISSING:
        prompt = None

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = _get_data_field(data, data_key)
                if multimodal_data is _MISSING:
                    multimodal_data = None
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        item = content.pop(0)
                        # Support rich image config from https://github.com/QwenLM/Qwen3-VL/blob/main/README.md
                        # "images": [{"type": "image", "image": "path/to/img/01.jpeg", "max_pixels": 50176, "min_pixels": 50176}, {...}]
                        if isinstance(item, dict):
                            content_list.append(item)
                        # "images": ["path/to/img/01.jpeg", "url", "base64enc"]
                        else:
                            content_list.append({"type": mt.name, mt.name: item})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
            )

    return prompt


_LABEL_REQUIRED_REWARD_FAMILIES = frozenset({"rar", "dapo", "math", "code", "code_rlvr"})


def _normalize_reward_family(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and len(value) == 1:
        family = value[0]
        if isinstance(family, str) and family.strip():
            return family.strip()
    raise ValueError("specific_metadata.reward_family must be one non-empty string")


def _usable_label(value) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _normalize_new_row(data: dict) -> dict:
    """Convert the post-training source schema into Slime's canonical row."""

    messages = data.get("messages")
    specific_metadata = data.get("specific_metadata")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if not isinstance(specific_metadata, dict):
        raise ValueError("specific_metadata must be an object")

    normalized_messages = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"messages[{message_index}].role must be a non-empty string")
        if "content" not in message:
            raise ValueError(f"messages[{message_index}].content is missing")
        content = message["content"]
        if not isinstance(content, (str, list, dict)):
            raise ValueError(
                f"messages[{message_index}].content must be a string, list, or object"
            )
        normalized_messages.append(
            {
                "role": role.strip(),
                "content": deepcopy(content),
            }
        )

    if normalized_messages[-1]["role"] != "user":
        raise ValueError("messages must end with a user turn for generation")

    reward_family = _normalize_reward_family(specific_metadata.get("reward_family"))
    raw_label = specific_metadata.get("label", _MISSING)
    reference_answer = specific_metadata.get("reference_answer", _MISSING)
    if reward_family == "rar":
        if _usable_label(reference_answer):
            label = reference_answer
        elif isinstance(raw_label, str) and _usable_label(raw_label):
            label = raw_label
        else:
            label = None
    elif _usable_label(raw_label):
        label = raw_label
    else:
        label = None

    if reward_family in _LABEL_REQUIRED_REWARD_FAMILIES and not _usable_label(label):
        raise ValueError(f"label is required for reward_family={reward_family!r}")

    metadata = dict(specific_metadata)
    metadata["reward_family"] = reward_family
    for key in ("id", "source_id", "dataset_name", "task", "lang", "license"):
        if key in data:
            metadata.setdefault(key, data[key])
    source_metadata = data.get("source_metadata", _MISSING)
    if source_metadata is not _MISSING:
        if not isinstance(source_metadata, dict):
            raise ValueError("source_metadata must be an object when present")
        metadata.setdefault("provenance", {"source_metadata": deepcopy(source_metadata)})

    normalized = dict(data)
    normalized.update(
        {
            "prompt": normalized_messages,
            "label": label,
            "metadata": metadata,
        }
    )
    return normalized


def _is_new_format_row(data: dict) -> bool:
    return "messages" in data or "specific_metadata" in data


_MIXTURE_KEYS = frozenset({"seed", "total_samples", "sources"})
_MIXTURE_SOURCE_KEYS = frozenset({"name", "path", "weight"})


def _normalize_mixture_config(mixture, base_path, default_seed: int) -> dict:
    """Validate and resolve a YAML-configured prompt-data mixture."""

    if not isinstance(mixture, Mapping):
        raise ValueError("prompt data mixture must be an object")
    unknown = sorted(set(mixture) - _MIXTURE_KEYS)
    if unknown:
        raise ValueError(f"prompt data mixture has unknown key(s): {', '.join(unknown)}")

    sources = mixture.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("prompt data mixture.sources must be a non-empty list")

    base = Path(base_path)
    base_dir = base if base.is_dir() else base.parent
    normalized_sources = []
    names = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"prompt data mixture.sources[{index}] must be an object")
        unknown = sorted(set(source) - _MIXTURE_SOURCE_KEYS)
        if unknown:
            raise ValueError(
                f"prompt data mixture.sources[{index}] has unknown key(s): {', '.join(unknown)}"
            )
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(f"prompt data mixture.sources[{index}].path must be a non-empty string")
        weight = source.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError(
                f"prompt data mixture.sources[{index}].weight must be a positive number"
            )
        name = source.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"prompt data mixture.sources[{index}].name must be a non-empty string"
                )
            name = name.strip()
            if name in names:
                raise ValueError(f"prompt data mixture source name is duplicated: {name!r}")
            names.add(name)

        raw_path, row_slice = _parse_generalized_path(source_path.strip())
        if not os.path.isabs(raw_path):
            raw_path = os.fspath(base_dir / raw_path)
        if row_slice is not None:
            start = "" if row_slice.start is None else row_slice.start
            end = "" if row_slice.stop is None else row_slice.stop
            resolved_path = f"{raw_path}@[{start}:{end}]"
        else:
            resolved_path = raw_path
        normalized_source = {"path": resolved_path, "weight": float(weight)}
        if name is not None:
            normalized_source["name"] = name
        normalized_sources.append(normalized_source)

    seed = mixture.get("seed", default_seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("prompt data mixture.seed must be an integer")
    total_samples = mixture.get("total_samples")
    if total_samples is not None and (
        isinstance(total_samples, bool) or not isinstance(total_samples, int) or total_samples <= 0
    ):
        raise ValueError("prompt data mixture.total_samples must be a positive integer")

    return {
        "seed": seed,
        "total_samples": total_samples,
        "sources": normalized_sources,
    }


def _usable_mixture_row(data: dict) -> bool:
    """Return whether a row can enter a mixture without normalizing it twice later."""

    if not _is_new_format_row(data):
        return True
    try:
        _normalize_new_row(data)
    except ValueError:
        return False
    return True


def _count_mixture_rows(path: str) -> int:
    """Count usable rows in one mixture source without retaining large records."""

    return sum(
        1 for data, _location in _iter_rows_with_source(path) if _usable_mixture_row(data)
    )


def _allocate_mixture_counts(total_samples: int, weights: list[float]) -> list[int]:
    """Allocate an exact total using the largest-remainder method."""

    weight_total = sum(weights)
    exact = [total_samples * weight / weight_total for weight in weights]
    counts = [int(value) for value in exact]
    remainder = total_samples - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _largest_without_replacement_mix(
    available_counts: list[int], weights: list[float]
) -> tuple[int, list[int]]:
    """Find the largest weighted mixture that fits every source once."""

    weight_total = sum(weights)
    upper_bound = min(
        int(available / (weight / weight_total))
        for available, weight in zip(available_counts, weights, strict=True)
    )
    for total_samples in range(upper_bound, 0, -1):
        requested_counts = _allocate_mixture_counts(total_samples, weights)
        if all(
            requested <= available
            for requested, available in zip(requested_counts, available_counts, strict=True)
        ):
            return total_samples, requested_counts
    raise ValueError("prompt data mixture cannot allocate a positive no-replacement sample")


def _iter_selected_mixture_rows(path: str, available: int, requested: int, seed: int):
    """Yield a deterministic sample from one source without replacement."""

    if requested <= 0:
        return
    if requested > available:
        raise ValueError(
            f"prompt data mixture requests {requested} rows from a source with only "
            f"{available} usable rows"
        )
    order = list(range(available))
    random.Random(seed).shuffle(order)
    selected = set(order[:requested])
    row_index = 0
    for data, location in _iter_rows_with_source(path):
        if not _usable_mixture_row(data):
            continue
        if row_index in selected:
            yield data, location
        row_index += 1


def _iter_mixture_rows(path: str, mixture, default_seed: int):
    """Yield rows according to exact source weights, without retaining raw rows."""

    normalized = _normalize_mixture_config(mixture, path, default_seed)
    sources = normalized["sources"]
    available_counts = [_count_mixture_rows(source["path"]) for source in sources]
    empty = [
        source["name"] if "name" in source else source["path"]
        for source, count in zip(sources, available_counts, strict=True)
        if count == 0
    ]
    if empty:
        raise ValueError(f"prompt data mixture source(s) contain no usable rows: {', '.join(empty)}")

    total_samples = normalized["total_samples"]
    if total_samples is None:
        total_samples, requested_counts = _largest_without_replacement_mix(
            available_counts, [source["weight"] for source in sources]
        )
    else:
        requested_counts = _allocate_mixture_counts(
            total_samples, [source["weight"] for source in sources]
        )
        if any(
            requested > available
            for requested, available in zip(requested_counts, available_counts, strict=True)
        ):
            details = ", ".join(
                f"{source.get('name', source['path'])}: requested {requested}, available {available}"
                for source, requested, available in zip(
                    sources, requested_counts, available_counts, strict=True
                )
            )
            raise ValueError(
                "prompt data mixture would oversample a source; reduce total_samples "
                f"or adjust weights ({details})"
            )
    logger.info(
        "Prompt data mixture sources=%s available=%s requested=%s",
        [source.get("name", source["path"]) for source in sources],
        available_counts,
        requested_counts,
    )
    for index, (source, available, requested) in enumerate(
        zip(sources, available_counts, requested_counts, strict=True)
    ):
        yield from _iter_selected_mixture_rows(
            source["path"], available, requested, normalized["seed"] + index
        )


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        mixture=None,
    ):
        origin_samples = []
        skipped_rows = 0
        row_iterator = (
            _iter_rows_with_source(os.fspath(path))
            if mixture is None
            else _iter_mixture_rows(os.fspath(path), mixture, seed)
        )
        for data, location in row_iterator:
            if _is_new_format_row(data):
                try:
                    data = _normalize_new_row(data)
                except ValueError as error:
                    skipped_rows += 1
                    logger.warning("Skipping invalid dataset row at %s: %s", location, error)
                    continue
                row_prompt_key = "prompt"
                row_label_key = "label"
                row_metadata_key = "metadata"
            else:
                row_prompt_key = prompt_key
                row_label_key = label_key
                row_metadata_key = metadata_key

            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, row_prompt_key, as_conversation, multimodal_keys)

            raw_metadata = _get_data_field(data, row_metadata_key)
            if raw_metadata is _MISSING or raw_metadata is None:
                metadata = {}
            elif not isinstance(raw_metadata, dict):
                raise ValueError(f"metadata field {row_metadata_key!r} must be an object")
            else:
                metadata = dict(raw_metadata)
            tools = None
            tool_value = _get_data_field(data, tool_key)
            if tool_value is not _MISSING:
                tools = tool_value
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = tokenizer.apply_chat_template(
                    prompt,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor:
                from slime.utils.processing_utils import process_vision_info

                assert isinstance(
                    prompt, list
                ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=(
                        None
                        if row_label_key is None
                        else (
                            None
                            if (label_value := _get_data_field(data, row_label_key)) is _MISSING
                            else label_value
                        )
                    ),
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if not origin_samples:
            raise ValueError(f"No usable prompt rows loaded from '{path}'.")
        if skipped_rows:
            logger.warning(
                "Loaded %s prompt rows from %s; skipped %s invalid new-format rows.",
                len(origin_samples),
                path,
                skipped_rows,
            )

        if mixture is not None:
            random.Random(_normalize_mixture_config(mixture, os.fspath(path), seed)["seed"]).shuffle(
                origin_samples
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
