from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

DEGENERATION_METADATA_KEY = "degenerate_response"
DEGENERATION_REASON_KEY = "degeneration_reason"
DEGENERATION_METRICS_KEY = "degeneration_metrics"

DEFAULT_TRAILING_FENCE_THRESHOLD = 64
DEFAULT_REPEATED_LINE_THRESHOLD = 64
DEFAULT_TOP_CHAR_FRACTION_THRESHOLD = 0.60
DEFAULT_MAX_CHAR_RUN_THRESHOLD = 64
DEFAULT_TAIL_CHARS = 4096


@dataclass(frozen=True)
class DegenerationSignal:
    detected: bool
    reason: str | None
    tail_fence_count: int
    tail_repeated_line_count: int
    top_char_fraction: float
    max_char_run: int

    def to_metadata(self) -> dict[str, int | float | str | bool | None]:
        return asdict(self)


def trailing_fence_count(text: str) -> int:
    lines = [line.strip() for line in (text or "").rstrip().splitlines()]
    count = 0
    for line in reversed(lines):
        if line != "```":
            break
        count += 1
    return count


def tail_repeated_line_count(text: str) -> int:
    lines = [line.strip() for line in (text or "").rstrip().splitlines() if line.strip()]
    if not lines:
        return 0
    last = lines[-1]
    count = 0
    for line in reversed(lines):
        if line != last:
            break
        count += 1
    return count


def top_non_space_char_fraction(text: str, *, tail_chars: int = DEFAULT_TAIL_CHARS) -> float:
    tail = (text or "")[-tail_chars:]
    chars = [char for char in tail if not char.isspace()]
    if not chars:
        return 0.0
    return Counter(chars).most_common(1)[0][1] / len(chars)


def max_same_char_run(text: str, *, tail_chars: int = DEFAULT_TAIL_CHARS) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for char in (text or "")[-tail_chars:]:
        if char == previous:
            current += 1
        else:
            current = 1
            previous = char
        best = max(best, current)
    return best


def detect_degenerate_response(text: str) -> DegenerationSignal:
    fences = trailing_fence_count(text)
    repeated_lines = tail_repeated_line_count(text)
    top_char_fraction = top_non_space_char_fraction(text)
    max_run = max_same_char_run(text)

    reason = None
    if fences >= DEFAULT_TRAILING_FENCE_THRESHOLD:
        reason = "trailing_fence"
    elif repeated_lines >= DEFAULT_REPEATED_LINE_THRESHOLD:
        reason = "repeated_tail_line"
    elif top_char_fraction >= DEFAULT_TOP_CHAR_FRACTION_THRESHOLD:
        reason = "top_char_fraction"
    elif max_run >= DEFAULT_MAX_CHAR_RUN_THRESHOLD:
        reason = "max_char_run"

    return DegenerationSignal(
        detected=reason is not None,
        reason=reason,
        tail_fence_count=fences,
        tail_repeated_line_count=repeated_lines,
        top_char_fraction=top_char_fraction,
        max_char_run=max_run,
    )


def mark_degenerate_response(metadata: dict[str, Any], signal: DegenerationSignal) -> None:
    metadata[DEGENERATION_METADATA_KEY] = True
    metadata[DEGENERATION_REASON_KEY] = signal.reason
    metadata[DEGENERATION_METRICS_KEY] = signal.to_metadata()


def degeneration_signal_from_metadata(metadata: dict[str, Any] | None) -> DegenerationSignal | None:
    if not isinstance(metadata, dict) or not metadata.get(DEGENERATION_METADATA_KEY):
        return None
    metrics = metadata.get(DEGENERATION_METRICS_KEY)
    if not isinstance(metrics, dict):
        return DegenerationSignal(
            detected=True,
            reason=metadata.get(DEGENERATION_REASON_KEY) or "unknown",
            tail_fence_count=0,
            tail_repeated_line_count=0,
            top_char_fraction=0.0,
            max_char_run=0,
        )
    return DegenerationSignal(
        detected=bool(metrics.get("detected", True)),
        reason=metrics.get("reason") or metadata.get(DEGENERATION_REASON_KEY),
        tail_fence_count=int(metrics.get("tail_fence_count", 0)),
        tail_repeated_line_count=int(metrics.get("tail_repeated_line_count", 0)),
        top_char_fraction=float(metrics.get("top_char_fraction", 0.0)),
        max_char_run=int(metrics.get("max_char_run", 0)),
    )


def apply_degeneration_penalty(reward: float | dict[str, Any], *, penalty: float) -> float | dict[str, Any]:
    if isinstance(reward, dict):
        updated = dict(reward)
        base_score = float(updated.get("score", 0.0))
        updated["score"] = base_score - penalty
        updated["degeneration_penalty"] = penalty
        updated["pre_degeneration_score"] = base_score
        return updated
    return float(reward) - penalty
