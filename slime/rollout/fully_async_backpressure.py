"""Adaptive sample-rate and queue backpressure for fully async rollout."""

from __future__ import annotations

from collections import deque
import math
import time

DEFAULT_RATE_MULTIPLIER = 0.95
DEFAULT_RATE_WINDOW_SECONDS = 300.0
MIN_RATE_WINDOW_SECONDS = 30.0
WARNING_INTERVAL_SECONDS = 30.0


class AdaptiveBackpressureController:
    """Limit new generation using downstream sample rates and queue hysteresis."""

    def __init__(
        self,
        *,
        rate_multiplier: float = DEFAULT_RATE_MULTIPLIER,
        rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
        high_watermark_samples: int,
        low_watermark_fraction: float = 0.5,
        clock=time.monotonic,
    ) -> None:
        if not math.isfinite(rate_multiplier) or not 0 < rate_multiplier <= 1:
            raise ValueError(f"backpressure rate multiplier must be in (0, 1], got {rate_multiplier}")
        if not math.isfinite(rate_window_seconds) or rate_window_seconds < MIN_RATE_WINDOW_SECONDS:
            raise ValueError(
                f"fully async backpressure rate window must be at least {MIN_RATE_WINDOW_SECONDS:.0f} seconds, "
                f"got {rate_window_seconds}"
            )
        if high_watermark_samples < 1:
            raise ValueError(f"backpressure high watermark must be positive, got {high_watermark_samples}")
        if not 0 < low_watermark_fraction < 1:
            raise ValueError(f"backpressure low watermark fraction must be in (0, 1), got {low_watermark_fraction}")

        self.rate_multiplier = float(rate_multiplier)
        self.rate_window_seconds = float(rate_window_seconds)
        self.high_watermark_samples = int(high_watermark_samples)
        self.low_watermark_samples = max(1, int(math.floor(high_watermark_samples * low_watermark_fraction)))
        self._clock = clock
        self._started_at = clock()
        self._generated_events: deque[tuple[float, int]] = deque()
        self._rewarded_events: deque[tuple[float, int]] = deque()
        self.reward_capacity_samples_per_s: float | None = None
        self.trainer_samples_per_s: float | None = None
        self.trainer_step_time: float | None = None
        self.trainer_effective_batch_size: int | None = None
        self.queue_pause_reasons: tuple[str, ...] = ()
        self.rate_limited = False
        self._tokens = 0.0
        self._token_rate: float | None = None
        self._last_token_update = self._started_at

    def record_generated(self, count: int, *, now: float | None = None) -> None:
        if count > 0:
            self._generated_events.append((self._clock() if now is None else now, count))

    def record_rewarded(self, count: int, *, now: float | None = None) -> None:
        if count > 0:
            self._rewarded_events.append((self._clock() if now is None else now, count))

    def report_trainer_performance(self, *, step_time: float, effective_batch_size: int) -> bool:
        if (
            not isinstance(step_time, (int, float))
            or isinstance(step_time, bool)
            or not math.isfinite(step_time)
            or step_time <= 0
            or isinstance(effective_batch_size, bool)
            or not isinstance(effective_batch_size, (int, float))
            or not math.isfinite(effective_batch_size)
            or effective_batch_size <= 0
        ):
            return False
        self.trainer_step_time = float(step_time)
        self.trainer_effective_batch_size = int(effective_batch_size)
        self.trainer_samples_per_s = float(effective_batch_size) / float(step_time)
        return True

    def observe_queues(self, *, reward_samples: int, training_samples: int) -> tuple[str, ...]:
        if self.queue_pause_reasons:
            if reward_samples <= self.low_watermark_samples and training_samples <= self.low_watermark_samples:
                self.queue_pause_reasons = ()
            return self.queue_pause_reasons

        reasons = []
        if reward_samples >= self.high_watermark_samples:
            reasons.append("reward_backpressure")
        if training_samples >= self.high_watermark_samples:
            reasons.append("trainer_backpressure")
        self.queue_pause_reasons = tuple(reasons)
        return self.queue_pause_reasons

    def observe_reward_capacity(self, *, saturated: bool, now: float | None = None) -> None:
        """Update evaluator capacity only when enough reward work is available.

        Freezing the estimate when demand drains prevents a 0.95 multiplier from
        recursively reducing its own observed downstream rate every window.
        """

        if not saturated:
            return
        generation_rate, reward_rate = self.rates(now=now)
        if reward_rate is None or reward_rate <= 0:
            return
        if self.reward_capacity_samples_per_s is None:
            self.reward_capacity_samples_per_s = reward_rate
            return
        if generation_rate is not None and generation_rate > reward_rate:
            self.reward_capacity_samples_per_s = reward_rate
        else:
            self.reward_capacity_samples_per_s = max(self.reward_capacity_samples_per_s, reward_rate)

    def rates(self, *, now: float | None = None) -> tuple[float | None, float | None]:
        now = self._clock() if now is None else now
        self._prune(now)
        if now - self._started_at < self.rate_window_seconds:
            return None, None
        return (
            sum(count for _timestamp, count in self._generated_events) / self.rate_window_seconds,
            sum(count for _timestamp, count in self._rewarded_events) / self.rate_window_seconds,
        )

    def target_rate(self, *, now: float | None = None) -> float | None:
        self.rates(now=now)
        if self.reward_capacity_samples_per_s is None or self.trainer_samples_per_s is None:
            return None
        return self.rate_multiplier * min(self.reward_capacity_samples_per_s, self.trainer_samples_per_s)

    def try_admit(self, *, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        if self.queue_pause_reasons:
            self.rate_limited = False
            return False

        target_rate = self.target_rate(now=now)
        if target_rate is None:
            self.rate_limited = False
            return True

        self._refill_tokens(now, target_rate)
        if self._tokens < 1.0:
            self.rate_limited = True
            return False
        self._tokens -= 1.0
        self.rate_limited = False
        return True

    def pressure_reasons(self, *, now: float | None = None) -> tuple[str, ...]:
        generation_rate, reward_rate = self.rates(now=now)
        reasons = set(self.queue_pause_reasons)
        if generation_rate is not None and reward_rate is not None and generation_rate > reward_rate:
            reasons.add("reward_backpressure")
        if (
            generation_rate is not None
            and self.trainer_samples_per_s is not None
            and generation_rate > self.trainer_samples_per_s
        ):
            reasons.add("trainer_backpressure")
        return tuple(sorted(reasons))

    def snapshot(self, *, now: float | None = None) -> dict[str, float | int]:
        now = self._clock() if now is None else now
        generation_rate, reward_rate = self.rates(now=now)
        target_rate = self.target_rate(now=now)
        reasons = self.pressure_reasons(now=now)
        return {
            "fully_async/generation_samples_per_s": -1.0 if generation_rate is None else generation_rate,
            "fully_async/reward_samples_per_s": -1.0 if reward_rate is None else reward_rate,
            "fully_async/reward_capacity_samples_per_s": (
                -1.0 if self.reward_capacity_samples_per_s is None else self.reward_capacity_samples_per_s
            ),
            "fully_async/trainer_samples_per_s": (
                -1.0 if self.trainer_samples_per_s is None else self.trainer_samples_per_s
            ),
            "fully_async/target_generation_samples_per_s": -1.0 if target_rate is None else target_rate,
            "fully_async/backpressure_warmup": int(target_rate is None),
            "fully_async/backpressure_paused": int(bool(self.queue_pause_reasons) or self.rate_limited),
            "fully_async/reward_backpressure": int("reward_backpressure" in reasons),
            "fully_async/trainer_backpressure": int("trainer_backpressure" in reasons),
            "fully_async/backpressure_rate_window_seconds": self.rate_window_seconds,
            "fully_async/backpressure_high_watermark_samples": self.high_watermark_samples,
            "fully_async/backpressure_low_watermark_samples": self.low_watermark_samples,
        }

    def _prune(self, now: float) -> None:
        cutoff = now - self.rate_window_seconds
        while self._generated_events and self._generated_events[0][0] < cutoff:
            self._generated_events.popleft()
        while self._rewarded_events and self._rewarded_events[0][0] < cutoff:
            self._rewarded_events.popleft()

    def _refill_tokens(self, now: float, target_rate: float) -> None:
        if self._token_rate is not None:
            elapsed = max(0.0, now - self._last_token_update)
            capacity = max(1.0, self._token_rate)
            self._tokens = min(capacity, self._tokens + elapsed * self._token_rate)

        if self._token_rate is None:
            self._tokens = max(1.0, target_rate) if target_rate > 0 else 0.0
        self._token_rate = target_rate
        self._last_token_update = now
        self._tokens = min(max(1.0, target_rate), self._tokens)
