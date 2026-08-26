from __future__ import annotations

import pytest

from slime.rollout.fully_async_backpressure import (
    AdaptiveBackpressureController,
    DEFAULT_RATE_MULTIPLIER,
    DEFAULT_RATE_WINDOW_SECONDS,
    MIN_RATE_WINDOW_SECONDS,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _controller(clock: FakeClock, **overrides) -> AdaptiveBackpressureController:
    values = {
        "rate_window_seconds": MIN_RATE_WINDOW_SECONDS,
        "high_watermark_samples": 100,
        "clock": clock,
    }
    values.update(overrides)
    return AdaptiveBackpressureController(**values)


def test_backpressure_defaults_and_minimum_window():
    assert DEFAULT_RATE_MULTIPLIER == 0.95
    assert DEFAULT_RATE_WINDOW_SECONDS == 300.0

    with pytest.raises(ValueError, match="at least 30 seconds"):
        AdaptiveBackpressureController(rate_window_seconds=29.9, high_watermark_samples=10)


def test_target_uses_95_percent_of_slower_downstream_rate_after_warmup():
    clock = FakeClock()
    controller = _controller(clock)
    controller.record_generated(120)
    controller.record_rewarded(60)
    assert controller.report_trainer_performance(step_time=30, effective_batch_size=90)

    clock.now = 29.9
    assert controller.target_rate() is None
    assert controller.try_admit()

    clock.now = 30.0
    controller.observe_reward_capacity(saturated=True)
    generation_rate, reward_rate = controller.rates()
    assert generation_rate == pytest.approx(4.0)
    assert reward_rate == pytest.approx(2.0)
    assert controller.trainer_samples_per_s == pytest.approx(3.0)
    assert controller.target_rate() == pytest.approx(1.9)
    assert controller.pressure_reasons() == ("reward_backpressure", "trainer_backpressure")


def test_token_bucket_limits_admission_without_cancelling_work():
    clock = FakeClock()
    controller = _controller(clock)
    controller.record_rewarded(60)
    controller.report_trainer_performance(step_time=10, effective_batch_size=100)
    clock.now = 30.0
    controller.observe_reward_capacity(saturated=True)

    assert controller.try_admit()
    assert not controller.try_admit()
    clock.now += 1 / 1.9
    assert controller.try_admit()


def test_queue_watermarks_pause_and_resume_with_hysteresis():
    clock = FakeClock()
    controller = _controller(clock, high_watermark_samples=100)

    assert controller.observe_queues(reward_samples=100, training_samples=0) == ("reward_backpressure",)
    assert not controller.try_admit()
    assert controller.observe_queues(reward_samples=51, training_samples=0) == ("reward_backpressure",)
    assert controller.observe_queues(reward_samples=50, training_samples=50) == ()
    assert controller.try_admit()

    assert controller.observe_queues(reward_samples=0, training_samples=100) == ("trainer_backpressure",)


def test_invalid_trainer_measurement_is_ignored():
    controller = _controller(FakeClock())

    assert not controller.report_trainer_performance(step_time=0, effective_batch_size=64)
    assert not controller.report_trainer_performance(step_time=10, effective_batch_size=None)
    assert controller.trainer_samples_per_s is None


def test_reward_capacity_freezes_when_generation_becomes_demand_limited():
    clock = FakeClock()
    controller = _controller(clock)
    controller.record_generated(120)
    controller.record_rewarded(60)
    controller.report_trainer_performance(step_time=10, effective_batch_size=100)
    clock.now = 30.0
    controller.observe_reward_capacity(saturated=True)
    assert controller.target_rate() == pytest.approx(1.9)

    clock.now = 60.1
    controller.record_generated(57)
    controller.record_rewarded(57)
    controller.observe_reward_capacity(saturated=False)

    assert controller.reward_capacity_samples_per_s == pytest.approx(2.0)
    assert controller.target_rate() == pytest.approx(1.9)
