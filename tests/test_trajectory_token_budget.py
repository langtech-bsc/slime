from types import SimpleNamespace

import pytest

from slime.rollout.sglang_rollout import clamp_sampling_params_for_sample
from slime.utils.types import Sample

NUM_GPUS = 0


def _args(**overrides):
    values = dict(
        rollout_max_response_len=12288,
        rollout_max_context_len=16384,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_clamp_sampling_params_uses_remaining_trajectory_budget():
    sample = Sample(index=1, prompt="prompt")
    sample.response_length = 5000

    clamped = clamp_sampling_params_for_sample(
        _args(),
        sample,
        {"max_new_tokens": 12288},
        prompt_token_count=4096,
    )

    assert clamped["max_new_tokens"] == 7288


@pytest.mark.unit
def test_clamp_sampling_params_uses_remaining_context_budget():
    sample = Sample(index=1, prompt="prompt")
    sample.response_length = 0

    clamped = clamp_sampling_params_for_sample(
        _args(),
        sample,
        {"max_new_tokens": 12288},
        prompt_token_count=15000,
    )

    assert clamped["max_new_tokens"] == 1384


@pytest.mark.unit
def test_clamp_sampling_params_returns_zero_when_trajectory_exhausted():
    sample = Sample(index=1, prompt="prompt")
    sample.response_length = 12288

    clamped = clamp_sampling_params_for_sample(
        _args(),
        sample,
        {"max_new_tokens": 12288},
        prompt_token_count=4096,
    )

    assert clamped["max_new_tokens"] == 0
