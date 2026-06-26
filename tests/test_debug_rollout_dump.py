import pytest

from slime.ray.rollout import _limit_debug_rollout_samples
from slime.utils.types import Sample


@pytest.mark.unit
def test_limit_debug_rollout_samples_by_group_index():
    samples = [
        Sample(group_index=0, index=0, response="a0"),
        Sample(group_index=0, index=1, response="a1"),
        Sample(group_index=1, index=2, response="b0"),
        Sample(group_index=1, index=3, response="b1"),
    ]

    limited = _limit_debug_rollout_samples(samples, max_per_group=1, n_samples_per_prompt=2)

    assert [sample.response for sample in limited] == ["a0", "b0"]


@pytest.mark.unit
def test_limit_debug_rollout_samples_falls_back_to_prompt_chunks():
    samples = [Sample(index=i, response=f"s{i}") for i in range(6)]

    limited = _limit_debug_rollout_samples(samples, max_per_group=1, n_samples_per_prompt=2)

    assert [sample.response for sample in limited] == ["s0", "s2", "s4"]
