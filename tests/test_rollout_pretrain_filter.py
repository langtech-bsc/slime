from types import SimpleNamespace

import pytest

from slime.ray.rollout import _filter_rollout_groups_for_training
from slime.utils.types import Sample

NUM_GPUS = 0


def make_args(**overrides):
    values = dict(
        rollout_max_response_len=100,
        rollout_max_context_len=128,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=128,
        max_rollout_weight_staleness=3,
        rollout_group_min_survival_rate=0.8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_sample(index: int, *, response_length: int = 10, total_length: int = 20, weight_version: str = "10"):
    sample = Sample(index=index, group_index=0)
    sample.tokens = list(range(total_length))
    sample.response_length = response_length
    sample.weight_versions = [weight_version]
    sample.reward = 1.0
    return sample


@pytest.mark.unit
def test_filter_groups_flat_samples_by_group_index_before_rollout_id():
    samples = [make_sample(i, weight_version="10") for i in range(5)]
    for sample in samples:
        sample.rollout_id = sample.index
    samples[0].weight_versions = ["1"]

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        samples,
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [samples[1:]]
    assert metrics["original_groups"] == 1
    assert metrics["kept_groups"] == 1
    assert metrics["dropped_stale_samples"] == 1


@pytest.mark.unit
def test_filter_keeps_group_when_at_least_80_percent_survive():
    group = [make_sample(i, weight_version="10") for i in range(5)]
    group[0].weight_versions = ["1"]

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group[1:]]
    assert metrics["dropped_stale_samples"] == 1
    assert metrics["kept_groups"] == 1
    assert metrics["kept_samples"] == 4


@pytest.mark.unit
def test_filter_drops_group_below_80_percent_survival():
    group = [make_sample(i, weight_version="10") for i in range(5)]
    group[0].weight_versions = ["1"]
    group[1].response_length = 101

    with pytest.raises(ValueError, match="No rollout groups survived"):
        _filter_rollout_groups_for_training(
            make_args(),
            [group],
            trainer_weight_version=10,
            train_parallel_config={"cp_size": 1},
        )


@pytest.mark.unit
def test_filter_drops_sample_over_train_capacity():
    group = [make_sample(0, total_length=129), make_sample(1), make_sample(2), make_sample(3), make_sample(4)]

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group[1:]]
    assert metrics["dropped_length_samples"] == 1


@pytest.mark.unit
def test_generate_requires_trainer_weight_version_when_staleness_enabled():
    from types import SimpleNamespace

    from slime.ray.rollout import RolloutManager

    manager = SimpleNamespace(args=make_args(max_rollout_weight_staleness=3))
    generate = RolloutManager.__ray_actor_class__.generate

    with pytest.raises(ValueError, match="requires trainer_weight_version"):
        generate(manager, 0)
