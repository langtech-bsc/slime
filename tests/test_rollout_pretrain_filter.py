from types import SimpleNamespace

import pytest

from slime.ray.rollout import _filter_rollout_groups_for_training, _log_rollout_filter_data
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


def make_sample_with_versions(index: int, versions: list[str]):
    sample = make_sample(index)
    sample.weight_versions = versions
    return sample


def test_rollout_filter_logs_aggregate_staleness_warning(monkeypatch, caplog):
    monkeypatch.setattr("slime.ray.rollout.compute_rollout_step", lambda _args, _rollout_id: 7)
    monkeypatch.setattr("slime.ray.rollout.logging_utils.log", lambda *_args, **_kwargs: None)

    with caplog.at_level("WARNING", logger="slime.ray.rollout"):
        _log_rollout_filter_data(
            3,
            SimpleNamespace(),
            {
                "dropped_stale_samples": 4,
                "original_samples": 64,
                "kept_samples": 60,
                "mean_rollout_weight_staleness": 5.5,
                "max_rollout_weight_staleness": 8,
            },
        )

    assert "dropped stale samples" in caplog.text
    assert "dropped_stale_samples=4" in caplog.text


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


@pytest.mark.unit
def test_filter_keeps_multi_version_sample_when_mean_and_max_staleness_are_within_policy():
    group = [make_sample(i, weight_version="10") for i in range(4)]
    group.append(make_sample_with_versions(4, ["6", "9"]))

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group]
    assert metrics["effective_max_rollout_weight_staleness"] == 6
    assert metrics["dropped_stale_samples"] == 0


@pytest.mark.unit
def test_filter_drops_multi_version_sample_when_max_staleness_is_high():
    group = [make_sample(i, weight_version="10") for i in range(4)]
    group.append(make_sample_with_versions(4, ["3", "10", "10", "10"]))

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group[:4]]
    assert metrics["effective_max_rollout_weight_staleness"] == 6
    assert metrics["dropped_stale_samples"] == 1


@pytest.mark.unit
def test_filter_drops_reward_timeout_sample_before_training():
    group = [make_sample(i, weight_version="10") for i in range(5)]
    group[0].reward = {"score": 0.0, "timeout": True, "failure_type": "timeout"}

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group[1:]]
    assert metrics["dropped_reward_timeout_samples"] == 1
    assert metrics["kept_samples"] == 4


@pytest.mark.unit
def test_filter_drops_sample_over_response_cap():
    group = [make_sample(0, response_length=101), make_sample(1), make_sample(2), make_sample(3), make_sample(4)]

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        [group],
        trainer_weight_version=10,
        train_parallel_config={"cp_size": 1},
    )

    assert kept == [group[1:]]
    assert metrics["dropped_length_samples"] == 1


@pytest.mark.unit
def test_filtered_batch_schedules_partial_global_batch_size():
    """512→511 after staleness filter should schedule one partial step."""
    from slime.utils.dp_schedule import build_dp_schedule

    groups = []
    for group_index in range(32):
        group = [make_sample(group_index * 16 + i, weight_version="34") for i in range(16)]
        groups.append(group)
    groups[0][0].weight_versions = ["28"]

    kept, metrics = _filter_rollout_groups_for_training(
        make_args(),
        groups,
        trainer_weight_version=34,
        train_parallel_config={"cp_size": 1},
    )
    samples = [sample for group in kept for sample in group]
    rollout_ids = [sample.index for sample in samples]
    total_lengths = [len(sample.tokens) for sample in samples]

    assert metrics["kept_samples"] == 511
    assert metrics["dropped_stale_samples"] == 1
    _, _, _, gbs_per_step = build_dp_schedule(
        SimpleNamespace(
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=16384,
            micro_batch_size=1,
            balance_data=False,
            balance_by_flops=False,
            hidden_size=16,
            num_attention_heads=2,
            num_query_groups=2,
            vocab_size=32,
            ffn_hidden_size=64,
            num_experts=None,
            num_layers=2,
            kv_channels=8,
        ),
        {"dp_size": 8, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        total_lengths,
        global_batch_size=512,
        rollout_indices=rollout_ids,
    )
    assert gbs_per_step == [511]
