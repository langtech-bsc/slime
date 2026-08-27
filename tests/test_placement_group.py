import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.placement_group import (
    _create_placement_group,
    _get_placement_group_layout,
    create_placement_groups,
    resolve_num_rollout,
)

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
        "use_critic": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


def test_async_opd_reserves_disjoint_teacher_bundles(monkeypatch):
    args = _args(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus=12,
        use_opd=True,
        opd_type="megatron_async",
        opd_teacher_num_nodes=1,
        opd_teacher_num_gpus_per_node=4,
    )
    monkeypatch.setattr(
        "slime.ray.placement_group._create_placement_group",
        lambda count: ("pg", list(range(count)), list(range(count))),
    )

    groups = create_placement_groups(args)

    assert groups["rollout"][1][:12] == list(range(4, 16))
    assert groups["opd_teacher"][1] == list(range(16, 20))


def test_resolve_num_rollout_from_epoch_and_dataset(monkeypatch):
    class FakeDataSource:
        def __init__(self, args):
            self.args = args

        def __len__(self):
            return 10

    monkeypatch.setattr("slime.utils.misc.load_function", lambda path: FakeDataSource)

    args = Namespace(
        num_rollout=None,
        num_epoch=2,
        rollout_batch_size=5,
        data_source_path="fake.DataSource",
        rollout_global_dataset=True,
    )

    assert resolve_num_rollout(args) == 2
    assert args.num_rollout == 4


def test_resolve_num_rollout_is_noop_when_already_set(monkeypatch):
    def fail_load(_path):
        raise AssertionError("data source should not be loaded when num_rollout is set")

    monkeypatch.setattr("slime.utils.misc.load_function", fail_load)

    args = Namespace(num_rollout=8, num_epoch=1, data_source_path="fake.DataSource")
    assert resolve_num_rollout(args) is None
    assert args.num_rollout == 8


def test_resolve_num_rollout_is_noop_without_epoch(monkeypatch):
    def fail_load(_path):
        raise AssertionError("data source should not be loaded when num_epoch is unset")

    monkeypatch.setattr("slime.utils.misc.load_function", fail_load)

    args = Namespace(num_rollout=None, num_epoch=None, data_source_path="fake.DataSource")
    assert resolve_num_rollout(args) is None
    assert args.num_rollout is None


def test_resolve_num_rollout_requires_global_dataset():
    args = Namespace(
        num_rollout=None,
        num_epoch=1,
        rollout_batch_size=4,
        data_source_path="fake.DataSource",
        rollout_global_dataset=False,
    )
    with pytest.raises(AssertionError, match="rollout_global_dataset"):
        resolve_num_rollout(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
