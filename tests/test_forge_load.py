from pathlib import Path
from unittest import mock

import pytest
import torch

from slime.rollout import forge_load
from slime.utils.rollout_dump_utils import _write_rollout_dump


@pytest.mark.unit
def test_forge_load_reads_chunked_directory(tmp_path):
    samples = [{"index": 0, "response": "hello", "reward": 1.0, "status": "completed"}]
    dump_dir = tmp_path / "rollout_0"
    _write_rollout_dump(
        local_dir=dump_dir,
        rollout_id=0,
        evaluation=False,
        samples=samples,
        chunk_bytes=1024,
    )

    class Args:
        load_forge_rollout_data = str(tmp_path / "rollout_{rollout_id}")
        eval_reward_key = None
        reward_key = None

    output = forge_load.generate_rollout(Args(), rollout_id=0, data_source=None, evaluation=False)
    assert len(output.samples) == 1
    assert output.samples[0].response == "hello"


@pytest.mark.unit
def test_forge_load_reads_legacy_single_file(tmp_path):
    samples = [{"index": 0, "response": "legacy", "status": "completed"}]
    legacy_path = tmp_path / "0.pt"
    torch.save({"rollout_id": 0, "samples": samples}, legacy_path)

    class Args:
        load_forge_rollout_data = str(tmp_path / "{rollout_id}.pt")
        eval_reward_key = None
        reward_key = None

    output = forge_load.generate_rollout(Args(), rollout_id=0, data_source=None, evaluation=False)
    assert output.samples[0].response == "legacy"


@pytest.mark.unit
def test_forge_load_eval_uses_rollout_eval_directory(tmp_path):
    samples = [{"index": 0, "response": "eval", "reward": {"score": 0.5}, "status": "completed"}]
    dump_dir = tmp_path / "rollout_eval_1"
    _write_rollout_dump(
        local_dir=dump_dir,
        rollout_id=1,
        evaluation=True,
        samples=samples,
        chunk_bytes=1024,
    )

    class Args:
        load_forge_rollout_data = str(tmp_path / "rollout_{rollout_id}")
        eval_reward_key = "score"
        reward_key = "score"

    output = forge_load.generate_rollout(Args(), rollout_id=1, data_source=None, evaluation=True)
    assert output.data["forge_eval"]["samples"][0].response == "eval"
