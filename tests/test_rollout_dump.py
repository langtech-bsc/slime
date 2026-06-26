import json
import pickle
from pathlib import Path
from unittest import mock

import pytest
import torch

from queue import Full

from slime.utils.rollout_dump_utils import (
    DEFAULT_CHUNK_BYTES,
    LOCAL_QUEUE_MAXSIZE,
    AsyncRolloutDumper,
    RolloutDumpJob,
    _write_rollout_dump,
    load_rollout_dump,
    resolve_rollout_dump_load_path,
    resolve_rollout_dump_local_root,
    rollout_dump_exists,
    split_samples_into_chunks,
)


def _make_samples(count: int, payload_size: int = 1024) -> list[dict]:
    blob = "x" * payload_size
    return [{"index": i, "response": f"{blob}-{i}"} for i in range(count)]


@pytest.mark.unit
def test_split_samples_into_chunks_respects_target():
    samples = _make_samples(20, payload_size=32 * 1024)
    chunks = split_samples_into_chunks(samples, target_bytes=128 * 1024)
    assert len(chunks) > 1
    for chunk in chunks:
        chunk_bytes = sum(len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)) for sample in chunk)
        assert chunk_bytes <= 128 * 1024 or len(chunk) == 1


@pytest.mark.unit
def test_write_and_load_round_trip(tmp_path):
    samples = _make_samples(8, payload_size=16 * 1024)
    local_dir = tmp_path / "rollout_0"
    _write_rollout_dump(
        local_dir=local_dir,
        rollout_id=0,
        evaluation=False,
        samples=samples,
        chunk_bytes=64 * 1024,
    )
    manifest = json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == len(samples)
    assert len(manifest["parts"]) >= 1
    loaded = load_rollout_dump(local_dir)
    assert loaded == samples


@pytest.mark.unit
def test_load_legacy_single_file(tmp_path):
    samples = [{"index": 0, "response": "hello"}]
    legacy_path = tmp_path / "rollout_0.pt"
    torch.save({"rollout_id": 0, "samples": samples}, legacy_path)
    assert load_rollout_dump(legacy_path) == samples


@pytest.mark.unit
def test_resolve_rollout_dump_load_path_supports_legacy_template(tmp_path):
    template = str(tmp_path / "rollout_{rollout_id}.pt")
    resolved = resolve_rollout_dump_load_path(template, rollout_id=3)
    assert resolved == tmp_path / "rollout_3"


@pytest.mark.unit
def test_async_rollout_dumper_gpfs_copy_best_effort(tmp_path):
    local_root = tmp_path / "local"
    gpfs_root = tmp_path / "gpfs"
    dumper = AsyncRolloutDumper(local_root=local_root, gpfs_root=gpfs_root, chunk_bytes=DEFAULT_CHUNK_BYTES)
    samples = _make_samples(2, payload_size=1024)
    dumper.enqueue(
        RolloutDumpJob(
            rollout_key="0",
            rollout_id=0,
            evaluation=False,
            samples=samples,
        )
    )
    dumper.close()

    local_dir = local_root / "rollout_0"
    gpfs_dir = gpfs_root / "rollout_0"
    assert (local_dir / "manifest.json").is_file()
    assert load_rollout_dump(local_dir) == samples
    assert (gpfs_dir / "manifest.json").is_file()
    assert load_rollout_dump(gpfs_dir) == samples


@pytest.mark.unit
def test_async_rollout_dumper_gpfs_failure_is_non_fatal(tmp_path):
    local_root = tmp_path / "local"
    gpfs_root = tmp_path / "gpfs"
    dumper = AsyncRolloutDumper(local_root=local_root, gpfs_root=gpfs_root, chunk_bytes=DEFAULT_CHUNK_BYTES)
    samples = [{"index": 0, "response": "ok"}]
    with mock.patch("slime.utils.rollout_dump_utils.shutil.copy2", side_effect=OSError("gpfs down")):
        dumper.enqueue(
            RolloutDumpJob(
                rollout_key="1",
                rollout_id=1,
                evaluation=False,
                samples=samples,
            )
        )
        dumper.close()
    assert load_rollout_dump(local_root / "rollout_1") == samples


@pytest.mark.unit
def test_rollout_dump_exists_supports_file_and_directory(tmp_path):
    samples = [{"index": 0}]
    legacy_path = tmp_path / "legacy.pt"
    torch.save({"rollout_id": 0, "samples": samples}, legacy_path)
    assert rollout_dump_exists(legacy_path)

    local_dir = tmp_path / "rollout_0"
    _write_rollout_dump(
        local_dir=local_dir,
        rollout_id=0,
        evaluation=False,
        samples=samples,
        chunk_bytes=DEFAULT_CHUNK_BYTES,
    )
    assert rollout_dump_exists(local_dir)
    assert rollout_dump_exists(tmp_path / "rollout_0.pt")
    assert not rollout_dump_exists(tmp_path / "missing_rollout_99.pt")


@pytest.mark.unit
def test_async_rollout_dumper_drops_when_local_queue_full(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    local_root = tmp_path / "local"
    dumper = AsyncRolloutDumper(local_root=local_root, gpfs_root=None, chunk_bytes=DEFAULT_CHUNK_BYTES)
    samples = [{"index": 0, "response": "x"}]
    for rollout_id in range(LOCAL_QUEUE_MAXSIZE + 1):
        dumper.enqueue(
            RolloutDumpJob(
                rollout_key=str(rollout_id),
                rollout_id=rollout_id,
                evaluation=False,
                samples=samples,
            )
        )
    dumper.close()
    assert any("local queue full" in record.message for record in caplog.records)
    assert len(list(local_root.iterdir())) == LOCAL_QUEUE_MAXSIZE


@pytest.mark.unit
def test_async_rollout_dumper_skips_gpfs_copy_when_queue_full(tmp_path, caplog):
    import logging
    from queue import Full

    caplog.set_level(logging.WARNING)
    local_root = tmp_path / "local"
    gpfs_root = tmp_path / "gpfs"
    dumper = AsyncRolloutDumper(local_root=local_root, gpfs_root=gpfs_root, chunk_bytes=64 * 1024)
    samples = _make_samples(4, payload_size=32 * 1024)
    real_put = dumper._gpfs_queue.put_nowait
    calls = {"n": 0}

    def limited_put(item):
        calls["n"] += 1
        if calls["n"] > 1:
            raise Full()
        return real_put(item)

    dumper._gpfs_queue.put_nowait = limited_put
    dumper.enqueue(
        RolloutDumpJob(
            rollout_key="0",
            rollout_id=0,
            evaluation=False,
            samples=samples,
        )
    )
    dumper.close()
    assert any("GPFS queue full" in record.message for record in caplog.records)
    assert load_rollout_dump(local_root / "rollout_0") == samples
    copied_parts = list((gpfs_root / "rollout_0").glob("*"))
    assert len(copied_parts) < len(list((local_root / "rollout_0").glob("*")))


@pytest.mark.unit
def test_resolve_rollout_dump_local_root_falls_back_to_tempdir(tmp_path):
    class Args:
        rollout_dump_local_dir = None
        save = str(tmp_path / "run")

    with mock.patch("slime.utils.rollout_dump_utils._path_is_writable", return_value=False):
        root = resolve_rollout_dump_local_root(Args())
    assert str(root).startswith(__import__("tempfile").gettempdir())
