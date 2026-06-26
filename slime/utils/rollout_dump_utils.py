import json
import logging
import os
import pickle
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue

import torch

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024
MANIFEST_FILENAME = "manifest.json"
LOCAL_QUEUE_MAXSIZE = 4
GPFS_QUEUE_MAXSIZE = 256

_scratch_fallback_warned = False


@dataclass(frozen=True)
class RolloutDumpJob:
    rollout_key: str
    rollout_id: int
    evaluation: bool
    samples: list[dict]


def _estimate_sample_bytes(sample: dict) -> int:
    return len(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))


def split_samples_into_chunks(samples: list[dict], target_bytes: int) -> list[list[dict]]:
    if not samples:
        return []
    if target_bytes <= 0:
        return [samples]

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    for sample in samples:
        sample_size = _estimate_sample_bytes(sample)
        if current and current_size + sample_size > target_bytes:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(sample)
        current_size += sample_size
    if current:
        chunks.append(current)
    return chunks


def _rollout_dir_name(rollout_key: str) -> str:
    return f"rollout_{rollout_key}"


def resolve_rollout_dump_local_root(args) -> Path:
    if getattr(args, "rollout_dump_local_dir", None):
        return Path(args.rollout_dump_local_dir)
    user = os.environ.get("USER", "slime")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        run_tag = job_id
    elif getattr(args, "save", None):
        run_tag = os.path.basename(os.path.normpath(args.save))
    else:
        run_tag = "local"
    root = Path("/scratch") / user / "slime_rollout_dumps" / run_tag
    if not _path_is_writable(root):
        global _scratch_fallback_warned
        if not _scratch_fallback_warned:
            logger.warning("Rollout dump local root %s is not writable; falling back to tempdir", root)
            _scratch_fallback_warned = True
        root = Path(tempfile.gettempdir()) / "slime_rollout_dumps" / run_tag
    return root


def resolve_rollout_dump_gpfs_root(args) -> Path | None:
    if getattr(args, "rollout_dump_gpfs_dir", None):
        return Path(args.rollout_dump_gpfs_dir)
    if getattr(args, "save", None):
        return Path(args.save) / "rollout_dumps"
    return None


def rollout_dump_dir(local_root: Path, rollout_key: str) -> Path:
    return local_root / _rollout_dir_name(rollout_key)


def _path_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _write_manifest(path: Path, *, rollout_id: int, evaluation: bool, parts: list[str], sample_count: int) -> None:
    manifest = {
        "rollout_id": rollout_id,
        "evaluation": evaluation,
        "sample_count": sample_count,
        "parts": parts,
    }
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_rollout_dump(
    *,
    local_dir: Path,
    rollout_id: int,
    evaluation: bool,
    samples: list[dict],
    chunk_bytes: int,
) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_samples_into_chunks(samples, chunk_bytes)
    part_names: list[str] = []
    written_paths: list[Path] = []

    for part_index, chunk in enumerate(chunks):
        part_name = f"part_{part_index:03d}.pt"
        part_path = local_dir / part_name
        _atomic_torch_save(
            {"rollout_id": rollout_id, "part_index": part_index, "samples": chunk},
            part_path,
        )
        part_names.append(part_name)
        written_paths.append(part_path)

    manifest_path = local_dir / MANIFEST_FILENAME
    _write_manifest(
        manifest_path,
        rollout_id=rollout_id,
        evaluation=evaluation,
        parts=part_names,
        sample_count=len(samples),
    )
    written_paths.append(manifest_path)
    return written_paths


def load_rollout_dump(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_file():
        blob = torch.load(path, weights_only=False)
        return blob["samples"]

    if not path.is_dir():
        legacy_pt = path.with_suffix(".pt")
        if legacy_pt.is_file():
            blob = torch.load(legacy_pt, weights_only=False)
            return blob["samples"]
        raise FileNotFoundError(f"Rollout dump not found at {path} or {legacy_pt}")

    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Rollout dump manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list[dict] = []
    for part_name in manifest["parts"]:
        part_blob = torch.load(path / part_name, weights_only=False)
        samples.extend(part_blob["samples"])
    return samples


def rollout_dump_exists(path: Path) -> bool:
    path = Path(path)
    if path.is_file():
        return True
    if path.is_dir():
        return (path / MANIFEST_FILENAME).is_file()
    legacy_pt = path.with_suffix(".pt")
    if legacy_pt.is_file():
        return True
    if path.suffix == ".pt":
        return (path.with_suffix("") / MANIFEST_FILENAME).is_file()
    return False


def resolve_rollout_dump_load_path(template: str, rollout_id: int, *, evaluation: bool = False) -> Path:
    rollout_key = ("eval_" if evaluation else "") + str(rollout_id)
    if "{rollout_id}" in template:
        resolved = template.format(rollout_id=rollout_key)
    else:
        resolved = template
    path = Path(resolved)
    if path.suffix == ".pt" and not path.is_file():
        return path.with_suffix("")
    return path


class AsyncRolloutDumper:
    def __init__(
        self,
        *,
        local_root: Path,
        gpfs_root: Path | None,
        chunk_bytes: int,
    ) -> None:
        self._local_root = local_root
        self._gpfs_root = gpfs_root
        self._chunk_bytes = chunk_bytes
        self._local_queue: Queue[RolloutDumpJob | None] = Queue(maxsize=LOCAL_QUEUE_MAXSIZE)
        self._gpfs_queue: Queue[tuple[Path, Path] | None] = Queue(maxsize=GPFS_QUEUE_MAXSIZE)
        self._local_thread = threading.Thread(target=self._local_writer_loop, name="rollout-dump-local", daemon=True)
        self._gpfs_thread = threading.Thread(target=self._gpfs_sync_loop, name="rollout-dump-gpfs", daemon=True)
        self._local_thread.start()
        if self._gpfs_root is not None:
            self._gpfs_thread.start()
        else:
            logger.warning("Rollout dump GPFS sync disabled because args.save is unset")

    @classmethod
    def from_args(cls, args) -> "AsyncRolloutDumper":
        return cls(
            local_root=resolve_rollout_dump_local_root(args),
            gpfs_root=resolve_rollout_dump_gpfs_root(args),
            chunk_bytes=getattr(args, "rollout_dump_chunk_bytes", DEFAULT_CHUNK_BYTES),
        )

    def enqueue(self, job: RolloutDumpJob) -> None:
        try:
            self._local_queue.put_nowait(job)
        except Full:
            logger.warning(
                "Rollout dump local queue full; dropping rollout_id=%s evaluation=%s (%s samples)",
                job.rollout_id,
                job.evaluation,
                len(job.samples),
            )

    def close(self, *, drain_timeout_s: float = 30.0) -> None:
        try:
            self._local_queue.put(None, timeout=drain_timeout_s)
        except Full:
            logger.warning("Rollout dump local queue full during close; GPFS copies may be incomplete")
            self._local_queue.put(None)
        self._local_thread.join(timeout=drain_timeout_s)
        if self._gpfs_root is not None:
            try:
                self._gpfs_queue.put(None, timeout=drain_timeout_s)
            except Full:
                self._gpfs_queue.put(None)
            self._gpfs_thread.join(timeout=drain_timeout_s)

    def _local_writer_loop(self) -> None:
        while True:
            job = self._local_queue.get()
            try:
                if job is None:
                    if self._gpfs_root is not None:
                        self._gpfs_queue.put(None)
                    return
                self._write_job(job)
            finally:
                self._local_queue.task_done()

    def _write_job(self, job: RolloutDumpJob) -> None:
        local_dir = rollout_dump_dir(self._local_root, job.rollout_key)
        logger.info(
            "Enqueue rollout dump write local_dir=%s rollout_id=%s samples=%s",
            local_dir,
            job.rollout_id,
            len(job.samples),
        )
        written_paths = _write_rollout_dump(
            local_dir=local_dir,
            rollout_id=job.rollout_id,
            evaluation=job.evaluation,
            samples=job.samples,
            chunk_bytes=self._chunk_bytes,
        )
        if self._gpfs_root is None:
            return
        gpfs_dir = self._gpfs_root / _rollout_dir_name(job.rollout_key)
        for local_path in written_paths:
            gpfs_path = gpfs_dir / local_path.name
            try:
                self._gpfs_queue.put_nowait((local_path, gpfs_path))
            except Full:
                logger.warning("Rollout dump GPFS queue full; skipping copy for %s", local_path)

    def _gpfs_sync_loop(self) -> None:
        while True:
            item = self._gpfs_queue.get()
            try:
                if item is None:
                    return
                local_path, gpfs_path = item
                try:
                    gpfs_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path, gpfs_path)
                except OSError as exc:
                    logger.warning("Best-effort rollout dump GPFS copy failed %s -> %s: %s", local_path, gpfs_path, exc)
            finally:
                self._gpfs_queue.task_done()
