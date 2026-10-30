#!/usr/bin/env python
"""Stream one UnrealEgo2 environment into unified records, keeping PNGs on disk.

The official rgb parts are one gzip-compressed tar split into ~10 GB chunks. This reads
them as a single stream and writes one sequence of PNGs at a time under ``work_root/png``
(on disk, not tmpfs). A small process pool turns that sequence into 384^2 records, then
the PNGs are deleted. Pose json for the environment is extracted once (one small part).

Compressed parts are left in place until the caller deletes them after a successful exit,
so an interrupt resumes without re-downloading. Exit 0 means every split sequence of this
environment is in the index. Exit 2 means free space dropped below ``--min-free-gb``;
parts are kept. Anything else is a failure and parts are kept.

Sequences in the tar are contiguous (checked on the UnrealEgo-RW parts of the same
packaging). A sequence that resumes after another one aborts the environment and drops
that sequence from the index, instead of writing a short record.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import traceback
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Hide the training GPUs before torch is imported (preprocess imports torch).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import preprocess_unrealego_raw as raw  # noqa: E402
from egorayformer.data.records import seq_id_to_dirname  # noqa: E402

IMG_SUBDIR = raw.IMG_SUBDIR
JSON_SUBDIR = raw.JSON_SUBDIR
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_SEQ_BYTES = 8 * 1024 * 1024 * 1024
_FLUSH_EVERY = 20


class StagingError(RuntimeError):
    """A data or usage problem. Parts are kept and the traceback is not useful."""


class DiskLow(RuntimeError):
    """Free space fell below the margin. Parts are kept; rerun to resume."""


class ConcatFiles(io.RawIOBase):
    """Read several files in order as one byte stream. Used to rejoin split tar.gz parts."""

    def __init__(self, paths: Sequence[str]):
        self._paths = [os.fspath(p) for p in paths]
        if not self._paths:
            raise ValueError("no archive parts")
        self._i = 0
        self._fh: Optional[io.BufferedReader] = open(self._paths[0], "rb")

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunks: List[bytes] = []
            while True:
                blob = self.read(1024 * 1024)
                if not blob:
                    return b"".join(chunks)
                chunks.append(blob)
        if size == 0:
            return b""
        out = bytearray()
        while len(out) < size and self._fh is not None:
            blob = self._fh.read(size - len(out))
            if blob:
                out += blob
                continue
            self._fh.close()
            self._i += 1
            if self._i >= len(self._paths):
                self._fh = None
            else:
                self._fh = open(self._paths[self._i], "rb")
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def _parts(name: str) -> List[str]:
    name = name.lstrip("./")
    if not name or name.startswith("/") or "\\" in name:
        raise StagingError(f"refusing archive member {name!r}")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise StagingError(f"refusing archive member {name!r}")
    return parts


def sequence_id(member_name: str, env: str, marker: str) -> Optional[str]:
    """Path from ``env`` up to (not including) ``marker``, or None when the member is unrelated."""
    parts = _parts(member_name)
    if marker not in parts or env not in parts:
        return None
    head = parts[: parts.index(marker)]
    if env not in head:
        return None
    seq = "/".join(head[head.index(env):])
    if not seq:
        return None
    return seq


def in_env(seq_id: str, env: str) -> bool:
    return seq_id == env or seq_id.startswith(env + "/")


def pending_download_bytes(parts: Sequence[Tuple[str, str, int]], nominal_bytes: int) -> int:
    """Bytes still to fetch. ``parts`` is ``(directory, env, count)``. A part with its ``.ok_`` marker counts as zero,
    and a partial file counts only for what is missing from ``nominal_bytes``."""
    total = 0
    for directory, env, count in parts:
        for i in range(count):
            part = f"{env}.tar.gz{i:03d}"
            if os.path.isfile(os.path.join(directory, f".ok_{part}")):
                continue
            path = os.path.join(directory, part)
            have = os.path.getsize(path) if os.path.isfile(path) else 0
            total += max(0, nominal_bytes - have)
    return total


def pending_download_gb(parts: Sequence[Tuple[str, str, int]], part_gb: int) -> int:
    nbytes = pending_download_bytes(parts, part_gb * 1024 ** 3)
    if nbytes <= 0:
        return 0
    return (nbytes + 1024 ** 3 - 1) // 1024 ** 3


def list_parts(directory: str, env: str, count: int) -> List[str]:
    paths = []
    for i in range(count):
        part = f"{env}.tar.gz{i:03d}"
        path = os.path.join(directory, part)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise StagingError(f"missing archive part {path}")
        if not os.path.isfile(os.path.join(directory, f".ok_{part}")):
            raise StagingError(f"{path} has no .ok_ marker (download incomplete)")
        paths.append(path)
    return paths


def _open_tar(paths: Sequence[str]) -> Tuple[ConcatFiles, tarfile.TarFile]:
    # tarfile's gzip reader calls read(). A BufferedReader around a pure-Python
    # RawIOBase raises NotImplementedError, so the parts are handed over directly.
    stream = ConcatFiles(paths)
    try:
        tar = tarfile.open(fileobj=stream, mode="r|gz")
    except Exception:
        stream.close()
        raise
    return stream, tar


def _discard_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
    """Read and drop a member body. A streaming tar desynchronises if a body is left unread."""
    if not member.isreg():
        return
    src = tar.extractfile(member)
    if src is not None:
        while src.read(1024 * 1024):
            pass


def _copy_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: str, limit: int) -> int:
    src = tar.extractfile(member)
    if src is None:
        raise StagingError(f"cannot read {member.name}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    total = 0
    try:
        with src, open(dest, "wb") as out:
            while True:
                blob = src.read(1024 * 1024)
                if not blob:
                    break
                total += len(blob)
                if total > limit:
                    raise StagingError(f"{member.name} is larger than {limit} bytes")
                out.write(blob)
    except Exception:
        if os.path.isfile(dest):
            os.remove(dest)
        raise
    return total


def extract_pose(parts: Sequence[str], pose_root: str, env: str) -> int:
    """Extract this environment's json under ``pose_root/<seq>/json``. Returns the file count."""
    os.makedirs(pose_root, exist_ok=True)
    stream, tar = _open_tar(parts)
    n = 0
    try:
        for member in tar:
            if not member.isreg():
                continue
            seq = sequence_id(member.name, env, JSON_SUBDIR) if member.name.endswith(".json") else None
            if seq is None:
                _discard_member(tar, member)
                continue
            tail = _parts(member.name)
            dest = os.path.join(pose_root, seq, *tail[tail.index(JSON_SUBDIR):])
            _assert_inside(pose_root, dest)
            _copy_member(tar, member, dest, MAX_FILE_BYTES)
            n += 1
    finally:
        tar.close()
        stream.close()
    if n == 0:
        raise StagingError(f"pose archive for {env} contains no json")
    return n


def _assert_inside(root: str, path: str) -> None:
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    if path_abs != root_abs and not path_abs.startswith(root_abs + os.sep):
        raise StagingError(f"{path} escapes {root}")


def _load_entries(index_path: str, img_size: int) -> Dict[str, Dict]:
    if not os.path.isfile(index_path):
        return {}
    with open(index_path, "r") as f:
        info = json.load(f)
    have = tuple(info.get("image_size") or [])
    if have != (img_size, img_size):
        raise StagingError(f"{index_path} is {have} px, this run writes {(img_size, img_size)}")
    return {s["id"]: s for s in info.get("sequences", [])}


def _write_index(dir_path: str, split: str, rig: str, img_size: int, orig_size: int, entries: Dict[str, Dict]) -> None:
    os.makedirs(dir_path, exist_ok=True)
    sequences = sorted(entries.values(), key=lambda s: s["id"])
    info = {
        "dataset": "unrealego2",
        "split": split,
        "rig": rig,
        "image_size": [img_size, img_size],
        "orig_size": [orig_size, orig_size],
        "num_views": 2,
        "skeleton": "unrealego16",
        "num_frames": int(sum(s["num_frames"] for s in sequences)),
        "sequences": sequences,
    }
    tmp = os.path.join(dir_path, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(info, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, os.path.join(dir_path, "index.json"))


def _free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


class _InlineResult:
    def __init__(self, value=None, error: Optional[BaseException] = None):
        self._value = value
        self._error = error

    def ready(self) -> bool:
        return True

    def get(self):
        if self._error is not None:
            raise self._error
        return self._value


class _Job:
    def __init__(self, seq_id: str, seq_dir: str, result):
        self.seq_id = seq_id
        self.seq_dir = seq_dir
        self.result = result


class RgbStreamer:
    """Pull sequences out of the rgb tar. At most ``max_inflight`` sequence directories exist at once."""

    def __init__(self, png_root, pose_root, split_of, entries, split_dirs, img_size, orig_size, max_inflight, min_free_gb, pool):
        self.png_root = png_root
        self.split_of = split_of
        self.entries = entries
        self.split_dirs = split_dirs
        self.img_size = img_size
        self.orig_size = orig_size
        self.max_inflight = max_inflight
        self.min_free_gb = min_free_gb
        self.pool = pool
        self.pending: List[_Job] = []
        self.seen: Set[str] = set()
        self.current: Optional[str] = None
        self.current_dir: Optional[str] = None
        self.current_skip = False
        self.current_bytes = 0
        self.live_dirs = 0
        self.peak_dirs = 0
        self.processed = 0
        self.skipped = 0
        self.frames = 0
        self._since_flush = 0
        self._stopped = False
        raw._init_worker(dict(
            dataset="unrealego2", rgb_root=png_root, pose_root=pose_root,
            out_split=next(iter(split_dirs.values())),
            out_splits={seq: split_dirs[sp] for seq, sp in split_of.items()},
            img_size=img_size, rig="unrealego2",
        ))

    def _note_live(self, delta: int) -> None:
        self.live_dirs += delta
        if self.live_dirs > self.peak_dirs:
            self.peak_dirs = self.live_dirs

    def _ensure_free(self) -> None:
        self.flush()
        free = _free_gb(self.png_root if os.path.isdir(self.png_root) else os.path.dirname(self.png_root))
        if free < self.min_free_gb:
            raise DiskLow(f"free space {free:.0f}G is below {self.min_free_gb:.0f}G; stopping with parts kept")

    def _wait_slot(self) -> None:
        while len(self.pending) >= self.max_inflight:
            self._reap(block=True)

    def _reap(self, block: bool) -> bool:
        if not self.pending:
            return False
        job = self.pending[0]
        if not block and not job.result.ready():
            return False
        self.pending.pop(0)
        try:
            seq_id, dirname, n, err = job.result.get()
        finally:
            if os.path.isdir(job.seq_dir):
                shutil.rmtree(job.seq_dir)
            self._note_live(-1)
        if seq_id != job.seq_id:
            raise StagingError(f"worker returned {seq_id}, expected {job.seq_id}")
        if self._stopped:
            return True
        if n <= 0:
            raise StagingError(f"{seq_id} produced no frames")
        split = self.split_of[seq_id]
        self.entries[split][seq_id] = {"id": seq_id, "dir": dirname, "num_frames": int(n)}
        self.processed += 1
        self.frames += int(n)
        self._since_flush += 1
        if err is not None and self.processed <= 3:
            print(f"[stream] {seq_id}: json-vs-projected pts2d mean abs err {err:.2f} px")
        if self._since_flush >= _FLUSH_EVERY:
            self.flush()
        return True

    def _submit(self, seq_id: str, seq_dir: str) -> None:
        if self.pool is None:
            try:
                value = raw.process_sequence(seq_id)
                result = _InlineResult(value)
            except Exception as exc:  # recorded by _reap / raised from get()
                result = _InlineResult(error=exc)
            self.pending.append(_Job(seq_id, seq_dir, result))
            self._reap(block=True)  # one worker: free the directory before the next sequence
            return
        self.pending.append(_Job(seq_id, seq_dir, self.pool.apply_async(raw.process_sequence, (seq_id,))))

    def _drop(self, seq_id: str) -> None:
        split = self.split_of.get(seq_id)
        if split is None:
            return
        ent = self.entries[split].pop(seq_id, None)
        dirname = ent["dir"] if ent else seq_id_to_dirname(seq_id)
        shutil.rmtree(os.path.join(self.split_dirs[split], dirname), ignore_errors=True)
        self.flush()

    def begin(self, seq_id: str) -> None:
        if seq_id in self.seen or seq_id == self.current:
            self._stopped = True
            self.abort()
            self._drop(seq_id)
            raise StagingError(f"{seq_id} is not contiguous in the rgb archive; dropped it and stopped")
        self._ensure_free()
        self._wait_slot()
        self.current = seq_id
        self.current_bytes = 0
        self.current_dir = None
        known = seq_id in self.split_of
        already = known and seq_id in self.entries[self.split_of[seq_id]]
        self.current_skip = (not known) or already
        if self.current_skip:
            self.skipped += 1
            return
        seq_dir = os.path.join(self.png_root, seq_id)
        _assert_inside(self.png_root, seq_dir)
        os.makedirs(seq_dir, exist_ok=True)
        self.current_dir = seq_dir
        self._note_live(1)

    def add_file(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        if self.current is None or self.current_skip:
            _discard_member(tar, member)
            return
        if not member.isfile():
            return
        tail = _parts(member.name)
        marker_at = tail.index(IMG_SUBDIR)
        dest = os.path.join(self.png_root, self.current, *tail[marker_at:])
        _assert_inside(self.png_root, dest)
        self.current_bytes += _copy_member(tar, member, dest, MAX_FILE_BYTES)
        if self.current_bytes > MAX_SEQ_BYTES:
            raise StagingError(f"{self.current} exceeds {MAX_SEQ_BYTES} bytes of PNGs")

    def finish(self) -> None:
        seq, directory, skip = self.current, self.current_dir, self.current_skip
        self.current = None
        self.current_dir = None
        if seq is None:
            return
        self.seen.add(seq)
        if skip or directory is None:
            return
        self._submit(seq, directory)

    def drain(self) -> None:
        self.finish()
        while self.pending:
            self._reap(block=True)

    def abort(self) -> None:
        pool = self.pool
        self.pool = None
        self.pending.clear()
        if pool is not None:
            pool.terminate()
            pool.join()

    def close_pool(self) -> None:
        pool = self.pool
        self.pool = None
        if pool is not None:
            pool.close()
            pool.join()

    def flush(self) -> None:
        for split, dir_path in self.split_dirs.items():
            if not self.entries[split] and not os.path.isfile(os.path.join(dir_path, "index.json")):
                continue
            _write_index(dir_path, split, "unrealego2", self.img_size, self.orig_size, self.entries[split])
        self._since_flush = 0

    def consume(self, part_paths: Sequence[str], env: str) -> None:
        os.makedirs(self.png_root, exist_ok=True)
        stream, tar = _open_tar(part_paths)
        try:
            for member in tar:
                if not member.isreg() and not member.isdir():
                    if member.issym() or member.islnk():
                        raise StagingError(f"refusing link member {member.name}")
                    continue
                seq = sequence_id(member.name, env, IMG_SUBDIR)
                if seq is None:
                    _discard_member(tar, member)
                    continue
                if not in_env(seq, env):
                    raise StagingError(f"member {member.name} is outside {env}")
                if seq != self.current:
                    self.finish()
                    self.begin(seq)
                if member.isreg():
                    self.add_file(tar, member)
        finally:
            tar.close()
            stream.close()
        self.drain()


def _npy_frames(path: str) -> int:
    arr = np.load(path, mmap_mode="r")
    try:
        return int(arr.shape[0]) if arr.ndim >= 1 else -1
    finally:
        mmap = getattr(arr, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _record_ok(split_dir: str, ent: Dict) -> bool:
    """The index entry has both view files and meta, and the stored length matches ``num_frames``."""
    try:
        n = int(ent.get("num_frames") or 0)
    except (TypeError, ValueError):
        return False
    if n <= 0 or not ent.get("dir"):
        return False
    seq_dir = os.path.join(split_dir, ent["dir"])
    meta = os.path.join(seq_dir, "meta.npz")
    if not os.path.isfile(meta) or os.path.getsize(meta) <= 0:
        return False
    for name in ("left.npy", "right.npy"):
        path = os.path.join(seq_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            return False
        if _npy_frames(path) != n:
            return False
    return True


def records_complete(out_root: str, split_dir: str, splits: Sequence[str], env: str, img_size: int) -> bool:
    """Every split sequence of ``env`` has a matching record under ``out_root``."""
    try:
        split_of = _split_map(split_dir, splits, env)
    except StagingError:
        return False
    entries: Dict[str, Dict[str, Dict]] = {}
    for split in splits:
        try:
            entries[split] = _load_entries(os.path.join(out_root, "unrealego2", split, "index.json"), img_size)
        except (StagingError, OSError, json.JSONDecodeError):
            return False
    for seq, split in split_of.items():
        ent = entries.get(split, {}).get(seq)
        if ent is None or not _record_ok(os.path.join(out_root, "unrealego2", split), ent):
            return False
    return True


def resume_decision(marker_path: str, img_size: int, out_root: str, split_dir: str, splits: Sequence[str], env: str) -> str:
    """``skip`` only when this output directory already holds the environment. ``size_mismatch`` when the marker
    names this output at a different resolution. Otherwise ``reprocess``."""
    out_root = os.path.abspath(out_root)
    if os.path.isfile(marker_path):
        lines = open(marker_path).read().splitlines()
        size = lines[0].strip() if lines else ""
        path = lines[1].strip() if len(lines) > 1 else ""
        same_output = (not path) or os.path.abspath(path) == out_root
        if same_output and size and size != str(img_size):
            return "size_mismatch"
    if records_complete(out_root, split_dir, splits, env, img_size):
        return "skip"
    return "reprocess"


def _split_map(split_dir: str, splits: Sequence[str], env: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for split in splits:
        path = os.path.join(split_dir, f"{split}.txt")
        if not os.path.isfile(path):
            raise StagingError(f"missing split file {path}")
        for seq in raw.parse_split_file(path):
            if not in_env(seq, env):
                continue
            if seq in found and found[seq] != split:
                raise StagingError(f"{seq} is listed in both {found[seq]} and {split}")
            found[seq] = split
    if not found:
        raise StagingError(f"no sequences for {env} in {', '.join(splits)}")
    return found


def _separate(path: str, other: str, what: str) -> None:
    path_abs, other_abs = os.path.abspath(path), os.path.abspath(other)
    if path_abs == other_abs or path_abs.startswith(other_abs + os.sep) or other_abs.startswith(path_abs + os.sep):
        raise StagingError(f"{what} ({path}) must not overlap {other}")


def process_environment(
    *,
    env: str,
    rgb_dir: str,
    pose_dir: str,
    rgb_count: int,
    pose_count: int,
    split_dir: str,
    splits: Sequence[str],
    out_root: str,
    work_root: str,
    img_size: int = 384,
    orig_size: int = 1024,
    workers: int = 2,
    max_inflight: int = 2,
    min_free_gb: float = 50,
) -> int:
    """Process one environment. See the module docstring for exit codes. Stats land on ``process_environment.stats``."""
    stats = {"processed": 0, "skipped": 0, "peak_dirs": 0}
    process_environment.stats = stats
    if workers < 1 or max_inflight < 1:
        raise StagingError("workers and max_inflight must be >= 1")
    os.makedirs(work_root, exist_ok=True)
    lock_fh = open(os.path.join(work_root, ".lock"), "a")
    try:
        try:
            import fcntl
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another process is staging in {work_root}", file=sys.stderr)
            return 1
        try:
            code = _process_locked(
                env=env, rgb_dir=rgb_dir, pose_dir=pose_dir, rgb_count=rgb_count, pose_count=pose_count,
                split_dir=split_dir, splits=splits, out_root=out_root, work_root=work_root, img_size=img_size,
                orig_size=orig_size, workers=workers, max_inflight=max_inflight, min_free_gb=min_free_gb, stats=stats,
            )
        except DiskLow as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except StagingError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception:
            traceback.print_exc()
            return 1
        return code
    finally:
        lock_fh.close()


def _process_locked(*, env, rgb_dir, pose_dir, rgb_count, pose_count, split_dir, splits, out_root, work_root,
                    img_size, orig_size, workers, max_inflight, min_free_gb, stats) -> int:
    split_of = _split_map(split_dir, splits, env)
    split_dirs = {sp: os.path.join(out_root, "unrealego2", sp) for sp in splits}
    for directory in split_dirs.values():
        _separate(work_root, directory, "work_root")
    entries = {sp: _load_entries(os.path.join(split_dirs[sp], "index.json"), img_size) for sp in splits}
    wanted = set(split_of)
    for split, dir_path in split_dirs.items():
        kept = {}
        for seq_id, ent in entries[split].items():
            if seq_id in wanted and not _record_ok(dir_path, ent):
                print(f"[stream] {seq_id} is indexed but its records are incomplete; it will be processed again")
                continue
            kept[seq_id] = ent
        entries[split] = kept
    done = {seq for seq in wanted if seq in entries[split_of[seq]]}
    png_root = os.path.join(work_root, "png", env)
    pose_root = os.path.join(work_root, "pose")
    _separate(png_root, out_root, "png scratch")
    if wanted <= done:
        print(f"[stream] {env}: all {len(wanted)} sequences already in the index")
        stats.update(processed=0, skipped=len(wanted), peak_dirs=0)
        shutil.rmtree(png_root, ignore_errors=True)
        shutil.rmtree(pose_root, ignore_errors=True)
        return 0

    if _free_gb(work_root) < min_free_gb:
        raise DiskLow(f"free space {_free_gb(work_root):.0f}G is below {min_free_gb:.0f}G; not starting {env}")

    pose_parts = list_parts(pose_dir, env, pose_count)
    rgb_parts = list_parts(rgb_dir, env, rgb_count)
    marker = os.path.join(pose_root, f".complete_{env}")
    if not os.path.isfile(marker):
        if os.path.isdir(pose_root):
            shutil.rmtree(pose_root)
        n_json = extract_pose(pose_parts, pose_root, env)
        with open(marker, "w") as f:
            f.write(f"{n_json}\n")
        print(f"[stream] {env}: extracted {n_json} pose json files")

    if os.path.isdir(png_root):
        shutil.rmtree(png_root)
    os.makedirs(png_root, exist_ok=True)
    pool = None
    streamer: Optional[RgbStreamer] = None
    success = False
    try:
        if workers > 1:
            pool = Pool(workers, initializer=raw._init_worker, initargs=(dict(
                dataset="unrealego2", rgb_root=png_root, pose_root=pose_root,
                out_split=split_dirs[splits[0]],
                out_splits={seq: split_dirs[sp] for seq, sp in split_of.items()},
                img_size=img_size, rig="unrealego2",
            ),), maxtasksperchild=32)
        streamer = RgbStreamer(
            png_root, pose_root, split_of, entries, split_dirs, img_size, orig_size, max_inflight, min_free_gb, pool,
        )
        print(f"[stream] {env}: {len(rgb_parts)} rgb parts, {len(wanted)} sequences "
              f"({len(done)} already done), inflight {max_inflight}")
        streamer.consume(rgb_parts, env)
        missing = sorted(wanted - streamer.seen)
        if missing:
            raise StagingError(f"{len(missing)} split sequences were not in the rgb archive (e.g. {missing[0]})")
        streamer.flush()
        stats.update(processed=streamer.processed, skipped=streamer.skipped, peak_dirs=streamer.peak_dirs)
        print(f"[stream] {env}: wrote {streamer.processed} sequences, {streamer.frames} frames, "
              f"skipped {streamer.skipped}, peak {streamer.peak_dirs} sequence dirs")
        success = True
        return 0
    except DiskLow:
        if streamer is not None:
            streamer.drain()
        raise
    finally:
        if streamer is not None:
            stats.update(processed=streamer.processed, skipped=streamer.skipped, peak_dirs=streamer.peak_dirs)
            if not success and streamer.pool is not None:
                streamer.abort()
            else:
                streamer.close_pool()
            try:
                streamer.flush()
            except Exception:
                traceback.print_exc()
        elif pool is not None:
            pool.terminate()
            pool.join()
        shutil.rmtree(png_root, ignore_errors=True)
        if success and os.path.isdir(pose_root):
            shutil.rmtree(pose_root)
        elif os.path.isfile(marker):
            # The parts may be replaced before the next run. Drop the marker so pose json is extracted again.
            os.remove(marker)


def _pending_gb_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--rgb-count", type=int, required=True)
    ap.add_argument("--pose-count", type=int, required=True)
    ap.add_argument("--part-gb", type=int, required=True)
    args = ap.parse_args(list(argv))
    print(pending_download_gb([
        (args.rgb_dir, args.env, args.rgb_count),
        (args.pose_dir, args.env, args.pose_count),
    ], args.part_gb))
    return 0


def _resume_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    args = ap.parse_args(list(argv))
    print(resume_decision(args.marker, args.img_size, args.out_root, args.split_dir, args.splits, args.env))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--pending-gb":
        return _pending_gb_main(argv[1:])
    if argv and argv[0] == "--resume-decision":
        return _resume_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--pose-dir", required=True)
    ap.add_argument("--rgb-count", type=int, required=True)
    ap.add_argument("--pose-count", type=int, required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--orig-size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-inflight", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=50)
    args = ap.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    return process_environment(
        env=args.env, rgb_dir=args.rgb_dir, pose_dir=args.pose_dir, rgb_count=args.rgb_count,
        pose_count=args.pose_count, split_dir=args.split_dir, splits=args.splits, out_root=args.out_root,
        work_root=args.work_root, img_size=args.img_size, orig_size=args.orig_size, workers=args.workers,
        max_inflight=args.max_inflight, min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    sys.exit(main())
