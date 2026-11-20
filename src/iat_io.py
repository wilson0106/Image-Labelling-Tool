#!/usr/bin/env python3
"""I/O helpers for IAT exports.

The helpers in this module are intentionally hardware-agnostic.  File copying is
an I/O workload; CUDA/GPU acceleration does not improve it.  A small bounded
thread pool overlaps filesystem latency without assuming SSD, RAID, or any
specific storage technology.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from collections.abc import Sized
from itertools import islice
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class CopyJob:
    src: Path
    dest: Path
    token: object = None


@dataclass
class CopyReport:
    requested: int
    copied: int = 0
    failed: int = 0
    bytes_copied: int = 0
    elapsed_seconds: float = 0.0
    workers: int = 1
    succeeded_tokens: set = field(default_factory=set)
    failed_tokens: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    max_in_flight: int = 0

    @property
    def files_per_second(self) -> float:
        return self.copied / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def mib_per_second(self) -> float:
        return (self.bytes_copied / (1024.0 * 1024.0)) / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def recommended_io_workers() -> int:
    """Return a conservative cross-platform worker count for file copies.

    Two workers are a safe default for unknown storage: enough to overlap metadata
    latency on SSD/network filesystems without flooding mechanical disks.  Power
    users can override it with ILT_EXPORT_WORKERS.
    """
    raw = os.environ.get("ILT_EXPORT_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(8, int(raw)))
        except ValueError:
            pass
    return 2


def _copy_one(job: CopyJob, preserve_metadata: bool = True):
    size = 0
    try:
        size = int(job.src.stat().st_size)
    except OSError:
        pass
    if preserve_metadata:
        shutil.copy2(job.src, job.dest)
    else:
        shutil.copyfile(job.src, job.dest)
    return job.token, size


def copy_jobs(
    jobs: Iterable[CopyJob],
    *,
    progress: Optional[Callable[[int, int, str], None]] = None,
    workers: Optional[int] = None,
    preserve_metadata: bool = True,
    progress_interval_seconds: float = 0.12,
    chunk_size: Optional[int] = None,
    max_pending_factor: int = 4,
) -> CopyReport:
    """Copy files with bounded memory, concurrency and progress reporting.

    The previous implementation sorted and submitted every job at once.  That is
    harmless for small exports but creates one ``Future`` per image on large
    projects.  This implementation consumes the iterable in locality-sorted
    chunks and keeps only a small window of futures in flight.

    ``chunk_size`` controls how many jobs are materialized/sorted at a time.
    ``max_pending_factor`` bounds submitted-but-not-yet-consumed futures to a
    small multiple of the worker count.  Neither setting changes file contents
    or collision decisions made by callers.
    """
    try:
        known_total = len(jobs) if isinstance(jobs, Sized) else None
    except Exception:
        known_total = None

    requested_hint = max(0, int(known_total or 0))
    worker_count = recommended_io_workers() if workers is None else max(1, int(workers))
    if known_total is not None and known_total > 0:
        worker_count = min(worker_count, int(known_total))
    window = max(worker_count, worker_count * max(1, int(max_pending_factor)))
    local_chunk = int(chunk_size or max(64, window * 4))
    local_chunk = max(window, local_chunk)

    report = CopyReport(requested=requested_hint, workers=worker_count)
    iterator = iter(jobs)
    start = time.perf_counter()
    done_count = 0
    seen_count = 0
    last_progress = 0.0

    def current_total() -> int:
        if known_total is not None:
            return int(known_total)
        return max(seen_count, done_count)

    def notify(text="", force=False):
        nonlocal last_progress
        if progress is None:
            return
        now = time.perf_counter()
        total = current_total()
        if force or (now - last_progress) >= progress_interval_seconds:
            progress(done_count, total, text)
            last_progress = now

    def mark_success(job: CopyJob, size: int):
        nonlocal done_count
        done_count += 1
        report.copied += 1
        report.bytes_copied += int(size or 0)
        if job.token is not None:
            report.succeeded_tokens.add(job.token)
        notify(Path(job.src).name)

    def mark_failure(job: CopyJob, exc: BaseException):
        nonlocal done_count
        done_count += 1
        report.failed += 1
        if job.token is not None:
            report.failed_tokens.add(job.token)
        if len(report.errors) < 50:
            report.errors.append((str(job.src), str(job.dest), f"{type(exc).__name__}: {exc}"))
        notify(Path(job.src).name)

    try:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ilt-copy") if worker_count > 1 else _NullExecutor() as pool:
            pending = {}
            while True:
                chunk = list(islice(iterator, local_chunk))
                if not chunk:
                    break
                seen_count += len(chunk)
                report.requested = int(known_total) if known_total is not None else seen_count
                chunk.sort(
                    key=lambda j: (
                        os.path.normcase(str(Path(j.src).parent)),
                        os.path.normcase(Path(j.src).name),
                    )
                )

                # Create each destination directory once per chunk rather than once
                # per file.  The set stays bounded with the chunk.
                for parent in {Path(job.dest).parent for job in chunk}:
                    parent.mkdir(parents=True, exist_ok=True)

                if worker_count <= 1:
                    for job in chunk:
                        try:
                            _, size = _copy_one(job, preserve_metadata)
                            mark_success(job, size)
                        except BaseException as exc:
                            mark_failure(job, exc)
                    continue

                chunk_iter = iter(chunk)

                def refill():
                    while len(pending) < window:
                        try:
                            job = next(chunk_iter)
                        except StopIteration:
                            break
                        pending[pool.submit(_copy_one, job, preserve_metadata)] = job
                        report.max_in_flight = max(report.max_in_flight, len(pending))

                refill()
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        job = pending.pop(future)
                        try:
                            _, size = future.result()
                            mark_success(job, size)
                        except BaseException as exc:
                            mark_failure(job, exc)
                    refill()
    except BaseException:
        for future in list(locals().get("pending", {})):
            try:
                future.cancel()
            except Exception:
                pass
        raise
    finally:
        report.requested = int(known_total) if known_total is not None else seen_count
        report.elapsed_seconds = max(0.0, time.perf_counter() - start)
        notify(force=True)

    return report


class _NullExecutor:
    """Tiny context manager used to share the copy loop with the serial path."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
