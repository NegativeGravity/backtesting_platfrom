from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from backend.utils.ids import new_id

logger = logging.getLogger(__name__)
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
JobCallable = Callable[[threading.Event], tuple[Path, dict[str, Any]]]


@dataclass
class BacktestJob:
    job_id: str
    status: JobStatus = "queued"
    run_id: str | None = None
    run_dir: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class BacktestJobManager:
    def __init__(self, max_workers: int | None = None, max_jobs: int = 256) -> None:
        worker_count = max_workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="backtest-job")
        self._jobs: dict[str, BacktestJob] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(self, job_fn: JobCallable) -> BacktestJob:
        job = BacktestJob(job_id=new_id("job"))
        with self._lock:
            self._evict_old_jobs_locked()
            if len(self._jobs) >= self._max_jobs:
                raise RuntimeError("Backtest job queue is full.")
            self._jobs[job.job_id] = job

        future = self._executor.submit(self._run_job, job.job_id, job_fn)
        with self._lock:
            self._futures[job.job_id] = future
        return self._copy(job)

    def get(self, job_id: str) -> BacktestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else self._copy(job)

    def cancel(self, job_id: str) -> BacktestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            job.cancel_event.set()
            future = self._futures.get(job_id)
            if job.status == "queued" and future is not None and future.cancel():
                job.error = "Cancelled before execution."
            elif job.status in {"queued", "running"}:
                job.error = "Cancellation requested."

            if job.status in {"queued", "running"}:
                job.status = "cancelled"
                job.updated_at = time.time()

            return self._copy(job)

    def _run_job(self, job_id: str, job_fn: JobCallable) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                return
            job.status = "running"
            job.updated_at = time.time()
            cancel_event = job.cancel_event

        try:
            run_dir, summary = job_fn(cancel_event)
            with self._lock:
                job = self._jobs[job_id]
                if job.status == "cancelled" or job.cancel_event.is_set():
                    return
                job.status = "completed"
                job.run_id = run_dir.name
                job.run_dir = str(run_dir)
                job.summary = summary
                job.updated_at = time.time()
        except Exception as exc:
            logger.exception("Backtest job failed: %s", job_id)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.error = "Cancelled."
                elif job.status != "cancelled":
                    job.status = "failed"
                    job.error = str(exc)
                job.updated_at = time.time()

    def _evict_old_jobs_locked(self) -> None:
        terminal = {"completed", "failed", "cancelled"}
        if len(self._jobs) < self._max_jobs:
            return
        old_ids = sorted(
            (job_id for job_id, job in self._jobs.items() if job.status in terminal),
            key=lambda item: self._jobs[item].updated_at,
        )
        for job_id in old_ids[: max(1, len(old_ids) // 4)]:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)

    @staticmethod
    def _copy(job: BacktestJob) -> BacktestJob:
        return BacktestJob(
            job_id=job.job_id,
            status=job.status,
            run_id=job.run_id,
            run_dir=job.run_dir,
            summary=dict(job.summary),
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
