"""
Spectre invocation, license-pool management, and parallel subarray dispatch
for SPICECiM-X v3.

This module is the core of the parallelization update.  The previous code
ran each subarray serially in the foreground; here we:

  * model Spectre licenses as a bounded resource (``LicensePool``),
  * represent each (subarray, branch) pair as a ``SpectreJob``,
  * dispatch jobs to a thread pool whose width equals the license count
    (Spectre is an external subprocess, so threads suffice — no GIL pain),
  * optionally pipeline post-processing: as soon as a sim finishes, its
    psfascii dump is parsed on a separate process pool, overlapping with
    the next sim.

Failure handling: each job retries up to ``retries`` times before being
recorded as failed.  A failed job does NOT crash the rest of the batch — its
output is reported as None and the caller can decide whether to abort.

Per-job isolation: every job gets its own working directory under
``tmp_root/{tag}/{subarray_id}/{branch}`` so that concurrent jobs never share
input deck or Verilog-A subckt files (which would otherwise trample each
other since both branches reuse fixed filenames).

Public API
----------
class LicensePool             — bounded slot manager, context-manager-friendly.
@dataclass SpectreJob         — one (subarray, branch) unit of work.
class JobScheduler            — runs a batch of jobs with concurrency control.
class SimResult               — what a scheduler returns per job.
build_spectre_command(...)    — pure-string builder for the spectre CLI.
run_spectre(...)              — subprocess.run wrapper with retries + cwd.
"""

from __future__ import annotations

import multiprocessing
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Spectre CLI helpers
# ---------------------------------------------------------------------------
def build_spectre_command(
    input_deck:    str,
    raw_dir:       str,
    log_path:      str,
    extra_args:    Sequence[str] = (),
    *,
    binary:        str  = "spectre",
    flag_64:       bool = True,
    format_:       str  = "psfascii",
) -> List[str]:
    """
    Build the Spectre command line for a single sim.

    Parameters
    ----------
    input_deck : absolute path to the .scs input deck.
    raw_dir    : where Spectre will write `tran.tran.tran` / `*.dc` files.
    log_path   : where to redirect Spectre's own log output.
    extra_args : tuple of additional CLI args (e.g. ``+preset=mx``,
                 ``+mt=16``).  Passed through verbatim.
    binary     : path or name of the spectre executable.
    flag_64    : append ``-64`` for 64-bit mode.
    format_    : raw output format; ``psfascii`` (transient default) or
                 the appropriate format for DC analysis.

    Returns the argv list ready to hand to ``subprocess.run``.

    All path arguments are converted to absolute paths so that the cwd
    Spectre is invoked with (typically the per-job working directory) does
    not affect how it resolves them — see Spectre Reference Manual on
    invocation, where path arguments are taken to be relative to the
    process working directory.
    """
    cmd: List[str] = [binary]
    if flag_64:
        cmd.append("-64")
    cmd.extend(extra_args)
    cmd.append(os.path.abspath(input_deck))
    cmd.extend([
        "=log",        os.path.abspath(log_path),
        "-format",     format_,
        "-raw",        os.path.abspath(raw_dir),
    ])
    return cmd


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and (on POSIX) its whole process group, then reap it."""
    try:
        if os.name != "nt":
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:                                    # noqa: BLE001
        pass


def run_spectre(
    cmd:        Sequence[str],
    cwd:        str,
    *,
    retries:        int   = 2,
    retry_sleep_s:  float = 5.0,
    timeout_s:      Optional[float] = None,
) -> subprocess.CompletedProcess:
    """
    Invoke Spectre, retrying transient failures.

    Returns the final CompletedProcess (success or last failure).  Raises
    ``subprocess.TimeoutExpired`` if the per-call timeout is hit.

    Stdout/stderr are dropped on success (Spectre's own ``=log`` already
    captures them).  On failure, the last 2 KB of stderr is preserved on
    the CompletedProcess so the scheduler can surface it in error messages.
    """
    os.makedirs(cwd, exist_ok=True)
    abs_cwd = os.path.abspath(cwd)
    attempts = 0
    last: Optional[subprocess.CompletedProcess] = None
    while attempts <= retries:
        attempts += 1
        # Own process group, so a timeout kills Spectre's children as well
        # (spectre is often a wrapper script around the real simulator).
        kw = {"start_new_session": True} if os.name != "nt" else {}
        proc = subprocess.Popen(cmd, cwd=abs_cwd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, **kw)
        try:
            _, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            # Bubble up — retries probably won't help with timeouts.
            raise
        except BaseException:
            _kill_tree(proc)
            raise
        last = subprocess.CompletedProcess(cmd, proc.returncode, None, err)
        if last.returncode == 0:
            return last
        if attempts > retries:
            return last
        time.sleep(retry_sleep_s)
    assert last is not None
    return last


# ---------------------------------------------------------------------------
# License pool
# ---------------------------------------------------------------------------
class LicensePool:
    """
    A bounded resource pool that gates concurrent Spectre invocations to
    ``size`` slots.  Implemented as a counting semaphore so a thread can
    block on ``acquire`` and other threads can wait without busy-looping.

    Usage::

        pool = LicensePool(size=16)
        with pool.slot():
            run_spectre(...)

    The pool keeps a small bit of bookkeeping (peak concurrency, total
    acquisitions) accessible via ``stats()`` — useful for confirming the
    scheduler is actually keeping all seats busy.
    """

    def __init__(self, size: int):
        if size < 1:
            raise ValueError(f"LicensePool size must be >= 1, got {size}")
        self._size = size
        self._sem = threading.Semaphore(size)
        self._lock = threading.Lock()
        self._inflight = 0
        self._peak = 0
        self._total = 0

    @property
    def size(self) -> int:
        return self._size

    def acquire(self, timeout: Optional[float] = None) -> bool:
        ok = self._sem.acquire(timeout=timeout) if timeout is not None else self._sem.acquire()
        if ok:
            with self._lock:
                self._inflight += 1
                self._total += 1
                self._peak = max(self._peak, self._inflight)
        return ok

    def release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
        self._sem.release()

    class _Slot:
        """Context manager returned by ``LicensePool.slot()``."""

        def __init__(self, pool: "LicensePool"):
            self._pool = pool

        def __enter__(self):
            self._pool.acquire()
            return self._pool

        def __exit__(self, exc_type, exc, tb):
            self._pool.release()
            return False

    def slot(self) -> "LicensePool._Slot":
        return LicensePool._Slot(self)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": self._size,
                "inflight": self._inflight,
                "peak_concurrent": self._peak,
                "total_acquisitions": self._total,
            }


# ---------------------------------------------------------------------------
# Job representation
# ---------------------------------------------------------------------------
@dataclass
class SpectreJob:
    """
    One unit of work: simulate one branch (pos or neg) of one subarray.

    The job is intentionally a *plain data record* — all simulator-specific
    work happens inside the ``sim_fn`` callable the scheduler invokes.  This
    keeps SpectreJob picklable for the post-process ProcessPoolExecutor.

    Attributes
    ----------
    job_id : opaque string used for log messages (e.g. ``Q_h0/Array_0_1/pos``).
    cmd    : argv list to invoke Spectre with.
    cwd    : working directory the subprocess should run in.
    raw_dir: where Spectre writes its raw output — also where the
             post-processor will look for ``tran.tran.tran`` or ``*.dc``.
    branch : ``'pos'`` | ``'neg'``.
    subarray_row, subarray_col : indices into the layer's subarray grid.
    extra  : any caller payload (e.g. the .npy path) the post-processor
             needs.  Must be picklable if pipeline_postprocess=True.
    """
    job_id:       str
    cmd:          List[str]
    cwd:          str
    raw_dir:      str
    branch:       str
    subarray_row: int
    subarray_col: int
    extra:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimResult:
    """
    Per-job result.  Either ``error`` is None and ``post`` holds whatever
    the post-processor returned, or ``error`` is a string describing the
    failure (sim or postprocess).
    """
    job:    SpectreJob
    ok:     bool
    error:  Optional[str] = None
    post:   Any = None          # whatever the post-process callback returned
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
PostprocessFn = Callable[[SpectreJob], Any]


class JobScheduler:
    """
    Dispatches a batch of SpectreJobs across a LicensePool, with optional
    pipelined post-processing.

    Parameters
    ----------
    license_pool : LicensePool sized at ``license_count``.  All Spectre
                   invocations go through it.
    sim_workers  : ThreadPoolExecutor concurrency.  Should equal or exceed
                   ``license_pool.size`` (extra threads will just block on
                   ``pool.acquire``, which is harmless).
    post_workers : ProcessPoolExecutor concurrency for the post-processor.
                   Independent of license_count — CSV / psfascii parsing is
                   CPU-bound and benefits from real processes.
    retries      : per-Spectre-invocation retry count (delegated to
                   ``run_spectre``).
    pipeline_postprocess : if True, post-processing of subarray N begins
                   as soon as its sim finishes, in parallel with sim N+1.
                   If False, all sims complete first, then post-process runs
                   as a batch.
    """

    def __init__(
        self,
        license_pool:           LicensePool,
        *,
        sim_workers:            Optional[int] = None,
        post_workers:           int = 4,
        retries:                int = 2,
        retry_sleep_s:          float = 5.0,
        timeout_s:              Optional[float] = None,
        pipeline_postprocess:   bool = True,
    ):
        self.pool = license_pool
        self.sim_workers = sim_workers or license_pool.size
        self.post_workers = post_workers
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s
        self.timeout_s = timeout_s
        self.pipeline = pipeline_postprocess
        # Reusable executors created once per .run_layer() call.

    # ---------------------- internal sim worker -----------------------------
    def _run_one_sim(self, job: SpectreJob) -> SimResult:
        """Acquire a license, run Spectre, release.  Threaded-safe."""
        t0 = time.time()
        with self.pool.slot():
            try:
                cp = run_spectre(
                    job.cmd, cwd=job.cwd,
                    retries=self.retries,
                    retry_sleep_s=self.retry_sleep_s,
                    timeout_s=self.timeout_s,
                )
                if cp.returncode != 0:
                    err_tail = (cp.stderr or b"")[-2048:].decode("utf-8", "replace")
                    return SimResult(
                        job=job, ok=False,
                        error=f"spectre exited {cp.returncode}; tail:\n{err_tail}",
                        elapsed_s=time.time() - t0,
                    )
            except subprocess.TimeoutExpired:
                return SimResult(
                    job=job, ok=False,
                    error=f"spectre timeout after {self.timeout_s}s",
                    elapsed_s=time.time() - t0,
                )
            except Exception as e:                          # noqa: BLE001
                return SimResult(
                    job=job, ok=False, error=f"{type(e).__name__}: {e}",
                    elapsed_s=time.time() - t0,
                )
        return SimResult(job=job, ok=True, elapsed_s=time.time() - t0)

    # ---------------------- main dispatch -----------------------------------
    def run_layer(
        self,
        jobs:             Sequence[SpectreJob],
        postprocess_fn:   Optional[PostprocessFn] = None,
        *,
        progress_cb:      Optional[Callable[[int, int, str], None]] = None,
    ) -> List[SimResult]:
        """
        Run a batch of SpectreJobs.

        Parameters
        ----------
        jobs : sequence of SpectreJob.
        postprocess_fn : if provided, called as ``postprocess_fn(job)`` for
            every job whose sim succeeded.  Its return value is stored in
            ``SimResult.post``.  If pipeline_postprocess=True the call is
            scheduled on a process pool concurrently with subsequent sims.
        progress_cb : optional callback ``(done, total, phase_label)``,
            invoked from the dispatching thread.

        Returns the SimResult list in *job submission order*.
        """
        n = len(jobs)
        if n == 0:
            return []
        results: List[Optional[SimResult]] = [None] * n
        index = {id(j): i for i, j in enumerate(jobs)}

        # --- post-process executor (processes), only if pipelining ---
        # CRITICAL: create the ProcessPoolExecutor BEFORE the ThreadPoolExecutor
        # starts spawning worker threads, and use the 'spawn' context.  The
        # default 'fork' start method copies the parent's locked-GIL state
        # when threads are already running, which deadlocks the child.
        # 'spawn' starts each worker from a fresh interpreter.
        post_pool: Optional[ProcessPoolExecutor] = None
        if self.pipeline and postprocess_fn is not None:
            ctx = multiprocessing.get_context("spawn")
            post_pool = ProcessPoolExecutor(
                max_workers=self.post_workers,
                mp_context=ctx,
            )

        # --- sim executor (threads) ---
        sim_pool = ThreadPoolExecutor(max_workers=self.sim_workers,
                                      thread_name_prefix="spectre")

        post_futs: Dict[Future, int] = {}
        sim_futs:  Dict[Future, int] = {
            sim_pool.submit(self._run_one_sim, j): index[id(j)] for j in jobs
        }

        done_sim = 0
        try:
            for sim_fut in as_completed(sim_futs):
                i = sim_futs[sim_fut]
                try:
                    res = sim_fut.result()
                except Exception as e:                       # noqa: BLE001
                    res = SimResult(job=jobs[i], ok=False,
                                    error=f"{type(e).__name__}: {e}")
                results[i] = res
                done_sim += 1
                if progress_cb is not None:
                    progress_cb(done_sim, n, "sim")

                # Schedule post-process if we should
                if res.ok and postprocess_fn is not None:
                    if post_pool is not None:
                        f = post_pool.submit(postprocess_fn, res.job)
                        post_futs[f] = i
                    else:
                        # Non-pipelined: do it inline (after all sims, below)
                        pass

            # Drain post-pool if pipelining
            if post_pool is not None:
                done_pp = 0
                total_pp = len(post_futs)
                for f in as_completed(post_futs):
                    i = post_futs[f]
                    try:
                        results[i].post = f.result()         # type: ignore[union-attr]
                    except Exception as e:                   # noqa: BLE001
                        results[i].ok = False                # type: ignore[union-attr]
                        results[i].error = f"postprocess: {type(e).__name__}: {e}"  # type: ignore[union-attr]
                    done_pp += 1
                    if progress_cb is not None:
                        progress_cb(done_pp, total_pp, "postprocess")
            elif postprocess_fn is not None:
                # Non-pipelined path: run post-process serially on the main thread
                done_pp = 0
                pp_total = sum(1 for r in results if r is not None and r.ok)
                for r in results:
                    if r is None or not r.ok:
                        continue
                    try:
                        r.post = postprocess_fn(r.job)
                    except Exception as e:                   # noqa: BLE001
                        r.ok = False
                        r.error = f"postprocess: {type(e).__name__}: {e}"
                    done_pp += 1
                    if progress_cb is not None:
                        progress_cb(done_pp, pp_total, "postprocess")
        finally:
            # On an error / Ctrl-C, don't launch the Spectre jobs still queued.
            sim_pool.shutdown(wait=True, cancel_futures=True)
            if post_pool is not None:
                post_pool.shutdown(wait=True, cancel_futures=True)

        # All slots should be filled by now
        out: List[SimResult] = []
        for r in results:
            assert r is not None, "scheduler dropped a job"
            out.append(r)
        return out


class SpectreJobsFailed(RuntimeError):
    """Raised when Spectre jobs of a layer failed and ``sim.on_job_failure``
    is ``'raise'``."""


def check_failed_jobs(results: Sequence[SimResult], *, tag: str, n_jobs: int,
                      policy: str = "raise") -> List[SimResult]:
    """Report the jobs of one layer that failed (simulation or post-process).

    ``policy='raise'`` raises :class:`SpectreJobsFailed` listing them;
    ``policy='nan'`` returns them so the caller can mark their output columns
    NaN.  Never lets a failed job silently contribute zero current.
    """
    failed = [r for r in results if not r.ok or r.post is None]
    if not failed:
        return []
    lines = [f"  - {r.job.job_id}: {(r.error or 'no post-processed output').strip()[:400]}"
             for r in failed[:10]]
    msg = (f"[{tag}] {len(failed)}/{n_jobs} Spectre job(s) failed:\n" + "\n".join(lines)
           + ("\n  ..." if len(failed) > 10 else ""))
    if policy == "raise":
        raise SpectreJobsFailed(
            msg + "\n(set [SIM] on_job_failure = nan to continue with NaN outputs)")
    print(msg + "\n  -> their output columns are NaN (sim.on_job_failure = nan)",
          flush=True)
    return failed
