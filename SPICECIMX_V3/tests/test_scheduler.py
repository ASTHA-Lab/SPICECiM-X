"""
JobScheduler honours license_count and LicensePool tracks peak concurrency.

A short ``python -c "time.sleep(x)"`` subprocess stands in for Spectre, so the
test is portable (Linux + Windows).
"""

import os
import sys
import time

import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.sim.spectre import LicensePool, JobScheduler, SpectreJob  # noqa: E402
# Importable from a real module so the 'spawn' post-process pool can pickle it.
from tests.scheduler_helpers import fake_postprocess                   # noqa: E402


def make_sleep_job(i, duration_s, tmproot, rc=0):
    cwd = os.path.join(str(tmproot), f"job_{i:02d}")
    os.makedirs(cwd, exist_ok=True)
    code = f"import time, sys; time.sleep({duration_s}); sys.exit({rc})"
    return SpectreJob(
        job_id=f"job_{i:02d}", cmd=[sys.executable, "-c", code],
        cwd=cwd, raw_dir=cwd, branch="pos" if i % 2 == 0 else "neg",
        subarray_row=i // 4, subarray_col=i % 4,
    )


def run_case(n_jobs, license_count, tmproot, dur_s=0.3, pipeline=True):
    jobs = [make_sleep_job(i, dur_s, tmproot) for i in range(n_jobs)]
    pool = LicensePool(license_count)
    sched = JobScheduler(pool, sim_workers=license_count, post_workers=2,
                         retries=0, retry_sleep_s=0.0,
                         pipeline_postprocess=pipeline)
    t0 = time.time()
    res = sched.run_layer(jobs, postprocess_fn=fake_postprocess)
    return res, pool.stats(), time.time() - t0


@pytest.mark.parametrize("n_jobs,licenses,pipeline", [
    (6, 1, False),     # serial baseline
    (8, 4, True),      # parallel x4, pipelined post-process (spawn pool)
    (8, 4, False),     # parallel x4, inline post-process
    (3, 8, False),     # more licenses than jobs
])
def test_scheduler_respects_license_count(tmp_path, n_jobs, licenses, pipeline):
    res, stats, _ = run_case(n_jobs, licenses, tmp_path, pipeline=pipeline)
    assert len(res) == n_jobs
    assert all(r.ok for r in res), [r.error for r in res if not r.ok]
    # results come back in submission order, each post-processed
    assert [r.job.job_id for r in res] == [f"job_{i:02d}" for i in range(n_jobs)]
    assert all(r.post == {"job_id": r.job.job_id, "burn": 50_000} for r in res)
    assert stats["peak_concurrent"] == min(licenses, n_jobs)
    assert stats["total_acquisitions"] == n_jobs
    assert stats["inflight"] == 0


def test_scheduler_reports_failed_job(tmp_path):
    jobs = [make_sleep_job(0, 0.0, tmp_path), make_sleep_job(1, 0.0, tmp_path, rc=3)]
    sched = JobScheduler(LicensePool(2), retries=0, retry_sleep_s=0.0,
                         pipeline_postprocess=False)
    res = sched.run_layer(jobs, postprocess_fn=fake_postprocess)
    assert res[0].ok and res[0].post is not None
    assert not res[1].ok and res[1].post is None
    assert "exited 3" in res[1].error


def test_license_pool_rejects_zero():
    with pytest.raises(ValueError):
        LicensePool(0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
