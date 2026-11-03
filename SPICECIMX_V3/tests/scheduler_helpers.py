"""Helper functions for the scheduler test, importable from worker processes."""


def fake_postprocess(job):
    """Simulate post-processing — small CPU burn so pipelining is visible."""
    s = 0
    for _ in range(50_000):
        s += 1
    return {"job_id": job.job_id, "burn": s}
