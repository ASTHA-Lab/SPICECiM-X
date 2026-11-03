"""
End-to-end transient-backend flow without a real Spectre.

``tests/fake_spectre.py`` is run as ``[sys.executable, fake_spectre.py, ...]``
(no shell wrapper, so this works on Windows too).  It writes a psfascii
``tran.tran.tran`` whose settled currents are the analytic ``V @ G_branch``
for its subarray block and branch, with the differential mapping
``G± = c·max(±B, 0) + Gmin``.  The real weight tiling, parallel scheduler
(with the spawn post-process pool), psfascii post-processor and sign-aware
assembly must then reproduce the ideal differential current, and the affine
de-map must recover ``A @ B``.
"""

import os
import sys

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.config import HwCfg, SimCfg                     # noqa: E402
from spicecimx.core.imc_matmul import (                        # noqa: E402
    derive_affine, ideal_current, compute_offset_terms,
)
from spicecimx.sim.transient_backend import (                  # noqa: E402
    _postprocess_transient_subarray,
)
from spicecimx.sim.spectre import (                            # noqa: E402
    SpectreJob, LicensePool, JobScheduler, check_failed_jobs,
)
from spicecimx.sim.templates import (                          # noqa: E402
    array_indices_from_npy, pad_rows_to_subarrays,
)
from spicecimx.hw.netlist import distribute_weights            # noqa: E402

FAKE_SPECTRE = os.path.join(PKG_ROOT, "tests", "fake_spectre.py")


def hw_window(hw):
    alpha = hw.Ik1 / hw.Vref
    Gmax = float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    Gmin = float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    return Gmin, Gmax


def run_fake_transient(A, B, hw, sim, root):
    """Mirror _transient_imc with fake_spectre; return (I_SPICE, scales)."""
    rs, cs = hw.subarray_rows, hw.subarray_cols
    m, _ = A.shape
    n = B.shape[1]
    Gmin, Gmax = hw_window(hw)
    scales = derive_affine(A, B, vmin=hw.vmin, vmax=hw.vmax, Gmin=Gmin, Gmax=Gmax)
    V = pad_rows_to_subarrays(scales.a * A + scales.b, rs)
    inc = 3 * sim.pulse_width + sim.trise + sim.tfall

    hwsimdb_dir = os.path.join(root, "hwsimdb")
    tmp_dir = os.path.join(root, "tmp")
    tag = "T1"
    distribute_weights(layer_name=tag, layer_weights=B,
                       synaptic_array_size=(rs, cs),
                       pe_size=(hw.pe_rows, hw.pe_cols),
                       tile_size=(hw.tile_rows, hw.tile_cols),
                       base_dir=hwsimdb_dir)

    jobs = []
    for dirpath, _dirs, files in os.walk(os.path.join(hwsimdb_dir, tag)):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(dirpath, fname)
            sa_row, sa_col = array_indices_from_npy(fname)
            v_slice_path = os.path.join(tmp_dir, tag, f"Array_{sa_row}_{sa_col}_V.npy")
            os.makedirs(os.path.dirname(v_slice_path), exist_ok=True)
            np.save(v_slice_path, V[:, sa_row * rs:(sa_row + 1) * rs])
            for branch in ("pos", "neg"):
                job_cwd = os.path.join(tmp_dir, tag, f"Array_{sa_row}_{sa_col}", branch)
                os.makedirs(job_cwd, exist_ok=True)
                raw_dir = os.path.join(os.path.splitext(npy_path)[0] + f"_{branch}", "psf")
                settings = {
                    "NPY": npy_path, "V_NPY": v_slice_path, "BRANCH": branch,
                    "NCOL": cs, "INC": f"{inc:.6e}", "M": m,
                    "C": f"{scales.c!r}", "D": f"{scales.d!r}",
                    "GMIN": f"{Gmin!r}", "TMIN_GVAL": f"{Gmin!r}",
                    "SL": sim.settling_window_lo, "SH": sim.settling_window_hi,
                }
                cmd = [sys.executable, FAKE_SPECTRE,
                       *(f"SPICECIM_FAKE_{k}={v}" for k, v in settings.items()),
                       "-raw", raw_dir]
                jobs.append(SpectreJob(
                    job_id=f"{tag}/Array_{sa_row}_{sa_col}/{branch}",
                    cmd=cmd, cwd=job_cwd, raw_dir=raw_dir, branch=branch,
                    subarray_row=sa_row, subarray_col=sa_col,
                    extra={"ncolumn": cs, "start_time": 0.0, "increment": inc,
                           "settling_lo": sim.settling_window_lo,
                           "settling_hi": sim.settling_window_hi,
                           "reducer": "median"},
                ))

    sched = JobScheduler(LicensePool(sim.license_count),
                         sim_workers=sim.license_count, post_workers=2,
                         retries=0, retry_sleep_s=0.0, pipeline_postprocess=True)
    results = sched.run_layer(jobs, postprocess_fn=_postprocess_transient_subarray)
    assert check_failed_jobs(results, tag=tag, n_jobs=len(jobs), policy="raise") == []

    num_h = (n + cs - 1) // cs
    I_full = np.zeros((m, num_h * cs))
    for r in results:
        pp = r.post
        c0 = pp["sa_col"] * cs
        sign = +1 if pp["branch"] == "pos" else -1
        I_full[:, c0:c0 + cs] += sign * pp["currents"][:m]
    return I_full[:, :n], scales, len(jobs)


@pytest.mark.parametrize("m,k,n", [(6, 8, 8), (10, 16, 24), (5, 20, 12)])
def test_transient_flow_recovers_AatB(tmp_path, m, k, n):
    hw = HwCfg(subarray_rows=8, subarray_cols=8, pe_rows=2, pe_cols=2,
               tile_rows=2, tile_cols=2, vmin=0.0, vmax=1.0)
    sim = SimCfg(license_count=4, pulse_width=1e-9, trise=1e-12, tfall=1e-12,
                 settling_window_lo=0.40, settling_window_hi=0.65)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))

    I_SPICE, sc, n_jobs = run_fake_transient(A, B, hw, sim, str(tmp_path))
    rs, cs = hw.subarray_rows, hw.subarray_cols
    assert n_jobs == 2 * (-(-k // rs)) * (-(-n // cs))
    assert np.all(np.isfinite(I_SPICE))

    # Assembled current == the differential ideal ac·A@B + bc·J@B
    I_ref = ideal_current(A, B, sc, differential=True)
    rel_i = np.max(np.abs(I_SPICE - I_ref)) / np.max(np.abs(I_ref))
    assert rel_i < 1e-6, f"current rel err {rel_i:.2e}"

    # ... and the term2-only de-map recovers A @ B
    C = (I_SPICE - compute_offset_terms(A, B, sc)["sum"]) / (sc.a * sc.c)
    Y = A @ B
    rel = np.max(np.abs(C - Y)) / (Y.max() - Y.min())
    assert rel < 1e-6, f"A@B rel err {rel:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
