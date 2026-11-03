"""
v2 vs v3 affine de-mapping under DIFFERENTIAL readout.

The SPICE backends return ``I_diff = I+ - I-`` from two crossbars that share
the same baseline conductance, so ``I_diff = ac·A@B + bc·J@B``: the d-terms
(term1 = ad·A@J, term3 = bd·k·J) cancel.  Hence:

  * v3's de-map (``compute_offset_terms(differential=True)`` -> term2 only)
    recovers A@B exactly;
  * v2's term2-only de-map is algebraically the same and also exact;
  * the single-ended 3-term expansion (``differential=False``) is NOT needed
    for differential currents -- applying it adds a large constant offset.
"""

import os
import sys

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.core.imc_matmul import (                     # noqa: E402
    derive_affine, ideal_current, compute_offset_terms,
)


def v2_demap(I, A, B, sc):
    """v2's de-mapping: subtract term2 = bc·J@B only."""
    m, k = A.shape
    S_B = np.ones((m, 1)) @ (np.ones((k, 1)).T @ B)
    return (I - sc.b * sc.c * S_B) / (sc.a * sc.c)


def v3_demap(I, A, B, sc, differential=True):
    off = compute_offset_terms(A, B, sc, differential=differential)
    return (I - off["sum"]) / (sc.a * sc.c)


CASES = [
    (4, 6, 5, 1e-7, 5e-5),
    (8, 8, 8, 1e-8, 1e-4),
    (16, 16, 16, 5e-8, 4e-5),
    (16, 32, 12, 2e-6, 7e-5),
    (32, 64, 32, 1e-9, 1e-3),
]


def _setup(m, k, n, Gmin, Gmax, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    sc = derive_affine(A, B, vmin=0.0, vmax=1.0, Gmin=Gmin, Gmax=Gmax)
    Y = A @ B
    return A, B, sc, Y, float(np.max(Y) - np.min(Y))


@pytest.mark.parametrize("m,k,n,Gmin,Gmax", CASES)
def test_differential_demap_v2_and_v3_exact(m, k, n, Gmin, Gmax):
    A, B, sc, Y, rng_y = _setup(m, k, n, Gmin, Gmax, seed=42)
    assert abs(sc.d) > 0                      # the d-term is genuinely non-zero
    I_diff = ideal_current(A, B, sc, differential=True)
    err_v3 = np.max(np.abs(v3_demap(I_diff, A, B, sc) - Y)) / rng_y
    err_v2 = np.max(np.abs(v2_demap(I_diff, A, B, sc) - Y)) / rng_y
    assert err_v3 < 1e-12, f"v3 rel err {err_v3:.3e}"
    assert err_v2 < 1e-12, f"v2 rel err {err_v2:.3e}"


@pytest.mark.parametrize("m,k,n,Gmin,Gmax", CASES)
def test_single_ended_expansion_not_needed_for_differential(m, k, n, Gmin, Gmax):
    A, B, sc, Y, rng_y = _setup(m, k, n, Gmin, Gmax, seed=7)
    I_diff = ideal_current(A, B, sc, differential=True)
    # Wrong: single-ended (3-term) de-map on a differential current.
    err_se = np.max(np.abs(v3_demap(I_diff, A, B, sc, differential=False) - Y)) / rng_y
    assert err_se > 1e-2, f"single-ended de-map unexpectedly accurate ({err_se:.3e})"
    # Self-consistency: the single-ended de-map IS exact on single-ended current.
    I_se = ideal_current(A, B, sc, differential=False)
    err_ok = np.max(np.abs(v3_demap(I_se, A, B, sc, differential=False) - Y)) / rng_y
    assert err_ok < 1e-9, f"single-ended round trip rel err {err_ok:.3e}"
    # The extra terms are exactly the ones that cancel differentially.
    off = compute_offset_terms(A, B, sc, differential=False)
    np.testing.assert_allclose(off["sum"] - off["term2"],
                               off["term1"] + off["term3"], rtol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
