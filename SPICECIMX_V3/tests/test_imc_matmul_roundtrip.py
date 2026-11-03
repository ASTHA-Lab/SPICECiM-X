"""
Round-trip tests for spicecimx.core.imc_matmul.

With a noiseless backend that returns the analytic *differential* ideal
current ``I = ac·A@B + bc·J@B`` and an identity TIA, the affine de-mapping in
imc_matmul must recover A @ B to machine precision.  A backend that returns
the sign-flipped current must be caught by the polarity cache.

Run standalone:  python tests/test_imc_matmul_roundtrip.py
"""

import os
import sys

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.core.imc_matmul import (                     # noqa: E402
    imc_matmul, PolarityCache, ideal_current,
)


def ideal_spice_backend(A, B, scales, tag, bname, dump_dir):
    """Pretend the crossbar is perfect: I_diff = V @ (G+ - G-) analytically."""
    return ideal_current(A, B, scales)


def identity_tia(I_spice, tag):
    """No-op TIA+ADC: I_hat = I_spice."""
    return np.zeros_like(I_spice), I_spice, {"k_nom": 1.0}


def run_case(m, k, n, vmin, vmax, Gmin, Gmax, seed=0, flip=False, pc=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((k, n))
    sign = -1.0 if flip else 1.0
    C = imc_matmul(
        A, B, Gmax=Gmax, Gmin=Gmin, vmin=vmin, vmax=vmax,
        sim_backend=lambda **kw: sign * ideal_spice_backend(**kw),
        polarity_cache=pc if pc is not None else PolarityCache(),
        tia_apply=identity_tia, tag=f"unit_{m}_{k}_{n}", bname="test",
    )
    Y = A @ B
    return float(np.max(np.abs(C - Y))) / (float(np.max(np.abs(Y))) + 1e-12)


CASES = [
    # (m, k, n, vmin, vmax, Gmin, Gmax)
    (4, 6, 5, 0.0, 1.0, 1e-7, 5e-5),       # typical: d != 0
    (8, 8, 8, 0.0, 0.5, 0.0, 1.0),         # Gmin = 0
    (16, 32, 12, -0.5, 0.5, 1e-7, 5e-5),   # symmetric V range
    (3, 4, 7, 0.1, 0.9, 2e-6, 7e-5),       # asymmetric V range
    (16, 16, 16, 0.0, 1.0, 5e-8, 4e-5),    # nominal subarray size
]


@pytest.mark.parametrize("m,k,n,vmin,vmax,Gmin,Gmax", CASES)
def test_roundtrip_recovers_AatB(m, k, n, vmin, vmax, Gmin, Gmax):
    rel = run_case(m, k, n, vmin, vmax, Gmin, Gmax)
    assert rel < 1e-9, f"relative error {rel:.3e}"


@pytest.mark.parametrize("m,k,n,vmin,vmax,Gmin,Gmax", CASES[:2])
def test_roundtrip_with_flipped_backend(m, k, n, vmin, vmax, Gmin, Gmax):
    pc = PolarityCache()
    rel = run_case(m, k, n, vmin, vmax, Gmin, Gmax, flip=True, pc=pc)
    assert rel < 1e-9, f"relative error {rel:.3e}"
    assert pc.get(f"test/unit_{m}_{k}_{n}") == -1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
