"""
Unit tests for spicecimx.sim.hybrid_backend._pick_transient_sample.

These do not require Spectre or any SPICE infrastructure — they validate
the sampling helper that decides which subarrays get a transient
validation pass under hybrid mode.
"""
from __future__ import annotations

import os
import sys

# Ensure package is importable when run from the repo root
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from spicecimx.sim.hybrid_backend import _pick_transient_sample


def test_zero_fraction_returns_empty():
    assert _pick_transient_sample(n_subarrays=10, fraction=0.0, seed=0) == set()


def test_zero_subarrays_returns_empty():
    assert _pick_transient_sample(n_subarrays=0, fraction=0.5, seed=0) == set()


def test_full_fraction_returns_all():
    out = _pick_transient_sample(n_subarrays=8, fraction=1.0, seed=0)
    assert out == set(range(8))


def test_fraction_above_one_clamped():
    out = _pick_transient_sample(n_subarrays=8, fraction=5.0, seed=0)
    assert out == set(range(8))


def test_fraction_rounds_up_to_at_least_one():
    """Tiny fractions still pick at least one subarray."""
    out = _pick_transient_sample(n_subarrays=100, fraction=1e-6, seed=0)
    assert len(out) == 1


def test_reproducibility_with_seed():
    a = _pick_transient_sample(n_subarrays=50, fraction=0.2, seed=42)
    b = _pick_transient_sample(n_subarrays=50, fraction=0.2, seed=42)
    assert a == b
    # Different seed -> (very likely) different sample
    c = _pick_transient_sample(n_subarrays=50, fraction=0.2, seed=43)
    assert not (a == c)  # extremely unlikely to collide at this size


def test_sample_size_matches_fraction():
    n_sub = 100
    out = _pick_transient_sample(n_subarrays=n_sub, fraction=0.25, seed=0)
    # round(100 * 0.25) = 25
    assert len(out) == 25


def test_sample_indices_in_range():
    n_sub = 20
    out = _pick_transient_sample(n_subarrays=n_sub, fraction=0.5, seed=1)
    assert all(0 <= i < n_sub for i in out)


if __name__ == "__main__":
    # Allow running directly without pytest
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
    print(f"{'-'*50}\n{'FAIL' if failed else 'PASS'}  {failed} failure(s)")
    sys.exit(1 if failed else 0)
