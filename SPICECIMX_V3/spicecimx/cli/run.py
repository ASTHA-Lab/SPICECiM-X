"""
Minimal demo runner for SPICECiM-X v3.

This is NOT a drop-in replacement for v2's `main.py`.  It is a
documentation-by-example showing how to build a context, pick a backend,
and run one or more matrix multiplies through `imc_matmul`.  Use it as a
starting template; replace the toy `np.random.standard_normal` inputs
with the real ViT activations and weights from your inference loop.

Usage
-----
    python -m spicecimx.cli.run \
        --mode dc \
        --licenses 4 \
        --hwsimdb ./hwsimdb \
        --tmp     ./tmp

    python -m spicecimx.cli.run --selftest   # numpy-only sanity check, no SPICE
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPICECiM-X v3 demo runner")
    p.add_argument("--mode", choices=("transient", "dc", "hybrid"),
                   default="dc",
                   help="Simulation mode (default: dc)")
    p.add_argument("--licenses", type=int, default=1,
                   help="Number of Spectre licenses available (default: 1)")
    p.add_argument("--hwsimdb", default="./hwsimdb",
                   help="Root directory for SPICE outputs and weight tiles")
    p.add_argument("--tmp",     default="./tmp",
                   help="Root directory for per-job temporary working files")
    p.add_argument("--batch-id", default="demo_run",
                   help="Tag for this batch (used in output paths)")
    p.add_argument("--m", type=int, default=4, help="Token rows in A")
    p.add_argument("--k", type=int, default=16, help="Inner dimension")
    p.add_argument("--n", type=int, default=16, help="Output dimension")
    p.add_argument("--selftest", action="store_true",
                   help="Skip Spectre; run a numpy-backed sanity check")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args(argv)


def _no_op_tia(I, tag):
    """Trivial TIA: pass current through as 'voltage', no quantization error."""
    return I, I, {}


def selftest(m: int, k: int, n: int, seed: int) -> int:
    """Sanity check the imc_matmul kernel without invoking Spectre."""
    from spicecimx.config import HwCfg
    from spicecimx.core.imc_matmul import imc_matmul, PolarityCache, ideal_current

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, k))
    W = rng.standard_normal((k, n))

    # numpy-passthrough backend: pretend SPICE returns the analytic ideal
    # current.  This exercises the affine de-mapping plus the imc_matmul
    # scaffolding (polarity cache, scale derivation, TIA, ADC).
    def numpy_backend(A, B, scales, tag, bname, dump_dir):
        return ideal_current(A, B, scales)

    pcache = PolarityCache()
    hw = HwCfg(subarray_rows=k, subarray_cols=n)  # one big subarray
    from spicecimx.hw.device_emit import conductance_calc
    alpha = hw.Ik1 / hw.Vref
    Gmax, Gmin = conductance_calc(alpha, hw.Tox, hw.Tref, hw.Tmax, hw.Tmin)

    Y = imc_matmul(
        A=A, B=W,
        Gmax=Gmax, Gmin=Gmin,
        vmin=hw.vmin, vmax=hw.vmax,
        sim_backend=numpy_backend,
        polarity_cache=pcache,
        tia_apply=_no_op_tia,
        tag="selftest", bname="selftest", dump_dir=None,
    )
    Y_ref = A @ W
    rel = np.max(np.abs(Y - Y_ref)) / max(np.max(np.abs(Y_ref)), 1e-12)
    print(f"selftest:  A {A.shape} @ B {W.shape} -> Y {Y.shape}")
    print(f"  max relative error vs A@W: {rel:.3e}")
    if rel > 1e-8:
        print(f"  FAIL (expected < 1e-8)")
        return 1
    print("  PASS")
    return 0


def real_run(args) -> int:
    """Run one matmul with the configured backend.  Requires Spectre."""
    from spicecimx.config import HwCfg, SimCfg, PathsCfg
    from spicecimx.sim import build_context, make_backend
    from spicecimx.core.imc_matmul import imc_matmul, PolarityCache

    hw = HwCfg(subarray_rows=16, subarray_cols=16)
    sim = SimCfg(
        mode=args.mode,
        license_count=args.licenses,
        postprocess_workers=4,
        pipeline_postprocess=True,
    )
    paths = PathsCfg(
        hwsimdb_root=args.hwsimdb,
        tmp_root=args.tmp,
    )

    ctx = build_context(hw, sim, paths, batch_id=args.batch_id)
    backend = make_backend(ctx)
    pcache = PolarityCache()
    from spicecimx.hw.device_emit import conductance_calc
    alpha = hw.Ik1 / hw.Vref
    Gmax, Gmin = conductance_calc(alpha, hw.Tox, hw.Tref, hw.Tmax, hw.Tmin)

    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((args.m, args.k))
    W = rng.standard_normal((args.k, args.n))

    print(f"Running {args.mode} mode, {args.licenses} license(s), "
          f"A {A.shape} @ W {W.shape}")
    Y = imc_matmul(
        A=A, B=W,
        Gmax=Gmax, Gmin=Gmin,
        vmin=hw.vmin, vmax=hw.vmax,
        sim_backend=backend,
        polarity_cache=pcache,
        tia_apply=_no_op_tia,
        tag="demo_layer", bname=args.batch_id, dump_dir=None,
    )
    Y_ref = A @ W
    rel = np.max(np.abs(Y - Y_ref)) / max(np.max(np.abs(Y_ref)), 1e-12)
    print(f"  Y shape: {Y.shape}")
    print(f"  max relative error vs ideal A@W: {rel:.3e}")
    print(f"  pool peak concurrency: {ctx.pool.stats()['peak_concurrent']}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest(args.m, args.k, args.n, args.seed)
    return real_run(args)


if __name__ == "__main__":
    sys.exit(main())
