"""
Hybrid simulation backend for SPICECiM-X v3.

Strategy
--------
Use DC mode for the *primary* inference currents (fast, physics-faithful at
DC), and OPTIONALLY run transient mode on a small sampled subset of
subarrays to produce a validation report:

  * For each sampled subarray, compute the (M, cs) DC current matrix and
    the (M, cs) transient settled-current matrix.
  * Report per-subarray and global agreement metrics (max abs err, mean
    abs err, signed bias).
  * The returned ``I_SPICE`` is taken from DC — the transient sample is
    diagnostic only.

This is intentionally minimal.  Two real-research extensions that build on
this skeleton without changing its interface:

  1. *Adaptive calibration*: fit a per-subarray scalar correction
     ``alpha`` such that ``alpha * I_dc[subarray] ≈ I_tran[subarray]``,
     then apply it to all subarrays of the same layer.
  2. *Hotspot retry*: re-run transient on subarrays where
     ``|I_dc - I_numpy| / |I_numpy| > threshold`` to localize where DC's
     physics deviates from the software ideal.

The sampling rule is configured via ``sim.hybrid_transient_fraction`` (in
[0, 1]) and ``sim.hybrid_transient_seed`` (RNG seed for reproducibility).
A fraction of 0 disables transient sampling entirely — i.e. hybrid mode
degenerates to pure DC.

Output report
-------------
Each ``imc_matmul`` call writes ``hybrid_report.csv`` into the same
hwsimdb tag directory as the .npy files, with columns:

    tag, sa_row, sa_col, n_points,
    dc_max_abs, tran_max_abs,
    diff_max_abs, diff_mean_abs, diff_signed_bias

so downstream notebooks can aggregate across layers.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from spicecimx.config import HwCfg, SimCfg, PathsCfg
from spicecimx.core.imc_matmul import AffineScales
from spicecimx.sim.transient_backend import (
    TransientBackendContext, _transient_imc,
)
from spicecimx.sim.dc_backend import _dc_imc


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------
def _pick_transient_sample(
    n_subarrays: int,
    fraction: float,
    seed: int,
) -> Set[int]:
    """Pick a deterministic random subset of subarray indices."""
    if fraction <= 0.0 or n_subarrays == 0:
        return set()
    fraction = min(1.0, fraction)
    k = max(1, int(round(n_subarrays * fraction)))
    rng = random.Random(seed)
    return set(rng.sample(range(n_subarrays), k))


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
def make_hybrid_backend(ctx: TransientBackendContext) -> Callable[..., np.ndarray]:
    """Return a callable ``backend(A, B, scales, tag, bname, dump_dir) -> I_SPICE``."""
    return lambda A, B, scales, tag, bname, dump_dir: _hybrid_imc(
        A=A, B=B, scales=scales, tag=tag, bname=bname, dump_dir=dump_dir,
        ctx=ctx,
    )


def _hybrid_imc(
    A:        np.ndarray,
    B:        np.ndarray,
    scales:   AffineScales,
    tag:      str,
    bname:    str,
    dump_dir: Optional[str],
    *,
    ctx:      TransientBackendContext,
) -> np.ndarray:
    """
    Run DC for the primary I_SPICE and (optionally) transient on a sampled
    subset for validation.

    Implementation note: we re-use ``_dc_imc`` and ``_transient_imc`` rather
    than re-implementing dispatch.  The transient pass uses a *cropped*
    weight matrix containing only the sampled subarrays' columns, so it
    only pays the cost of the sample.
    """
    # 1. Always run DC for the full matmul.
    I_dc = _dc_imc(
        A=A, B=B, scales=scales, tag=tag, bname=bname,
        dump_dir=dump_dir, ctx=ctx,
    )

    # 2. Decide whether to run transient sampling.
    fraction = float(getattr(ctx.sim, "hybrid_transient_fraction", 0.0))
    seed     = int(getattr(ctx.sim, "hybrid_transient_seed", 0))
    if fraction <= 0.0:
        return I_dc

    rs, cs = ctx.hw.subarray_rows, ctx.hw.subarray_cols
    m, k = A.shape
    _, n = B.shape
    num_v = (k + rs - 1) // rs
    num_h = (n + cs - 1) // cs
    n_sub = num_v * num_h
    if n_sub == 0:
        return I_dc

    sample_idx = _pick_transient_sample(n_sub, fraction, seed)
    if not sample_idx:
        return I_dc

    # 3. Build a cropped (A, B) covering only the sampled subarrays.
    #    For each sampled (sa_row, sa_col):
    #       A_cols [sa_row*rs : sa_row*rs+rs]
    #       B_rows same  ;  B_cols [sa_col*cs : sa_col*cs+cs]
    #    Because the IMC kernel is linear, slicing inputs and weights
    #    horizontally / vertically yields the analytic same slice of the
    #    output.  We package this as one transient run with grouping-aware
    #    block-diagonal B that only spans the sampled subarrays.
    #
    # To keep this simple, we run *one transient call per sampled subarray*
    # — each call processes a (m, rs) A-slice times a (rs, cs) B-slice.
    # The number of sampled subarrays is small (sample fraction is small),
    # so this is acceptable.

    sa_rows_cols = [
        (idx // num_h, idx % num_h) for idx in sorted(sample_idx)
    ]

    # Per-row diagnostics
    rows: List[Dict[str, Any]] = []
    for sa_row, sa_col in sa_rows_cols:
        r0, r1 = sa_row * rs, min((sa_row + 1) * rs, k)
        c0, c1 = sa_col * cs, min((sa_col + 1) * cs, n)
        if r1 <= r0 or c1 <= c0:
            continue
        A_sub = A[:, r0:r1]
        B_sub = B[r0:r1, c0:c1]
        # Pad to (rs, cs) shape so the kernels are happy.
        if A_sub.shape[1] < rs:
            A_sub = np.hstack([A_sub, np.zeros((m, rs - A_sub.shape[1]))])
        if B_sub.shape != (rs, cs):
            padded = np.zeros((rs, cs), dtype=B_sub.dtype)
            padded[:B_sub.shape[0], :B_sub.shape[1]] = B_sub
            B_sub = padded

        # Simulate the SAME single subarray (one row tile x one column tile)
        # in both modes.  The full-layer I_dc sums every row tile of a column,
        # so it is not comparable with one subarray's transient current.
        sub_tag = f"{tag}__hybrid_sa_{sa_row}_{sa_col}"
        try:
            I_tran_sub = _transient_imc(
                A=A_sub, B=B_sub, scales=scales,
                tag=sub_tag, bname=bname, dump_dir=dump_dir, ctx=ctx,
            )
            I_dc_slice = _dc_imc(
                A=A_sub, B=B_sub, scales=scales,
                tag=sub_tag + "_dc", bname=bname, dump_dir=dump_dir, ctx=ctx,
            )
        except Exception as e:                              # noqa: BLE001
            print(f"  [{tag}/hybrid] sample sa({sa_row},{sa_col}) failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue

        diff = I_tran_sub - I_dc_slice
        rows.append({
            "tag":             tag,
            "sa_row":          sa_row,
            "sa_col":          sa_col,
            "n_points":        int(I_dc_slice.size),
            "dc_max_abs":      float(np.nanmax(np.abs(I_dc_slice))),
            "tran_max_abs":    float(np.nanmax(np.abs(I_tran_sub))),
            "diff_max_abs":    float(np.nanmax(np.abs(diff))),
            "diff_mean_abs":   float(np.nanmean(np.abs(diff))),
            "diff_signed_bias":float(np.nanmean(diff)),
        })

    if rows:
        layer_name = f"{bname}_{tag}" if bname else tag
        report_path = os.path.join(ctx.hwsimdb_dir, layer_name, "hybrid_report.csv")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        write_header = not os.path.exists(report_path)
        with open(report_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        # Headline summary
        all_diff = np.array([r["diff_max_abs"] for r in rows])
        all_bias = np.array([r["diff_signed_bias"] for r in rows])
        print(f"  [{tag}/hybrid] sampled {len(rows)}/{n_sub} subarrays "
              f"via transient:", flush=True)
        print(f"    max(|I_tran - I_dc|) = {all_diff.max():.3e}", flush=True)
        print(f"    mean signed bias     = {all_bias.mean():+.3e}", flush=True)
        print(f"    report: {report_path}", flush=True)

    return I_dc
