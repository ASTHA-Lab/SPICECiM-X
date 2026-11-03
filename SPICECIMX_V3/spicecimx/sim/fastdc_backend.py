"""
fastdc DC simulation backend for SPICECiM-X.

Replaces SPICE-per-MAC with a pure-NumPy crossbar evaluation that uses a
pre-built cell I-V table (``CellITable``).  The cell table itself is
SPICE-derived (see :mod:`spicecimx.sim.char_run`), so every per-cell
current is exactly what SPICE would have computed at that operating
point — but the crossbar MAC reduction (sum across the inner dim) is
done by NumPy in microseconds instead of by Spectre in seconds.

The backend implements the same callable contract as
:func:`spicecimx.sim.dc_backend.make_dc_backend`::

    backend(A, B, scales, tag, bname, dump_dir) -> I_SPICE  # (m, n)

so it slots into ``vit_forward_imc`` without any changes upstream.

What this captures
------------------
* The full 1T1R cell nonlinearity in V_BL (tunneling I-V).
* Series-transistor effects (the table sees the real 1T1R cell, not a
  bare ReRAM).
* Cell-to-cell differences in programmed thickness.

What this does NOT capture
--------------------------
* Wire IR drop along the bit-line (negligible for ≤16-cell columns,
  but in principle present).
* Cell-to-cell coupling via parasitic caps / shared wire resistance.
* Process variation: the table is one nominal corner.  For PVT studies
  build multiple tables and switch via the config.

For 1T1R arrays where the access transistor dominates the per-cell
impedance — which is the regime users care about — these omissions are
under 1 % and below ADC quantisation.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np

from spicecimx.core.imc_matmul import AffineScales
from spicecimx.hw.char_table import CellITable
from spicecimx.hw.device_emit import convert_to_conductance
from spicecimx.hw.netlist import distribute_weights
from spicecimx.sim.templates import array_indices_from_npy


# ---------------------------------------------------------------------------
# Backend context — minimal mirror of TransientBackendContext
# ---------------------------------------------------------------------------
class FastDcBackendContext:
    """
    Holds the table and the config slices the backend needs.  Constructed
    once at the top of the inference run and passed into every
    ``imc_matmul`` call via ``make_fastdc_backend``.
    """
    def __init__(self, hw, sim, paths, hwsimdb_dir, tmp_dir,
                 cell_table: CellITable, batch_id: str = "fastdc",
                 v_linearize: bool = True):
        self.hw          = hw
        self.sim         = sim
        self.paths       = paths
        self.hwsimdb_dir = os.path.abspath(hwsimdb_dir)
        self.tmp_dir     = os.path.abspath(tmp_dir)
        self.cell_table  = cell_table
        self.batch_id    = batch_id
        self.v_linearize = v_linearize
        # Optional hardware DAC (PhysicalConverters), set by infer.py
        self.physical_converters = None
        # Tile interconnect (see transient_backend.TransientBackendContext)
        self.interconnect = None
        self.ic_params = None
        self.loading_records = []
        # Build the 1-D V-linearisation curve once: pre-warp the target
        # voltage to compensate for the cell's nonlinear I-V.  This is
        # critical for accuracy on multi-block ViTs — without it, per-layer
        # rmse is 10-20% (cell V-nonlinearity ~ factor 2x in this technology),
        # and compounds through softmax/GELU to wreck classification.
        if v_linearize:
            self.v_linearizer = cell_table.build_v_linearization()
            h_max = float(self.v_linearizer.h[-1])
            print(f"[fastdc] V-linearization on  (h(V_max)/V_max = {h_max:.3f}, "
                  f"cell super-linearity factor)")
        else:
            self.v_linearizer = None
            print("[fastdc] V-linearization OFF — affine demap will see "
                  "cell nonlinearity directly (~10-20% per-layer rmse expected)")


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def make_fastdc_backend(
    ctx: FastDcBackendContext,
) -> Callable[..., np.ndarray]:
    """Return a callable matching the dc/transient backend signature."""
    return lambda A, B, scales, tag, bname, dump_dir: _fastdc_imc(
        A=A, B=B, scales=scales, tag=tag, bname=bname, dump_dir=dump_dir,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Branch evaluation (per subarray-branch crossbar)
# ---------------------------------------------------------------------------
def _eval_branch(
    T_branch: np.ndarray,            # (rs, cs) thicknesses for this branch
    V_subarray: np.ndarray,          # (m, rs) row voltages
    cell_table: CellITable,
) -> np.ndarray:
    """
    Compute the (m, cs) bit-line currents for ONE subarray-branch by
    bilinear-interp lookups on the cell I-V table.

    The sign of the returned current follows the table (negative for the
    user's NM0:s probe at positive V_BL).  The caller sums pos/neg
    contributions accordingly.
    """
    return cell_table.bitline_current(T_branch, V_subarray)


# ---------------------------------------------------------------------------
# Main backend body
# ---------------------------------------------------------------------------
def _fastdc_imc(
    A:        np.ndarray,
    B:        np.ndarray,
    scales:   AffineScales,
    tag:      str,
    bname:    str,
    dump_dir: Optional[str],
    *,
    ctx:      FastDcBackendContext,
) -> np.ndarray:
    """
    Compute I_SPICE = I+ − I- for one matmul using the cell table.

    Parameters mirror :func:`spicecimx.sim.dc_backend._dc_imc`; the
    returned matrix has the same sign convention (post-differential),
    so the upstream affine de-mapping in ``imc_matmul`` works unchanged.
    """
    hw    = ctx.hw
    paths = ctx.paths
    table = ctx.cell_table

    rs, cs = hw.subarray_rows, hw.subarray_cols
    m, k   = A.shape
    _, n   = B.shape

    # --- 1. Voltage matrix --------------------------------------------------
    V = scales.a * A + scales.b           # (m, k)

    # --- 2. Per-subarray thicknesses ----------------------------------------
    # Tile B into per-subarray .npy files on disk, exactly as the DC
    # backend does — same layout, same naming, so the post-walk works.
    layer_name = f"{bname}_{tag}" if bname else tag     # block-unique folder
    distribute_weights(
        layer_name=layer_name, layer_weights=B,
        synaptic_array_size=(rs, cs),
        pe_size=(hw.pe_rows, hw.pe_cols),
        tile_size=(hw.tile_rows, hw.tile_cols),
        base_dir=ctx.hwsimdb_dir,
    )
    layer_dir = os.path.join(ctx.hwsimdb_dir, layer_name)

    # --- 3. Conductance window for the conversion ---------------------------
    # Use cell-level Gmin/Gmax (1T1R series) — the table is already 1T1R,
    # so the affine map and the table agree at the boundary points.
    alpha = hw.Ik1 / hw.Vref
    from spicecimx.hw.device_emit import effective_cell_window
    G_R_max_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    G_R_min_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    _, Gmin = effective_cell_window(G_R_max_bare, G_R_min_bare, hw.R_access)

    # --- 4. Accumulate I_SPICE = sum(pos) − sum(neg) ------------------------
    I_full   = np.zeros((m, n), dtype=np.float64)
    nan_mask = np.zeros((m, n), dtype=bool)
    from spicecimx.sim.transient_backend import _loading_active
    loading = _loading_active(ctx)
    R, C = -(-k // rs), -(-n // cs)
    if loading:
        I0_col = {b: np.zeros((m, R, C * cs)) for b in ("pos", "neg")}
        V_pad = np.zeros((m, R * rs))
        B_pad = np.zeros((R * rs, C * cs))

    n_jobs = 0
    for root, _dirs, files in os.walk(layer_dir, topdown=True):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(root, fname)
            sa_row, sa_col = array_indices_from_npy(fname)
            B_block = np.load(npy_path)               # (rs, cs)

            T_pos, T_neg = convert_to_conductance(
                B_block, scales.c, Gmin,
                hw.Tmin, hw.Tox, hw.Tref, alpha,
                R_access=hw.R_access, Tmax=hw.Tmax,
            )

            # Inner-dim slice of V for this subarray-row band
            r0 = sa_row * rs
            r1 = min(r0 + rs, k)
            V_sub_full = np.zeros((m, rs), dtype=V.dtype)
            V_sub_full[:, : r1 - r0] = V[:, r0:r1]    # zero-pad if needed

            # V-linearization: drive cells at V_phys = h⁻¹(V_target) so the
            # table lookup returns G_lin(T) · V_target instead of the cell's
            # native nonlinear current.  After this the BL current is
            # exactly what the affine demap expects.
            if ctx.v_linearizer is not None:
                V_phys = ctx.v_linearizer.inverse(V_sub_full)
            else:
                V_phys = V_sub_full
            # Hardware DAC: the cells see the measured DAC output for the
            # quantised drive code (no-op with a software DAC).
            pcv = ctx.physical_converters
            if pcv is not None and getattr(pcv, "dac_hw", False):
                V_phys = pcv.dac_apply(V_phys)
                V_phys[:, r1 - r0:] = 0.0      # padded rows stay undriven (0 V)

            I_pos = _eval_branch(T_pos, V_phys, table)   # (m, cs)
            I_neg = _eval_branch(T_neg, V_phys, table)   # (m, cs)

            # Differential: |I_pos| − |I_neg|.  The table stores the
            # NM0:s sign (negative for V_BL>0); abs() converts to the
            # "current flowing INTO the bit-line node" magnitude used by
            # the affine demap.  The two branches share the same sign
            # convention, so taking abs is equivalent to negating both
            # and yields the same differential.
            contrib = np.abs(I_pos) - np.abs(I_neg)    # (m, cs)

            c0 = sa_col * cs
            c1 = min(c0 + cs, n)
            I_full[:, c0:c1] += contrib[:, : c1 - c0]
            if loading:
                I0_col["pos"][:, sa_row, sa_col * cs:(sa_col + 1) * cs] = np.abs(I_pos)
                I0_col["neg"][:, sa_row, sa_col * cs:(sa_col + 1) * cs] = np.abs(I_neg)
                V_pad[:, sa_row * rs:(sa_row + 1) * rs] = V_phys
                B_pad[sa_row * rs:(sa_row + 1) * rs,
                      sa_col * cs:(sa_col + 1) * cs] = B_block
            n_jobs += 1

    if n_jobs == 0:
        raise RuntimeError(
            f"fastdc: no subarray .npy files found under {layer_dir}; "
            "did the netlist-emission step run?"
        )

    # Brief breadcrumb so per-layer progress is visible in the user log
    print(f"  [fastdc] {tag}: {n_jobs} subarrays evaluated by table")

    if loading:
        # Shared tile lines (arch.loading): table column currents as the
        # ideal-boundary I0, the pre-warped linear model for the row currents
        # (G * h(V_phys) = G * V_target), and the coupled line solve's
        # loaded currents as the result (no extra table passes).
        from spicecimx.arch.loading import LayerLoading, hprime, summarize
        G = {"pos": scales.c * np.maximum(B_pad, 0.0) + Gmin,
             "neg": scales.c * np.maximum(-B_pad, 0.0) + Gmin}
        LL = LayerLoading(G, ctx.ic_params)
        lin = ctx.v_linearizer
        hV = lin.forward(V_pad) if lin is not None else V_pad
        I0_row = np.empty((m, R * rs, 2 * C))
        for q in range(C):
            for b in ("pos", "neg"):
                I0_row[:, :, LL.tap(q, b)] = hV * G[b][:, q * cs:(q + 1) * cs].sum(axis=1)
        sol = LL.solve(I0_row, I0_col, hprime(lin, V_pad))
        I_load = (sol.I_col["pos"].sum(axis=1) - sol.I_col["neg"].sum(axis=1))[:, :n]
        ctx.loading_records.append(summarize(
            layer_name, LL, sol, I0_col, {}, passes=1, residual_V=float("nan"), m=m))
        I_full = I_load

    return I_full