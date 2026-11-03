"""
DC simulation backend for SPICECiM-X v3.

This is the function the v3 ``imc_matmul`` calls when ``sim.mode == 'dc'``.
It produces the same (m, n) ``I_SPICE`` matrix as the transient backend but
solves each image as a separate DC operating point rather than a transient
pulse window.  The trade-offs:

  * No time domain → no settling, no peripheral RC, no PWL overhead.  A DC
    inference solves much faster per operating point (often >10x).
  * Models *only* the ReRAM I-V curve and the NMOS access transistor at DC.
    Capacitances and Verilog-A dynamic state (Eact_*, velocity_k1, ...) are
    not exercised.  Use this mode for accuracy diagnostics, not for energy
    or timing claims.

Per subarray-branch we issue *one* Spectre invocation whose deck contains:

  * a parametric DC vsource per BL — ``vsource dc=table(img_idx, ...)``
    selects the right voltage for each image from a per-image lookup table,
  * the parametric DC analysis ``dc param=img_idx start=0 stop=M-1 step=1``,
    which produces M operating points in one go.

Each DC sweep point's iprobe currents are read out of the binary PSF dump
by ``PsfDcReader`` (already validated byte-exact in Phase 0).

The assembly logic — sign-aware pos/neg differencing, summation across
sa_rows, column-slicing across sa_cols — is identical to the transient
backend, so we share a small helper for the I_full accumulation.

Public API
----------
TransientBackendContext      -- reused from transient_backend (the context
                                 is mode-agnostic; only the backend callable
                                 differs).
make_dc_backend(ctx)         -> Callable[..., np.ndarray]
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from spicecimx.config import HwCfg, SimCfg, PathsCfg
from spicecimx.core.imc_matmul import AffineScales
from spicecimx.hw.netlist import distribute_weights, generate_subarray
from spicecimx.hw.device_emit import (
    build_cell_emit_spec,
    convert_to_conductance,
    write_device_models_dc,
)
from spicecimx.io.parse_dc import PsfDcReader
from spicecimx.sim.templates import (
    render_dc_sources_text,
    render_dc_altergroups,
    array_indices_from_npy,
    render_testbench_dc,
    pad_rows_to_subarrays,
)
from spicecimx.sim.spectre import (
    JobScheduler, SpectreJob, build_spectre_command, check_failed_jobs,
)
# Reuse the existing context type — it's mode-agnostic.
from spicecimx.sim.transient_backend import TransientBackendContext


# ---------------------------------------------------------------------------
# Post-processor (top-level so spawn-context pools can pickle it)
# ---------------------------------------------------------------------------
def _postprocess_dc_subarray(job: SpectreJob) -> Dict[str, Any]:
    """
    Reduce one subarray-branch's DC dump into a per-image current matrix
    of shape (M, ncolumn).

    Spectre's ``sweep`` analysis writes ONE PSF file per iteration, named
    ``<sweep_name>-<iter:03d>_<inner_name>.dc``.  For our deck — where
    the sweep is ``dc_sweep`` and the inner analysis is ``dc_op`` — the
    M files land at::

        psf/dc_sweep-000_dc_op.dc
        psf/dc_sweep-001_dc_op.dc
        ...
        psf/dc_sweep-{M-1:03d}_dc_op.dc

    Plus a ``dc_sweep_dc_op.sweep`` master index file (Spectre uses this
    to track which iteration produced which file; we don't need it).

    Each per-iteration file is a single-operating-point PSF that
    PsfDcReader handles via its ``_parse_single_op_section4`` path
    (sections 0, 1, 4 only — no SWEEP or TRACE section).  Row i of the
    output comes from file ``dc_sweep-{i:03d}_dc_op.dc``.

    The paramset+sweep mechanism is still much faster than the previous
    alter-chain approach because:
      * Spectre elaborates the deck ONCE (not M times)
      * The matrix structure / topology is reused across iterations
      * Newton solves use the previous iteration's solution as the IC,
        which converges in 1-3 iterations rather than the 10-20 of a
        cold start

    Returns a dict with keys identical to the transient post-processor:
        'sa_row', 'sa_col', 'branch', 'currents'.
    """
    extra = job.extra
    ncolumn: int = int(extra["ncolumn"])
    M:       int = int(extra["M"])

    out = np.full((M, ncolumn), np.nan, dtype=np.float64)
    parse_failures = 0
    missing_files = 0

    for i in range(M):
        # Spectre's exact filename. Try the documented pattern first; if
        # missing, fall back to the older single-combined-file names from
        # earlier iterations of this code (so we keep working if the user
        # has stale runs around).
        candidates = [
            f"dc_sweep-{i:03d}_dc_op.dc",
            f"dc_sweep-{i}_dc_op.dc",       # in case Spectre version doesn't zero-pad
        ]
        dc_path = None
        for cand in candidates:
            p = os.path.join(job.raw_dir, cand)
            if os.path.exists(p):
                dc_path = p
                break
        if dc_path is None:
            missing_files += 1
            continue
        try:
            rdr = PsfDcReader(dc_path)
        except Exception as e:
            parse_failures += 1
            if parse_failures <= 3:
                print(f"  [postproc] {job.job_id}: failed to parse "
                      f"{os.path.basename(dc_path)}: {type(e).__name__}: {e}")
            continue
        for col in range(ncolumn):
            net = f"IPRB{col}:in"
            try:
                vals = rdr.column(net)            # shape (1,) for single op
            except KeyError:
                continue
            if vals.size > 0:
                out[i, col] = float(vals[0])

    if missing_files or parse_failures:
        n_ok = M - missing_files - parse_failures
        print(f"  [postproc] {job.job_id}: {n_ok}/{M} ok, "
              f"{missing_files} missing iteration files, "
              f"{parse_failures} parse failures")

    return {
        "sa_row": job.subarray_row, "sa_col": job.subarray_col,
        "branch": job.branch, "currents": out,
    }


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
def make_dc_backend(ctx: TransientBackendContext) -> Callable[..., np.ndarray]:
    """Return a callable ``backend(A, B, scales, tag, bname, dump_dir) -> I_SPICE``."""
    return lambda A, B, scales, tag, bname, dump_dir: _dc_imc(
        A=A, B=B, scales=scales, tag=tag, bname=bname, dump_dir=dump_dir,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Main backend body
# ---------------------------------------------------------------------------
def _dc_imc(
    A:        np.ndarray,
    B:        np.ndarray,
    scales:   AffineScales,
    tag:      str,
    bname:    str,
    dump_dir: Optional[str],
    *,
    ctx:      TransientBackendContext,
) -> np.ndarray:
    """Run the full DC flow for one matmul.  Returns I_SPICE of shape (m, n)."""
    hw    = ctx.hw
    sim   = ctx.sim
    paths = ctx.paths
    sched = ctx.scheduler

    rs, cs = hw.subarray_rows, hw.subarray_cols
    m, k   = A.shape
    _, n   = B.shape

    # --- 1. Voltage matrix ----------------------------------------------------
    # V_target = a*A + b is what a *linear* cell would need to see for the
    # affine demap to recover A@B.  But the real 1T1R cell is super-linear
    # in V (G_eff @V=1V is ~2x G_eff @V=0.1V), so driving the cell with
    # V_target directly leaves a ~18% per-layer error that compounds across
    # the ViT.  When a cell-I-V table is loaded (paths.char_table_path),
    # pre-warp V_target through h⁻¹(·) so the cell physically responds as
    # G_lin(T)*V_target — i.e. exactly the linear-cell behaviour the demap
    # was derived for.  Without the table this no-ops, matching the
    # pre-linearization behaviour for backward compatibility.
    V_target = scales.a * A + scales.b                       # (m, k)
    if getattr(ctx, "v_linearizer", None) is not None:
        V = ctx.v_linearizer.inverse(V_target)               # V_phys = h⁻¹(V_target)
    else:
        V = V_target
    # Hardware DAC: drive the cells with the measured DAC output for the
    # quantised code (no-op with a software DAC).
    pcv = getattr(ctx, "physical_converters", None)
    if pcv is not None and getattr(pcv, "dac_hw", False):
        V = pcv.dac_apply(V)
    # Whole subarrays only: unused IN pins of the last row tile get 0 V
    # instead of floating (padded rows carry zero weights -> no current).
    V = pad_rows_to_subarrays(V, rs)
    sweep_var = "img_idx"  # legacy arg, no longer used by templates

    # PDK-driven cell emission (transistor geometry, VA params, cell prefix)
    # and crossbar parasitics — the same as the transient backend and the
    # array characterisation, so all three see the configured hardware.
    if ctx.pdk is not None:
        cell_spec = build_cell_emit_spec(hw, ctx.pdk)
        cell_prefix = ctx.pdk.cell_prefix
    else:
        cell_spec = None
        cell_prefix = "rram_cell_1T1R_180N_STD"

    # --- 3. Tile B onto disk --------------------------------------------------
    # Block-unique folder (tags repeat in every transformer block).
    layer_name = f"{bname}_{tag}" if bname else tag
    distribute_weights(
        layer_name=layer_name, layer_weights=B,
        synaptic_array_size=(rs, cs),
        pe_size=(hw.pe_rows, hw.pe_cols),
        tile_size=(hw.tile_rows, hw.tile_cols),
        base_dir=ctx.hwsimdb_dir,
    )
    layer_dir = os.path.join(ctx.hwsimdb_dir, layer_name)

    # --- 4. Build the SpectreJob list ----------------------------------------
    alpha = hw.Ik1 / hw.Vref
    # Use cell-level Gmin (1T1R series combination). At HRS the transistor
    # is negligible so this equals the bare-ReRAM Gmin to within ppm, but
    # we keep the explicit formula here for symmetry with the cell-level
    # Gmax used to derive c.  See device_emit.effective_cell_window.
    from spicecimx.hw.device_emit import effective_cell_window
    G_R_max_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    G_R_min_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    _, Gmin = effective_cell_window(G_R_max_bare, G_R_min_bare, hw.R_access)

    iprobe_save_list = [f"IPRB{i}:in" for i in range(cs)]

    # Seed netlist (shared, read-only) lives in the layer's tmp dir.  Rewritten
    # on every call so a changed size / std-cell file / parasitics setting can
    # never reuse a stale netlist.  ABSOLUTE path so the symlinks created
    # below resolve regardless of where they are followed from.
    seed_path = os.path.abspath(os.path.join(ctx.tmp_dir, layer_name, "seed_netlist.scs"))
    os.makedirs(os.path.dirname(seed_path), exist_ok=True)
    with open(seed_path, "w") as fh:
        fh.write(generate_subarray(rs, cs, paths.stdcell_netlist,
                                   cell_prefix=cell_prefix,
                                   parasitics=ctx.parasitics))

    # Each subarray row only sees its own rs consecutive VI columns; we
    # render both the source declarations AND the altergroups per-slice.
    # The source text from render_dc_sources_text is:
    #   "parameters vi0=0 vi1=0 ... vi{cols-1}=0\n\nVI0 (...) ...\nVI1 (...)\n..."
    # We re-emit a smaller declaration + slice the VI lines.
    _asset_cache: Dict[int, tuple] = {}

    def _sa_row_assets(sa_row: int) -> tuple:
        """Return (per-sa_row source text, per-sa_row paramset text); rendered
        once per subarray row and shared by its columns and both branches."""
        if sa_row not in _asset_cache:
            V_slice = V[:, sa_row * rs:(sa_row + 1) * rs]
            src_text, _ = render_dc_sources_text(V_slice, sweep_var=sweep_var, scol=rs)
            _asset_cache[sa_row] = (src_text, render_dc_altergroups(V_slice))
        return _asset_cache[sa_row]

    jobs: List[SpectreJob] = []
    for root, dirs, files in os.walk(layer_dir, topdown=True):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(root, fname)
            sa_row, sa_col = array_indices_from_npy(fname)
            B_block = np.load(npy_path)

            # Map weights to per-cell thickness (same continuous formula as
            # the transient flow — but the DC subcircuit ignores dynamic
            # parameters so we only emit the static initial state).
            T_pos, T_neg = convert_to_conductance(
                B_block, scales.c, Gmin,
                hw.Tmin, hw.Tox, hw.Tref, alpha,
                R_access=hw.R_access, Tmax=hw.Tmax,
            )

            for branch, T_branch in (("pos", T_pos), ("neg", T_neg)):
                job_cwd = os.path.join(
                    ctx.tmp_dir, layer_name,
                    f"Array_{sa_row}_{sa_col}", branch,
                )
                os.makedirs(job_cwd, exist_ok=True)

                if branch == "pos":
                    dev_path, _ = write_device_models_dc(
                        T_pos=T_branch, T_neg=T_branch,
                        baseline_va_path=paths.device_model_dc,
                        target_dir=job_cwd,
                        pos_name="device_model.scs",
                        neg_name="_unused.scs",
                        spec=cell_spec,
                    )
                else:
                    _, dev_path = write_device_models_dc(
                        T_pos=T_branch, T_neg=T_branch,
                        baseline_va_path=paths.device_model_dc,
                        target_dir=job_cwd,
                        pos_name="_unused.scs",
                        neg_name="device_model.scs",
                        spec=cell_spec,
                    )

                # Symlink the shared seed_netlist.scs in (or copy if symlink
                # fails — e.g. Windows or cross-filesystem). Use lexists so a
                # broken symlink from an earlier failed run is detected too.
                local_seed = os.path.join(job_cwd, "seed_netlist.scs")
                if os.path.lexists(local_seed):
                    try:
                        os.unlink(local_seed)
                    except OSError:
                        pass
                try:
                    os.symlink(seed_path, local_seed)
                except OSError:
                    shutil.copy2(seed_path, local_seed)

                # Render the DC testbench
                deck_path = os.path.join(job_cwd, "input.scs")
                sa_src, sa_alt = _sa_row_assets(sa_row)
                render_testbench_dc(
                    template_path=paths.golden_input_dc,
                    output_path=deck_path,
                    n_images=m,
                    input_sources=sa_src,
                    altergroups=sa_alt,
                    device_model_path=dev_path,
                    transistor_model=paths.transistor_model,
                    process_corner=sim.process_corner,
                    vdd=sim.vdd, vss=sim.vss,
                    temperature=sim.temperature,
                    tnom=sim.tnom,
                    iprobe_save_list=iprobe_save_list,
                    sweep_var=sweep_var,
                )

                # Raw DC output goes alongside the .npy.  Spectre writes
                # 'dc.dc' (analysis-name.analysis-type).  We request psfbin
                # because PsfDcReader handles the binary format.
                raw_dir = os.path.join(
                    os.path.splitext(npy_path)[0] + f"_{branch}", "psf",
                )
                # Empty raw dir: a stale dc_sweep-NNN file from an earlier
                # image/run must never stand in for a non-converged point.
                shutil.rmtree(raw_dir, ignore_errors=True)
                os.makedirs(raw_dir, exist_ok=True)
                log_path = os.path.join(raw_dir, "spectre.out")

                cmd = build_spectre_command(
                    input_deck=deck_path,
                    raw_dir=raw_dir,
                    log_path=log_path,
                    extra_args=sim.spectre_args,
                    format_="psfbin",
                )
                jobs.append(SpectreJob(
                    job_id=f"{layer_name}/Array_{sa_row}_{sa_col}/{branch}",
                    cmd=cmd, cwd=job_cwd, raw_dir=raw_dir,
                    branch=branch,
                    subarray_row=sa_row, subarray_col=sa_col,
                    extra={
                        "npy_path": npy_path,
                        "ncolumn":  cs,
                        "M":        m,
                    },
                ))

    # --- 5. Dispatch ----------------------------------------------------------
    n_jobs = len(jobs)
    print(f"[{bname}/{tag}] dispatching {n_jobs} DC jobs "
          f"({n_jobs // 2} subarrays × 2 branches) "
          f"across {ctx.pool.size} license(s)...", flush=True)

    _lock = threading.Lock()
    _last = [0]
    def _progress(done: int, total: int, phase: str) -> None:
        step = max(1, total // 10)
        if done == total or done - _last[0] >= step:
            with _lock:
                _last[0] = done
                print(f"  [{tag}/{phase}] {done}/{total}", flush=True)

    results = sched.run_layer(
        jobs,
        postprocess_fn=_postprocess_dc_subarray,
        progress_cb=_progress,
    )

    # --- 6. Assemble I_SPICE --------------------------------------------------
    failed = check_failed_jobs(results, tag=f"{bname}/{tag}", n_jobs=n_jobs,
                               policy=getattr(sim, "on_job_failure", "raise"))

    num_h = (n + cs - 1) // cs
    I_full = np.zeros((m, num_h * cs), dtype=np.float64)
    nan_mask = np.zeros_like(I_full, dtype=bool)
    for r in failed:                     # whole column slice unknown
        c0 = r.job.subarray_col * cs
        nan_mask[:, c0:c0 + cs] = True

    for r in results:
        if not r.ok or r.post is None:
            continue
        pp = r.post
        sa_col   = pp["sa_col"]
        branch   = pp["branch"]
        currents = pp["currents"]              # (M, cs)
        if currents.shape[0] < m:
            pad = np.full((m - currents.shape[0], cs), np.nan)
            currents = np.vstack([currents, pad])
        elif currents.shape[0] > m:
            currents = currents[:m, :]

        c0 = sa_col * cs
        c1 = c0 + cs
        sign = +1 if branch == "pos" else -1
        local_nan = np.isnan(currents)
        contribution = np.where(local_nan, 0.0, sign * currents)
        I_full[:, c0:c1] += contribution
        nan_mask[:, c0:c1] |= local_nan

    I_full[nan_mask] = np.nan
    return I_full[:, :n]