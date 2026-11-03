"""
Transient simulation backend for SPICECiM-X v3.

This is the function the v3 ``imc_matmul`` calls when ``sim.mode == 'transient'``.
It orchestrates the entire SPICE round-trip for one matrix-multiply:

    1. Affine-scale A to a voltage matrix V (vmin .. vmax).
    2. Generate the per-BL PWL voltage deck for the whole batch of M rows.
    3. Tile B into rs×cs subarrays under hwsimdb/<tag>/Tile{T}/PE{P}/...
    4. For each subarray .npy:
        a. map B-block to per-cell filament thickness (T_pos, T_neg)
        b. emit per-branch Verilog-A subcircuit decks
        c. render the testbench from _golden_input.scs
        d. build a SpectreJob (pos and neg) and add it to the batch
    5. Hand the whole job list to the JobScheduler — it fans out across
       Spectre licenses and pipelines post-processing per-subarray.
    6. Assemble per-subarray (M, cs) current matrices into the final
       (M, n_total) result with sign (+pos, -neg) and sum across sa-rows.
    7. Return I_SPICE of shape (m, n) — caller (``imc_matmul``) handles
       polarity, TIA+ADC, and affine de-mapping.

The backend is a *callable factory*: you build it once per run via
``make_transient_backend(...)`` and pass the resulting callable as
``imc_matmul(..., sim_backend=...)``.  This keeps simulator-specific
configuration out of the imc_matmul signature.

The post-processor for each subarray is a top-level function in this module
so that it pickles correctly through ProcessPoolExecutor with the 'spawn'
start method.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spicecimx.config import HwCfg, SimCfg, PathsCfg
from spicecimx.core.imc_matmul import AffineScales
from spicecimx.hw.netlist import distribute_weights
from spicecimx.hw.device_emit import (
    convert_to_conductance,
    write_device_models_transient,
    build_cell_emit_spec,
)
from spicecimx.io.parse_transient import (
    TransientReadSettings,
    read_psfascii_tran,
    build_time_windows,
    settled_value,
    energy_from_arrays,
)
from spicecimx.sim.templates import (
    render_pwl_sources_text,
    slice_pwl_sources_for_subarray,
    array_indices_from_npy,
    render_testbench_transient,
    pad_rows_to_subarrays,
)
from spicecimx.sim.spectre import (
    LicensePool,
    JobScheduler,
    SpectreJob,
    build_spectre_command,
    check_failed_jobs,
)


# ---------------------------------------------------------------------------
# Post-processor — top-level function so spawn-context ProcessPoolExecutor
# can pickle it.
# ---------------------------------------------------------------------------
def _postprocess_transient_subarray(job: SpectreJob) -> Dict[str, Any]:
    """
    Reduce one subarray-branch's tran.tran.tran into a per-window
    settled-current matrix of shape (M, ncolumn).

    Returns a dict with keys:
        'sa_row'   : subarray row index in the layer grid
        'sa_col'   : subarray col index in the layer grid
        'branch'   : 'pos' | 'neg'
        'currents' : np.ndarray of shape (M, ncolumn) — settled current per
                     window per IPRB net.  May contain NaNs if a window or
                     net had no samples.
    """
    extra = job.extra
    ncolumn:    int   = int(extra["ncolumn"])
    start_time: float = float(extra["start_time"])
    increment:  float = float(extra["increment"])
    sl:         float = float(extra["settling_lo"])
    sh:         float = float(extra["settling_hi"])
    reducer:    str   = str(extra["reducer"])

    tran_path = os.path.join(job.raw_dir, "tran.tran.tran")
    if not os.path.exists(tran_path):
        # Sim "succeeded" but no output file — degrade gracefully
        empty = np.full((1, ncolumn), np.nan, dtype=np.float64)
        return {
            "sa_row": job.subarray_row, "sa_col": job.subarray_col,
            "branch": job.branch, "currents": empty,
        }

    params, values = read_psfascii_tran(tran_path)
    windows = build_time_windows(params, values, start_time, increment)
    M = len(windows)
    if M == 0:
        empty = np.full((1, ncolumn), np.nan, dtype=np.float64)
        return {
            "sa_row": job.subarray_row, "sa_col": job.subarray_col,
            "branch": job.branch, "currents": empty,
        }

    out = np.full((M, ncolumn), np.nan, dtype=np.float64)
    for col in range(ncolumn):
        net = f"IPRB{col}:in"
        for w, (s, e) in enumerate(windows):
            v = settled_value(params, values, s, e, net, sl, sh, reducer)
            if v is not None:
                out[w, col] = v
    res = {
        "sa_row": job.subarray_row, "sa_col": job.subarray_col,
        "branch": job.branch, "currents": out,
        # (avg P, E, P_static, E_static, E_dynamic) of this deck — the same
        # integration as the per-layer reducer, done here where the waveform
        # is already parsed (the loaded RSN sums it per pass).
        "energy": energy_from_arrays(params, values),
    }
    # Loaded RSN: the current each input source delivers into its row
    # (terminal current VI<i>:p, positive INTO the source -> negate).
    nrow = int(extra.get("nrow", 0) or 0)
    if nrow:
        rows = np.full((M, nrow), np.nan, dtype=np.float64)
        for i in range(nrow):
            net = f"VI{i}:p"
            for w, (s, e) in enumerate(windows):
                v = settled_value(params, values, s, e, net, sl, sh, reducer)
                if v is not None:
                    rows[w, i] = -v
        res["row_currents"] = rows
    return res


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
@dataclass
class TransientBackendContext:
    """All the long-lived state the backend needs.

    A single TransientBackendContext is built once per ViT forward pass and
    reused across every ``imc_matmul`` call.  Sharing the LicensePool means
    every layer-tag competes for the same set of Spectre seats.

    Optional cell/array-table V-linearizer
    --------------------------------------
    When ``v_linearizer`` is set (any mode with paths.array_table_path or
    paths.char_table_path configured), the transient, DC, and FastDC backends
    all apply V_phys = h^-1(V_target) before emitting voltages (PWL amplitudes
    for transient, the paramset block for DC, the table lookup for FastDC).
    The cell/array then produces currents that match the linear-cell
    expectation the affine de-mapping was derived for, eliminating the
    ~10-20% per-layer error caused by the cell's V-nonlinearity (G_eff
    roughly doubles between V_BL=0.1V and V_BL=1V).  Prefer the array-level
    linearizer (captures BL_DRIVER IR drop / loading) over the single-cell
    one when both are available; infer.py handles that selection.
    """
    hw:        HwCfg
    sim:       SimCfg
    paths:     PathsCfg
    pool:      LicensePool
    scheduler: JobScheduler
    transient_read: TransientReadSettings
    # batch-scoped working dirs
    hwsimdb_dir: str
    tmp_dir:     str
    # PDK (transistor geometry, VA params, cell prefix) and crossbar parasitics
    # — both config-driven; ``Any`` to avoid importing config dataclasses here.
    pdk:        Optional[Any] = None
    parasitics: Optional[Any] = None
    # Optional voltage pre-warp (built from a SPICE cell-I-V table).
    # Type left as ``Any`` so this module doesn't drag the hw subpackage
    # into its import chain — the field is only consumed by dc_backend.
    v_linearizer: Optional[Any] = None
    # Optional physical converters (feature 5).  When set and the DAC role is
    # hardware, the ideal crossbar drive V is replaced by the measured DAC
    # output (dac_apply) before the PWL sources are emitted.  Built by
    # infer.py via peripheral_chain.characterize_converters; ``Any`` to avoid
    # importing the analog subpackage here.
    physical_converters: Optional[Any] = None
    # hwsimdb retention policy ([SIM] hwsimdb_retention: all|last_block|none)
    # and the block currently being simulated (for last_block pruning).
    hwsimdb_retention: str = "all"
    last_bname: str = ""
    # Tile interconnect ([INTERCONNECT]); ``ic_params`` is the resolved
    # arch.floorplan.InterconnectParams.  When loading is on, every layer is
    # simulated as a loaded RSN (arch.loading) and its summary is appended to
    # ``loading_records`` (written to the dump by infer.py).
    interconnect: Optional[Any] = None
    ic_params:    Optional[Any] = None
    loading_records: List[Dict[str, Any]] = field(default_factory=list)


def build_context(
    hw: HwCfg, sim: SimCfg, paths: PathsCfg,
    batch_id: str,
    pdk: Optional[Any] = None,
    parasitics: Optional[Any] = None,
) -> TransientBackendContext:
    """
    Build a ``TransientBackendContext`` for a single inference batch.

    Working directories are scoped by ``batch_id`` (typically derived from
    ``--images_dir``) so multiple concurrent runs of the tool never share
    files.
    """
    # Absolutize once at construction so every path built off these
    # (job_cwd, raw_dir, deck_path, log_path, seed_path, ...) is absolute
    # — Spectre's CLI args and includes resolve from process cwd, and the
    # per-job cwd we set later differs from the project cwd.
    hwsimdb_dir = os.path.abspath(os.path.join(paths.hwsimdb_root, batch_id))
    tmp_dir     = os.path.abspath(os.path.join(paths.tmp_root,     batch_id))
    os.makedirs(hwsimdb_dir, exist_ok=True)
    os.makedirs(tmp_dir,     exist_ok=True)

    pool = LicensePool(sim.license_count)
    sched = JobScheduler(
        license_pool=pool,
        sim_workers=sim.license_count,
        post_workers=sim.postprocess_workers,
        retries=2,
        retry_sleep_s=5.0,
        timeout_s=(float(sim.spectre_timeout_s)
                   if getattr(sim, "spectre_timeout_s", 0) else None),
        pipeline_postprocess=sim.pipeline_postprocess,
    )

    tr_settings = TransientReadSettings(
        start_time   = 0.0,
        increment    = 3 * sim.pulse_width + sim.trise + sim.tfall,
        settling_lo  = sim.settling_window_lo,
        settling_hi  = sim.settling_window_hi,
        reducer      = sim.settling_reducer,
    )

    return TransientBackendContext(
        hw=hw, sim=sim, paths=paths,
        pool=pool, scheduler=sched,
        transient_read=tr_settings,
        hwsimdb_dir=hwsimdb_dir, tmp_dir=tmp_dir,
        hwsimdb_retention=str(getattr(sim, "hwsimdb_retention", "all")
                              or "all").strip().lower(),
        pdk=pdk, parasitics=parasitics,
    )


def make_transient_backend(ctx: TransientBackendContext) -> Callable[..., np.ndarray]:
    """
    Return a callable ``backend(A, B, scales, tag, bname, dump_dir) -> I_SPICE``
    suitable for passing to ``imc_matmul``.
    """
    return lambda A, B, scales, tag, bname, dump_dir: _transient_imc(
        A=A, B=B, scales=scales, tag=tag, bname=bname, dump_dir=dump_dir,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Main backend body
# ---------------------------------------------------------------------------
def _reduce_and_prune_layer(layer_dir: str) -> None:
    """Disk-space guard: make sure the layer's *_pwr/_eng CSVs exist, then
    delete everything else (Array_* waveform dirs, netlists, npy tiles).

    The energy report only needs the tiny CSVs; the raw transient dumps are
    the ~GB/layer part.  CSVs already present are kept as-is."""
    import shutil as _shutil
    if not os.path.isdir(layer_dir):
        return
    tag = os.path.basename(os.path.normpath(layer_dir))
    eng_csv = os.path.join(layer_dir, f"{tag}_eng.csv")
    if not os.path.exists(eng_csv):
        try:
            from spicecimx.io.parse_transient import (
                process_power_energy_in_folder)
            process_power_energy_in_folder(
                layer_dir, os.path.join(layer_dir, f"{tag}.csv"))
        except Exception as e:
            print(f"[retention] pwr/eng reduce failed for {tag} "
                  f"({type(e).__name__}: {e}); keeping raw data.")
            return
    if not os.path.exists(eng_csv):
        print(f"[retention] no energy CSV produced for {tag}; keeping raw data.")
        return
    freed = 0
    for entry in os.listdir(layer_dir):
        full = os.path.join(layer_dir, entry)
        if entry.lower().endswith(".csv"):
            continue
        try:
            if os.path.isdir(full):
                freed += sum(os.path.getsize(os.path.join(r, f))
                             for r, _, fs in os.walk(full) for f in fs)
                _shutil.rmtree(full, ignore_errors=True)
            else:
                freed += os.path.getsize(full)
                os.remove(full)
        except OSError:
            pass
    human = (f"{freed/1e9:.2f} GB" if freed >= 1e9 else f"{freed/1e6:.0f} MB")
    print(f"[retention] pruned {tag}: kept CSVs, freed ~{human}")


def _prune_block(ctx: "TransientBackendContext", bname: str) -> None:
    """Reduce + prune every hwsimdb layer folder of block ``bname`` and drop
    its tmp netlist dirs."""
    import shutil as _shutil
    if not bname:
        return
    prefix = f"{bname}_"
    if os.path.isdir(ctx.hwsimdb_dir):
        for entry in sorted(os.listdir(ctx.hwsimdb_dir)):
            if entry.startswith(prefix) and                     os.path.isdir(os.path.join(ctx.hwsimdb_dir, entry)):
                _reduce_and_prune_layer(os.path.join(ctx.hwsimdb_dir, entry))
    if os.path.isdir(ctx.tmp_dir):
        for entry in sorted(os.listdir(ctx.tmp_dir)):
            if entry.startswith(prefix):
                _shutil.rmtree(os.path.join(ctx.tmp_dir, entry),
                               ignore_errors=True)


def _transient_imc(
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
    Run the full transient flow for one matmul.  Returns I_SPICE of shape
    ``(A.shape[0], B.shape[1])``.

    See the module docstring for the step-by-step.
    """
    hw      = ctx.hw
    sim     = ctx.sim
    paths   = ctx.paths
    sched   = ctx.scheduler
    rs, cs  = hw.subarray_rows, hw.subarray_cols
    m, k    = A.shape
    _, n    = B.shape

    # PDK-driven cell emission spec (transistor geometry, VA params, cell
    # prefix) — built from config; falls back to defaults when ctx.pdk is None.
    if ctx.pdk is not None:
        cell_spec = build_cell_emit_spec(hw, ctx.pdk)
        cell_prefix = ctx.pdk.cell_prefix
    else:
        cell_spec = None
        cell_prefix = "rram_cell_1T1R_180N_STD"

    # --- 1. Voltage matrix ----------------------------------------------------
    # V_target = a*A + b is the *linear-cell* drive the affine de-mapping in
    # imc_matmul was derived for.  The physical 1T1R cell (and the loaded
    # subarray) is super-linear in V_BL — G_eff at V_BL=1V is ~2x G_eff at
    # V_BL=0.1V — so driving the PWL sources with V_target directly leaves a
    # large per-layer error (~10-20%) that compounds across the ViT.  This is
    # exactly the error the user observed as "very high error in transient".
    #
    # When an array-level (or, as a fallback, single-cell) V-linearizer is
    # configured (paths.array_table_path / char_table_path, wired in by
    # infer.py onto ctx.v_linearizer), pre-warp V_target through h_array^-1(.)
    # so the array physically responds as G0_array * V_target — the linear-cell
    # behaviour the de-mapping expects.  This is the SAME compensation the DC
    # and FastDC backends already apply; it was declared on the context but
    # never applied here, which is why only transient kept the nonlinearity.
    # Without a linearizer this no-ops (back-compat).
    V_target = scales.a * A + scales.b       # shape (m, k)
    if getattr(ctx, "v_linearizer", None) is not None:
        V = ctx.v_linearizer.inverse(V_target)   # V_phys = h_array^-1(V_target)
        try:
            slf = ctx.v_linearizer.super_linearity_factor
            print(f"[{bname}/{tag}] transient V-linearization ON "
                  f"(array h(V), super-linearity {slf:.3f}); "
                  f"V_target[{V_target.min():.3f},{V_target.max():.3f}] -> "
                  f"V_phys[{V.min():.3f},{V.max():.3f}]", flush=True)
        except Exception:
            pass
    else:
        V = V_target
        print(f"[{bname}/{tag}] transient V-linearization OFF "
              f"(no array_table_path/char_table_path); affine demap will see "
              f"cell V-nonlinearity directly (~10-20% per-layer rmse expected)",
              flush=True)

    # --- 1b. Physical DAC stage (feature 5) ----------------------------------
    # If the DAC role is hardware (declared H in [SUBCIRCUITS]), replace the
    # ideal drive V with the DAC's *measured* output: quantise V to an N-bit
    # code and look up V_dac[code] from the SPICE-characterised transfer.  This
    # injects the real DAC INL/DNL/offset/gain into the crossbar drive — the
    # "simulate the DAC, feed its output to the crossbar" step.  No-op when the
    # DAC is software (the default) or no converters were characterised.
    pcv = getattr(ctx, "physical_converters", None)
    if pcv is not None and getattr(pcv, "dac_hw", False):
        V_before = V
        V = pcv.dac_apply(V)
        try:
            dnl = float(np.max(np.abs(V - V_before)))
            print(f"[{bname}/{tag}] physical DAC stage ON "
                  f"({pcv.dac_bits}b, measured transfer); max |V_dac - V_ideal| "
                  f"= {dnl:.4f} V", flush=True)
        except Exception:
            pass

    # Pad the drive to whole subarrays: the last row tile of a layer whose k
    # is not a multiple of rs would otherwise leave IN pins undriven
    # (floating); 0 V on a padded row draws no current (its weights are 0).
    V = pad_rows_to_subarrays(V, rs)

    # --- 2. PWL deck for the whole batch -------------------------------------
    pwl_text, M_rows, sim_time = render_pwl_sources_text(
        V, sim.pulse_width, sim.trise, sim.tfall, rs,
    )
    assert M_rows == m, f"PWL emitted {M_rows} rows, expected {m}"

    # --- 3. Tile B onto disk --------------------------------------------------
    # ── hwsimdb retention: when a new block starts, the previous block's
    # sims are complete (each layer's outputs were needed to proceed), so in
    # last_block mode we reduce its _pwr/_eng CSVs and free its raw data.
    if ctx.hwsimdb_retention == "last_block" and bname and             ctx.last_bname and bname != ctx.last_bname:
        _prune_block(ctx, ctx.last_bname)
    if bname:
        ctx.last_bname = bname

    # Block-unique folder name: tags repeat across transformer blocks
    # (h0_Q, WO, FFN_W1, ... appear once PER BLOCK), so keying the hwsimdb
    # folder on the bare tag made every block overwrite the previous one --
    # the summed crossbar energy then covered ONE block instead of all of
    # them.  Prefixing the block name keeps each block's decks and CSVs.
    layer_name = f"{bname}_{tag}" if bname else tag
    distribute_weights(
        layer_name=layer_name,
        layer_weights=B,
        synaptic_array_size=(rs, cs),
        pe_size=(hw.pe_rows, hw.pe_cols),
        tile_size=(hw.tile_rows, hw.tile_cols),
        base_dir=ctx.hwsimdb_dir,
    )
    layer_dir = os.path.join(ctx.hwsimdb_dir, layer_name)

    # --- 4. Build the SpectreJob list ----------------------------------------
    alpha = hw.Ik1 / hw.Vref
    # Use cell-level Gmin/Gmax (1T1R series combination). At HRS the
    # transistor is negligible so cell Gmin ≈ bare Gmin, but at LRS the
    # cell Gmax is compressed by 1/(1+R_access*Gmax_bare). See
    # device_emit.effective_cell_window.
    from spicecimx.hw.device_emit import effective_cell_window
    G_R_max_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    G_R_min_bare = float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    Gmax, Gmin = effective_cell_window(G_R_max_bare, G_R_min_bare, hw.R_access)


    loading = _loading_active(ctx)
    col_save = [f"IPRB{i}:in" for i in range(cs)]
    row_save: List[str] = []      # row currents come from KCL (_row_currents)

    # The seed_netlist.scs (full subarray + std cells) is shared read-only by
    # every branch of every subarray of this layer.  Rewritten on every call
    # so a changed subarray size / std-cell file / parasitics setting can never
    # reuse a stale netlist.  Absolute path so symlinks resolve from any cwd.
    # The loaded RSN passes use a second seed whose columns end on SLT<i>
    # (driven by the boundary sources) instead of VSS.
    from spicecimx.hw.netlist import generate_subarray
    seeds: Dict[str, str] = {}
    for term in (("ground", "source") if loading else ("ground",)):
        fn = "seed_netlist.scs" if term == "ground" else "seed_netlist_loaded.scs"
        seeds[term] = os.path.abspath(os.path.join(ctx.tmp_dir, layer_name, fn))
        os.makedirs(os.path.dirname(seeds[term]), exist_ok=True)
        with open(seeds[term], "w") as fh:
            fh.write(generate_subarray(
                rs, cs, paths.stdcell_netlist,
                cell_prefix=cell_prefix,
                parasitics=ctx.parasitics,
                sl_termination=term,
            ))

    # --- 4. Device decks (once per subarray-branch, reused by every pass) ----
    blocks: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(layer_dir, topdown=True):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue
            npy_path = os.path.join(root, fname)
            sa_row, sa_col = array_indices_from_npy(fname)
            B_block = np.load(npy_path)

            # Map weights to device thickness (continuous mapping with the same
            # 'c' the affine-derivation produced; matches v2 semantics).
            T_pos, T_neg = convert_to_conductance(
                B_block, scales.c, Gmin,
                hw.Tmin, hw.Tox, hw.Tref, alpha,
                R_access=hw.R_access, Tmax=hw.Tmax,
            )
            info = {"sa_row": sa_row, "sa_col": sa_col, "npy_path": npy_path,
                    "B": B_block, "cwd": {}}
            # Each (subarray, branch) gets its own working directory so
            # concurrent jobs cannot trample each other's input decks.
            for branch, T_branch in (("pos", T_pos), ("neg", T_neg)):
                job_cwd = os.path.join(
                    ctx.tmp_dir, layer_name,
                    f"Array_{sa_row}_{sa_col}", branch,
                )
                os.makedirs(job_cwd, exist_ok=True)
                # Emit one branch's device-model deck (only this branch).
                write_device_models_transient(
                    T_pos=T_branch, T_neg=T_branch,
                    baseline_va_path=paths.device_model_transient,
                    target_dir=job_cwd,
                    res_val=hw.res_val, cap_val=hw.cap_val,
                    pos_name=("device_model.scs" if branch == "pos" else "_unused.scs"),
                    neg_name=("_unused.scs" if branch == "pos" else "device_model.scs"),
                    spec=cell_spec,
                )
                info["cwd"][branch] = job_cwd
            blocks.append(info)

    def _std_raw_dir(info, branch):
        # Raw output goes alongside the .npy, matching v2's layout so
        # downstream tools (power/energy reducer, settle meter) still work.
        return os.path.join(os.path.splitext(info["npy_path"])[0] + f"_{branch}", "psf")

    def _make_job(info, branch, raw_dir, sources, terminal, pass_no):
        job_cwd = info["cwd"][branch]
        # The testbench includes 'seed_netlist.scs' relatively — symlink (or
        # copy) the right seed into each job_cwd.  lexists handles broken
        # symlinks from prior failed runs.
        local_seed = os.path.join(job_cwd, "seed_netlist.scs")
        if os.path.lexists(local_seed):
            try:
                os.unlink(local_seed)
            except OSError:
                pass
        try:
            os.symlink(seeds[terminal], local_seed)
        except OSError:
            shutil.copy2(seeds[terminal], local_seed)
        deck_path = os.path.join(job_cwd, "input.scs")
        render_testbench_transient(
            template_path=paths.golden_input_transient,
            output_path=deck_path,
            sim_time=sim_time,
            input_sources=sources,
            device_model_path=os.path.join(job_cwd, "device_model.scs"),
            transistor_model=paths.transistor_model,
            process_corner=sim.process_corner,
            vdd=sim.vdd, vss=sim.vss, temperature=sim.temperature,
            tnom=sim.tnom,
            iprobe_save_list=col_save + row_save,
            maxstep=getattr(sim, "tran_maxstep", 0.0),
        )
        # Start from an empty raw dir: the same layer folder is reused by
        # every image in per-image mode, and a stale waveform must never be
        # read in place of a failed simulation.
        shutil.rmtree(raw_dir, ignore_errors=True)
        os.makedirs(raw_dir, exist_ok=True)
        cmd = build_spectre_command(
            input_deck=deck_path, raw_dir=raw_dir,
            log_path=os.path.join(raw_dir, "spectre.out"),
            extra_args=sim.spectre_args,
        )
        r, c = info["sa_row"], info["sa_col"]
        return SpectreJob(
            job_id=f"{layer_name}/Array_{r}_{c}/{branch}"
                   + (f"/pass{pass_no}" if pass_no > 1 else ""),
            cmd=cmd, cwd=job_cwd, raw_dir=raw_dir,
            branch=branch, subarray_row=r, subarray_col=c,
            extra={
                "npy_path":    info["npy_path"],
                "ncolumn":     cs,
                "nrow":        len(row_save),
                "start_time":  ctx.transient_read.start_time,
                "increment":   ctx.transient_read.increment,
                "settling_lo": ctx.transient_read.settling_lo,
                "settling_hi": ctx.transient_read.settling_hi,
                "reducer":     ctx.transient_read.reducer,
            },
        )

    _lock = threading.Lock()

    def _run(jobs, label):
        n_jobs = len(jobs)
        print(f"[{bname}/{tag}] {label}dispatching {n_jobs} SPICE jobs "
              f"({n_jobs // 2} subarrays × 2 branches) "
              f"across {ctx.pool.size} license(s)...", flush=True)
        last = [0]

        def _progress(done: int, total: int, phase: str) -> None:
            # Coalesce: only print on every 10% step
            step = max(1, total // 10)
            if done == total or done - last[0] >= step:
                with _lock:
                    last[0] = done
                    print(f"  [{tag}/{phase}] {done}/{total}", flush=True)

        results = sched.run_layer(
            jobs,
            postprocess_fn=_postprocess_transient_subarray,
            progress_cb=_progress,
        )
        failed = check_failed_jobs(results, tag=tag, n_jobs=n_jobs,
                                   policy=getattr(ctx.sim, "on_job_failure", "raise"))
        return results, failed

    num_arrays_horizontally = (n + cs - 1) // cs
    num_arrays_vertically   = (k + rs - 1) // rs

    if not loading:
        # --- 5. Dispatch (ideal RSN boundaries: the original flow) ---------
        jobs = [_make_job(info, branch, _std_raw_dir(info, branch),
                          slice_pwl_sources_for_subarray(pwl_text, info["sa_row"], rs),
                          "ground", 1)
                for info in blocks for branch in ("pos", "neg")]
        results, failed = _run(jobs, "")
        I_SPICE = _assemble_from_results(results, failed, m, n, cs,
                                         num_arrays_horizontally)
    else:
        I_SPICE = _loaded_rsn_passes(
            ctx=ctx, blocks=blocks, V=V, scales=scales, Gmin=Gmin,
            m=m, n=n, rs=rs, cs=cs, R=num_arrays_vertically,
            C=num_arrays_horizontally, layer_name=layer_name, layer_dir=layer_dir,
            tag=tag, bname=bname, make_job=_make_job, run=_run,
            std_raw_dir=_std_raw_dir,
        )

    # hwsimdb retention 'none': this layer's numeric outputs are in hand, so
    # reduce its _pwr/_eng CSVs and free the raw sim data immediately.
    if ctx.hwsimdb_retention == "none":
        _reduce_and_prune_layer(layer_dir)
        import shutil as _shutil
        _shutil.rmtree(os.path.join(ctx.tmp_dir, layer_name),
                       ignore_errors=True)

    return I_SPICE


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------
def _assemble_from_results(results, failed, m, n, cs, num_arrays_horizontally):
    """Sum the per-subarray (M, cs) currents into (m, n) with sign
    (+pos, -neg); every sa_row adds into the same column slice."""
    # Accumulator: (M, num_arrays_horizontally * cs)
    I_full_padded = np.zeros((m, num_arrays_horizontally * cs), dtype=np.float64)
    nan_mask      = np.zeros_like(I_full_padded, dtype=bool)
    # A failed (subarray, branch) leaves its whole column slice unknown.
    for r in failed:
        c0 = r.job.subarray_col * cs
        nan_mask[:, c0:c0 + cs] = True

    for r in results:
        if not r.ok or r.post is None:
            continue
        pp:        Dict[str, Any] = r.post
        sa_col:    int = pp["sa_col"]
        branch:    str = pp["branch"]
        currents = _length_match(pp["currents"], m)
        c0 = sa_col * cs
        c1 = c0 + cs
        sign = +1 if branch == "pos" else -1

        # pos - neg in this fashion: add pos, subtract neg
        local_nan = np.isnan(currents)
        contribution = np.where(local_nan, 0.0, sign * currents)
        I_full_padded[:, c0:c1] += contribution
        nan_mask[:, c0:c1] |= local_nan

    # Mark columns whose contributions were all NaN
    I_full_padded[nan_mask] = np.nan
    # Trim padding columns down to n
    return I_full_padded[:, :n]


def _length_match(a: np.ndarray, m: int) -> np.ndarray:
    """Pad (NaN) or truncate the window axis of a (M_meas, x) array to m."""
    if a.shape[0] < m:
        return np.vstack([a, np.full((m - a.shape[0], a.shape[1]), np.nan)])
    return a[:m, :]


def _loading_active(ctx) -> bool:
    ic = getattr(ctx, "interconnect", None)
    return bool(ic is not None and getattr(ic, "enable", False)
                and getattr(ic, "loading", False)
                and getattr(ctx, "ic_params", None) is not None)


def _pass_currents(results, m, R, C, rs, cs):
    """Per-pass column currents {b: (m, R, C*cs)} of every subarray."""
    I_col = {b: np.full((m, R, C * cs), np.nan) for b in ("pos", "neg")}
    for r in results:
        if not r.ok or r.post is None:
            continue
        pp = r.post
        p, q, b = pp["sa_row"], pp["sa_col"], pp["branch"]
        I_col[b][:, p, q * cs:(q + 1) * cs] = _length_match(pp["currents"], m)
    return I_col


def _row_currents(I_col, V_cell, G, lin, R, C, rs, cs):
    """Row (bus-tap) currents (m, R*rs, 2C) consistent with the measured
    column currents.

    The row split inside a subarray comes from the cell model
    G * h(V_row - d_in); every subarray-window is then scaled so its rows sum
    to the SPICE column total (KCL).  The input sources' own terminal current
    is NOT used: in the first vector of every deck it also carries the
    decoder / BL-driver start-up transient (up to ~2x the cell current),
    which is not current the shared bus delivers to the cells."""
    hV = lin.forward(V_cell) if (lin is not None and hasattr(lin, "forward")) else V_cell
    m = I_col["pos"].shape[0]
    out = np.zeros((m, R * rs, 2 * C))
    for p in range(R):
        rows = slice(p * rs, (p + 1) * rs)
        for q in range(C):
            cols = slice(q * cs, (q + 1) * cs)
            for t, b in enumerate(("pos", "neg")):
                tap = 2 * q + t
                model = hV[:, rows, tap] * G[b][rows, cols].sum(axis=1)   # (m, rs)
                tot_model = model.sum(axis=1, keepdims=True)
                tot_meas = np.nansum(I_col[b][:, p, cols], axis=1, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.where(np.abs(tot_model) > 0, tot_meas / tot_model, 0.0)
                out[:, rows, tap] = model * k
    return out


def _deck_energy(results) -> Tuple[float, float, float]:
    """(total, static, dynamic) crossbar-deck energy of one pass, summed from
    the per-job post-processor values (``energy_from_arrays``, the same
    integration and static/dynamic split as the report's reducer)."""
    tot = st = dy = 0.0
    for r in results:
        e = (r.post or {}).get("energy") if getattr(r, "ok", False) else None
        if e is None:
            return float("nan"), float("nan"), float("nan")
        tot += e[1]; st += e[3]; dy += e[4]
    return tot, st, dy


def _loaded_rsn_passes(*, ctx, blocks, V, scales, Gmin, m, n, rs, cs, R, C,
                       layer_name, layer_dir, tag, bname, make_job, run,
                       std_raw_dir) -> np.ndarray:
    """Boundary relaxation over the shared tile lines (see arch.loading)."""
    import json
    from spicecimx.arch.loading import (LayerLoading, LoadingResult, hprime,
                                        summarize)

    ic, P = ctx.interconnect, ctx.ic_params
    sim = ctx.sim
    max_passes = int(max(1, ic.loading_max_passes))
    tol = float(ic.loading_tol_V)
    k_pad, n_pad = R * rs, C * cs
    V = V[:, :k_pad]

    # Programmed (linear) conductances of the whole layer, padded.
    B_pad = np.zeros((k_pad, n_pad))
    for info in blocks:
        r, c = info["sa_row"], info["sa_col"]
        B_pad[r * rs:(r + 1) * rs, c * cs:(c + 1) * cs] = info["B"]
    G = {"pos": scales.c * np.maximum(B_pad, 0.0) + Gmin,
         "neg": scales.c * np.maximum(-B_pad, 0.0) + Gmin}
    LL = LayerLoading(G, P)
    s = hprime(getattr(ctx, "v_linearizer", None), V)

    pwl = lambda X, **kw: render_pwl_sources_text(
        X, sim.pulse_width, sim.trise, sim.tfall, X.shape[1], **kw)[0]
    pass_root = lambda j: os.path.join(ctx.tmp_dir, layer_name, f"_pass{j}")
    pass_raw = lambda j, info, b: os.path.join(
        pass_root(j), f"Array_{info['sa_row']}_{info['sa_col']}_{b}", "psf")

    # pass 1: ideal boundaries (the RSN as the paper defines it)
    jobs = [make_job(info, b, pass_raw(1, info, b), pwl(V[:, info["sa_row"] * rs:
                                                         (info["sa_row"] + 1) * rs]),
                     "ground", 1)
            for info in blocks for b in ("pos", "neg")]
    results, failed = run(jobs, "pass 1/ideal: ")
    lin = getattr(ctx, "v_linearizer", None)
    I0_col = _pass_currents(results, m, R, C, rs, cs)
    V_tap = np.repeat(V[:, :, None], 2 * C, axis=2)          # drive at every bus tap
    I0_row = _row_currents(I0_col, V_tap, G, lin, R, C, rs, cs)
    E1 = _deck_energy(results)
    E_acc = E1

    ok = (not failed and all(np.all(np.isfinite(I0_col[b])) for b in I0_col))
    res_stats: Dict[str, Any] = {}
    if not ok:
        print(f"[{bname}/{tag}] loading skipped: pass 1 has failed jobs; "
              f"reporting the ideal RSN for this layer")
        accepted_j, I_col, I_row = 1, I0_col, np.nan_to_num(I0_row)
        d_in = np.zeros_like(I_row)
        d_out = {b: np.zeros_like(I0_col[b]) for b in I0_col}
        residual, passes_run = float("nan"), 1
    else:
        sol = LL.solve(I0_row, I0_col, s)
        d_in, d_out = sol.d_in, sol.d_out
        accepted_j, I_col, I_row = 1, sol.I_col, sol.I_row
        residual, passes_run = float("nan"), 1
        print(f"[{bname}/{tag}] loading: R={R} C={C} bus {LL.len_in_um:.0f} um "
              f"({LL.bus.seg.sum():.1f} ohm + R_drv {P.r_drv:g}), column "
              f"{LL.len_out_um:.0f} um ({LL.col.seg.sum():.1f} ohm + R_tia "
              f"{P.r_tia:.2f}); predicted max drop {d_in.max()*1e3:.2f} mV, "
              f"max column raise {max(d_out[b].max() for b in d_out)*1e3:.2f} mV",
              flush=True)
        I_pred = sol.I_col
        for j in range(2, max_passes + 1):
            # Thevenin column termination: own path resistance in the deck,
            # the other subarrays' contribution as the source (V_oc).
            R_self, V_oc = LL.column_thevenin(d_out, I_pred)
            jobs = []
            for info in blocks:
                r, c = info["sa_row"], info["sa_col"]
                rows = slice(r * rs, (r + 1) * rs)
                for b in ("pos", "neg"):
                    t = LL.tap(c, b)
                    src = (pwl(V[:, rows] - d_in[:, rows, t])
                           + "".join(f"RSL{i} (SLT{i} SLX{i}) resistor "
                                     f"r={max(R_self[r], 1e-6):.6g}\n" for i in range(cs))
                           + pwl(V_oc[b][:, r, c * cs:(c + 1) * cs],
                                 name="VSL", node="SLX", ref="VSS"))
                    jobs.append(make_job(info, b, pass_raw(j, info, b), src,
                                         "source", j))
            results, failed = run(jobs, f"pass {j}/loaded: ")
            Ij_col = _pass_currents(results, m, R, C, rs, cs)
            Ij_row = _row_currents(Ij_col, V_tap - d_in, G, lin, R, C, rs, cs)
            passes_run = j
            if failed or not all(np.all(np.isfinite(Ij_col[b])) for b in Ij_col):
                print(f"[{bname}/{tag}] loaded pass {j} incomplete; keeping pass "
                      f"{accepted_j}")
                break
            accepted_j, I_col, I_row = j, Ij_col, Ij_row
            E_acc = _deck_energy(results)
            # the boundary the decks actually saw: V_oc + own drop
            d_out = {b: V_oc[b] + R_self[None, :, None] * Ij_col[b] for b in Ij_col}
            chk_in, chk_out = LL.boundary_from_currents(Ij_row, Ij_col)
            # per-window worst deviation (V) of the exact line response from
            # the applied boundary
            res_w = np.maximum(
                np.max(np.abs(chk_in - d_in).reshape(m, -1), axis=1),
                np.max(np.stack([np.abs(chk_out[b] - d_out[b]).reshape(m, -1)
                                 for b in d_out], axis=0), axis=(0, 2)))
            residual = float(np.max(res_w))
            res_stats = {"residual_median_V": float(np.median(res_w)),
                         "residual_p90_V": float(np.percentile(res_w, 90)),
                         "residual_worst_window": int(np.argmax(res_w)),
                         "boundary_applied_max_V": max(float(np.max(d_out[b])) for b in d_out)}
            print(f"[{bname}/{tag}] pass {j}: boundary residual max "
                  f"{residual*1e3:.3f} mV (window {res_stats['residual_worst_window']}), "
                  f"median {res_stats['residual_median_V']*1e3:.3f} mV "
                  f"(tol {tol*1e3:.3f} mV)", flush=True)
            if residual <= tol or j == max_passes:
                break
            g_row, g_col = LL.secant_conductances(
                I0_row, I0_col, Ij_row, Ij_col, d_in, d_out,
                LL._row_g(s), {b: LL._col_g(s, b) for b in ("pos", "neg")})
            sol = LL.solve(I0_row, I0_col, s, g_row=g_row, g_col=g_col,
                           coupled=False)
            d_in, d_out, I_pred = sol.d_in, sol.d_out, sol.I_col

    # The accepted pass becomes the layer's crossbar waveforms (energy
    # reducer, settle meter); with max_passes = 1 the ideal waveforms are
    # kept and only the predicted currents carry the loading.
    for info in blocks:
        for b in ("pos", "neg"):
            dst = std_raw_dir(info, b)
            shutil.rmtree(os.path.dirname(dst), ignore_errors=True)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(pass_raw(accepted_j, info, b), dst)

    t_on = sim.pulse_width + 0.5 * (sim.trise + sim.tfall)
    energy = LL.line_energy(V, I_row, I_col, d_out, t_on)
    if accepted_j > 1:
        # The accepted decks already dissipate each tap's own path resistance
        # (the Thevenin RSL inside the deck, counted by pwr=total): remove it
        # from the analytic line I^2R so nothing is counted twice.
        R_self = np.diag(LL.col.Z)[::-1]
        in_deck = sum(float(np.sum(R_self[None, :, None] * I_col[b] ** 2))
                      for b in I_col) * t_on
        energy["ir_out_in_deck_J"] = in_deck
        energy["ir_out_J"] = max(0.0, energy["ir_out_J"] - in_deck)
        energy["total_J"] = (energy["ir_in_J"] + energy["ir_out_J"]
                             + energy["cv2_in_J"] + energy["cv2_out_J"])
    rec = summarize(f"{bname}_{tag}" if bname else tag, LL,
                    LoadingResult(d_in=d_in, d_out=d_out, I_row=I_row,
                                  I_col=I_col, iterations=0, converged=True),
                    I0_col, energy, passes=passes_run, residual_V=residual, m=m,
                    E_deck_ideal_J=E1[0], E_deck_loaded_J=E_acc[0])
    rec.update({"E_deck_ideal_static_J": E1[1], "E_deck_ideal_dynamic_J": E1[2],
                "E_deck_loaded_static_J": E_acc[1],
                "E_deck_loaded_dynamic_J": E_acc[2],
                "accepted_pass": accepted_j, **res_stats})
    ctx.loading_records.append(rec)
    try:
        with open(os.path.join(layer_dir, "loading.json"), "w") as f:
            json.dump(rec, f, indent=2)
    except OSError:
        pass

    I_full = (np.nansum(I_col["pos"], axis=1) - np.nansum(I_col["neg"], axis=1))
    bad = np.any(np.isnan(I_col["pos"]), axis=1) | np.any(np.isnan(I_col["neg"]), axis=1)
    I_full[bad] = np.nan
    return I_full[:, :n]
