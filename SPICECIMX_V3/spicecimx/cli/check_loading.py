"""
SPICE validation of the loaded-RSN model against a monolithic layer deck.

    python -m spicecimx.cli.check_loading --config config.ini \
        [--k 64] [--n 32] [--m 4] [--seed 0] [--out loading_check]

Builds one random (k x n) layer and simulates it three ways with the
configured PDK / corner / [INTERCONNECT] parameters:

  ideal RSN    every 16x16 subarray deck with ideal boundaries (the paper's
               original decomposition — pass 1 of the transient backend);
  loaded RSN   the transient backend with shared-line loading (boundary
               relaxation, arch.loading);
  monolithic   ONE Spectre deck holding every subarray of the layer (+ and -
               arrays), its BL/WL decoders, the input buses (driver R_out +
               wire R per segment, wire C at every tap) and the column lines
               (wire R + C, TIA input R into virtual ground) — the reference
               the RSN decomposition is meant to reproduce.

Reports the column-current error of both RSN variants against the
monolithic deck, the energies (deck + shared-line terms) and the MVM settle
time, and writes ``loading_check.json`` under --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import numpy as np

from spicecimx.config import load_cfg


def _monolithic_netlist(cfg, P, Gsel, R, C, rs, cs, workdir, bus_rc=True):
    """Write seed_netlist.scs + device_model.scs for the whole layer.

    ``Gsel`` maps (p, q, b) -> the branch device-deck text written by the
    backend (cells named <cell_prefix>_R<j>C<i>), renamed here per subarray."""
    from spicecimx.hw.netlist import (generate_crossbar_netlist,
                                      generate_bl_wl_decoder_netlist,
                                      parse_standard_cells)
    from spicecimx.arch.floorplan import input_bus_segments, column_line_segments
    std_path = cfg.paths.stdcell_netlist
    std = parse_standard_cells(std_path)
    nb = int(math.ceil(math.log2(rs)))
    dec, dec_name, _ = generate_bl_wl_decoder_netlist(nb, std)
    prefix = cfg.pdk.cell_prefix
    parts = [open(std_path).read(), dec]
    dev_parts, ahdl = [], None
    top = []
    for (p, q, b), dev_text in Gsel.items():
        s = f"{p}_{q}_{b}"
        lines = dev_text.splitlines()
        body = []
        for ln in lines:
            if ln.strip().startswith("ahdl_include"):
                ahdl = ahdl or ln
                continue
            body.append(ln.replace(f"{prefix}_R", f"c{s}_R"))
        dev_parts.append("\n".join(body))
        xb, name, wl, bl, sl = generate_crossbar_netlist(
            rs, cs, cell_prefix=f"c{s}", parasitics=cfg.parasitics)
        parts.append(xb.replace(name, f"xbar_{s}"))
        top.append(f"X{s} (" + " ".join(f"BL_{s}_{i}" for i in range(rs)) + " "
                   + " ".join(f"SL_{s}_{j}" for j in range(cs)) + " "
                   + " ".join(f"WL_{s}_{i}" for i in range(rs)) + f") xbar_{s}")
        t = 2 * q + (0 if b == "pos" else 1)
        if P.input_shared:      # the pass gates hang directly on the bus
            drv = [f"bus_{p * rs + i}_{t}" for i in range(2 ** nb)]
        elif not bus_rc:        # ideal input bus: every driver = the source
            drv = [f"SRC{p * rs + i}" for i in range(2 ** nb)]
        else:                   # row driver per subarray: unity buffer of its tap
            drv = [f"drv_{s}_{i}" for i in range(2 ** nb)]
            for i in range(rs):
                top.append(f"E_{s}_{i} (drv_{s}_{i} 0 bus_{p * rs + i}_{t} 0) vcvs gain=1")
                top.append(f"CT_{s}_{i} (bus_{p * rs + i}_{t} 0) capacitor "
                           f"c={P.tap_c_F:.6g}")
        top.append(f"D{s} (AE " + " ".join(f"A{i}" for i in range(nb)) + " VDD VSS "
                   + " ".join(f"BL_{s}_{i}" for i in range(2 ** nb)) + " "
                   + " ".join(drv) + " VDDP PGMEN "
                   + " ".join(f"WL_{s}_{i}" for i in range(2 ** nb)) + f") {dec_name}")
        for j in range(cs):
            top.append(f"IPRB_{s}_{j} (SL_{s}_{j} col_{b}_{q * cs + j}_{p}) iprobe")
    # decoder address inputs tied low (as in generate_subarray)
    for i in range(nb):
        top.append(f"VADDR{i} (A{i} VSS) vsource dc=0 type=dc")
    # input buses
    seg_in = input_bus_segments(C, P)
    for i in range(R * rs if (bus_rc or P.input_shared) else 0):
        top.append(f"RDRV{i} (SRC{i} bd_{i}) resistor r={max(P.r_drv, 1e-6):.6g}")
        prev = f"bd_{i}"
        for t, L in enumerate(seg_in):
            top.append(f"RB_{i}_{t} ({prev} bus_{i}_{t}) resistor "
                       f"r={max(L * P.r_um, 1e-6):.6g}")
            top.append(f"CB_{i}_{t} (bus_{i}_{t} 0) capacitor c={L * P.c_um_F:.6g}")
            prev = f"bus_{i}_{t}"
    # column lines (tap 0 = row-block R-1, nearest the TIA)
    seg_out = column_line_segments(R, P)
    for b in ("pos", "neg"):
        for jj in range(C * cs):
            top.append(f"RT_{b}_{jj} (cn_{b}_{jj} tia_{b}_{jj}) resistor "
                       f"r={max(P.r_tia, 1e-6):.6g}")
            top.append(f"IT_{b}_{jj} (tia_{b}_{jj} VSS) iprobe")
            prev = f"cn_{b}_{jj}"
            for t, L in enumerate(seg_out):
                node = f"col_{b}_{jj}_{R - 1 - t}"
                top.append(f"RC_{b}_{jj}_{t} ({prev} {node}) resistor "
                           f"r={max(L * P.r_um, 1e-6):.6g}")
                top.append(f"CC_{b}_{jj}_{t} ({node} 0) capacitor c={L * P.c_um_F:.6g}")
                prev = node
    with open(os.path.join(workdir, "seed_netlist.scs"), "w") as f:
        f.write("\n\n".join(parts) + "\n\n" + "\n".join(top) + "\n")
    with open(os.path.join(workdir, "device_model.scs"), "w") as f:
        f.write((ahdl or "") + "\n" + "\n".join(dev_parts) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser("SPICECiM-X loaded-RSN vs monolithic check")
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="loading_check")
    ap.add_argument("--licenses", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--maxstep", type=float, default=0.0,
                    help="tran maxstep (s) for every deck (0 = simulator "
                         "default; Spectre X presets ignore it)")
    ap.add_argument("--bus", choices=("ideal", "rc"), default="ideal",
                    help="monolithic input bus: 'ideal' (drivers = sources; the "
                         "energy comparison then isolates the column-line "
                         "loading) or 'rc' (DAC R_out + wire RC + driver "
                         "buffers; ideal sources recover part of the C V^2 "
                         "a real CMOS driver spends, so the analytic bus term "
                         "is not compared)")
    args = ap.parse_args(argv)

    from spicecimx.core.imc_matmul import derive_affine
    from spicecimx.arch.floorplan import resolve_interconnect
    from spicecimx.sim import transient_backend as tb
    from spicecimx.sim.templates import render_pwl_sources_text, render_testbench_transient
    from spicecimx.sim.spectre import build_spectre_command, run_spectre
    from spicecimx.io.parse_transient import (read_psfascii_tran, build_time_windows,
                                              settled_value, settle_time_in_window,
                                              process_power_energy_in_folder)
    from spicecimx.hw.device_emit import effective_cell_window

    cfg = load_cfg(args.config)
    if args.licenses:
        cfg.sim.license_count = args.licenses
    hw, sim, ic = cfg.hw, cfg.sim, cfg.interconnect
    ic.enable, ic.loading = True, True
    sim.tran_maxstep = args.maxstep
    if hw.R_access is None or hw.R_access < 0:
        hw.R_access = 0.0
    out = os.path.abspath(args.out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    cfg.paths.hwsimdb_root = os.path.join(out, "hwsimdb")
    cfg.paths.tmp_root = os.path.join(out, "tmp")
    P = resolve_interconnect(ic, hw, cfg.pdk, cfg.peripherals)
    rs, cs = hw.subarray_rows, hw.subarray_cols
    R, C = -(-args.k // rs), -(-args.n // cs)

    rng = np.random.default_rng(args.seed)
    A = rng.uniform(0.0, 1.0, (args.m, args.k))
    B = rng.standard_normal((args.k, args.n))
    alpha = hw.Ik1 / hw.Vref
    Gmax, Gmin = effective_cell_window(
        float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref))),
        float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref))), hw.R_access)
    sc = derive_affine(A, B, vmin=hw.vmin, vmax=hw.vmax, Gmin=Gmin, Gmax=Gmax)

    def backend(loading: bool, sub: str):
        ic.loading = loading
        ctx = tb.build_context(hw, sim, cfg.paths, batch_id=sub, pdk=cfg.pdk,
                               parasitics=cfg.parasitics)
        ctx.interconnect, ctx.ic_params = ic, P
        t0 = time.time()
        I = tb._transient_imc(A=A, B=B, scales=sc, tag="L", bname="chk",
                              dump_dir=None, ctx=ctx)
        return I, ctx, time.time() - t0

    I_load, ctx_l, t_l = backend(True, "loaded")
    rec = ctx_l.loading_records[0]
    I_ideal, ctx_i, t_i = backend(False, "ideal")

    # ---- monolithic deck --------------------------------------------------
    mono = os.path.join(out, "monolithic")
    os.makedirs(mono)
    dev = {}
    for root, _d, files in os.walk(os.path.join(ctx_l.tmp_dir, "chk_L")):
        if "device_model.scs" in files and os.path.basename(root) in ("pos", "neg"):
            arr = os.path.basename(os.path.dirname(root))       # Array_p_q
            _, p, q = arr.split("_")
            dev[(int(p), int(q), os.path.basename(root))] = open(
                os.path.join(root, "device_model.scs")).read()
    _monolithic_netlist(cfg, P, dict(sorted(dev.items())), R, C, rs, cs, mono,
                        bus_rc=(args.bus == "rc"))
    V = sc.a * A + sc.b
    V = np.hstack([V, np.zeros((args.m, R * rs - args.k))])
    src, M, sim_time = render_pwl_sources_text(V, sim.pulse_width, sim.trise,
                                               sim.tfall, R * rs, node="SRC")
    saves = ([f"IT_{b}_{jj}:in" for b in ("pos", "neg") for jj in range(C * cs)]
             + [f"IPRB_{p}_{q}_{b}_{j}:in" for p in range(R) for q in range(C)
                for b in ("pos", "neg") for j in range(cs)])
    render_testbench_transient(
        template_path=cfg.paths.golden_input_transient,
        output_path=os.path.join(mono, "input.scs"), sim_time=sim_time,
        input_sources=src, device_model_path="device_model.scs",
        transistor_model=cfg.paths.transistor_model,
        process_corner=sim.process_corner, vdd=sim.vdd, vss=sim.vss,
        temperature=sim.temperature, tnom=sim.tnom, iprobe_save_list=saves,
        maxstep=sim.tran_maxstep)
    raw = os.path.join(mono, "Array_mono", "psf")
    os.makedirs(raw)
    t0 = time.time()
    cp = run_spectre(build_spectre_command(os.path.join(mono, "input.scs"), raw,
                                           os.path.join(raw, "spectre.out"),
                                           extra_args=sim.spectre_args),
                     cwd=mono, retries=0, timeout_s=args.timeout)
    t_mono = time.time() - t0
    if cp.returncode != 0:
        raise RuntimeError(f"monolithic Spectre run failed (rc={cp.returncode}); "
                           f"see {os.path.join(raw, 'spectre.out')}")
    prm, val = read_psfascii_tran(os.path.join(raw, "tran.tran.tran"))
    win = build_time_windows(prm, val, 0.0, 3 * sim.pulse_width + sim.trise + sim.tfall)
    I_mono_b = {b: np.full((args.m, C * cs), np.nan) for b in ("pos", "neg")}
    settle = []
    for b in ("pos", "neg"):
        for jj in range(C * cs):
            for w, (s, e) in enumerate(win[:args.m]):
                v = settled_value(prm, val, s, e, f"IT_{b}_{jj}:in",
                                  sim.settling_window_lo, sim.settling_window_hi, "median")
                I_mono_b[b][w, jj] = np.nan if v is None else v
                ts = settle_time_in_window(prm, val, s, e, f"IT_{b}_{jj}:in",
                                           cfg.latency.settle_band,
                                           sim.settling_window_lo, sim.settling_window_hi)
                if ts is not None:
                    settle.append(ts)
    I_mono = (I_mono_b["pos"] - I_mono_b["neg"])[:, :args.n]
    process_power_energy_in_folder(mono, os.path.join(mono, "mono.csv"))
    from spicecimx.analog.pwr_energy import _extract_total_from_csv
    E_mono = float(_extract_total_from_csv(os.path.join(mono, "mono_eng.csv"),
                                           "Total_eng") or float("nan"))

    def err(I):
        d = I - I_mono
        span = float(np.nanmax(I_mono) - np.nanmin(I_mono))
        return {"max_abs_A": float(np.nanmax(np.abs(d))),
                "rmse_pct_of_range": 100 * float(np.sqrt(np.nanmean(d ** 2))) / span,
                "max_pct_of_range": 100 * float(np.nanmax(np.abs(d))) / span}

    from spicecimx.io.parse_transient import collect_worst_settle_time, TransientReadSettings
    st = TransientReadSettings(settling_lo=sim.settling_window_lo,
                               settling_hi=sim.settling_window_hi,
                               increment=3 * sim.pulse_width + sim.trise + sim.tfall)
    rsn_settle = collect_worst_settle_time(ctx_l.hwsimdb_dir, st,
                                           cfg.latency.settle_band).get("analog_settle_s", 0.0)
    line_settle = max(rec["tau_in_s"], rec["tau_out_s"]) * math.log(1 / cfg.latency.settle_band)
    lineE = rec["line_energy_J"]
    # The terms present in the monolithic deck: the column lines always; the
    # input bus only with --bus rc (and then not like-for-like, see --bus).
    line_in_mono = lineE["ir_out_J"] + lineE["cv2_out_J"] + lineE.get("ir_in_J", 0.0) \
        + (lineE["cv2_in_J"] if args.bus == "rc" else 0.0)
    res = {
        "node_nm": P.node_nm, "corner": sim.process_corner, "vdd": sim.vdd,
        "layer": {"k": args.k, "n": args.n, "m": args.m, "R": R, "C": C,
                  "cells": 2 * R * C * rs * cs},
        "params": P.as_dict(),
        "current_error_vs_monolithic": {"ideal_rsn": err(I_ideal[:, :args.n]),
                                        "loaded_rsn": err(I_load[:, :args.n])},
        "energy_J": {
            "monolithic": E_mono,
            "ideal_rsn_decks": rec["E_deck_ideal_J"],
            "loaded_rsn_decks": rec["E_deck_loaded_J"],
            "loaded_rsn_lines": lineE,
            "loaded_rsn_total": rec["E_deck_loaded_J"] + line_in_mono,
            "input_bus_cv2_analytic": lineE["cv2_in_J"],
        },
        "bus": args.bus,
        "settle_s": {"monolithic_worst": max(settle) if settle else float("nan"),
                     "rsn_worst": rsn_settle,
                     "rsn_plus_line_model": rsn_settle + line_settle},
        "loading": {k_: rec[k_] for k_ in ("d_in_max_V", "d_out_max_V", "residual_V",
                                           "passes", "current_change_rel")},
        "runtime_s": {"loaded_rsn": t_l, "ideal_rsn": t_i, "monolithic": t_mono},
    }
    E = res["energy_J"]
    res["energy_error_pct"] = {
        "ideal_rsn": 100 * (E["ideal_rsn_decks"] - E_mono) / E_mono,
        "loaded_rsn": 100 * (E["loaded_rsn_total"] - E_mono) / E_mono,
    }
    with open(os.path.join(out, "loading_check.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    ce = res["current_error_vs_monolithic"]
    print("=" * 72)
    print(f" loaded-RSN check  {P.node_nm:.0f} nm {sim.process_corner}  layer "
          f"{args.k}x{args.n} ({R}x{C} subarrays, {res['layer']['cells']} cells), "
          f"{args.m} vectors")
    print(f"  current error vs monolithic (rmse % of range): ideal RSN "
          f"{ce['ideal_rsn']['rmse_pct_of_range']:.3f}  |  loaded RSN "
          f"{ce['loaded_rsn']['rmse_pct_of_range']:.3f}")
    print(f"  energy: monolithic {E_mono*1e12:.3f} pJ | ideal RSN "
          f"{E['ideal_rsn_decks']*1e12:.3f} pJ ({res['energy_error_pct']['ideal_rsn']:+.2f}%)"
          f" | loaded RSN {E['loaded_rsn_total']*1e12:.3f} pJ "
          f"({res['energy_error_pct']['loaded_rsn']:+.2f}%)")
    print(f"  settle: monolithic {res['settle_s']['monolithic_worst']*1e12:.1f} ps | RSN "
          f"{rsn_settle*1e12:.1f} ps | RSN+line {res['settle_s']['rsn_plus_line_model']*1e12:.1f} ps")
    print(f"  runtime: monolithic {t_mono:.1f} s, loaded RSN {t_l:.1f} s")
    print(f"  written: {os.path.join(out, 'loading_check.json')}")
    print("=" * 72)
    return res


if __name__ == "__main__":
    main()
