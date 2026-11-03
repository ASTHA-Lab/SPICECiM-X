"""
CLI: build the array-level I-V characterisation table.

Usage::

    python -m spicecimx.cli.characterize_array --config config.ini [--force]

The table is written to ``paths.array_table_path``.  By default the
build is skipped if the existing file's stored ``config_hash`` matches
the current configuration (transistor model + corner + vdd + memristor
params + T/V grid + anchor sizes).  Use ``--force`` to override.

Inference (``cli.infer``) auto-builds the table on the first run if
``paths.array_table_path`` is unset OR the file's hash doesn't match —
so most users will never invoke this CLI directly.  It exists for
explicit characterisation (e.g. when sweeping config to study a process
corner or a new memristor model).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from spicecimx.config import load_cfg
from spicecimx.hw.array_table import ArrayCharTable
from spicecimx.sim.array_char_run import (
    ArrayCharSpec, characterize_array, compute_config_hash,
)


def _spec_from_cfg(cfg) -> ArrayCharSpec:
    return ArrayCharSpec(
        T_min=cfg.hw.Tmin, T_max=cfg.hw.Tmax, T_step=cfg.sim.array_char_T_step,
        V_min=cfg.sim.array_char_V_min, V_max=cfg.sim.array_char_V_max,
        V_step=cfg.sim.array_char_V_step,
        N_primary=cfg.sim.array_char_size_primary,
        N_secondary=cfg.sim.array_char_size_secondary,
        probe_column=cfg.sim.array_char_probe_column,
    )


def _hw_params(cfg) -> dict:
    # Only what enters the characterisation deck.  R_access does not (it is
    # *derived* from the table), so changing it must not force a rebuild.
    return {
        "Tmin": cfg.hw.Tmin, "Tmax": cfg.hw.Tmax,
        "Tref": cfg.hw.Tref, "Tox": cfg.hw.Tox,
        "Ik1":  cfg.hw.Ik1,  "Vref": cfg.hw.Vref,
    }


def build_or_load(cfg, *, force: bool = False, verbose: bool = True) -> ArrayCharTable:
    """
    Return an ``ArrayCharTable`` matching ``cfg`` — either by loading the
    cached file at ``cfg.paths.array_table_path`` (if its stored hash
    matches the current config) or by running characterisation.

    Used both by this CLI and (via auto-invocation) by ``cli.infer`` on
    first run.
    """
    spec = _spec_from_cfg(cfg)
    hw_params = _hw_params(cfg)
    table_path = cfg.paths.array_table_path
    if not os.path.isabs(table_path):
        table_path = os.path.join(cfg.paths.project_dir, table_path)

    cfg_hash = compute_config_hash(
        transistor_model=cfg.paths.transistor_model,
        process_corner=cfg.sim.process_corner,
        vdd=cfg.sim.vdd, vss=cfg.sim.vss, temperature=cfg.sim.temperature,
        tnom=cfg.sim.tnom,
        hw_params=hw_params, spec=spec,
        veriloga_path=cfg.paths.device_model_dc,
        stdcell_path=cfg.paths.stdcell_netlist,
        pdk_params=vars(cfg.pdk), parasitics_params=vars(cfg.parasitics),
    )

    if not force and os.path.exists(table_path):
        try:
            tab = ArrayCharTable.load_npz(table_path)
            if tab.config_hash == cfg_hash:
                if verbose:
                    print(f"[array_char] cache HIT: {table_path}")
                    print(f"             N=({tab.N_primary},{tab.N_secondary}), "
                          f"nT={tab.nT}, nV={tab.nV}, hash={cfg_hash[:12]}…")
                return tab
            else:
                if verbose:
                    print(f"[array_char] cache STALE at {table_path}")
                    print(f"             stored hash {tab.config_hash[:12]}… "
                          f"!= current {cfg_hash[:12]}… → rebuilding")
        except Exception as e:
            if verbose:
                print(f"[array_char] cache load failed ({e}); rebuilding")

    if verbose:
        print(f"[array_char] building anchors @ N=({spec.N_primary},{spec.N_secondary}); "
              f"this is a ONE-TIME cost cached at {table_path}")

    work_dir = Path(cfg.paths.tmp_root) / "_array_char"
    tab = characterize_array(
        spec=spec,
        transistor_model=cfg.paths.transistor_model,
        veriloga_path=cfg.paths.device_model_dc,
        stdcell_path=cfg.paths.stdcell_netlist,
        work_dir=str(work_dir),
        process_corner=cfg.sim.process_corner,
        vdd=cfg.sim.vdd, vss=cfg.sim.vss, temperature=cfg.sim.temperature,
        tnom=cfg.sim.tnom,
        spectre_extra_args=cfg.sim.spectre_args,
        config_hash=cfg_hash,
        timeout_s=cfg.sim.array_char_timeout_s,
        pdk=cfg.pdk, hw=cfg.hw, parasitics=cfg.parasitics,
    )

    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
    tab.save_npz(table_path)
    if verbose:
        print(f"[array_char] wrote {table_path}  ({tab.nT}x{tab.nV} per anchor)")
        h_prim = tab.build_v_linearization(N=tab.N_primary)
        h_secn = tab.build_v_linearization(N=tab.N_secondary)
        print(f"[array_char] super-linearity @ N={tab.N_primary}: "
              f"h(V_max)/V_max = {h_prim.super_linearity_factor:.3f}")
        print(f"[array_char] super-linearity @ N={tab.N_secondary}: "
              f"h(V_max)/V_max = {h_secn.super_linearity_factor:.3f}")
    return tab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build array-level I-V char table.")
    ap.add_argument("--config", required=True, help="config.ini path")
    ap.add_argument("--force",  action="store_true",
                    help="rebuild even if cached hash matches")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    build_or_load(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())