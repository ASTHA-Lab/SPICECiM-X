"""
Build the fastdc cell I-V characterisation table.

Run once per device/PVT corner; the result is a ~100 KB .npz that the
fastdc backend loads at inference time.

Usage::

    python -m spicecimx.cli.characterize --config config.ini \\
        [--veriloga supports/sky130_dc_memristor_dc.v] [--output supports/cell_table.npz]

The grid (T range, V_BL range, step sizes) is taken from
``[SIM] fastdc_T_min``/``T_max``/``T_step`` and ``fastdc_V_min``/``V_max``
/``V_step`` in config.ini, or from the dataclass defaults if not set.
The transistor model and process corner come from ``[PATHS]`` and
``[SIM]`` respectively.

Spectre runs ONE elaboration + ``nT * nV`` Newton solves (so about
161 × 101 = 16 261 ops for the default grid).  Each Newton solve is
~1 ms with IC continuation, so total wall time is typically under
30 s.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from spicecimx.config import load_cfg
from spicecimx.sim.char_run import CharSpec, characterize_cell


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the fastdc cell I-V characterisation table."
    )
    p.add_argument("--config", type=str, default="config.ini",
                   help="Path to config.ini (default: ./config.ini).")
    p.add_argument("--veriloga", type=str, default=None,
                   help="ReRAM DC Verilog-A file (default: $SPICECIMX_VERILOGA, "
                        "else [PATHS] device_model_dc).")
    p.add_argument("--output", type=str, default=None,
                   help="Where to write the .npz (default: "
                        "paths.char_table_path from config).")
    p.add_argument("--work_dir", type=str, default=None,
                   help="Scratch dir for the deck + raw outputs "
                        "(default: <tmp>/char).")
    p.add_argument("--spectre", type=str, default="spectre",
                   help="Spectre binary name (default: 'spectre').")
    p.add_argument("--corner", type=str, default=None,
                   help="PDK section name (default: from sim.process_corner).")
    p.add_argument("--T_min", type=float, default=None)
    p.add_argument("--T_max", type=float, default=None)
    p.add_argument("--T_step", type=float, default=None)
    p.add_argument("--V_min", type=float, default=None)
    p.add_argument("--V_max", type=float, default=None)
    p.add_argument("--V_step", type=float, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_cfg(args.config)

    spec = CharSpec(
        T_min  = args.T_min  if args.T_min  is not None else cfg.sim.fastdc_T_min,
        T_max  = args.T_max  if args.T_max  is not None else cfg.sim.fastdc_T_max,
        T_step = args.T_step if args.T_step is not None else cfg.sim.fastdc_T_step,
        V_min  = args.V_min  if args.V_min  is not None else cfg.sim.fastdc_V_min,
        V_max  = args.V_max  if args.V_max  is not None else cfg.sim.fastdc_V_max,
        V_step = args.V_step if args.V_step is not None else cfg.sim.fastdc_V_step,
    )

    # The DC ReRAM model: --veriloga > $SPICECIMX_VERILOGA > [PATHS] device_model_dc
    veriloga = (args.veriloga or os.environ.get("SPICECIMX_VERILOGA")
                or cfg.paths.device_model_dc)
    if not veriloga or not os.path.exists(veriloga):
        sys.exit(f"ERROR: ReRAM DC Verilog-A model not found ({veriloga!r}); set "
                 f"[PATHS] device_model_dc or pass --veriloga.")

    output = args.output or cfg.paths.char_table_path
    if not os.path.isabs(output):
        output = os.path.join(cfg.paths.project_dir, output)
    if not output.lower().endswith(".npz"):
        # the table is always written as .npz; a .csv char_table_path can't be
        # overwritten in its own format
        output = os.path.splitext(output)[0] + ".npz"
        print(f"[char] NOTE: writing {output}; set [PATHS] char_table_path to it "
              f"to use the new table.")

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.path.join(cfg.paths.project_dir, cfg.paths.tmp_root, "char")

    corner = args.corner or getattr(cfg.sim, "process_corner", "stat")
    transistor_model = cfg.paths.transistor_model
    if not os.path.isabs(transistor_model):
        transistor_model = os.path.join(cfg.paths.project_dir, transistor_model)

    print(f"[char] config: T={spec.T_min:.2e}..{spec.T_max:.2e} step {spec.T_step:.2e}")
    print(f"[char]         V={spec.V_min:.2f}..{spec.V_max:.2f} step {spec.V_step:.2f}")
    print(f"[char]         transistor_model = {transistor_model}")
    print(f"[char]         veriloga         = {veriloga}")
    print(f"[char]         work_dir         = {work_dir}")
    print(f"[char]         output           = {output}")

    table = characterize_cell(
        spec,
        transistor_model=transistor_model,
        veriloga_path=veriloga,
        work_dir=work_dir,
        process_corner=corner,
        temperature=cfg.sim.temperature,
        tnom=cfg.sim.tnom,
        spectre_binary=args.spectre,
        timeout_s=cfg.sim.cell_char_timeout_s,
        pdk=cfg.pdk, hw=cfg.hw,
    )
    table.save_npz(output)
    print(f"[char] table saved: {output}")
    print(f"[char] shape: ({table.nT}, {table.nV}), "
          f"I range: {table.I.min():.3e} .. {table.I.max():.3e} A")


if __name__ == "__main__":
    main()