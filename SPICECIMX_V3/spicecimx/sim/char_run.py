"""
One-time SPICE characterisation of a single 1T1R cell, used to build the
``CellITable`` consumed by the fastdc backend.

The driver emits a tiny Spectre deck with a single 1T1R cell, a paramset
listing all ``(tinit, V_bl)`` combinations on the configured grid, and a
sweep wrapper that iterates the inner DC operating-point analysis across
every paramset row.  Spectre runs this as ONE elaboration + ``nT × nV``
Newton solves (each ~1 ms), so the whole characterisation completes in
seconds even for fine grids.

The cell schematic mirrors the user's reference testbench
(``input_1T1R.scs``)::

    BL ── I7(memristor, Tfilament=tinit) ── net1 ── NM0(W=2u, L=180n, WL=2.4V) ── 0
    V_BL is driven at the BL node by V0
    WL is held at 2.4V by V1

The measurement is ``I(NM0:s)`` — current into the source of the access
NMOS, which equals the current through the BL/memristor branch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from spicecimx.hw.char_table import CellITable
from spicecimx.io.parse_dc import PsfDcReader
from spicecimx.sim.spectre import build_spectre_command, run_spectre


# ---------------------------------------------------------------------------
# Deck emission
# ---------------------------------------------------------------------------
_CELL_NETLIST = """\
simulator lang=spectre

// 1T1R cell — access NMOS in series with the Verilog-A ReRAM.
// Topology matches the user's reference rram_cell_1T1R_180N_DC_Test:
//
//   BL ── I7 ── net1 ── NM0(D=net1, G=WL, S=0, B=0) ── 0
//
// I_NM0:s is what we save; it equals the BL-branch current.
NM0 (net1 WL 0 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u m=(1)*(1)
I7  (BL net1) sky130_dc_memristor_dc \\
        area_ox=1.024e-13 Tox=5e-09 \\
        Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 \\
        Tfilament_init=tinit \\
        I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 \\
        Temperature_0=300
V1  (WL 0) vsource dc=2.4    type=dc
V0  (BL 0) vsource dc=V_bl   type=dc
"""


def _build_cell_netlist(pdk, hw) -> str:
    """Config-driven single-cell characterisation netlist (replaces the
    hardcoded _CELL_NETLIST when [PDK]/[HW] are available).  Same topology
    (NMOS source grounded, WL gate drive, swept tinit/V_bl); transistor
    geometry, memristor primitive + params and WL read voltage all sourced
    from config."""
    from spicecimx.hw.device_emit import transistor_geom_inline, _va_params_dc
    return (
        "simulator lang=spectre\n\n"
        "// 1T1R cell — access NMOS in series with the Verilog-A ReRAM.\n"
        f"NM0 (net1 WL 0 0) {transistor_geom_inline(pdk)}\n"
        f"I7  (BL net1) {pdk.va_cell_dc} \\\n"
        f"        {_va_params_dc(hw, pdk)} Tfilament_init=tinit\n"
        f"V1  (WL 0) vsource dc={pdk.wl_read_V}    type=dc\n"
        "V0  (BL 0) vsource dc=V_bl   type=dc\n"
    )

_DECK_TEMPLATE = """\
// SPICECiM-X fastdc characterisation deck.  One elaboration, nT*nV Newton
// solves, one combined raw output suite under raw_dir/.
simulator lang=spectre
global 0
include "{transistor_model}" section={pcorner}
include "cell.scs"

parameters tinit={tinit0} V_bl={V_bl0}

simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 \\
    iabstol=1e-12 temp={temp} tnom={tnom} scalem=1.0 scale=1.0 gmin=1e-12 \\
    rforce=1 maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \\
    checklimitdest=psf

{paramset_block}

ahdl_include "{veriloga_path}"
save NM0:s
saveOptions options save=selected
"""


def _emit_paramset_block(T_grid: np.ndarray, V_grid: np.ndarray) -> str:
    """
    Build the ``cell_grid paramset { ... }`` block plus a ``sweep`` wrapper
    over a single inner ``dc`` operating-point analysis.

    Each paramset row pairs one (tinit, V_bl) combination.  Spectre will
    emit one PSF file per row, named ``cell_sweep-{NNN:03d}_op.dc``.
    """
    nT = len(T_grid); nV = len(V_grid)
    rows = []
    for ti in range(nT):
        for vi in range(nV):
            rows.append(f"  {T_grid[ti]:.6e} {V_grid[vi]:.6e}")
    rows_text = "\n".join(rows)

    return (
        "cell_grid paramset {\n"
        "  tinit V_bl\n"
        f"{rows_text}\n"
        "}\n"
        "\n"
        "cell_sweep sweep paramset=cell_grid {\n"
        "  op dc oppoint=rawfile maxiters=150 maxsteps=10000\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def _parse_char_output(
    raw_dir: Path, nT: int, nV: int, probe_name: str = "NM0:s",
) -> np.ndarray:
    """
    Read the per-iteration PSF files emitted by the sweep and reassemble
    them into the (nT, nV) current matrix.

    Iteration index r corresponds to (ti, vi) = (r // nV, r % nV) per
    the row order in ``_emit_paramset_block``.
    """
    total = nT * nV
    I_flat = np.full(total, np.nan, dtype=np.float64)
    missing = 0
    for r in range(total):
        for cand in (f"cell_sweep-{r:03d}_op.dc",
                     f"cell_sweep-{r}_op.dc"):
            p = raw_dir / cand
            if p.exists():
                rdr = PsfDcReader(p)
                try:
                    vals = rdr.column(probe_name)
                except KeyError:
                    # Fall back to any column ending in ":s" if probe
                    # name varies by Spectre version.
                    for c in rdr.columns:
                        if c.endswith(":s"):
                            vals = rdr.column(c); break
                    else:
                        missing += 1; break
                if vals.size:
                    I_flat[r] = float(vals[0])
                break
        else:
            missing += 1

    if missing:
        print(f"[char] WARNING: {missing}/{total} sweep iterations missing "
              f"in {raw_dir} — those entries will be NaN")

    return I_flat.reshape(nT, nV)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class CharSpec:
    """Grid spec for the fastdc characterisation."""
    T_min: float = 3.3e-9
    T_max: float = 4.9e-9
    T_step: float = 0.01e-9
    V_min: float = 0.0
    V_max: float = 1.0
    V_step: float = 0.01

    def T_grid(self) -> np.ndarray:
        nT = int(round((self.T_max - self.T_min) / self.T_step)) + 1
        return np.linspace(self.T_min, self.T_max, nT)

    def V_grid(self) -> np.ndarray:
        nV = int(round((self.V_max - self.V_min) / self.V_step)) + 1
        return np.linspace(self.V_min, self.V_max, nV)


def characterize_cell(
    spec: CharSpec,
    *,
    transistor_model: str,
    veriloga_path: str,
    work_dir: str | Path,
    process_corner: str = "stat",
    temperature: float = 27.0,
    tnom: float = 27.0,
    spectre_binary: str = "spectre",
    spectre_extra_args: Tuple[str, ...] = (),
    timeout_s: float = 3600.0,
    pdk=None, hw=None,
) -> CellITable:
    """
    Run a single Spectre characterisation and return the parsed
    ``CellITable``.  Caller is responsible for caching (``save_npz``).

    Parameters
    ----------
    spec
        Grid specification — defines the (T, V_BL) sweep ranges/steps.
    transistor_model
        Absolute path to the gpdk (or other PDK) Spectre model file
        containing the ``nmos1`` device.
    veriloga_path
        Absolute path to the ReRAM Verilog-A file.
    work_dir
        Where the deck, raw output, and log file land.  Will be created.
    process_corner
        PDK section name to ``include ... section=<corner>``.
    """
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    raw_dir = work / "psf"
    raw_dir.mkdir(parents=True, exist_ok=True)

    T_grid = spec.T_grid()
    V_grid = spec.V_grid()
    nT, nV = len(T_grid), len(V_grid)
    total = nT * nV
    print(f"[char] characterising 1T1R: {nT} T-points × {nV} V-points = "
          f"{total} operating points")

    # 1) Cell netlist (config-driven when [PDK]/[HW] available; else legacy)
    cell_text = _build_cell_netlist(pdk, hw) if (pdk is not None and hw is not None) \
        else _CELL_NETLIST
    (work / "cell.scs").write_text(cell_text)

    # 2) Top deck
    deck = _DECK_TEMPLATE.format(
        transistor_model=os.path.abspath(transistor_model),
        pcorner=process_corner,
        tinit0=spec.T_min,
        V_bl0=spec.V_min,
        temp=temperature,
        tnom=tnom,
        paramset_block=_emit_paramset_block(T_grid, V_grid),
        veriloga_path=os.path.abspath(veriloga_path),
    )
    deck_path = work / "input.scs"
    deck_path.write_text(deck)

    # 3) Run Spectre
    log_path = work / "spectre.out"
    cmd = build_spectre_command(
        input_deck=str(deck_path),
        raw_dir=str(raw_dir),
        log_path=str(log_path),
        extra_args=tuple(spectre_extra_args),
        format_="psfbin",
        binary=spectre_binary,
    )
    print(f"[char] running: {' '.join(cmd[:3])} ... (see {log_path})")
    cp = run_spectre(cmd, cwd=str(work), timeout_s=timeout_s)
    if cp.returncode != 0:
        raise RuntimeError(f"[char] spectre exited {cp.returncode}; no table "
                           f"was written. See {log_path}")

    # 4) Parse outputs
    I_2d = _parse_char_output(raw_dir, nT, nV)

    # 5) A few non-converged HRS corners are interpolated; a failed run aborts
    #    instead of producing an all-zero table.
    from spicecimx.sim.array_char_run import _fill_sparse_nans
    I_2d = _fill_sparse_nans(I_2d, "[char]", log_path)

    return CellITable(T_grid=T_grid, V_grid=V_grid, I=I_2d)