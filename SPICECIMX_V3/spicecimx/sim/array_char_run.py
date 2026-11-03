"""
Array-level SPICE characterisation driver.

Builds a Spectre deck whose electrical conditions are IDENTICAL to the DC
inference deck (same ``generate_subarray`` topology, same supplies, same
VI->IN driving, decoder address inputs tied low inside the seed netlist),
differing only in that (a) every cell shares one ``Tfilament_init
=tinit`` parameter and (b) every BL input shares one ``V_bl`` parameter.
A 2-D paramset then sweeps (tinit, V_bl) in a single Spectre invocation
per anchor size.

Because the deck reproduces inference's exact conditions, the measured
column current ``I_col(T, V)`` captures whatever the real array does -
cell I-V nonlinearity, BL_DRIVER pass-gate IR drop, decoder behaviour,
SL return path - without us having to model any of it.  The derived
``h_array(V)`` is therefore the correct calibration curve for inference
by construction.

Outputs an ``ArrayCharTable`` (I_col at two anchor sizes).  Caching is the
caller's job (see ``spicecimx.cli.characterize_array``).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from spicecimx.hw.array_table import ArrayCharTable
from spicecimx.hw.netlist import generate_subarray
from spicecimx.hw.device_emit import (
    _DC_VA_CELL, _DC_VA_PARAMS, build_cell_emit_spec, transistor_geom_inline,
)
from spicecimx.io.parse_dc import PsfDcReader
from spicecimx.sim.spectre import build_spectre_command, run_spectre


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass
class ArrayCharSpec:
    """Grid + topology spec for the array characterisation."""
    T_min: float
    T_max: float
    T_step: float
    V_min: float
    V_max: float
    V_step: float
    N_primary:   int
    N_secondary: int
    probe_column: int = 0   # which IPRB to read (all identical under uniform conditions)

    def T_grid(self) -> np.ndarray:
        nT = int(round((self.T_max - self.T_min) / self.T_step)) + 1
        return np.linspace(self.T_min, self.T_max, nT)

    def V_grid(self) -> np.ndarray:
        nV = int(round((self.V_max - self.V_min) / self.V_step)) + 1
        return np.linspace(self.V_min, self.V_max, nV)


# ---------------------------------------------------------------------------
# Uniform-T device model (mirrors device_emit.emit_subckts_dc, neg branch)
# ---------------------------------------------------------------------------
def _emit_uniform_device_model(N: int, veriloga_path: str,
                               *, T_param: str = "tinit", spec=None) -> str:
    """
    Emit the per-cell DC subckts that the seed netlist instantiates by name,
    with every cell's ``Tfilament_init`` bound to the shared parameter
    ``T_param`` instead of a per-cell float.

    PDK text (cell prefix, transistor geometry, memristor primitive + params)
    comes from ``spec`` (a device_emit.CellEmitSpec built from [PDK]/[HW]).
    When ``spec`` is None the original hardcoded 180nm strings are used, so
    existing callers stay byte-identical.
    """
    abs_va = os.path.abspath(veriloga_path)
    parts = [f'ahdl_include "{abs_va}" \n']
    if spec is None:
        prefix, va_cell, va_params = "rram_cell_1T1R_180N_STD", _DC_VA_CELL, _DC_VA_PARAMS
        nmos = ("    NM0 (net1 WL SL 0) nmos1 w=(2u) l=180n "
                "as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n        m=(1)*(1)")
    else:
        prefix, va_cell, va_params = spec.cell_prefix, spec.va_cell_dc, spec.va_params_dc
        nmos = spec.nmos_neg
    for r in range(N):
        for c in range(N):
            name = f"{prefix}_R{r}C{c}"
            parts.append(
                f"// Cell name: {name}\n"
                f"// View name: schematic\n"
                f"subckt {name} BL SL WL\n"
                f"{nmos}\n"
                f"   I0 (BL net1) {va_cell} {va_params} "
                f"Tfilament_init={T_param}\n"
                f"ends {name}\n"
                "// End of subcircuit definition.\n\n"
            )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Testbench (matches _golden_input_dc.scs supplies exactly)
# ---------------------------------------------------------------------------
_CHAR_DECK = """\
// SPICECiM-X array characterisation deck (N={N}x{N}).
// Electrical conditions identical to the DC inference testbench
// (_golden_input_dc.scs): same supplies, same VI->IN driving, address
// inputs tied low in the seed.  Only differences: uniform cell T (tinit) and uniform input
// voltage (V_bl), swept by a 2-D paramset.
simulator lang=spectre
global 0
include "{transistor_model}" section={pcorner}
include "device_model.scs"
include "seed_netlist.scs"

parameters tinit={tinit0} V_bl={V_bl0}

{vi_sources}

Vvd (VDD 0) vsource dc={vdd} type=dc
Vvs (VSS 0) vsource dc={vss} type=dc
Vae (AE  0) vsource dc={vdd} type=dc
Vpgmen (PGMEN 0) vsource dc={vss} type=dc
Vddp   (VDDP  0) vsource dc={vdd} type=dc

simulatorOptions options psfversion="1.4.0" reltol=1e-5 vabstol=1e-7 \\
    iabstol=1e-13 temp={temp} tnom={tnom} scalem=1.0 scale=1.0 gmin=1e-12 \\
    rforce=1 maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 \\
    checklimitdest=psf

{paramset_block}

save IPRB{probe}:in
saveOptions options save=selected
"""


def _emit_vi_sources(N: int) -> str:
    """One vsource per IN pin, all sharing the V_bl parameter.

    Matches inference's ``VI{c} (IN{c} 0) vsource dc=... type=dc`` form,
    but every source uses the single shared V_bl so the whole array sees
    a uniform input - the condition under which all columns read identically.
    The decoder exposes ``2**ceil(log2 N)`` IN pins; drive them all.
    """
    n_in = 2 ** max(1, int(np.ceil(np.log2(N))))
    return "\n".join(f"VI{c} (IN{c} 0) vsource dc=V_bl type=dc" for c in range(n_in))


def _emit_paramset_block(T_grid: np.ndarray, V_grid: np.ndarray) -> str:
    """2-D paramset (tinit, V_bl) + a single inner DC op-point analysis."""
    rows = [f"  {t:.6e} {v:.6e}" for t in T_grid for v in V_grid]
    return (
        "char_grid paramset {\n"
        "  tinit V_bl\n"
        + "\n".join(rows) + "\n"
        "}\n\n"
        "char_sweep sweep paramset=char_grid {\n"
        "  op dc oppoint=rawfile maxiters=200 maxsteps=20000\n"
        "}\n"
    )


# ---------------------------------------------------------------------------
# PSF parsing
# ---------------------------------------------------------------------------
def _parse_anchor_output(raw_dir: Path, nT: int, nV: int, probe: int) -> np.ndarray:
    """Reassemble (nT, nV) current array from the per-sweep-row PSF files.

    Row order matches ``_emit_paramset_block``: r = ti*nV + vi.
    """
    total = nT * nV
    I_flat = np.full(total, np.nan, dtype=np.float64)
    probe_name = f"IPRB{probe}:in"
    missing = 0
    for r in range(total):
        for cand in (f"char_sweep-{r:03d}_op.dc", f"char_sweep-{r}_op.dc"):
            p = raw_dir / cand
            if p.exists():
                rdr = PsfDcReader(p)
                try:
                    vals = rdr.column(probe_name)
                except KeyError:
                    vals = None
                    for c in rdr.columns:
                        if c.startswith("IPRB") and c.endswith(":in"):
                            vals = rdr.column(c); break
                    if vals is None:
                        missing += 1; break
                if vals.size:
                    I_flat[r] = float(vals[0])
                break
        else:
            missing += 1
    if missing:
        print(f"[array_char] WARNING: {missing}/{total} sweep rows missing in {raw_dir}")
    return np.abs(I_flat.reshape(nT, nV))


# ---------------------------------------------------------------------------
# Run one anchor
# ---------------------------------------------------------------------------
def _run_one_anchor(
    N: int, spec: ArrayCharSpec, *,
    transistor_model: str, veriloga_path: str, stdcell_path: str,
    work_dir: Path, process_corner: str, vdd: float, vss: float,
    temperature: float, spectre_binary: str, spectre_extra_args, timeout_s: float,
    tnom: float = 27.0,
    cell_spec=None, cell_prefix: str = "rram_cell_1T1R_180N_STD", parasitics=None,
) -> np.ndarray:
    """Run characterisation for ONE anchor size; return (nT, nV) current."""
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "psf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    T_grid, V_grid = spec.T_grid(), spec.V_grid()
    nT, nV = len(T_grid), len(V_grid)
    print(f"[array_char] N={N}x{N}: {nT} T-points x {nV} V-points "
          f"= {nT*nV} operating points")

    # 1) Seed netlist - EXACT inference topology (crossbar + BL/WL decoder,
    #    iprobes-to-VSS), generated by the same function inference uses.
    (work_dir / "seed_netlist.scs").write_text(
        generate_subarray(N, N, stdcell_path,
                          cell_prefix=cell_prefix, parasitics=parasitics)
    )
    # 2) Uniform-T device model (cells reference the shared tinit parameter).
    (work_dir / "device_model.scs").write_text(
        _emit_uniform_device_model(N, veriloga_path, spec=cell_spec)
    )
    # 3) Testbench.
    deck = _CHAR_DECK.format(
        N=N, transistor_model=os.path.abspath(transistor_model),
        pcorner=process_corner, tinit0=spec.T_min, V_bl0=spec.V_min,
        temp=temperature, vdd=vdd, vss=vss, tnom=tnom,
        vi_sources=_emit_vi_sources(N),
        paramset_block=_emit_paramset_block(T_grid, V_grid),
        probe=spec.probe_column,
    )
    deck_path = work_dir / "input.scs"
    deck_path.write_text(deck)

    cmd = build_spectre_command(
        input_deck=str(deck_path),
        raw_dir=str(raw_dir),
        log_path=str(work_dir / "spectre.out"),
        extra_args=tuple(spectre_extra_args),
        format_="psfbin",
        binary=spectre_binary,
    )
    print(f"[array_char] running: {' '.join(cmd[:3])} ... "
          f"(see {work_dir / 'spectre.out'})")
    cp = run_spectre(cmd, cwd=str(work_dir), timeout_s=timeout_s)
    if cp.returncode != 0:
        raise RuntimeError(
            f"[array_char] spectre exited {cp.returncode} for N={N}; nothing was "
            f"cached. See {work_dir / 'spectre.out'}")

    I_2d = _parse_anchor_output(raw_dir, nT, nV, spec.probe_column)
    I_2d = _fill_sparse_nans(I_2d, f"[array_char] N={N}", work_dir / "spectre.out")
    return I_2d


def _fill_sparse_nans(I_2d: np.ndarray, label: str, log_path,
                      max_frac: float = 0.02) -> np.ndarray:
    """Tolerate a few non-converged operating points (HRS corners), never a
    failed characterisation: above ``max_frac`` NaN the build is aborted so
    a broken table can't be cached.  Isolated NaNs are filled by linear
    interpolation along V within their T row (0 A at V = 0)."""
    nan = np.isnan(I_2d)
    n_nan = int(nan.sum())
    if n_nan == 0:
        return I_2d
    if n_nan > max_frac * I_2d.size:
        raise RuntimeError(
            f"{label}: {n_nan}/{I_2d.size} operating points missing — the "
            f"characterisation failed; nothing was cached. See {log_path}")
    out = I_2d.copy()
    x = np.arange(out.shape[1])
    for i in range(out.shape[0]):
        row = out[i]
        bad = np.isnan(row)
        if bad.all():
            out[i] = 0.0
        elif bad.any():
            out[i, bad] = np.interp(x[bad], x[~bad], row[~bad])
    print(f"{label}: WARNING: {n_nan} non-converged point(s) interpolated")
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def characterize_array(
    spec: ArrayCharSpec, *,
    transistor_model: str, veriloga_path: str, stdcell_path: str,
    work_dir,
    process_corner: str = "stat",
    vdd: float = 2.4, vss: float = 0.0, temperature: float = 27.0,
    tnom: float = 27.0,
    spectre_binary: str = "spectre",
    spectre_extra_args: Tuple[str, ...] = (),
    timeout_s: float = 7200.0,
    config_hash: str = "",
    pdk=None, hw=None, parasitics=None,
) -> ArrayCharTable:
    """Run both anchor characterisations and return the assembled table.

    When ``pdk``/``hw`` are supplied the per-cell decks + crossbar are emitted
    from [PDK]/[HW] (fully portable); otherwise the legacy 180 nm hardcoded
    strings are used (back-compat).
    """
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cell_spec = build_cell_emit_spec(hw, pdk) if (pdk is not None and hw is not None) else None
    cell_prefix = pdk.cell_prefix if pdk is not None else "rram_cell_1T1R_180N_STD"
    kwargs = dict(
        spec=spec, transistor_model=transistor_model, veriloga_path=veriloga_path,
        stdcell_path=stdcell_path, process_corner=process_corner,
        vdd=vdd, vss=vss, temperature=temperature, tnom=tnom,
        spectre_binary=spectre_binary, spectre_extra_args=spectre_extra_args,
        timeout_s=timeout_s,
        cell_spec=cell_spec, cell_prefix=cell_prefix, parasitics=parasitics,
    )
    I_prim = _run_one_anchor(spec.N_primary,   work_dir=work / f"N{spec.N_primary}",   **kwargs)
    I_secn = _run_one_anchor(spec.N_secondary, work_dir=work / f"N{spec.N_secondary}", **kwargs)
    return ArrayCharTable(
        T_grid=spec.T_grid(), V_grid=spec.V_grid(),
        N_primary=spec.N_primary, N_secondary=spec.N_secondary,
        I_primary=I_prim, I_secondary=I_secn,
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------------
# Cache hashing
# ---------------------------------------------------------------------------
def compute_config_hash(
    *, transistor_model: str, process_corner: str, vdd: float, temperature: float,
    hw_params: dict, spec: ArrayCharSpec,
    vss: float = 0.0, tnom: float = 27.0,
    veriloga_path: str, stdcell_path: str,
    pdk_params: dict = None, parasitics_params: dict = None,
) -> str:
    """
    Deterministic hex digest of everything that affects array-char output.
    Files referenced by path are hashed by their CONTENTS (SHA256), so
    swapping the PDK, the memristor Verilog-A, or the std-cell library
    invalidates the cache automatically.  PDK geometry/VA params and crossbar
    parasitics are folded in too, so changing [PDK] or [PARASITICS] rebuilds.
    """
    h = hashlib.sha256()

    def add_str(s: str) -> None:
        h.update(s.encode("utf-8")); h.update(b"\x00")

    def add_file(path: str) -> None:
        try:
            with open(path, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        except OSError:
            h.update(b"<missing>" + str(path).encode("utf-8"))
        h.update(b"\x00")

    # Deck revision: bump when _CHAR_DECK changes so old caches rebuild.
    # (v2: PGMEN/VDDP driven like the inference decks.)
    add_str("char_deck=v2")
    add_str(process_corner)
    add_str(f"vdd={vdd:.6f}")
    add_str(f"temp={temperature:.6f}")
    add_str(json.dumps(hw_params, sort_keys=True))
    # Backward-compatible cache keying: only perturb the digest when these
    # actually change the emitted deck.  A cache built BEFORE [PDK]/[PARASITICS]
    # existed stays valid when the hardware is unchanged (PDK left at its
    # defaults, parasitics disabled) — so we add NOTHING in that case and the
    # digest is byte-identical to the legacy scheme.  A customized PDK or
    # enabled parasitics correctly forces a rebuild.
    if pdk_params is not None:
        try:
            from spicecimx.config import PdkCfg
            default_pdk = vars(PdkCfg())
        except Exception:
            default_pdk = None
        if default_pdk is None or pdk_params != default_pdk:
            add_str("pdk=" + json.dumps(pdk_params, sort_keys=True))
    # vss / tnom: only perturb when non-default so legacy caches (vss=0, tnom=27)
    # stay valid; changing supply rail or nominal temperature forces a rebuild.
    if abs(vss - 0.0) > 1e-12:
        add_str(f"vss={vss:.6f}")
    if abs(tnom - 27.0) > 1e-9:
        add_str(f"tnom={tnom:.6f}")
    if parasitics_params is not None and parasitics_params.get("enable"):
        add_str("parasitics=" + json.dumps(parasitics_params, sort_keys=True))
    add_str(json.dumps({
        "T_min": spec.T_min, "T_max": spec.T_max, "T_step": spec.T_step,
        "V_min": spec.V_min, "V_max": spec.V_max, "V_step": spec.V_step,
        "N_primary": spec.N_primary, "N_secondary": spec.N_secondary,
        "probe_column": spec.probe_column,
    }, sort_keys=True))
    add_file(transistor_model)
    add_file(veriloga_path)
    add_file(stdcell_path)
    return h.hexdigest()