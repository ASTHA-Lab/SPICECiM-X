"""
ReRAM device modeling and per-cell Verilog-A subcircuit emission.

For every cell in a 1T1R subarray we emit a unique ``rram_cell_1T1R_180N_STD_R{r}C{c}``
subcircuit whose body wraps an instance of the underlying Verilog-A model with
a programmed ``initial_state`` (transient) or ``Tfilament_init`` (DC) parameter.

The two flavors share the conductance→thickness mapping (`convert_to_conductance*`
family) and differ only in:
  * which Verilog-A primitive cell they instantiate
    (`sky130_fd_pr_reram__reram_cell` vs `sky130_dc_memristor_dc`),
  * which parameter name receives the programmed thickness,
  * whether the dynamic-model parameters (Eact_*, velocity_k1, ...) are
    emitted (transient) or skipped (DC, since the device is stateless).

A single subarray emits TWO decks per branch — one for the positive
conductance branch (cells where B[r,c] >= 0) and one for the negative branch
(cells where B[r,c] < 0).  In both, every cell physically exists; the
"unused" branch is held in HRS by setting its thickness to Tmin.

Public API
----------
conductance_calc(alpha, Tox, Tref, Tmax, Tmin) -> (Gmax, Gmin)
convert_to_conductance(B, DeltaG, Gmin, Tmin, Tox, Tref, alpha) -> (T_pos, T_neg)
convert_to_conductance_ptq(...)         -> (T_pos, T_neg)
convert_to_conductance_continuous(...)  -> (T_pos, T_neg)
quantize_weights_to_int(weights, bits)  -> ndarray of ints

emit_subckts_transient(T_pos, T_neg, baseline_va_path, *, ...) -> (pos_text, neg_text)
emit_subckts_dc(T_pos, T_neg, baseline_va_path, *, ...) -> (pos_text, neg_text)

write_device_models_transient(T_pos, T_neg, baseline_va_path, target_dir,
                              *, res_val=0, cap_val=0) -> (pos_path, neg_path)
write_device_models_dc(T_pos, T_neg, baseline_va_path, target_dir) -> (pos_path, neg_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PDK-driven emission spec (replaces the old module-level hardcoded constants).
# Build one from cfg.pdk + cfg.hw and pass it to the emit/write functions; the
# legacy constants below are kept only as the fallback default so existing
# callers that don't pass a spec still produce byte-identical output.
# ---------------------------------------------------------------------------
@dataclass
class CellEmitSpec:
    """Everything PDK/process-specific needed to emit a 1T1R cell subckt."""
    cell_prefix: str
    va_cell_transient: str
    va_cell_dc: str
    va_params_transient: str
    va_params_dc: str
    nmos_pos: str               # full NMOS instance line, 'pos' branch wiring
    nmos_neg: str               # full NMOS instance line, 'neg' branch wiring

    def cell_name(self, r: int, c: int) -> str:
        return f"{self.cell_prefix}_R{r}C{c}"


def _nmos_line(pdk, net_src: str) -> str:
    """Build the access-transistor instance line from PDK geometry.
    ``net_src`` is the source node (``net2`` for pos branch, ``SL`` for neg)."""
    return (f"    NM0 (net1 WL {net_src} 0) {pdk.transistor_model_name} "
            f"w=({pdk.transistor_w}) l={pdk.transistor_l} "
            f"as={pdk.transistor_as} ad={pdk.transistor_ad} "
            f"ps={pdk.transistor_ps} pd={pdk.transistor_pd} \\\n"
            f"        m={pdk.transistor_m}")


def transistor_geom_inline(pdk) -> str:
    """Just the geometry+model portion of the NMOS line, single-line form, for
    callers (e.g. the characterisation CLIs) that wrap it with their own nets:
    ``nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u m=(1)*(1)``."""
    return (f"{pdk.transistor_model_name} w=({pdk.transistor_w}) "
            f"l={pdk.transistor_l} as={pdk.transistor_as} ad={pdk.transistor_ad} "
            f"ps={pdk.transistor_ps} pd={pdk.transistor_pd} m={pdk.transistor_m}")


def _va_params_transient(hw, pdk) -> str:
    """Transient Verilog-A param string from [HW] physical params + [PDK]."""
    return (
        f"area_ox={pdk.va_area_ox:g} "
        f"Tox={hw.Tox:g} Tfilament_max={hw.Tmax:g} Tfilament_min={hw.Tmin:g} "
        f"Tfilament_0={hw.Tmin:g} Eact_generation={pdk.va_eact_generation:g} "
        f"Eact_recombination={pdk.va_eact_recombination:g} "
        f"I_k1={hw.Ik1:g} Tfilament_ref={hw.Tref:g} V_ref={hw.Vref:g} "
        f"velocity_k1={pdk.va_velocity_k1:g} gamma_k0={pdk.va_gamma_k0:g} "
        f"gamma_k1={pdk.va_gamma_k1:g} Temperature_0={pdk.va_temperature_0:g} "
        f"C_thermal={pdk.va_c_thermal:g} tau_thermal={pdk.va_tau_thermal:g} "
        f"t_step={pdk.va_t_step:g} "
        f"enable_switching={int(getattr(pdk, 'va_enable_switching', 0))}"
    )


def _va_params_dc(hw, pdk) -> str:
    """DC (stateless) Verilog-A param string from [HW] + [PDK]."""
    return (
        f"area_ox={pdk.va_area_ox:g} "
        f"Tox={hw.Tox:g} Tfilament_max={hw.Tmax:g} Tfilament_min={hw.Tmin:g} "
        f"I_k1={hw.Ik1:g} Tfilament_ref={hw.Tref:g} V_ref={hw.Vref:g} "
        f"Temperature_0={pdk.va_temperature_0:g}"
    )


def build_cell_emit_spec(hw, pdk) -> CellEmitSpec:
    """Construct a CellEmitSpec from the [HW] and [PDK] config sections."""
    return CellEmitSpec(
        cell_prefix=pdk.cell_prefix,
        va_cell_transient=pdk.va_cell_transient,
        va_cell_dc=pdk.va_cell_dc,
        va_params_transient=_va_params_transient(hw, pdk),
        va_params_dc=_va_params_dc(hw, pdk),
        nmos_pos=_nmos_line(pdk, "net2"),
        nmos_neg=_nmos_line(pdk, "SL"),
    )


# ---------------------------------------------------------------------------
# Conductance-window helpers
# ---------------------------------------------------------------------------
def conductance_calc(
    alpha: float, Tox: float, Tref: float, Tmax: float, Tmin: float
) -> Tuple[float, float]:
    """
    Convert (alpha, Tox, Tref, Tmax, Tmin) to the (Gmax, Gmin) conductance
    window of the BARE ReRAM device.  Inverse of the I(V) calibration used
    by the Verilog-A models.

    Note this is the BARE ReRAM window, not the realised 1T1R cell window.
    Call :func:`effective_cell_window` to derive the achievable cell-level
    bounds when an access transistor is present.
    """
    Gmax = alpha * np.exp(-((Tox - Tmax) / (Tox - Tref)))
    Gmin = alpha * np.exp(-((Tox - Tmin) / (Tox - Tref)))
    return float(Gmax), float(Gmin)


def effective_cell_window(
    G_R_max: float, G_R_min: float, R_access: float,
) -> Tuple[float, float]:
    """
    Map the bare-ReRAM conductance window to the realised 1T1R cell window
    when an access transistor of (linear-region) ON-resistance R_access is
    in series with each ReRAM:

        G_cell(G_R) = G_R / (1 + R_access * G_R)

    At HRS (small G_R) the transistor is negligible, so G_cell_min ≈ G_R_min.
    At LRS (large G_R) the transistor compresses the cell conductance.

    Returns (G_cell_max, G_cell_min).
    """
    if R_access <= 0.0:
        return float(G_R_max), float(G_R_min)
    G_cell_max = G_R_max / (1.0 + R_access * G_R_max)
    G_cell_min = G_R_min / (1.0 + R_access * G_R_min)
    return float(G_cell_max), float(G_cell_min)


def compensate_for_transistor(
    G_cell_desired: np.ndarray, R_access: float,
) -> np.ndarray:
    """
    Solve the cell equation for the ReRAM conductance that, in series with
    a fixed-R_access transistor, realises the desired total cell conductance:

        G_cell = G_R / (1 + R_access * G_R)
        =>  G_R = G_cell / (1 - R_access * G_cell)

    The output is the BARE ReRAM conductance to program; pass it to the
    thickness-inversion that goes into the Verilog-A model.

    Compensation is needed because the affine de-mapping in
    :mod:`spicecimx.core.imc_matmul` assumes G_cell (not G_R) is affine in
    the weight; without it, the LRS half of the conductance window suffers
    a Möbius-shaped non-linearity (~3-7% of full scale for typical 180nm
    R_access values) that compounds across subarray rows.
    """
    if R_access <= 0.0:
        return G_cell_desired
    denom = 1.0 - R_access * G_cell_desired
    # Guard against the singular point where the transistor would dominate
    # so completely that no finite G_R can achieve the requested G_cell.
    # In practice this should never fire (G_cell_max is derived from
    # effective_cell_window, which sits comfortably below 1/R_access).
    if np.any(denom <= 0.0):
        raise ValueError(
            "compensate_for_transistor: requested G_cell >= 1/R_access; "
            "the cell cannot reach that conductance because the access "
            "transistor saturates first. Reduce G_R_max or R_access."
        )
    return G_cell_desired / denom


def quantize_weights_to_int(weights: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric signed PTQ; output is int-valued in [-L, L] with L = 2^{b-1}-1."""
    levels_pos = 2 ** (bits - 1) - 1
    max_abs = float(np.max(np.abs(weights)))
    if max_abs < 1e-12:
        return np.zeros_like(weights, dtype=int)
    scaled = weights / max_abs * levels_pos
    q = np.clip(np.round(scaled), -levels_pos, levels_pos)
    return q.astype(int)


# ---------------------------------------------------------------------------
# Weight -> thickness mapping (three variants, same surface as v2)
# ---------------------------------------------------------------------------
def convert_to_conductance(
    B: np.ndarray, DeltaG: float, Gmin: float,
    Tmin: float, Tox: float, Tref: float, alpha: float,
    *,
    R_access: float = 0.0,
    Tmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a continuous weight matrix B to per-cell programming thickness T,
    using a constant DeltaG (the same ``c`` that imc_matmul derives).

    With ``Tmax`` given, T is clipped to [Tmin, Tmax]: the DC Verilog-A model
    rejects a thickness outside its range, and float rounding at G = Gmax
    can land a hair above Tmax.

    When ``R_access > 0`` the per-cell target ``DeltaG * |B| + Gmin`` is
    interpreted as the desired TOTAL 1T1R cell conductance.  We solve
    G_R = G_cell / (1 - R_access * G_cell) to back out the bare ReRAM
    conductance that, in series with the transistor, realises the target.
    With ``R_access == 0`` we recover the v2-style "bare ReRAM is affine"
    behavior.

    Returns (T_pos, T_neg):
      * T_pos[r,c] = T_del if B[r,c] >= 0 else Tmin   (Tmin = HRS = "off")
      * T_neg[r,c] = T_del if B[r,c]  < 0 else Tmin
    """
    G_cell_desired = DeltaG * np.abs(B) + Gmin
    G_R_required = compensate_for_transistor(G_cell_desired, R_access)
    T_del = Tox + (Tox - Tref) * np.log(G_R_required / alpha)
    if Tmax is not None:
        T_del = np.clip(T_del, Tmin, Tmax)
    T_pos = np.where(B >= 0, T_del, Tmin)
    T_neg = np.where(B <  0, T_del, Tmin)
    return T_pos, T_neg


def convert_to_conductance_ptq(
    quantized_weights: np.ndarray,
    Gmax: float, Gmin: float,
    Tmin: float, qbit: int,
    Tox: float, Tref: float, alpha: float,
    *,
    R_access: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform-step mapping for signed-int quantized weights."""
    levels_pos = 2 ** (qbit - 1) - 1
    if levels_pos <= 0:
        raise ValueError(f"qbit must be >= 2, got {qbit}")
    delG = (Gmax - Gmin) / levels_pos
    G_cell_desired = delG * np.abs(quantized_weights) + Gmin
    G_R_required = compensate_for_transistor(G_cell_desired, R_access)
    T_del = Tox + (Tox - Tref) * np.log(G_R_required / alpha)
    T_pos = np.where(quantized_weights >= 0, T_del, Tmin)
    T_neg = np.where(quantized_weights <  0, T_del, Tmin)
    return T_pos, T_neg


def convert_to_conductance_continuous(
    weights: np.ndarray,
    Gmax: float, Gmin: float,
    Tmin: float, Tox: float, Tref: float, alpha: float,
    *,
    R_access: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Self-normalizing continuous mapping (rescales by Wmax)."""
    abs_w = np.abs(weights)
    Wmax = float(abs_w.max()) if abs_w.max() != 0 else 1.0
    w_norm   = abs_w / Wmax
    G_cell_desired = Gmin + w_norm * (Gmax - Gmin)
    G_R_required = compensate_for_transistor(G_cell_desired, R_access)
    T_fil    = Tox + (Tox - Tref) * np.log(G_R_required / alpha)
    T_pos = np.where(weights >= 0, T_fil, Tmin)
    T_neg = np.where(weights <  0, T_fil, Tmin)
    return T_pos, T_neg


# ---------------------------------------------------------------------------
# Subcircuit emission — TRANSIENT
# ---------------------------------------------------------------------------
# Parameters shared by every transient cell instance (matches the v2 deck and
# the included `reram_cell.v` Verilog-A model).
_TRANSIENT_VA_CELL = "sky130_fd_pr_reram__reram_cell"
_TRANSIENT_VA_PARAMS = (
    "area_ox=1.024e-13 "
    "Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 "
    "Tfilament_0=3.3e-09 Eact_generation=1.501 Eact_recombination=1.5 "
    "I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 velocity_k1=150 "
    "gamma_k0=16.5 gamma_k1=-1.25 Temperature_0=300 "
    "C_thermal=3.1825e-16 tau_thermal=2.3e-10 t_step=1e-09"
)


def _default_spec() -> "CellEmitSpec":
    """Back-compat spec built from default [HW]/[PDK] — byte-identical to the
    original hardcoded strings.  Used when a caller passes spec=None."""
    from spicecimx.config import HwCfg, PdkCfg
    return build_cell_emit_spec(HwCfg(), PdkCfg())


def _emit_transient_one(
    cell_name: str, T_val: float, branch: str,
    res_val: float, cap_val: float, spec: "CellEmitSpec",
) -> str:
    """Build one transient subcircuit for a single 1T1R cell.

    `branch` selects the NMOS source/body wiring (the 'pos' branch routes the
    transistor source through an RC to SL; 'neg' ties it directly to SL).  All
    PDK-specific text (transistor geometry, VA cell + params) comes from
    ``spec``.
    """
    if branch == "pos":
        nmos_line = spec.nmos_pos
        rc_lines = (
            f"   R0 (net2 SL) resistor r={res_val} \n"
            f"C1 (SL 0) capacitor c={cap_val/2} \n"
            f"C0 (net2 0) capacitor c={cap_val/2}"
        )
    else:  # 'neg'
        nmos_line = spec.nmos_neg
        rc_lines = ""

    body = [
        f"// Cell name: {cell_name}",
        "// View name: schematic",
        f"subckt {cell_name} BL SL WL",
        nmos_line,
    ]
    if rc_lines:
        body.append(rc_lines)
    body += [
        f"   I0 (BL net1) {spec.va_cell_transient} {spec.va_params_transient} initial_state={T_val:.4e}",
        f"ends {cell_name}",
        "// End of subcircuit definition.\n",
    ]
    return "\n".join(body) + "\n"


def emit_subckts_transient(
    T_pos: np.ndarray, T_neg: np.ndarray,
    baseline_va_path: str,
    *,
    res_val: float = 0.0, cap_val: float = 0.0,
    spec: Optional["CellEmitSpec"] = None,
) -> Tuple[str, str]:
    """
    Emit the full per-branch SCS text (positive and negative) for a subarray.

    Returns (pos_text, neg_text), both starting with the `ahdl_include` line
    pointing at `baseline_va_path` and followed by one subckt per cell.  PDK
    text comes from ``spec`` (defaults to the [HW]/[PDK] defaults).
    """
    if T_pos.shape != T_neg.shape:
        raise ValueError(f"T_pos / T_neg shape mismatch: {T_pos.shape} vs {T_neg.shape}")
    spec = spec or _default_spec()
    # Spectre runs with cwd = job_cwd, deep under tmp/. ahdl_include must
    # be resolvable from there, so always emit an absolute path.
    abs_va = os.path.abspath(baseline_va_path)
    header = f'ahdl_include "{abs_va}" \n'
    pos_parts = [header]
    neg_parts = [header]
    rows, cols = T_pos.shape
    for r in range(rows):
        for c in range(cols):
            name = spec.cell_name(r, c)
            pos_parts.append(_emit_transient_one(name, float(T_pos[r, c]), "pos", res_val, cap_val, spec))
            neg_parts.append(_emit_transient_one(name, float(T_neg[r, c]), "neg", res_val, cap_val, spec))
    return "".join(pos_parts), "".join(neg_parts)


def write_device_models_transient(
    T_pos: np.ndarray, T_neg: np.ndarray,
    baseline_va_path: str,
    target_dir: str,
    *,
    res_val: float = 0.0, cap_val: float = 0.0,
    pos_name: str = "device_model_pos.scs",
    neg_name: str = "device_model_neg.scs",
    spec: Optional["CellEmitSpec"] = None,
) -> Tuple[str, str]:
    """Write the two transient decks into target_dir. Returns (pos_path, neg_path)."""
    os.makedirs(target_dir, exist_ok=True)
    pos_text, neg_text = emit_subckts_transient(
        T_pos, T_neg, baseline_va_path,
        res_val=res_val, cap_val=cap_val, spec=spec,
    )
    pos_path = os.path.join(target_dir, pos_name)
    neg_path = os.path.join(target_dir, neg_name)
    with open(pos_path, "w") as f: f.write(pos_text)
    with open(neg_path, "w") as f: f.write(neg_text)
    return pos_path, neg_path


# ---------------------------------------------------------------------------
# Subcircuit emission — DC
# ---------------------------------------------------------------------------
# The DC model is stateless: no Eact_*, no velocity, no C_thermal.
# It still exposes Tox / Tfilament_max / Tfilament_min for parity (the body
# uses them only to bound `Tfilament_init`).
_DC_VA_CELL = "sky130_dc_memristor_dc"
_DC_VA_PARAMS = (
    "area_ox=1.024e-13 "
    "Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 "
    "I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 "
    "Temperature_0=300"
)


def _emit_dc_one(cell_name: str, T_val: float, branch: str,
                 spec: "CellEmitSpec") -> str:
    """Build one DC subcircuit for a single 1T1R cell.

    NMOS wiring matches the transient version's per-branch choice; only the
    memristor primitive cell and its parameter list change.  PDK text from
    ``spec``.
    """
    if branch == "pos":
        nmos_line = spec.nmos_pos
        rc_lines = "   R0 (net2 SL) resistor r=0"
    else:
        nmos_line = spec.nmos_neg
        rc_lines = ""

    body = [
        f"// Cell name: {cell_name}",
        "// View name: schematic",
        f"subckt {cell_name} BL SL WL",
        nmos_line,
    ]
    if rc_lines:
        body.append(rc_lines)
    body += [
        f"   I0 (BL net1) {spec.va_cell_dc} {spec.va_params_dc} Tfilament_init={T_val:.4e}",
        f"ends {cell_name}",
        "// End of subcircuit definition.\n",
    ]
    return "\n".join(body) + "\n"


def emit_subckts_dc(
    T_pos: np.ndarray, T_neg: np.ndarray,
    baseline_va_path: str,
    *, spec: Optional["CellEmitSpec"] = None,
) -> Tuple[str, str]:
    """
    Emit DC-mode subcircuit decks (positive and negative branch).

    The DC Verilog-A model has no time-dependent state, so DC analysis
    solves each operating point in a single Newton step.  Use these with
    `_golden_input_dc.scs` and a parametric DC sweep over image index.
    """
    if T_pos.shape != T_neg.shape:
        raise ValueError(f"T_pos / T_neg shape mismatch: {T_pos.shape} vs {T_neg.shape}")
    spec = spec or _default_spec()
    abs_va = os.path.abspath(baseline_va_path)
    header = f'ahdl_include "{abs_va}" \n'
    pos_parts = [header]
    neg_parts = [header]
    rows, cols = T_pos.shape
    for r in range(rows):
        for c in range(cols):
            name = spec.cell_name(r, c)
            pos_parts.append(_emit_dc_one(name, float(T_pos[r, c]), "pos", spec))
            neg_parts.append(_emit_dc_one(name, float(T_neg[r, c]), "neg", spec))
    return "".join(pos_parts), "".join(neg_parts)


def write_device_models_dc(
    T_pos: np.ndarray, T_neg: np.ndarray,
    baseline_va_path: str,
    target_dir: str,
    *,
    pos_name: str = "device_model_pos.scs",
    neg_name: str = "device_model_neg.scs",
    spec: Optional["CellEmitSpec"] = None,
) -> Tuple[str, str]:
    """Write the two DC decks into target_dir. Returns (pos_path, neg_path)."""
    os.makedirs(target_dir, exist_ok=True)
    pos_text, neg_text = emit_subckts_dc(T_pos, T_neg, baseline_va_path, spec=spec)
    pos_path = os.path.join(target_dir, pos_name)
    neg_path = os.path.join(target_dir, neg_name)
    with open(pos_path, "w") as f: f.write(pos_text)
    with open(neg_path, "w") as f: f.write(neg_text)
    return pos_path, neg_path