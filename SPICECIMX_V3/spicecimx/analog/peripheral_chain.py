"""
================================================================================
  peripheral_chain.py — physical DAC / TIA+SUBTRACTOR / ADC chain (feature 5)
================================================================================

When a converter role is declared **hardware (H)** in ``[SUBCIRCUITS]`` (and the
mode is transient), the framework stops using the ideal software DAC/TIA/ADC and
instead drives the real subckt pasted into ``supports/standard_cell.netlist``.

How the chain is realised
-------------------------
Each converter is a (near-)static transfer function, so we **characterise the
real subckt once with SPICE** and then apply the measured transfer (and the
measured per-conversion energy) during inference.  This is identical in result
to co-simulating the converter on every conversion, but costs a handful of
extra Spectre runs instead of thousands, and it captures the real device
non-ideality (INL / DNL / offset / gain / supply energy) straight from the
netlist:

  Stage 1 — DAC:  drive ``DAC.INP<bits-1:0>`` through every code 0..2^bits-1,
                  measure the analog ``OUT`` -> table ``V_dac[code]`` and the
                  energy/conversion from the DAC supply.  In inference the
                  ideal crossbar drive V is quantised to a code and replaced by
                  ``V_dac[code]`` before the PWL sources are emitted, so the
                  crossbar sees the *real* DAC output (this is the "simulate the
                  DAC, feed its output to the crossbar" step).

  Stage 3 — TIA -> SUBTRACTOR -> ADC:  each differential column current is
                  converted to a voltage by its own ``TIA``; a distinct
                  ``SUBTRACTOR`` forms V_pos - V_neg; the ``ADC`` digitises the
                  difference.  The characterisation deck instantiates two TIAs
                  (one per +/- column), the subtractor, and the ADC in series,
                  sweeps the differential current, and reads back the ADC digital
                  ``OUT<bits-1:0>`` bus -> table ``code(I_diff)`` plus the
                  energy/conversion of the TIA pair, the subtractor, and the ADC
                  separately.  In inference the crossbar column differential
                  current is pushed through this measured transfer to get the
                  final digital code (and the de-quantised value for the affine
                  de-mapping) — replacing the software ``tia_adc_realistic``.

Everything about each subckt — its name, the pin list / order, which pins form
the digital bus, and the bit width — is taken from the parsed
``hw.subckt_registry.SubcktDef``; nothing is hard-coded, so a different cell
library (different names, different resolution) drops in unchanged.

This module is import-light (numpy only at module scope) and the live-SPICE
orchestration is dependency-injected via a ``run_decks`` callable, so the deck
emission, the PSF read-back, and the runtime ``apply`` paths are all unit-
testable without a simulator.

PIN SEMANTICS: ``DAC`` (digital INP bus, analog OUT), ``ADC`` (digital OUT bus,
analog IN, CLK), ``TIA`` (analog current IN -> analog voltage OUT) and
``SUBTRACTOR`` (two analog voltage inputs -> one analog OUT) are discovered
generically from the parsed pins.  Provide the TIA and SUBTRACTOR subckt
headers and the discovery is locked to them; multi-lane (bus) TIA/subtractor is
supported via ``ReadoutConfig.lane``.
================================================================================
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Helpers — bus pin ordering from a parsed SubcktDef                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def _digital_bus(subdef, prefer: Sequence[str]) -> str:
    """Pick the digital code bus base name from a SubcktDef.

    ``prefer`` lists candidate base names in priority order (e.g. ('INP',) for a
    DAC input, ('OUT',) for an ADC output).  Falls back to the widest bus."""
    for name in prefer:
        if name in subdef.buses:
            return name
    if subdef.buses:
        return max(subdef.buses.items(), key=lambda kv: len(kv[1]))[0]
    raise ValueError(f"subckt {subdef.name} has no bus pins to use as a code bus")


def _bus_pins_msb_first(subdef, base: str) -> List[int]:
    """Return the bus indices in MSB->LSB order (descending index)."""
    return sorted(subdef.buses[base], reverse=True)


def _scalar_like(subdef, *candidates: str) -> Optional[str]:
    """Find the first scalar pin whose name matches any candidate (case-insens.)."""
    low = {s.lower(): s for s in subdef.scalars}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _include_lines(transistor_model: str, process_corner: str,
                   includes: Sequence[str]) -> List[str]:
    lines = [f'include "{transistor_model}" section={process_corner}']
    lines += [f'include "{p}"' for p in includes]
    return lines


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  psfascii traces — read once, window by TIME                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def _traces(params: np.ndarray, values: np.ndarray,
            nets: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Split a flat psfascii (params, values) stream into the time vector and
    one sample vector per requested net (``"name"`` tokens, quoted in PSF)."""
    t = values[params == '"time"'].astype(float)
    out: Dict[str, np.ndarray] = {}
    for n in nets:
        v = values[params == f'"{n}"'].astype(float)
        if v.size:
            m = min(v.size, t.size)
            out[n] = v[:m]
    return t, out


def _mean_in(t: np.ndarray, v: Optional[np.ndarray], t_lo: float, t_hi: float
             ) -> Optional[float]:
    if v is None or v.size == 0:
        return None
    tt = t[:v.size]
    sel = (tt >= t_lo) & (tt <= t_hi)
    if not np.any(sel):
        # The simulator's adaptive step can skip a short, quiet window
        # entirely; the waveform is flat there, so interpolate mid-window.
        t_mid = 0.5 * (t_lo + t_hi)
        if tt.size < 2 or t_mid < tt[0] or t_mid > tt[-1]:
            return None
        return float(np.interp(t_mid, tt, v))
    return float(np.mean(v[sel]))


def _charge_in(t: np.ndarray, i: Optional[np.ndarray], t0: float, t1: float
               ) -> Optional[float]:
    """∫|i| dt over [t0, t1] (trapezoid on the simulator's own time steps)."""
    if i is None or i.size < 2:
        return None
    tt = t[:i.size]
    sel = (tt >= t0) & (tt <= t1)
    if np.count_nonzero(sel) < 2:
        return None
    x, y = tt[sel], np.abs(i[sel])
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _energy_per_conv(t: np.ndarray, i: Optional[np.ndarray],
                     windows: Sequence[Tuple[float, float, float]], vdd: float,
                     t_conv: Optional[float]) -> float:
    """Energy per conversion from a supply-current trace.

    ``windows`` are ``(t_start, t_settled, t_end)`` per step.  The settled
    current of the step (mean over ``[t_settled, t_end]``: bias plus any
    signal-dependent DC, e.g. a TIA sourcing the input current) is charged
    for ``t_conv`` — the converter's real conversion time from [LATENCY] —
    and only the transient charge above it (switching, comparator pulses)
    is added as dynamic energy.  So the result does not depend on how long
    the characterisation deck dwells on each step.  ``t_conv=None`` charges
    the settled current over the full dwell instead.
    """
    if i is None or i.size < 2:
        return float("nan")
    e = []
    for t0, t_set, t1 in windows:
        q = _charge_in(t, i, t0, t1)
        i_set = _mean_in(t, np.abs(i), t_set, t1)
        if q is None or i_set is None:
            continue
        q_dyn = max(q - i_set * (t1 - t0), 0.0)
        t_static = (t1 - t0) if t_conv is None else t_conv
        e.append(abs(vdd) * (q_dyn + i_set * t_static))
    return float(np.mean(e)) if e else float("nan")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Stage 1 — DAC characterisation deck + parser                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
@dataclass
class DacCharMeta:
    bits: int
    n_codes: int
    code_dwell: float          # seconds each code is held
    settle_frac: float         # fraction into the dwell where OUT is read
    out_node: str              # analog output node name in the deck
    vdd_node: str
    supply_src: str            # vsource whose current = DAC supply current
    code_windows: List[Tuple[int, float, float]]  # (code, t_lo, t_hi) read windows
    inp_base: str


def emit_dac_char_deck(
    dac_def, *,
    transistor_model: str, process_corner: str,
    vdd: float, vss: float, temperature: float,
    code_dwell: float = 1e-9, trise: float = 1e-11, tfall: float = 1e-11,
    settle_frac: float = 0.9, max_codes: Optional[int] = None,
    includes: Sequence[str] = ("standard_cell.scs",),
) -> Tuple[str, DacCharMeta]:
    """
    Build a Spectre transient deck that steps ``DAC.INP<bits-1:0>`` through every
    code and records the analog ``OUT``.  The digital bits are full-rail PWL
    sources (vss for 0, vdd for 1).  Supply current is measured through a
    dedicated vsource on the DAC's VDD pin so energy/conversion can be integrated.

    Returns ``(deck_text, DacCharMeta)``.
    """
    bits = dac_def.bit_width
    n_codes = 2 ** bits
    if max_codes is not None:
        n_codes = min(n_codes, max_codes)

    inp_base = _digital_bus(dac_def, ("INP", "IN", "D", "B"))
    msb_first = _bus_pins_msb_first(dac_def, inp_base)   # e.g. [7,6,...,0]
    out_node = _scalar_like(dac_def, "OUT", "AOUT", "VOUT", "O") or "OUT"
    vdd_pin = _scalar_like(dac_def, "VDD", "VDDA", "VCC") or "VDD"
    vss_pin = _scalar_like(dac_def, "VSS", "GND", "VSSA") or "VSS"

    # Map the subckt's port order onto our test nets.  Bus pins -> Dk nodes,
    # OUT -> out_node, VDD -> DVDD (through a current-sense vsource), VSS -> 0.
    port_nets: List[str] = []
    for p in dac_def.ports:
        if "<" in p:
            base, idx = p.split("<", 1)
            idx = int(idx.rstrip(">"))
            port_nets.append(f"D{idx}")          # one digital net per bit
        elif p == out_node:
            port_nets.append("DAC_OUT")
        elif p == vdd_pin:
            port_nets.append("DVDD")
        elif p == vss_pin:
            port_nets.append("0")
        else:
            port_nets.append(p)                  # pass through (e.g. extra rails)

    # Build per-bit PWL waveforms: bit b of code c is ((c>>b)&1).
    window = code_dwell
    code_windows: List[Tuple[int, float, float]] = []
    bit_waves: Dict[int, List[Tuple[float, float]]] = {idx: [] for idx in msb_first}
    t = 0.0
    for c in range(n_codes):
        for idx in msb_first:
            level = vdd if ((c >> idx) & 1) else vss
            bit_waves[idx].extend([
                (t, level), (t + window - trise - tfall, level),
            ])
        t_lo = t + settle_frac * window
        t_hi = t + window
        code_windows.append((c, t_lo, t_hi))
        t += window
    total_t = t

    lines = [
        "// AUTO-GENERATED DAC characterisation deck (feature 5)",
        "simulator lang=spectre",
        "global 0",
        *_include_lines(transistor_model, process_corner, includes),
        "",
    ]
    # digital bit sources
    for idx in msb_first:
        wave = " ".join(f"{tt:.9g} {vv:.6g}" for tt, vv in bit_waves[idx])
        lines.append(f"VD{idx} (D{idx} 0) vsource wave=[{wave}] type=pwl")
    # current-sense supply vsource on the DAC VDD pin
    lines.append(f"VDACVDD (DVDD 0) vsource dc={vdd} type=dc")
    # the DAC instance
    lines.append(f"XDAC ({' '.join(port_nets)}) {dac_def.name}")
    lines.append("")
    lines.append(f"simulatorOptions options temp={temperature} tnom=27 "
                 f"reltol=1e-3 vabstol=1e-6 iabstol=1e-12 gmin=1e-12")
    lines.append(f"tran tran stop={total_t:.6g} errpreset=conservative")
    lines.append("save DAC_OUT VDACVDD:p")
    lines.append("saveOptions options save=selected pwr=total currents=selected")

    meta = DacCharMeta(
        bits=bits, n_codes=n_codes, code_dwell=code_dwell,
        settle_frac=settle_frac, out_node="DAC_OUT", vdd_node="DVDD",
        supply_src="VDACVDD", code_windows=code_windows, inp_base=inp_base,
    )
    return "\n".join(lines) + "\n", meta


def parse_dac_transfer(
    read_fn: Callable, tran_path: str, meta: DacCharMeta, *,
    vdd: float, t_conv: Optional[float] = None,
    settled_value_fn: Optional[Callable] = None,       # unused; kept for API
) -> Tuple[np.ndarray, float]:
    """
    Parse a DAC characterisation tran dump into:
      * ``V_dac`` : (n_codes,) measured analog output per code (mean of OUT
        over each code's settled window)
      * ``e_per_conv_J`` : energy/conversion (see :func:`_energy_per_conv`)

    ``read_fn`` is injected (``parse_transient.read_psfascii_tran``) so this
    stays simulator-agnostic and testable.
    """
    params, values = read_fn(tran_path)
    sup = f"{meta.supply_src}:p"
    t, tr = _traces(params, values, (meta.out_node, sup))
    V_dac = np.full(meta.n_codes, np.nan, dtype=np.float64)
    for (code, t_lo, t_hi) in meta.code_windows:
        v = _mean_in(t, tr.get(meta.out_node), t_lo, t_hi)
        if v is not None:
            V_dac[code] = v
    steps = [(t_hi - meta.code_dwell, t_lo, t_hi) for (_c, t_lo, t_hi) in meta.code_windows]
    e_per_conv = _energy_per_conv(t, tr.get(sup), steps, vdd, t_conv)
    return V_dac, e_per_conv


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Stage 3 — TIA+SUBTRACTOR -> ADC characterisation deck + parser       ║
# ╚═══════════════════════════════════════════════════════════════════════╝
@dataclass
class ReadoutConfig:
    """Lane to characterise when a TIA/subtractor is multi-lane (bus) rather
    than single.  For scalar (single-column) TIA/subtractor this is ignored."""
    def __init__(self, lane: int = 0):
        self.lane = lane


def _analog_pin(subdef, candidates: Sequence[str], lane: int = 0) -> str:
    """Return the net name for a single analog pin: a scalar pin if one of the
    candidate names exists, else lane ``lane`` of the first matching bus."""
    s = _scalar_like(subdef, *candidates)
    if s is not None:
        return s
    for c in candidates:
        if c in subdef.buses:
            idx = sorted(subdef.buses[c], reverse=True)[::-1][lane]   # LSB-up lane
            return f"{c}<{idx}>"
    # last resort: first scalar / first bus lane
    if subdef.scalars:
        return subdef.scalars[0]
    base = next(iter(subdef.buses))
    return f"{base}<{sorted(subdef.buses[base])[lane]}>"


def _two_input_pins(subdef, lane: int = 0) -> Tuple[str, str]:
    """Find the (positive, negative) input pins of a subtractor, trying common
    naming pairs; falls back to the first two non-output scalar/bus pins."""
    pairs = [("INP", "INN"), ("IP", "IN"), ("VP", "VN"), ("A", "B"),
             ("IN1", "IN2"), ("INPUT1", "INPUT2"), ("VINP", "VINN")]
    for a, b in pairs:
        pa = _scalar_like(subdef, a)
        pb = _scalar_like(subdef, b)
        if pa and pb:
            return pa, pb
        if a in subdef.buses and b in subdef.buses:
            ia = sorted(subdef.buses[a], reverse=True)[::-1][lane]
            ib = sorted(subdef.buses[b], reverse=True)[::-1][lane]
            return f"{a}<{ia}>", f"{b}<{ib}>"
    # fallback: first two pins that aren't an obvious output/supply
    out = _scalar_like(subdef, "OUT", "VOUT", "O")
    supplies = {"VDD", "VSS", "GND", "VCC", "VDDA", "VSSA"}
    cand = [p for p in subdef.ports
            if "<" not in p and p != out and p.upper() not in supplies]
    if len(cand) >= 2:
        return cand[0], cand[1]
    raise ValueError(f"cannot identify two input pins on subtractor "
                     f"{subdef.name}; pass an explicit mapping")


@dataclass
class ReadoutCharMeta:
    adc_bits: int
    n_steps: int
    step_dwell: float
    settle_frac: float
    adc_out_base: str
    adc_out_bits_msb_first: List[int]
    i_values: List[float]                       # swept differential currents (A)
    step_windows: List[Tuple[int, float, float]]
    tia_supply_src: str                         # sensed supply of BOTH TIAs
    sub_supply_src: Optional[str]               # subtractor supply (None if sw)
    adc_supply_src: str
    clk_node: Optional[str]


def emit_readout_char_deck(
    tia_def, subtractor_def, adc_def, *,
    i_sweep: Sequence[float],
    transistor_model: str, process_corner: str,
    vdd: float, vss: float, temperature: float,
    clk_period: Optional[float] = None, step_dwell: float = 4e-9,
    settle_frac: float = 0.9, cfg: Optional[ReadoutConfig] = None,
    includes: Sequence[str] = ("standard_cell.scs",),
) -> Tuple[str, ReadoutCharMeta]:
    """
    Build the readout-chain characterisation deck for the **separated**
    architecture::

        I_pos -> TIA -> V_pos ┐
                              ├-> SUBTRACTOR -> V_diff -> ADC -> code
        I_neg -> TIA -> V_neg ┘

    Two TIA instances (one per differential column) convert the +/- column
    currents to voltages; a distinct SUBTRACTOR forms V_pos - V_neg; the ADC
    digitises it.  Each stage's supply is sensed so per-conversion energy can be
    integrated (the two TIAs share one sensed supply -> energy is the pair's
    energy/output).  Returns ``(deck_text, ReadoutCharMeta)``.

    The ADC clock defaults to one rising edge per step (``clk_period =
    step_dwell``), placed mid-step: the TIA/subtractor settle in the first
    half and a sampling (e.g. SAR) ADC converts before the read window at
    ``settle_frac``.  A faster clock would restart the conversion on every
    edge.
    """
    cfg = cfg or ReadoutConfig()
    if clk_period is None:
        clk_period = step_dwell
    lane = cfg.lane
    adc_bits = adc_def.bit_width
    adc_out_base = _digital_bus(adc_def, ("OUT", "D", "O", "Q"))
    adc_out_msb = _bus_pins_msb_first(adc_def, adc_out_base)

    # --- pin discovery (all from the parsed SubcktDefs) ---
    tia_in = _analog_pin(tia_def, ("IN", "IIN", "INP", "I", "VIN", "IINP"), lane)
    tia_out = _analog_pin(tia_def, ("OUT", "VOUT", "O", "AOUT"), lane)
    sub_inp, sub_inn = _two_input_pins(subtractor_def, lane)
    sub_out = _analog_pin(subtractor_def, ("OUT", "VOUT", "O"), lane)
    adc_in = _scalar_like(adc_def, "IN", "AIN", "VIN", "INP") or "IN"
    adc_clk = _scalar_like(adc_def, "CLK", "CK", "CLOCK")

    def _rails(subdef):
        return (_scalar_like(subdef, "VDD", "VDDA", "VCC") or "VDD",
                _scalar_like(subdef, "VSS", "GND", "VSSA") or "VSS")
    tia_vdd, tia_vss = _rails(tia_def)
    sub_vdd, sub_vss = _rails(subtractor_def)
    adc_vdd, adc_vss = _rails(adc_def)

    def _map(subdef, mapping: Dict[str, str], vdd_net: str) -> List[str]:
        """Map subckt ports->nets using ``mapping`` (port->net); VDD->vdd_net,
        VSS->0, unmapped pass through."""
        out = []
        for p in subdef.ports:
            if p in mapping:
                out.append(mapping[p])
            elif p in (tia_vdd, sub_vdd, adc_vdd) and p.upper().startswith("V") \
                    and "VSS" not in p.upper() and "GND" not in p.upper():
                out.append(vdd_net)
            elif "VSS" in p.upper() or p.upper() == "GND":
                out.append("0")
            else:
                out.append(p)
        return out

    # instance nets
    tia_p_nets = _map(tia_def, {tia_in: "TIA_P_IN", tia_out: "TIA_P_OUT",
                                tia_vdd: "TVDD"}, "TVDD")
    tia_n_nets = _map(tia_def, {tia_in: "TIA_N_IN", tia_out: "TIA_N_OUT",
                                tia_vdd: "TVDD"}, "TVDD")
    sub_nets = _map(subtractor_def, {sub_inp: "TIA_P_OUT", sub_inn: "TIA_N_OUT",
                                     sub_out: "SUB_OUT", sub_vdd: "SVDD"}, "SVDD")
    adc_map = {adc_in: "SUB_OUT", adc_vdd: "AVDD"}
    if adc_clk:
        adc_map[adc_clk] = "ACLK"
    for idx in adc_out_msb:
        adc_map[f"{adc_out_base}<{idx}>"] = f"ADC_{adc_out_base}_{idx}"
    adc_nets = _map(adc_def, adc_map, "AVDD")

    # --- current sweep waveforms (piecewise constant per step) ---
    n_steps = len(i_sweep)
    step_windows: List[Tuple[int, float, float]] = []
    ip_wave: List[Tuple[float, float]] = []
    in_wave: List[Tuple[float, float]] = []
    t = 0.0
    t_edge = step_dwell * 1e-3          # strictly increasing PWL time points
    for s, Idiff in enumerate(i_sweep):
        ip = max(float(Idiff), 0.0)
        inn = max(-float(Idiff), 0.0)
        ip_wave.extend([(t + (t_edge if s else 0.0), ip), (t + step_dwell, ip)])
        in_wave.extend([(t + (t_edge if s else 0.0), inn), (t + step_dwell, inn)])
        step_windows.append((s, t + settle_frac * step_dwell, t + step_dwell))
        t += step_dwell
    total_t = t

    def _wave(pairs):
        return " ".join(f"{a:.9g} {b:.6g}" for a, b in pairs)

    lines = [
        "// AUTO-GENERATED TIA -> SUBTRACTOR -> ADC readout characterisation deck",
        "simulator lang=spectre",
        "global 0",
        *_include_lines(transistor_model, process_corner, includes),
        "",
        f"IINP (0 TIA_P_IN) isource wave=[{_wave(ip_wave)}] type=pwl",
        f"IINN (0 TIA_N_IN) isource wave=[{_wave(in_wave)}] type=pwl",
        f"VTIAVDD (TVDD 0) vsource dc={vdd} type=dc",
        f"VSUBVDD (SVDD 0) vsource dc={vdd} type=dc",
        f"VADCVDD (AVDD 0) vsource dc={vdd} type=dc",
    ]
    if adc_clk:
        # one rising edge per period, mid-period
        lines.append(f"VACLK (ACLK 0) vsource type=pulse val0={vss} val1={vdd} "
                     f"period={clk_period:.9g} delay={clk_period / 2:.9g} "
                     f"width={clk_period / 2 - 2 * t_edge:.9g} "
                     f"rise={t_edge:.9g} fall={t_edge:.9g}")
    lines.append(f"XTIA_P ({' '.join(tia_p_nets)}) {tia_def.name}")
    lines.append(f"XTIA_N ({' '.join(tia_n_nets)}) {tia_def.name}")
    lines.append(f"XSUB ({' '.join(sub_nets)}) {subtractor_def.name}")
    lines.append(f"XADC ({' '.join(adc_nets)}) {adc_def.name}")
    lines.append("")
    lines.append(f"simulatorOptions options temp={temperature} tnom=27 "
                 f"reltol=1e-3 vabstol=1e-6 iabstol=1e-15 gmin=1e-12")
    lines.append(f"tran tran stop={total_t:.6g} errpreset=conservative")
    save_bits = " ".join(f"ADC_{adc_out_base}_{i}" for i in adc_out_msb)
    lines.append(f"save TIA_P_OUT TIA_N_OUT SUB_OUT {save_bits} "
                 f"VTIAVDD:p VSUBVDD:p VADCVDD:p")
    lines.append("saveOptions options save=selected pwr=total currents=selected")

    meta = ReadoutCharMeta(
        adc_bits=adc_bits, n_steps=n_steps, step_dwell=step_dwell,
        settle_frac=settle_frac, adc_out_base=adc_out_base,
        adc_out_bits_msb_first=adc_out_msb, i_values=list(map(float, i_sweep)),
        step_windows=step_windows, tia_supply_src="VTIAVDD",
        sub_supply_src="VSUBVDD", adc_supply_src="VADCVDD",
        clk_node=("ACLK" if adc_clk else None),
    )
    return "\n".join(lines) + "\n", meta


def parse_readout_transfer(
    read_fn: Callable, tran_path: str, meta: ReadoutCharMeta, *,
    vdd: float, logic_threshold: Optional[float] = None,
    t_conv: Optional[float] = None,
    settled_value_fn: Optional[Callable] = None,       # unused; kept for API
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Parse the TIA->SUBTRACTOR->ADC characterisation dump into:
      * ``i_values`` : (n_steps,) swept differential currents (A)
      * ``codes``    : (n_steps,) ADC digital code per current (-1 = unread)
      * ``e_tia_J``  : energy/output for the differential TIA *pair*
      * ``e_sub_J``  : energy/output for the subtractor
      * ``e_adc_J``  : energy/output for the ADC

    Digital bus thresholded at ``logic_threshold`` (default vdd/2).
    """
    thr = logic_threshold if logic_threshold is not None else vdd / 2.0
    params, values = read_fn(tran_path)
    bit_nets = [f"ADC_{meta.adc_out_base}_{idx}" for idx in meta.adc_out_bits_msb_first]
    sups = [f"{s}:p" for s in (meta.tia_supply_src, meta.sub_supply_src,
                              meta.adc_supply_src) if s]
    t, tr = _traces(params, values, bit_nets + sups)
    codes = np.full(meta.n_steps, -1, dtype=np.int64)
    i_arr = np.array(meta.i_values, dtype=np.float64)
    for (s, t_lo, t_hi) in meta.step_windows:
        code = 0
        ok = True
        for idx, net in zip(meta.adc_out_bits_msb_first, bit_nets):
            v = _mean_in(t, tr.get(net), t_lo, t_hi)
            if v is None:
                ok = False
                break
            code |= ((1 if v >= thr else 0) << idx)
        if ok:
            codes[s] = code
    steps = [(t_hi - meta.step_dwell, t_lo, t_hi) for (_s, t_lo, t_hi) in meta.step_windows]

    def _e(src):
        return (_energy_per_conv(t, tr.get(f"{src}:p"), steps, vdd, t_conv)
                if src else float("nan"))
    return (i_arr, codes, _e(meta.tia_supply_src), _e(meta.sub_supply_src),
            _e(meta.adc_supply_src))


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Runtime — apply measured transfers during inference                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
@dataclass
class PhysicalConverters:
    """Measured converter transfers + energies, applied during inference.

    Built by :func:`characterize_converters` (live SPICE) or directly from
    tables (tests / cached runs).  ``dac_apply`` and ``tia_adc_apply`` replace
    the ideal software converters; ``measured_energy`` feeds analog.pwr_energy.
    """
    # DAC
    dac_hw: bool = False
    dac_table: Optional[np.ndarray] = None      # V_dac[code], shape (2^bits,)
    dac_bits: int = 8
    dac_vmin: float = 0.0
    dac_vmax: float = 1.0
    e_dac_J: float = float("nan")
    # TIA + SUBTRACTOR + ADC (separated readout chain)
    adc_hw: bool = False
    tia_hw: bool = False
    sub_hw: bool = False
    adc_i_values: Optional[np.ndarray] = None   # (n_steps,) sweep currents
    adc_codes: Optional[np.ndarray] = None      # (n_steps,) codes
    adc_bits: int = 8
    i_fullscale: float = 0.0                    # ±sweep range of the readout char
    e_tia_J: float = float("nan")               # differential TIA pair / output
    e_sub_J: float = float("nan")               # subtractor / output
    e_adc_J: float = float("nan")

    # -- DAC: ideal drive voltage -> code -> measured DAC output ----------
    def dac_apply(self, V: np.ndarray) -> np.ndarray:
        """Quantise V to a code, then replace with the measured DAC output.
        No-op (returns V) when the DAC is software."""
        if not self.dac_hw or self.dac_table is None:
            return V
        span = self.dac_vmax - self.dac_vmin
        if span <= 0:
            return V
        codes = np.rint((np.asarray(V, float) - self.dac_vmin) / span
                        * (2 ** self.dac_bits - 1))
        codes = np.clip(codes, 0, 2 ** self.dac_bits - 1).astype(int)
        return self.dac_table[codes]

    # -- TIA+ADC: differential current -> digital code -> value -----------
    def tia_adc_apply(self, I_diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map crossbar differential current to (digital_code, dequantised_value)
        through the measured TIA+ADC transfer.  Returns (codes, values).
        Raises if called when not in hardware mode (caller must guard)."""
        if not (self.adc_hw and self.adc_i_values is not None):
            raise RuntimeError("tia_adc_apply called but ADC is not hardware-mode")
        I = np.asarray(I_diff, float)
        ok = self.adc_codes >= 0
        order = np.argsort(self.adc_i_values[ok])
        xs = self.adc_i_values[ok][order]
        cs = self.adc_codes[ok][order].astype(float)
        top = 2 ** self.adc_bits - 1
        # code of each current: the measured transfer, interpolated between
        # sweep points and held at the end codes outside the sweep (clipping)
        code_f = np.interp(I.ravel(), xs, cs,
                           left=cs[0], right=cs[-1]).reshape(I.shape)
        codes = np.clip(np.rint(code_f), 0, top).astype(int)
        # De-quantise back to current for the affine de-map with the nominal
        # (best-fit straight line) transfer over the unsaturated codes, so a
        # rising or falling chain both work and INL/offset/gain error of the
        # real converter shows up in the result instead of being calibrated
        # away point by point.
        lin = (cs > 0) & (cs < top)
        if np.count_nonzero(lin) >= 2:
            slope, icpt = np.polyfit(xs[lin], cs[lin], 1)
        else:
            slope, icpt = 0.0, 0.0
        if abs(slope) > 0:
            values = (codes - icpt) / slope
        else:
            values = I
        bad = ~np.isfinite(I)
        if bad.any():
            values = np.where(bad, np.nan, values)
        return codes, values

    # -- energy hand-off to pwr_energy ------------------------------------
    def measured_energy(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if self.dac_hw and self.e_dac_J == self.e_dac_J:
            out["dac"] = {"e_per_conv_J": self.e_dac_J, "source": "spice"}
        if self.tia_hw and self.e_tia_J == self.e_tia_J:
            out["tia"] = {"e_per_conv_J": self.e_tia_J, "source": "spice"}
        if self.sub_hw and self.e_sub_J == self.e_sub_J:
            out["subtractor"] = {"e_per_conv_J": self.e_sub_J, "source": "spice"}
        if self.adc_hw and self.e_adc_J == self.e_adc_J:
            out["adc"] = {"e_per_conv_J": self.e_adc_J, "source": "spice"}
        return out

    def write_measured_energy(self, dump_dir: str) -> Optional[str]:
        """Persist measured per-conversion energies for analog.pwr_energy to
        pick up (overrides the modelled energy for H-mode converters)."""
        meas = self.measured_energy()
        if not meas:
            return None
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, "measured_converter_energy.json")
        with open(path, "w") as f:
            json.dump(meas, f, indent=2)
        return path


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Orchestration — characterise the H-mode converters once per run      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def _conv_times(cfg) -> Tuple[Optional[float], Optional[float]]:
    """(DAC, ADC) conversion time in seconds from [LATENCY] — used to charge
    the measured static current per conversion."""
    lat = getattr(cfg, "latency", None)
    if lat is None or not getattr(lat, "clock_freq_hz", 0):
        return None, None
    tclk = 1.0 / float(lat.clock_freq_hz)
    bits = int(getattr(cfg.peripherals, "adc_bits", 8))
    n_adc = int(getattr(lat, "adc_cycles_per_conv", 0) or 0)
    if n_adc <= 0:
        typ = str(getattr(lat, "adc_type", "sar")).lower()
        n_adc = {"sar": bits + 1, "flash": 1, "pipeline": 1,
                 "integrating": 2 ** bits}.get(typ, bits + 1)
    n_dac = int(getattr(lat, "dac_cycles_per_conv", 1) or 1)
    return n_dac * tclk, n_adc * tclk


def characterize_converters(
    registry, cfg, *,
    run_decks: Callable[[List[Tuple[str, str]]], Dict[str, str]],
    read_fn: Callable, settled_value_fn: Optional[Callable] = None,
    workdir: str,
    i_sweep: Optional[Sequence[float]] = None,
    includes: Sequence[str] = ("standard_cell.scs",),
    dac_dwell: float = 20e-9,
    readout_dwell: float = 100e-9,
) -> PhysicalConverters:
    """
    Characterise every hardware-mode converter and return a PhysicalConverters.

    ``run_decks`` is injected: it takes ``[(name, deck_text), ...]``, runs each
    deck under SPICE, and returns ``{name: tran_path}``.  This keeps the heavy,
    scheduler-specific run logic out of this module (and makes the emission /
    parse paths unit-testable).  Only roles declared H in ``[SUBCIRCUITS]`` are
    characterised; software roles are skipped and their energy stays modelled.

    ``includes`` are the netlist files the decks include (the caller puts them
    next to the deck).  The DAC full scale is the crossbar drive range
    ``[HW] vmin..vmax``.  The readout chain is swept over
    ±``[PERIPHERALS] adc_i_fullscale``; at inference each layer's calibrated
    current range is scaled onto that sweep (a per-layer programmable TIA
    gain, the hardware counterpart of the software TIA calibration).
    """
    sim = cfg.sim
    pc = PhysicalConverters(
        dac_hw=registry.is_hardware("dac"),
        adc_hw=registry.is_hardware("adc"),
        tia_hw=registry.is_hardware("tia"),
        sub_hw=registry.is_hardware("subtractor"),
        dac_vmin=float(cfg.hw.vmin),
        dac_vmax=float(cfg.hw.vmax),
    )
    t_dac, t_adc = _conv_times(cfg)
    decks: List[Tuple[str, str]] = []
    dac_meta = read_meta = None

    if pc.dac_hw:
        dac_def = registry.definition("dac")
        pc.dac_bits = dac_def.bit_width
        deck, dac_meta = emit_dac_char_deck(
            dac_def, transistor_model=cfg.paths.transistor_model,
            process_corner=sim.process_corner, vdd=sim.vdd, vss=sim.vss,
            temperature=sim.temperature, code_dwell=dac_dwell,
            trise=dac_dwell * 1e-3, tfall=dac_dwell * 1e-3, includes=includes)
        decks.append(("dac", deck))

    # Readout chain is characterised when the ADC is hardware (the ADC needs a
    # real input).  The TIA and SUBTRACTOR subckts are used if their roles are
    # hardware; both must be present in the netlist for the chained deck.
    if pc.adc_hw:
        tia_def = registry.definition("tia")
        sub_def = registry.definition("subtractor")
        adc_def = registry.definition("adc")
        pc.adc_bits = adc_def.bit_width
        if tia_def is None or sub_def is None:
            raise ValueError(
                "[hw] adc = H needs TIA and SUBTRACTOR subckts for the "
                "TIA->SUB->ADC chain, but "
                + " and ".join(r for r, d in (("tia", tia_def), ("subtractor", sub_def))
                               if d is None)
                + " was not found in the std-cell / peripheral netlist.")
        else:
            if i_sweep is None:
                i_max = float(getattr(getattr(cfg, "peripherals", None),
                                      "adc_i_fullscale",
                                      getattr(sim, "adc_i_fullscale", 8e-4)))
                i_sweep = np.linspace(-i_max, i_max,
                                      2 ** min(pc.adc_bits, 10)).tolist()
            deck, read_meta = emit_readout_char_deck(
                tia_def, sub_def, adc_def, i_sweep=i_sweep,
                transistor_model=cfg.paths.transistor_model,
                process_corner=sim.process_corner, vdd=sim.vdd, vss=sim.vss,
                temperature=sim.temperature, step_dwell=readout_dwell,
                includes=includes)
            decks.append(("readout", deck))

    if not decks:
        return pc

    paths = run_decks(decks)

    if pc.dac_hw:
        if "dac" not in paths:
            raise RuntimeError("[hw] DAC characterisation deck failed")
        pc.dac_table, pc.e_dac_J = parse_dac_transfer(
            read_fn, paths["dac"], dac_meta, vdd=sim.vdd, t_conv=t_dac)
        if not np.all(np.isfinite(pc.dac_table)):
            raise RuntimeError(f"[hw] DAC transfer has "
                               f"{int(np.sum(~np.isfinite(pc.dac_table)))} unread codes")
    if pc.adc_hw:
        if "readout" not in paths:
            raise RuntimeError("[hw] TIA->SUB->ADC characterisation deck failed")
        i_vals, codes, e_tia, e_sub, e_adc = parse_readout_transfer(
            read_fn, paths["readout"], read_meta, vdd=sim.vdd, t_conv=t_adc)
        if np.count_nonzero(codes >= 0) < 2 or len(np.unique(codes[codes >= 0])) < 2:
            raise RuntimeError("[hw] readout chain produced no usable ADC codes "
                               f"(codes seen: {sorted(set(codes.tolist()))[:8]})")
        pc.adc_i_values, pc.adc_codes = i_vals, codes
        pc.e_tia_J, pc.e_sub_J, pc.e_adc_J = e_tia, e_sub, e_adc
        pc.i_fullscale = float(np.max(np.abs(i_vals)))

    pc.write_measured_energy(workdir)
    return pc


__all__ = [
    "DacCharMeta", "ReadoutCharMeta", "ReadoutConfig", "PhysicalConverters",
    "emit_dac_char_deck", "parse_dac_transfer",
    "emit_readout_char_deck", "parse_readout_transfer",
    "characterize_converters",
]