"""
Technology-node + process-corner scaling for peripheral energy and delay.
=========================================================================

Why this module exists
----------------------
The peripheral (DAC/ADC/TIA/MUX/subtractor) energy and latency models read
fixed config constants — they never see the operating point ([SIM] vdd /
process_corner / temperature) or the technology node, so their outputs were
identical across 40 nm vs 90 nm vs 180 nm and across ff/ss/tt.  This module
provides the physical scaling, following the approach used by the standard
CIM benchmarking frameworks:

  * NeuroSim / DNN+NeuroSim collect peripheral latency & energy PER
    TECHNOLOGY NODE from SPICE (PTM models) and fit them; node parameters
    (supply, capacitance, FO4 delay) come from ITRS/IRDS roadmaps.
  * First-order physics (also used in the NeuRRAM scaling projection):
        E_dynamic ∝ C · V²         (switched-capacitance energy)
        delay     ∝ C · V / I_on   (FO4 gate delay; ≈ linear in gate length)
    with C ∝ feature size and V = the node's nominal supply × corner factor.

Model summary
-------------
Supply:        V_nom(node) from the table below (log-interpolated between
               entries); corner multiplier ff = +10 %, ss = −10 % (industry
               convention; matches the corner supplies used in the runs).
Energy:        scale = (C(node)/C(ref)) · (V/V_ref)²   for dynamic blocks;
               static-bias blocks (TIA) scale linearly in V.
Delay:         scale = (node/ref)^e · corner_factor · temp_factor, with
               ff ≈ 0.85×, ss ≈ 1.30× (typical corner timing spread) and
               ≈ +0.1 %/°C above 25 °C (super-threshold mobility loss).
               Both are configurable.

Everything is opt-in; with scaling disabled the legacy fixed-constant
behaviour is reproduced bit-for-bit.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Nominal core supply per node (V).  ITRS/IRDS-style values as used by the
# NeuroSim technology tables and foundry norms.  Log-interpolated between
# entries so any node in [3, 250] nm works.  Override any value from config
# with [PERIPHERALS] nominal_vdd_V when your PDK differs.
# ---------------------------------------------------------------------------
NODE_VDD_NOM = {
    250.0: 2.5,
    180.0: 1.8,
    130.0: 1.3,
    90.0:  1.2,
    65.0:  1.1,
    45.0:  1.0,
    40.0:  1.1,   # 40 nm LP/G norm (e.g. TSMC N40: 1.1 V core)
    32.0:  1.0,
    28.0:  0.9,
    22.0:  0.8,
    16.0:  0.8,
    14.0:  0.8,
    10.0:  0.75,
    7.0:   0.7,
    5.0:   0.65,
    3.0:   0.6,
}

# Corner supply multipliers (industry ±10 % convention).
CORNER_VDD_FACTOR = {"ff": 1.10, "tt": 1.00, "ss": 0.90, "nn": 1.00,
                     "stat": 1.00, "fs": 1.00, "sf": 1.00}

# Corner delay multipliers (typical slow/fast timing spread at fixed V;
# the supply part of the corner is handled separately through V above).
CORNER_DELAY_FACTOR = {"ff": 0.85, "tt": 1.00, "ss": 1.30, "nn": 1.00,
                       "stat": 1.00, "fs": 1.08, "sf": 1.08}


def _norm_corner(corner: str) -> str:
    c = (corner or "tt").strip().lower()
    return c if c in CORNER_VDD_FACTOR else "tt"


def vdd_nominal(node_nm: float) -> float:
    """Nominal supply (V) for a node, log-log interpolated from the table."""
    if node_nm is None or node_nm <= 0:
        return float("nan")
    nodes = sorted(NODE_VDD_NOM)
    if node_nm <= nodes[0]:
        return NODE_VDD_NOM[nodes[0]]
    if node_nm >= nodes[-1]:
        return NODE_VDD_NOM[nodes[-1]]
    for lo, hi in zip(nodes, nodes[1:]):
        if lo <= node_nm <= hi:
            if node_nm in NODE_VDD_NOM:
                return NODE_VDD_NOM[node_nm]
            f = (math.log(node_nm) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return NODE_VDD_NOM[lo] * (1 - f) + NODE_VDD_NOM[hi] * f
    return NODE_VDD_NOM[nodes[-1]]


def corner_vdd(node_nm: float, corner: str,
               vdd_override: float = 0.0,
               nominal_override: float = 0.0) -> float:
    """Operating supply (V) at (node, corner).

    ``vdd_override`` (> 0) wins outright — it is the [SIM] vdd the user
    already sets per corner config, i.e. the most accurate value available.
    Else nominal (table or ``nominal_override``) × corner factor.
    """
    if vdd_override and vdd_override > 0:
        return float(vdd_override)
    v_nom = nominal_override if (nominal_override and nominal_override > 0) \
        else vdd_nominal(node_nm)
    return v_nom * CORNER_VDD_FACTOR[_norm_corner(corner)]


def cap_scale(node_nm: float, ref_node_nm: float) -> float:
    """Switched-capacitance ratio C(node)/C(ref) ≈ node/ref (C ∝ feature
    size: gate cap/µm is roughly flat, total width and wire length shrink
    linearly — the same first-order rule used in the NeuRRAM projection)."""
    if not node_nm or not ref_node_nm or node_nm <= 0 or ref_node_nm <= 0:
        return 1.0
    return float(node_nm) / float(ref_node_nm)


def energy_scale_dynamic(node_nm: float, corner: str, *,
                         ref_node_nm: float = 0.0,
                         vdd_override: float = 0.0,
                         nominal_override: float = 0.0) -> float:
    """E(node,corner)/E(ref,tt) for a dynamic (CV²) block.

    With ``ref_node_nm`` = 0 only the corner-supply part is applied (the
    capacitance is taken as specified for the current node); with a
    reference node the C ∝ node shrink is applied too.
    """
    v = corner_vdd(node_nm, corner, vdd_override, nominal_override)
    v_nom = nominal_override if (nominal_override and nominal_override > 0) \
        else vdd_nominal(node_nm)
    s = (v / v_nom) ** 2 if v_nom and v_nom > 0 else 1.0
    if ref_node_nm and ref_node_nm > 0:
        s *= cap_scale(node_nm, ref_node_nm)
        v_ref = vdd_nominal(ref_node_nm)
        if v_ref and v_ref > 0 and v_nom and v_nom > 0:
            s *= (v_nom / v_ref) ** 2
    return s


def delay_scale(node_nm: float, corner: str, temperature_C: float = 27.0, *,
                ref_node_nm: float = 0.0,
                node_exponent: float = 1.0,
                corner_factors: Optional[Dict[str, float]] = None,
                temp_coeff_per_C: float = 0.001,
                temp_ref_C: float = 25.0) -> float:
    """delay(node, corner, T)/delay(ref, tt, 25 °C).

    node part:    (node/ref)^e            (FO4 ≈ linear in drawn L; e tunable)
    corner part:  CORNER_DELAY_FACTOR     (drive-current spread; overridable)
    temp part:    1 + k·(T − 25 °C)       (mobility loss ≈ +0.1 %/°C default)
    """
    s = 1.0
    if ref_node_nm and ref_node_nm > 0 and node_nm and node_nm > 0:
        s *= (float(node_nm) / float(ref_node_nm)) ** float(node_exponent)
    cf = dict(CORNER_DELAY_FACTOR)
    if corner_factors:
        cf.update({k.lower(): float(v) for k, v in corner_factors.items()})
    s *= cf.get(_norm_corner(corner), 1.0)
    s *= max(0.1, 1.0 + float(temp_coeff_per_C) * (float(temperature_C) - float(temp_ref_C)))
    return s