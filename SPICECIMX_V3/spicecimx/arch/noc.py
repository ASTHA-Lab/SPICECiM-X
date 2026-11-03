"""
Inter-tile network-on-chip (NoC) traffic, energy and latency.

Every weight-stationary matmul of the ViT has its own tiles (see
``arch.floorplan.place_layers``).  The digital operators between matmuls
(softmax, LayerNorm, residual add, QK^T / SV, GELU) run on the host / global
buffer (GB) attached to one mesh router — the paper's architecture (Sec. IV:
non-IMC operators run digitally on the host).  For each matmul of one
inference:

    input phase   GB -> left-column tile of every tile row of the layer:
                  m x (rows of that tile row) activations x dac_bits
    output phase  bottom-row tile of every tile column -> GB:
                  m x (columns of that tile column) results x adc_bits

The analog input buses and column lines inside the layer macro (loading
model) take the data the rest of the way, so the NoC volume equals the DAC
and ADC conversion counts of the op ledger — nothing is double counted.

Multicast (``noc_multicast``, default on): the Q, K and V matmuls of every
head of a block read the same LayerNorm output, so the GB sends each tile-row
slice once along the XY multicast tree that reaches all of their left tiles
(links/routers of the union of the XY paths).  Off = every matmul gets its
own unicast copy (upper bound).

Analytic model (``noc = analytic``)
    XY dimension-order routing on a 2D mesh.  A unicast flit crossing ``h``
    hops traverses ``h + 1`` routers and ``h`` links of length tile_w/tile_h:
        E = bits * ((h+1) * E_router_bit + L_mm * E_link_bit_mm)
    a multicast flit pays every router and link of its tree once.  Each phase
    is limited by the single GB port (one flit per cycle):
        t = flits + h_max * (router_cycles + link_cycles) + router_cycles
    cycles at the NoC clock (serialisation + pipeline fill).

BookSim2 (``noc = booksim``)
    BookSim2 has no multicast, so it VALIDATES the per-hop model: the unicast
    version of the same traffic is simulated cycle-accurately (``arch.booksim``)
    and compared with the analytic unicast result; the reported NoC numbers
    stay the analytic ones (with the configured multicast).

Default energies: the router value is BookSim2's Orion model (2007 ITRS
32 nm techfile, 5-port mesh router, 64-bit flits) scaled by C*V^2 to the
node; the link is C_wire * V^2 * activity (0.5) per bit per mm.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from spicecimx.analog import tech_scale as _ts
from spicecimx.arch.floorplan import (
    ChipPlan, InterconnectParams, place_layers,
)

# BookSim2 power_module (techfile 32 nm, Vdd 0.9 V): data-dependent energy per
# bit through one 5-port, 4-VC x 4-flit, 64-bit mesh router (buffer write +
# read, crossbar, switch control, output DFF + control; clock and leakage
# excluded), from scripts/booksim/characterize_router.py: 8x8 mesh, uniform
# traffic, 4-flit packets -> 0.112 / 0.110 / 0.110 / 0.111 pJ/bit at loads
# 0.05 / 0.1 / 0.2 / 0.3 flit/node/cycle.
ROUTER_PJ_PER_BIT_32NM = 0.110
ROUTER_REF_NODE_NM = 32.0
ROUTER_REF_VDD = 0.9

_QKV = re.compile(r"^h\d+_[QKV]$")


def router_energy_pJ_per_bit(node_nm: float, vdd: float) -> float:
    """C*V^2 scaling of the BookSim2/Orion 32 nm router energy."""
    return (ROUTER_PJ_PER_BIT_32NM * _ts.cap_scale(node_nm, ROUTER_REF_NODE_NM)
            * (vdd / ROUTER_REF_VDD) ** 2)


def link_energy_pJ_per_bit_mm(c_F_per_um: float, vdd: float,
                              activity: float = 0.5) -> float:
    """Full-swing repeaterless link: activity * C_wire * V^2 per bit per mm."""
    return activity * c_F_per_um * 1e3 * vdd ** 2 * 1e12


@dataclass
class Transfer:
    layer_idx: int
    tag: str
    phase: str                 # 'in' | 'out'
    node: Tuple[int, int]      # tile that sends ('out') or receives ('in')
    bits: int
    slice_idx: int = 0         # tile row ('in') / tile column ('out') of the layer
    group: int = -1            # multicast group (first layer index), -1 = none


def multicast_groups(ledger_layers: List[Dict]) -> List[int]:
    """Group id per layer: runs of consecutive h<i>_{Q,K,V} matmuls (same
    input) share the id of their first layer; every other layer is -1."""
    out, cur = [], -1
    for i, L in enumerate(ledger_layers):
        if _QKV.match(str(L.get("tag", ""))):
            if cur < 0:
                cur = i
            out.append(cur)
        else:
            cur = -1
            out.append(-1)
    return out


def build_transfers(plan: ChipPlan, ledger_layers: List[Dict],
                    p: InterconnectParams, dac_bits: int, adc_bits: int,
                    multicast: bool = False) -> List[Transfer]:
    """One Transfer per (matmul, phase, edge tile)."""
    out: List[Transfer] = []
    groups = multicast_groups(ledger_layers) if multicast else [-1] * len(ledger_layers)
    rows_per_tile = p.tile_sa_rows * p.rs
    cols_per_tile = p.tile_sa_cols * p.cs
    for li, (pl, L) in enumerate(zip(plan.layers, ledger_layers)):
        m, k, n = int(L["m"]), int(L["k"]), int(L["n"])
        for r, xy in enumerate(pl.left_tiles()):
            rows = min(rows_per_tile, k - r * rows_per_tile)
            if rows > 0:
                out.append(Transfer(li, pl.tag, "in", xy, m * rows * int(dac_bits),
                                    slice_idx=r, group=groups[li]))
        for c, xy in enumerate(pl.bottom_tiles()):
            cols = min(cols_per_tile, n - c * cols_per_tile)
            if cols > 0:
                out.append(Transfer(li, pl.tag, "out", xy, m * cols * int(adc_bits),
                                    slice_idx=c))
    return out


def _xy_path(src: Tuple[int, int], dst: Tuple[int, int]):
    """Links (as node pairs) of the XY dimension-order route src -> dst."""
    (x, y), (tx, ty) = src, dst
    links = []
    while x != tx:
        nx = x + (1 if tx > x else -1)
        links.append(((x, y), (nx, y)))
        x = nx
    while y != ty:
        ny = y + (1 if ty > y else -1)
        links.append(((x, y), (x, ny)))
        y = ny
    return links


def _link_mm(link, plan: ChipPlan) -> float:
    (a, b) = link
    return plan.tile_w_mm if a[1] == b[1] else plan.tile_h_mm


def analytic_noc(plan: ChipPlan, transfers: List[Transfer], *,
                 flit_bits: int, router_cycles: int, link_cycles: int,
                 clock_hz: float, e_router_bit_pJ: float,
                 e_link_bit_mm_pJ: float) -> Dict:
    src0 = plan.buffer_xy
    E_router = E_link = 0.0
    bits_moved = 0          # bits leaving / entering the buffer
    bit_hops = 0.0
    # (phase key) -> {"tag", "phase", "bits", "h_max"}; multicast groups are
    # one phase keyed by their group id.
    phases: Dict[Tuple, Dict] = {}
    mc: Dict[Tuple[int, int], List[Transfer]] = {}     # (group, slice) -> dests
    for t in transfers:
        if t.phase == "in" and t.group >= 0:
            mc.setdefault((t.group, t.slice_idx), []).append(t)
            continue
        links = _xy_path(src0, t.node) if t.phase == "in" else _xy_path(t.node, src0)
        h = len(links)
        L_mm = sum(_link_mm(l, plan) for l in links)
        E_router += t.bits * (h + 1) * e_router_bit_pJ * 1e-12
        E_link += t.bits * L_mm * e_link_bit_mm_pJ * 1e-12
        bits_moved += t.bits
        bit_hops += t.bits * h
        ph = phases.setdefault((t.layer_idx, t.phase), {"tag": t.tag, "phase": t.phase,
                                                       "bits": 0, "h_max": 0})
        ph["bits"] += t.bits
        ph["h_max"] = max(ph["h_max"], h)
    for (g, sl), ts in mc.items():
        bits = ts[0].bits                               # same slice for every dest
        links = set()
        for t in ts:
            links.update(_xy_path(src0, t.node))
        routers = {a for a, _ in links} | {b for _, b in links} | {src0}
        L_mm = sum(_link_mm(l, plan) for l in links)
        E_router += bits * len(routers) * e_router_bit_pJ * 1e-12
        E_link += bits * L_mm * e_link_bit_mm_pJ * 1e-12
        bits_moved += bits
        h = max(len(_xy_path(src0, t.node)) for t in ts)
        bit_hops += bits * len(links)
        ph = phases.setdefault((g, "in_mc"), {"tag": f"{ts[0].tag}+qkv", "phase": "in",
                                            "bits": 0, "h_max": 0, "multicast_dests": 0})
        ph["bits"] += bits
        ph["h_max"] = max(ph["h_max"], h)
        ph["multicast_dests"] = max(ph["multicast_dests"], len(ts))
    cyc_total = 0
    rows = []
    for (li, phase), ph in sorted(phases.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        flits = int(math.ceil(ph["bits"] / float(flit_bits)))
        cyc = flits + ph["h_max"] * (router_cycles + link_cycles) + router_cycles
        cyc_total += cyc
        rows.append(dict(ph, layer_idx=li, flits=flits, cycles=cyc))
    t_clk = 1.0 / clock_hz
    return {
        "model": "analytic",
        "energy_J": E_router + E_link,
        "energy_router_J": E_router, "energy_link_J": E_link,
        "bits": bits_moved,
        "mean_hops": (bit_hops / bits_moved) if bits_moved else 0.0,
        "cycles": cyc_total, "latency_s": cyc_total * t_clk,
        "fill_latency_s": sum((r["h_max"] * (router_cycles + link_cycles)
                               + router_cycles) for r in rows) * t_clk,
        "phases": rows,
    }


def estimate_noc(ledger_layers: List[Dict], p: InterconnectParams, ic, *,
                 dac_bits: int, adc_bits: int, vdd: float,
                 workdir: Optional[str] = None) -> Dict:
    """Place the layers, build the traffic and evaluate the NoC.

    ``ledger_layers`` are the per-matmul records of ONE inference
    (``OpLedger.per_inference(...).layers``)."""
    shapes = [(L["tag"], int(L["k"]), int(L["n"])) for L in ledger_layers]
    plan = place_layers(shapes, p, mesh_cols=ic.noc_mesh_cols, buffer=ic.noc_buffer)
    e_r = (ic.noc_router_energy_pJ_per_bit if ic.noc_router_energy_pJ_per_bit >= 0
           else router_energy_pJ_per_bit(p.node_nm, vdd))
    e_l = (ic.noc_link_energy_pJ_per_bit_mm if ic.noc_link_energy_pJ_per_bit_mm >= 0
           else link_energy_pJ_per_bit_mm(p.c_um_F, vdd))
    kw = dict(flit_bits=ic.noc_flit_bits, router_cycles=ic.noc_router_cycles,
              link_cycles=ic.noc_link_cycles, clock_hz=ic.noc_clock_hz,
              e_router_bit_pJ=e_r, e_link_bit_mm_pJ=e_l)
    multicast = bool(getattr(ic, "noc_multicast", True))
    res = analytic_noc(plan, build_transfers(plan, ledger_layers, p, dac_bits,
                                             adc_bits, multicast=multicast), **kw)
    res["multicast"] = multicast
    uni_tr = build_transfers(plan, ledger_layers, p, dac_bits, adc_bits)
    uni = analytic_noc(plan, uni_tr, **kw)
    res["unicast_analytic"] = {k: uni[k] for k in ("energy_J", "energy_router_J",
                                                   "energy_link_J", "cycles",
                                                   "latency_s", "bits", "mean_hops")}
    if ic.noc == "booksim":
        from spicecimx.arch.booksim import run_booksim_phases
        ic._node_nm, ic._vdd = p.node_nm, vdd
        bs = run_booksim_phases(plan, uni_tr, ic, workdir=workdir)
        e_bs = bs["energy_router_J"] + uni["energy_link_J"]
        res["booksim_validation"] = {
            "traffic": "unicast",
            "energy_J": e_bs, "energy_router_J": bs["energy_router_J"],
            "latency_s": bs["latency_s"], "cycles": bs["cycles"],
            "energy_ratio_vs_analytic": e_bs / uni["energy_J"] if uni["energy_J"] else float("nan"),
            "latency_ratio_vs_analytic": (bs["latency_s"] / uni["latency_s"]
                                          if uni["latency_s"] else float("nan")),
            "booksim_energy_scale": bs["booksim_energy_scale"],
            "phases": bs["booksim_phases"],
        }
    res.update({
        "mesh": {"width": plan.width, "height": plan.height,
                 "buffer_xy": list(plan.buffer_xy),
                 "tiles_used": plan.n_tiles_used,
                 "tile_w_mm": plan.tile_w_mm, "tile_h_mm": plan.tile_h_mm,
                 "chip_w_mm": plan.width * plan.tile_w_mm,
                 "chip_h_mm": plan.height * plan.tile_h_mm},
        "e_router_pJ_per_bit": e_r, "e_link_pJ_per_bit_mm": e_l,
        "flit_bits": ic.noc_flit_bits, "clock_hz": ic.noc_clock_hz,
        "layers": [{"tag": pl.tag, "x0": pl.x0, "y0": pl.y0,
                    "tiles_r": pl.grid.tiles_r, "tiles_c": pl.grid.tiles_c}
                   for pl in plan.layers],
    })
    return res
