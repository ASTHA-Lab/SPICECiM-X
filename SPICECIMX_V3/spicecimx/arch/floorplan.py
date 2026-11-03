"""
Physical floorplan of a layer's subarray grid and of the whole chip.

Layer macro
-----------
A static-weight matmul ``(k x n)`` occupies ``R = ceil(k/rs)`` row-blocks by
``C = ceil(n/cs)`` column-blocks of RSN subarrays.  Each grid position holds
the differential pair (+array, -array) side by side.  The grid is laid out
physically as it is indexed:

    * the input bus of activation row i (row-block p) runs left -> right
      from its DAC/row driver across all C positions, tapping the BL driver
      of the +array and then the -array of every position;
    * the output (source) line of column j (column-block q, branch b) runs
      top -> bottom across all R row-blocks into one TIA virtual ground below
      the last row-block (analog partial-sum accumulation by KCL — the op
      ledger counts one TIA/ADC per output column).

``[HW] pe_rows x pe_cols`` subarray positions form a PE and
``tile_rows x tile_cols`` PEs a tile, so a tile is a
``(pe_rows*tile_rows) x (pe_cols*tile_cols)`` block of the layer grid; the
lines cross a ``tile_gap_um`` routing channel at every PE/tile boundary.

Chip
----
Tiles are the network-on-chip nodes.  ``place_layers`` shelf-packs the layer
macros (in execution order) onto a ``W x H`` mesh; the global buffer that
hosts the digital operators (softmax, LayerNorm, residual) is the mesh node
at the centre (or corner).  Inputs travel global buffer -> the layer's
left-column tiles (their DACs drive the analog input buses), outputs travel
the layer's bottom-row tiles (TIA/ADC) -> global buffer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Intermediate-metal (M2/M3-class) wire resistance per micrometre.  First-order
# R = rho_eff / (w * t) with ITRS-2007-style minimum widths / aspect ratios
# (180 nm Al; 90 nm and below Cu with size-effect resistivity):
#   180 nm: w 0.28 t 0.50 um, rho 3.3e-8  -> 0.24 ohm/um
#    90 nm: w 0.14 t 0.25 um, rho 2.6e-8  -> 0.74 ohm/um
#    45 nm: w 0.07 t 0.14 um, rho 3.1e-8  -> 3.2  ohm/um
#    40 nm: w 0.064 t 0.128,  rho 3.2e-8  -> 3.9  ohm/um
# Capacitance stays ~0.2 fF/um across nodes (the well-known constant).
WIRE_R_OHM_PER_UM = {180.0: 0.24, 130.0: 0.42, 90.0: 0.74, 65.0: 1.6,
                     45.0: 3.2, 40.0: 3.9, 32.0: 5.4, 28.0: 7.0, 22.0: 11.0}


def line_r_per_um(node_nm: float) -> float:
    """Default R (ohm/um) of the lines BETWEEN subarrays (input buses, column
    lines): semi-global metal at twice the minimum width and thickness of the
    table above, i.e. a quarter of its resistance.  The in-array bit/source
    lines are the RSN's own ([PARASITICS])."""
    return wire_r_per_um(node_nm) / 4.0


def wire_r_per_um(node_nm: float) -> float:
    """Minimum-pitch wire R (ohm/um) for a node, log-log interpolated."""
    nodes = sorted(WIRE_R_OHM_PER_UM)
    x = min(max(float(node_nm), nodes[0]), nodes[-1])
    for lo, hi in zip(nodes, nodes[1:]):
        if lo <= x <= hi:
            f = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return math.exp((1 - f) * math.log(WIRE_R_OHM_PER_UM[lo])
                            + f * math.log(WIRE_R_OHM_PER_UM[hi]))
    return WIRE_R_OHM_PER_UM[nodes[-1]]


def _length_um(s: str, default_um: float) -> float:
    """Parse a Spectre length string ('2u', '180n', '0.5u', '1e-6')."""
    if not s:
        return default_um
    t = str(s).strip().lower().strip("()")
    mult = {"u": 1.0, "n": 1e-3, "m": 1e3, "p": 1e-6}
    try:
        if t and t[-1] in mult:
            return float(t[:-1]) * mult[t[-1]]
        return float(t) * 1e6
    except ValueError:
        return default_um


@dataclass
class InterconnectParams:
    """Resolved (all-numeric) interconnect parameters for one run."""
    node_nm: float
    pitch_x_um: float           # cell pitch along the input bus (across columns)
    pitch_y_um: float           # cell pitch along the column line (across rows)
    subarray_gap_um: float
    tile_gap_um: float
    r_um: float                 # inter-subarray line ohm/um
    c_um_F: float               # wire F/um
    r_drv: float                # DAC / bus-driver output resistance (ohm)
    r_tia: float                # TIA virtual-ground input resistance (ohm)
    rs: int
    cs: int
    pe_rows: int
    pe_cols: int
    tile_rows: int
    tile_cols: int
    input_shared: bool = False  # unbuffered input bus (resistive loading)
    tap_c_F: float = 2e-15      # capacitance of one bus tap

    @property
    def tile_sa_rows(self) -> int:
        return self.pe_rows * self.tile_rows

    @property
    def tile_sa_cols(self) -> int:
        return self.pe_cols * self.tile_cols

    @property
    def subarray_w_um(self) -> float:
        return self.cs * self.pitch_x_um

    @property
    def subarray_h_um(self) -> float:
        return self.rs * self.pitch_y_um

    @property
    def tile_w_um(self) -> float:
        # tile_sa_cols differential pairs (2 arrays each) + their gaps
        return (self.tile_sa_cols * 2 * (self.subarray_w_um + self.subarray_gap_um)
                + self.tile_gap_um)

    @property
    def tile_h_um(self) -> float:
        return (self.tile_sa_rows * (self.subarray_h_um + self.subarray_gap_um)
                + self.tile_gap_um)

    def as_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["tile_w_um"] = self.tile_w_um
        d["tile_h_um"] = self.tile_h_um
        return d


def resolve_interconnect(ic, hw, pdk=None, peripherals=None) -> InterconnectParams:
    """Fill the ``auto`` (0 / <0) entries of an InterconnectCfg."""
    node = float(getattr(peripherals, "tech_node_nm", 0.0) or 45.0)
    F_um = node * 1e-3
    w_um = _length_um(getattr(pdk, "transistor_w", ""), 2.0)
    # 1T1R cell: access-transistor width + contacted spacing across the input
    # bus; two contacted poly pitches (~8F) along the column line.
    px = ic.cell_pitch_x_um if ic.cell_pitch_x_um > 0 else w_um + 4.0 * F_um
    py = ic.cell_pitch_y_um if ic.cell_pitch_y_um > 0 else 8.0 * F_um
    r_um = ic.wire_r_ohm_per_um if ic.wire_r_ohm_per_um > 0 else line_r_per_um(node)
    if ic.tia_r_in_ohm is not None and ic.tia_r_in_ohm >= 0:
        r_tia = float(ic.tia_r_in_ohm)
    else:
        # Feedback TIA: R_in = R_f / (1 + A_ol), R_f maps the full-scale
        # column current onto the TIA output window.
        vspan = (getattr(peripherals, "vmax_tia", 0.6)
                 - getattr(peripherals, "vmin_tia", 0.0)) or 0.6
        i_fs = getattr(peripherals, "adc_i_fullscale", 8e-4) or 8e-4
        r_tia = (vspan / i_fs) / (1.0 + max(0.0, ic.tia_open_loop_gain))
    return InterconnectParams(
        node_nm=node, pitch_x_um=px, pitch_y_um=py,
        subarray_gap_um=float(ic.subarray_gap_um),
        tile_gap_um=float(ic.tile_gap_um),
        r_um=float(r_um), c_um_F=float(ic.wire_c_fF_per_um) * 1e-15,
        r_drv=float(ic.driver_r_out_ohm), r_tia=float(r_tia),
        rs=int(hw.subarray_rows), cs=int(hw.subarray_cols),
        pe_rows=int(hw.pe_rows), pe_cols=int(hw.pe_cols),
        tile_rows=int(hw.tile_rows), tile_cols=int(hw.tile_cols),
        input_shared=(str(getattr(ic, "input_bus", "buffered")).lower() == "shared"),
        tap_c_F=float(getattr(ic, "bus_tap_c_fF", 2.0)) * 1e-15,
    )


# ---------------------------------------------------------------------------
# Per-layer line geometry
# ---------------------------------------------------------------------------
@dataclass
class LayerGrid:
    k: int
    n: int
    R: int                      # row-blocks
    C: int                      # column-blocks (differential pairs)
    tiles_r: int
    tiles_c: int

    @property
    def n_tiles(self) -> int:
        return self.tiles_r * self.tiles_c


def layer_grid(k: int, n: int, p: InterconnectParams) -> LayerGrid:
    R = -(-int(k) // p.rs)
    C = -(-int(n) // p.cs)
    return LayerGrid(k=int(k), n=int(n), R=R, C=C,
                     tiles_r=-(-R // p.tile_sa_rows),
                     tiles_c=-(-C // p.tile_sa_cols))


def input_bus_segments(C: int, p: InterconnectParams) -> List[float]:
    """Segment lengths (um) of one input-bus line, from the driver to the
    taps in order (+array of position 0, -array of 0, +array of 1, ...).

    Element 0 is the lead from the driver to the first tap; element t>0 joins
    tap t-1 to tap t.  ``2*C`` taps in total."""
    seg = [p.subarray_gap_um]
    step = p.subarray_w_um + p.subarray_gap_um
    for t in range(1, 2 * C):
        L = step
        q_prev, q = (t - 1) // 2, t // 2
        if q != q_prev and q % p.tile_sa_cols == 0:
            L += p.tile_gap_um                  # crosses a PE/tile boundary
        seg.append(L)
    return seg


def column_line_segments(R: int, p: InterconnectParams) -> List[float]:
    """Segment lengths (um) of one output column line, from the TIA up to the
    taps in order of distance (row-block R-1 first, then R-2, ...).

    Element 0 is the lead from the TIA to the nearest tap; ``R`` taps."""
    seg = [p.subarray_gap_um]
    step = p.subarray_h_um + p.subarray_gap_um
    for t in range(1, R):
        L = step
        rb = R - 1 - t                          # row-block of tap t
        if (rb + 1) % p.tile_sa_rows == 0:
            L += p.tile_gap_um
        seg.append(L)
    return seg


# ---------------------------------------------------------------------------
# Chip placement (tiles on a mesh)
# ---------------------------------------------------------------------------
@dataclass
class PlacedLayer:
    tag: str
    grid: LayerGrid
    x0: int                     # mesh column of the macro's left edge
    y0: int                     # mesh row of the macro's top edge

    def left_tiles(self) -> List[Tuple[int, int]]:
        return [(self.x0, self.y0 + r) for r in range(self.grid.tiles_r)]

    def bottom_tiles(self) -> List[Tuple[int, int]]:
        yb = self.y0 + self.grid.tiles_r - 1
        return [(self.x0 + c, yb) for c in range(self.grid.tiles_c)]


@dataclass
class ChipPlan:
    width: int
    height: int
    buffer_xy: Tuple[int, int]
    layers: List[PlacedLayer] = field(default_factory=list)
    tile_w_mm: float = 0.0
    tile_h_mm: float = 0.0

    @property
    def n_tiles_used(self) -> int:
        return sum(l.grid.n_tiles for l in self.layers)


def place_layers(shapes: List[Tuple[str, int, int]], p: InterconnectParams,
                 mesh_cols: int = 0, buffer: str = "center") -> ChipPlan:
    """Shelf-pack the layer macros (in the given order) onto a mesh.

    ``shapes`` is ``[(tag, k, n), ...]`` — one entry per weight-stationary
    matmul, each with its own tiles.  The mesh width defaults to the smallest
    square that holds every tile (grown until the shelves fit).  One extra
    node is reserved for the global buffer.
    """
    grids = [(tag, layer_grid(k, n, p)) for tag, k, n in shapes]
    total = sum(g.n_tiles for _, g in grids) + 1
    widest = max((g.tiles_c for _, g in grids), default=1)
    W = int(mesh_cols) if mesh_cols and mesh_cols > 0 else \
        max(widest, int(math.ceil(math.sqrt(total))))
    while True:
        placed, x, y, shelf_h = [], 0, 0, 0
        for tag, g in grids:
            if g.tiles_c > W:
                raise ValueError(f"layer {tag} needs {g.tiles_c} tile columns "
                                 f"> mesh width {W}")
            if x + g.tiles_c > W:
                x, y, shelf_h = 0, y + shelf_h, 0
            placed.append(PlacedLayer(tag=tag, grid=g, x0=x, y0=y))
            x += g.tiles_c
            shelf_h = max(shelf_h, g.tiles_r)
        H = y + shelf_h
        # the buffer takes the centre node: keep at least one free slot
        if (mesh_cols and mesh_cols > 0) or H <= W + 1:
            break
        W += 1
    H = max(H + 1, 1)                     # one spare row for the buffer node
    if buffer == "corner":
        bxy = (0, H - 1)
    else:
        bxy = (W // 2, H // 2)
    return ChipPlan(width=W, height=H, buffer_xy=bxy, layers=placed,
                    tile_w_mm=p.tile_w_um * 1e-3, tile_h_mm=p.tile_h_um * 1e-3)
