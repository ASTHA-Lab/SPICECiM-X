"""
Hardware netlist generation for SPICECiM-X v3.

Moved from utils.py.  No logic changes — only:
  * type hints added on the public surface,
  * indentation and string-building unified (using join() instead of += in hot loops),
  * docstrings expanded so each function says what it produces and what the
    pin / instance ordering invariants are (these matter — downstream
    weight_mapping and the PWL renderer depend on them).

Public API
----------
parse_standard_cells(path) -> dict[str, list[str]]
    Map cell name -> pin ordering (parsed from a standard_cell.netlist file).

generate_crossbar_netlist(rows, cols) -> (netlist, name, wl_pins, bl_pins, sl_pins)
generate_bl_wl_decoder_netlist(num_inputs, std_cells) -> (netlist, name, pins)
generate_sl_decoder_netlist(num_inputs, std_cells)    -> (netlist, name, pins)
    Note: as of the 2026-05 revision (see Revision History below) the SL
    decoder is no longer instantiated by ``generate_subarray`` in the
    inference flow.  The function is kept for backward compatibility with
    any callers (e.g., program/write-path tools) that still need it.

generate_subarray(rows, cols, stdcell_path) -> netlist_string
    Full subarray netlist as a string. Calls the three generators above
    plus the standard cell library; wires up the top-level instance plus
    iprobes on every SL.

distribute_weights(layer_name, layer_weights, sa_size, pe_size, tile_size,
                   base_dir) -> dict
    Tile a 2D weight matrix into rs×cs subarrays on disk under
    base_dir/layer_name/Tile{T}/PE{P}/Array_{r}_{c}.npy.

Revision History
----------------
2026-05-20 — SL decoder removed from inference seed netlist.
    Previously, each per-column iprobe ``IPRB{i}`` was wired between the
    crossbar's SL pin (``net{2i+1}i``) and the SL decoder's Y_i output
    (``net{2i}i``), with the decoder pulling Y_i to VSS through its
    internal pass gates.  In the analog-MAC operating mode all SL outputs
    sink simultaneously, so the pass gates carry the full column current
    (up to ~800 μA at LRS for a 16-row column).  Their finite Ron creates
    a non-zero source-line voltage at the cell — typically tens to a few
    hundred mV — which the single-cell characterisation deck (which ties
    NMOS source directly to ``0``) does not see.  After V-linearization
    eliminated the dominant per-layer error, this SL-pedestal mismatch
    became the next-largest contributor, leaving DC-mode RMSE near 5%
    while fastdc (which uses the SPICE characterisation table directly,
    grounded source) sat at 0.45%.

    Since this framework is inference-only (no write/program path runs
    through ``generate_subarray``), the SL decoder serves no functional
    role here.  ``generate_subarray`` now wires every iprobe directly to
    the top-level ``VSS`` net:

        IPRB{i} (net{2i+1}i VSS) iprobe

    matching the characterisation deck's ``(... 0 0)`` source termination
    exactly, and bringing the cell operating conditions into alignment
    with the I-V table used to build h(V).  Expected effect: DC-mode
    per-layer RMSE drops from ~5% to ~0.5%, matching fastdc.

    ``generate_sl_decoder_netlist`` is left in place (still callable) for
    any write-path tools that need it.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Standard cell library parser
# ---------------------------------------------------------------------------
def parse_standard_cells(file_path: str) -> Dict[str, List[str]]:
    """Read the std-cell netlist and return a {cell_name: [pin_order]} map.

    The pin order is critical for the decoder generators below — they
    splice user signals into the right slot by matching against the pin
    name returned here, not by absolute position.
    """
    cells: Dict[str, List[str]] = {}
    current = None
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("subckt"):
                parts = line.split()
                current = parts[1]
                cells[current] = parts[2:]
            elif line.startswith("ends") and current is not None:
                current = None
    return cells


# ---------------------------------------------------------------------------
# Crossbar
# ---------------------------------------------------------------------------
def generate_crossbar_netlist(
    size_bwl: int, size_sl: int,
    *,
    cell_prefix: str = "rram_cell_1T1R_180N_STD",
    parasitics=None,
) -> Tuple[str, str, List[str], List[str], List[str]]:
    """
    Build a size_bwl × size_sl 1T1R crossbar netlist.

    ``cell_prefix`` is the 1T1R cell subckt base name (must match device_emit's
    emitted cells).  ``parasitics`` is an optional ParasiticsCfg-like object;
    when ``parasitics.enable`` is true, each bit-line (row input) and source-line
    (column output) is split into per-cell segments joined by series wire
    resistors (with optional node caps to ground), reproducing crossbar IR-drop.
    When parasitics is None/disabled the netlist is identical to the ideal
    zero-parasitic crossbar.

    Returns:
        netlist, subcircuit_name, wl_pins, bl_pins, sl_pins
    """
    name = f"rram_array_{size_bwl}x{size_sl}"
    bl_pins = [f"BL{i}" for i in range(size_bwl)]
    sl_pins = [f"SL{i}" for i in range(size_sl)]
    wl_pins = [f"WL{i}" for i in range(size_bwl)]

    lines = [
        "// Library name: Custom Library",
        f"// Cell name: {name}",
        "// View name: schematic",
        f"subckt {name} {' '.join(bl_pins + sl_pins + wl_pins)}",
    ]

    use_par = bool(parasitics is not None and getattr(parasitics, "enable", False)
                   and (getattr(parasitics, "r_wire_per_cell_ohm", 0.0) > 0
                        or getattr(parasitics, "c_wire_per_cell_fF", 0.0) > 0))

    if not use_par:
        # ── ideal crossbar (unchanged) ──
        for i in range(size_sl):
            for j in range(size_bwl):
                cell = f"{cell_prefix}_R{j}C{i}"
                lines.append(
                    f"    I{i * size_bwl + j} ({bl_pins[j]} {sl_pins[i]} {wl_pins[j]}) {cell}"
                )
    else:
        # ── crossbar with interconnect parasitics ──
        rw = float(getattr(parasitics, "r_wire_per_cell_ohm", 0.0))
        cw = float(getattr(parasitics, "c_wire_per_cell_fF", 0.0)) * 1e-15
        at_origin = bool(getattr(parasitics, "driver_at_origin", True))

        # BL (row j) segment node as it crosses column i; col-0 tap = pin BL{j}
        def bl_node(j, i):
            return bl_pins[j] if i == 0 else f"blseg_{j}_{i}"
        # SL (col i) segment node as it crosses row j; row-0 tap = pin SL{i}
        def sl_node(i, j):
            return sl_pins[i] if j == 0 else f"slseg_{i}_{j}"

        # cell instances on their local segment nodes
        for i in range(size_sl):
            for j in range(size_bwl):
                cell = f"{cell_prefix}_R{j}C{i}"
                lines.append(
                    f"    I{i * size_bwl + j} ({bl_node(j, i)} {sl_node(i, j)} {wl_pins[j]}) {cell}"
                )
        # BL series wire resistors between adjacent columns, per row
        rk = 0
        for j in range(size_bwl):
            for i in range(size_sl - 1):
                lines.append(f"    RBL_{j}_{i} ({bl_node(j, i)} {bl_node(j, i+1)}) resistor r={rw}")
                if cw > 0:
                    lines.append(f"    CBL_{j}_{i} ({bl_node(j, i+1)} 0) capacitor c={cw}")
                rk += 1
        # SL series wire resistors between adjacent rows, per column
        for i in range(size_sl):
            for j in range(size_bwl - 1):
                lines.append(f"    RSL_{i}_{j} ({sl_node(i, j)} {sl_node(i, j+1)}) resistor r={rw}")
                if cw > 0:
                    lines.append(f"    CSL_{i}_{j} ({sl_node(i, j+1)} 0) capacitor c={cw}")
        # optional WL series resistance (affects transient settling only)
        if getattr(parasitics, "model_wl", False):
            rwl = float(getattr(parasitics, "r_wl_per_cell_ohm", rw))
            # WL{j} is a single shared net here (gate, no DC current); a series
            # string is emitted only for timing fidelity when requested.
            for j in range(size_bwl):
                for i in range(size_sl - 1):
                    pass  # WL taps share the gate net; placeholder for future per-tap WL nodes
        _ = at_origin  # reserved: driver-end selection (origin vs far end)

    lines.append(f"ends {name}")
    lines.append("// End of subcircuit definition.")
    return "\n".join(lines), name, wl_pins, bl_pins, sl_pins


# ---------------------------------------------------------------------------
# Decoder generators
# ---------------------------------------------------------------------------
def _and_tree(input_list: List[str], counter: int, std_cells: Dict[str, List[str]]) -> Tuple[str, str, int]:
    """
    Build a balanced AND-gate tree that ANDs `input_list` together using
    AND2_X1 cells from `std_cells`.  Returns (lines, final_net, counter).
    """
    if len(input_list) == 1:
        return "", input_list[0], counter
    out_lines: List[str] = []
    and_pins = std_cells["AND2_X1"]
    cur = input_list[:]
    while len(cur) > 1:
        new_net = f"net{counter}"
        a, b = cur[0], cur[1]
        formatted = " ".join(
            a if pin == and_pins[0]
            else b if pin == and_pins[1]
            else new_net if pin == "Y"
            else pin
            for pin in and_pins
        )
        out_lines.append(f"I_and{counter} ({formatted}) AND2_X1")
        counter += 1
        cur = [new_net] + cur[2:]
    return "\n".join(out_lines) + "\n", cur[0], counter


def generate_sl_decoder_netlist(
    num_address_pins: int, standard_cells: Dict[str, List[str]]
) -> Tuple[str, str, str]:
    """SL decoder: drives one of 2^num_address_pins outputs (Y0..Y{N-1})."""
    num_io = 2 ** num_address_pins
    name = f"BL_SL_DECODER_{num_address_pins}x{num_io}"
    pins = (
        " ".join(f"A{i}" for i in range(num_address_pins))
        + " VDD VSS "
        + " ".join(f"Y{i}" for i in range(num_io))
        + " "
        + " ".join(f"IO{i}" for i in range(num_io))
        + " VDDP PGMEN"
    )
    lines = [
        f"// Library name: Custom_Library",
        f"// Cell name: {name}",
        f"// View name: schematic",
        f"subckt {name} " + pins,
    ]

    # Inverters for each address line
    inv_pins = standard_cells["INVX1"]
    for i in range(num_address_pins):
        formatted = " ".join(
            f"A{i}" if pin == "A"
            else f"X{i}" if pin == "Y"
            else pin
            for pin in inv_pins
        )
        lines.append(f"I_inv{i} ({formatted}) INVX1")

    # AND-tree for every minterm
    and_outputs: List[str] = []
    counter = 0
    for i in range(num_io):
        binary_str = format(i, f"0{num_address_pins}b")[::-1]
        and_inputs = [
            f"{'X' if binary_str[j] == '0' else 'A'}{j}"
            for j in range(num_address_pins)
        ]
        tree_lines, final_net, counter = _and_tree(and_inputs, counter, standard_cells)
        if tree_lines:
            lines.append(tree_lines.rstrip())
        and_outputs.append(final_net)

    # BL drivers
    bl_pins = standard_cells["BL_DRIVER"]
    for i, and_out in enumerate(and_outputs):
        formatted = " ".join(
            f"Y{i}"      if pin == "OUT"
            else and_out if pin == "RDEN"
            else "PGMEN" if pin == "PGMEN"
            else "VDDP"  if pin == "VD_PGM"
            else f"IO{i}" if pin == "VD_READ"
            else pin
            for pin in bl_pins
        )
        lines.append(f"I_bl_driver{i} ({formatted}) BL_DRIVER")

    lines.append(f"ends BLDECODER_{num_address_pins}x{num_io}")
    lines.append("// End of subcircuit definition.")
    return "\n".join(lines) + "\n", name, pins


def generate_bl_wl_decoder_netlist(
    num_inputs: int, standard_cells: Dict[str, List[str]]
) -> Tuple[str, str, str]:
    """BL/WL decoder: 1-of-N decoder with an AE enable and per-line T outputs."""
    num_io = 2 ** num_inputs
    name = f"BL_WL_DECODER_{num_inputs}x{num_io}"
    pins = (
        "AE "
        + " ".join(f"A{i}" for i in range(num_inputs))
        + " VDD VSS "
        + " ".join(f"Y{i}" for i in range(num_io))
        + " "
        + " ".join(f"IO{i}" for i in range(num_io))
        + " VDDP PGMEN "
        + " ".join(f"T{i}" for i in range(num_io))
    )
    lines = [
        "// Library name: Custom_Library",
        f"// Cell name: {name}",
        "// View name: schematic",
        f"subckt {name} " + pins,
    ]

    inv_pins = standard_cells["INVX1"]
    for i in range(num_inputs):
        formatted = " ".join(
            f"A{i}" if pin == "A"
            else f"X{i}" if pin == "Y"
            else pin
            for pin in inv_pins
        )
        lines.append(f"I_inv{i} ({formatted}) INVX1")

    and_outputs: List[str] = []
    counter = 0
    for i in range(num_io):
        binary_str = format(i, f"0{num_inputs}b")[::-1]
        and_inputs = [
            f"{'X' if binary_str[j] == '0' else 'A'}{j}"
            for j in range(num_inputs)
        ]
        tree_lines, final_net, counter = _and_tree(and_inputs, counter, standard_cells)
        if tree_lines:
            lines.append(tree_lines.rstrip())
        and_outputs.append(final_net)

    bl_drv_pins = standard_cells["BL_DRIVER"]
    or_pins     = standard_cells["OR2_X1"]
    for i, and_out in enumerate(and_outputs):
        formatted_bl = " ".join(
            f"Y{i}"       if pin == "OUT"
            else and_out  if pin == "RDEN"
            else "PGMEN"  if pin == "PGMEN"
            else "VDDP"   if pin == "VD_PGM"
            else f"IO{i}" if pin == "VD_READ"
            else pin
            for pin in bl_drv_pins
        )
        lines.append(f"I_bl_driver{i} ({formatted_bl}) BL_DRIVER")
        formatted_or = " ".join(
            and_out  if pin == or_pins[0]
            else "AE" if pin == or_pins[1]
            else f"T{i}" if pin == "Y"
            else pin
            for pin in or_pins
        )
        lines.append(f"I_or{i} ({formatted_or}) OR2_X1")

    lines.append(f"ends {name}")
    lines.append("// End of subcircuit definition.")
    return "\n".join(lines) + "\n", name, pins


# ---------------------------------------------------------------------------
# Full subarray
# ---------------------------------------------------------------------------
def generate_subarray(
    input_length: int,
    num_tensor_out: int,
    stdcell_file_path: str,
    *,
    cell_prefix: str = "rram_cell_1T1R_180N_STD",
    parasitics=None,
    sl_termination: str = "ground",
) -> str:
    """
    Build the full subarray netlist string:
      <std cell library>
      <crossbar>
      <BL/WL decoder>
      <top-level I1/I2 instantiation + iprobes on every SL>

    ``cell_prefix`` names the 1T1R cell subckts (must match device_emit).
    ``parasitics`` (optional ParasiticsCfg) enables crossbar wire R/C.
    ``sl_termination``: ``"ground"`` returns every column straight to VSS
    (ideal virtual ground); ``"source"`` ends column i on node ``SLT{i}``
    instead, which the testbench drives with ``VSL{i}`` — the boundary
    voltage of the shared column line (loaded RSN, ``arch.loading``).

    The SL decoder is intentionally NOT instantiated — this is an
    inference-only build, and the cell source line is wired directly to
    ``VSS`` through the iprobe.  See the module-level Revision History
    (2026-05-20) for the motivation.

    Args:
        input_length:    BL/WL row count  (number of input rows)
        num_tensor_out:  SL column count  (number of tensor outputs)
        stdcell_file_path: path to standard_cell.netlist (used twice — once
                            included verbatim, once parsed for pin order)
    """
    bl_wl_row_size = input_length
    sl_col_size    = num_tensor_out

    bl_wl_input_size = int(math.ceil(math.log2(bl_wl_row_size)))

    with open(stdcell_file_path, "r") as f:
        stdcell_netlist = f.read()

    crossbar_netlist, xSC, xWL, xBL, xSL = generate_crossbar_netlist(
        bl_wl_row_size, sl_col_size,
        cell_prefix=cell_prefix, parasitics=parasitics,
    )
    std_cells = parse_standard_cells(stdcell_file_path)
    bl_decoder_netlist, bw_name, _bw_pins = generate_bl_wl_decoder_netlist(
        bl_wl_input_size, std_cells
    )
    # NOTE: SL decoder no longer generated for inference seed.  Cell
    # source lines are returned to VSS directly through the per-column
    # iprobes (see end of this function).  generate_sl_decoder_netlist
    # remains available for write/program-path tooling.

    out = (
        stdcell_netlist + "\n\n"
        + crossbar_netlist + "\n\n"
        + bl_decoder_netlist + "\n\n"
    )

    # Top-level instances — crossbar (I1) and BL/WL decoder (I2) only.
    # The crossbar's SL_i pin is wired to ``net{2i+1}i`` so the iprobe
    # sits in the column return path between the cell stack and VSS.
    out += (
        f"I1 ({' '.join(xBL)} "
        + " ".join(f"net{i*2+1}i" for i in range(len(xSL)))
        + f" {' '.join(xWL)}) {xSC}\n"
    )
    out += (
        f"I2 (AE "
        + " ".join(f"A{i}" for i in range(bl_wl_input_size))
        + " VDD VSS "
        + " ".join(f"BL{i}" for i in range(2**bl_wl_input_size))
        + " "
        + " ".join(f"IN{i}" for i in range(2**bl_wl_input_size))
        + " VDDP PGMEN "
        + " ".join(f"WL{i}" for i in range(2**bl_wl_input_size))
        + f") {bw_name}\n"
    )

    # iprobes on every SL — each connects the column's SL node directly
    # to the top-level VSS net.  This is the inference-mode return path
    # (was previously routed through the SL decoder; see Revision
    # History 2026-05-20).
    # Decoder address inputs are tied low: in read mode AE selects every row,
    # and a floating CMOS input leaves the decoder gates half on, drawing a
    # crowbar current that the transient energy would count as crossbar
    # static energy (~100x the real leakage).
    for i in range(bl_wl_input_size):
        out += f"VADDR{i} (A{i} VSS) vsource dc=0 type=dc\n"

    if sl_termination not in ("ground", "source"):
        raise ValueError(f"sl_termination must be 'ground'|'source', got {sl_termination!r}")
    for i in range(sl_col_size):
        end = "VSS" if sl_termination == "ground" else f"SLT{i}"
        out += f"IPRB{i} (net{i*2+1}i {end}) iprobe\n"

    return out


# ---------------------------------------------------------------------------
# Weight tiling
# ---------------------------------------------------------------------------
def distribute_weights(
    layer_name: str,
    layer_weights: np.ndarray,
    synaptic_array_size: Tuple[int, int],
    pe_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    base_dir: str,
) -> Dict[str, int]:
    """
    Split a 2D weight matrix into rs×cs synaptic arrays, pad edges with zeros,
    and save under:

        <base_dir>/<layer_name>/Tile<T>/PE<P>/Array_<sa_row>_<sa_col>.npy

    Returns a summary dict (counts of subarrays / PEs / tiles for sanity).
    Moved verbatim from utils.py; only docstring/type-hints touched.
    """
    if layer_weights.ndim != 2:
        raise ValueError("layer_weights must be a 2D NumPy array.")

    rs, cs = map(int, synaptic_array_size)
    rpe, cpe = map(int, pe_size)
    rt, ct = map(int, tile_size)
    if any(v <= 0 for v in (rs, cs, rpe, cpe, rt, ct)):
        raise ValueError("All size parameters must be positive integers.")

    m, n = layer_weights.shape
    nv = (m + rs - 1) // rs
    nh = (n + cs - 1) // cs
    total_arrays  = nv * nh
    arrays_per_pe = rpe * cpe
    pes_per_tile  = rt  * ct
    total_pes     = (total_arrays + arrays_per_pe - 1) // arrays_per_pe
    total_tiles   = (total_pes + pes_per_tile - 1)     // pes_per_tile

    layer_root = os.path.join(base_dir, layer_name)
    os.makedirs(layer_root, exist_ok=True)
    # Drop weight tiles left by an earlier call (another image, or a run with
    # a different tiling): the backends walk every Array_*.npy under the layer.
    for root, _dirs, files in os.walk(layer_root):
        for f in files:
            if f.startswith("Array_") and f.endswith(".npy"):
                os.remove(os.path.join(root, f))

    written = 0
    for tile_idx in range(total_tiles):
        pe_start = tile_idx * pes_per_tile
        pe_end   = min(pe_start + pes_per_tile, total_pes)
        for pe_in_tile_idx, pe_global_idx in enumerate(range(pe_start, pe_end)):
            start_array_idx = pe_global_idx * arrays_per_pe
            end_array_idx   = min(start_array_idx + arrays_per_pe, total_arrays)
            pe_dir = os.path.join(layer_root, f"Tile{tile_idx}", f"PE{pe_in_tile_idx}")
            os.makedirs(pe_dir, exist_ok=True)
            for array_idx in range(start_array_idx, end_array_idx):
                sa_row = array_idx // nh
                sa_col = array_idx %  nh
                r0, c0 = sa_row * rs, sa_col * cs
                r1, c1 = min(r0 + rs, m), min(c0 + cs, n)
                block = layer_weights[r0:r1, c0:c1]
                if block.shape != (rs, cs):
                    padded = np.zeros((rs, cs), dtype=layer_weights.dtype)
                    padded[:block.shape[0], :block.shape[1]] = block
                    save_block = padded
                else:
                    save_block = block
                np.save(os.path.join(pe_dir, f"Array_{sa_row}_{sa_col}.npy"), save_block)
                written += 1
                if written >= total_arrays:
                    break
            if written >= total_arrays: break
        if written >= total_arrays: break

    return {
        "m": m, "n": n, "rs": rs, "cs": cs,
        "num_arrays_vertically": nv, "num_arrays_horizontally": nh,
        "total_arrays": total_arrays,
        "arrays_per_pe": arrays_per_pe, "total_pes": total_pes,
        "pes_per_tile": pes_per_tile,   "total_tiles": total_tiles,
    }