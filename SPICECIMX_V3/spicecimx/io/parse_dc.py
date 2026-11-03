"""
Cadence Spectre PSF (binary) DC-analysis parser for SPICECiM-X.

This is a structured port of `dataext.py` with the same byte-level parsing
logic, exposed through a small object-oriented API and integrated with the
TransientReadSettings/DcReadSettings pattern used elsewhere in this package.

Format reference (big-endian throughout)
----------------------------------------
A Spectre PSF DC binary file is laid out as:

    HEADER section
    TYPE   section   (type_id -> (name, dtype, units))
    SWEEP  section   (sweep variable trace definition)
    TRACE  section   (signal/trace definitions)
    VALUE  section   (flat stream of 12-byte records: code|tid|float32)
    Section index    (N x (section_num:u32, start_off:u32))
    "Clarissa"       (8-byte literal marker)
    index_start:u32  (absolute offset of section index)

Public API
----------
psf_dc_to_csv(psf_path, csv_path) -> dict
    Convert a single .dc binary to a CSV; returns metadata.

PsfDcReader(psf_path)
    Class wrapper with .columns, .matrix, .n_points, .save_csv(path).

settled_dc_value(matrix, columns, target_net, reducer='last')
    Reduce a DC-sweep matrix to one scalar per target net.
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_U32 = struct.Struct(">I")
_F32 = struct.Struct(">f")
_F64_BE = struct.Struct(">d")


@dataclass(frozen=True)
class DcReadSettings:
    """Knobs for reducing a DC sweep result."""
    # 'last' = take the final operating point in the sweep (e.g. for a
    # parametric sweep across image index, the final image's value)
    # 'mean' = average over all sweep points
    # 'median' = robust reduction
    reducer: str = "last"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _read_u32(buf: bytes, pos: int) -> Tuple[int, int]:
    return _U32.unpack_from(buf, pos)[0], pos + 4


def _read_psf_string(buf: bytes, pos: int) -> Tuple[str, int]:
    """PSF strings: u32 length, then `length` bytes, then padded to 4-byte boundary."""
    n, pos = _read_u32(buf, pos)
    s = buf[pos : pos + n].decode("ascii", errors="replace")
    pos += (n + 3) & ~3  # align up to 4
    return s, pos


# ---------------------------------------------------------------------------
# Section walkers
# ---------------------------------------------------------------------------
def _parse_trailer(buf: bytes) -> Tuple[List[Tuple[int, int]], int]:
    """Locate the section index at the end of the file."""
    marker = buf.rfind(b"Clarissa")
    if marker < 0:
        raise ValueError("Not a Cadence PSF binary file (missing Clarissa marker)")
    index_start = _U32.unpack_from(buf, len(buf) - 4)[0]
    index_end = marker
    if not (0 < index_start < index_end):
        raise ValueError("PSF trailer index pointer is out of range")
    entries: List[Tuple[int, int]] = []
    pos = index_start
    while pos + 8 <= index_end:
        sec_num, start_off = struct.unpack_from(">II", buf, pos)
        entries.append((sec_num, start_off))
        pos += 8
    return entries, index_start


def _parse_type_section(buf: bytes, start: int, end: int) -> Dict[int, str]:
    """Return mapping type_id -> dtype_name (e.g. 'sweep', 'V', 'I', ...)."""
    types: Dict[int, str] = {}
    pos = start + 8
    if pos + 8 <= end and _U32.unpack_from(buf, pos)[0] == 0x16:
        pos += 8
    while pos + 12 <= end:
        code = _U32.unpack_from(buf, pos)[0]
        if code != 0x10:
            pos += 4
            continue
        pos += 4
        type_id, pos = _read_u32(buf, pos)
        name, pos = _read_psf_string(buf, pos)
        types[type_id] = name
        # Skip variable-length trailer of this type def
        while pos + 4 <= end:
            nxt = _U32.unpack_from(buf, pos)[0]
            if nxt in (0x10, 0x15):
                break
            pos += 4
    return types


def _parse_sweep_section(buf: bytes, start: int, end: int) -> Tuple[int, str]:
    """Return (sweep_trace_id, sweep_name)."""
    pos = start + 8
    while pos + 12 <= end:
        code = _U32.unpack_from(buf, pos)[0]
        if code == 0x10:
            pos += 4
            tid, pos = _read_u32(buf, pos)
            name, pos = _read_psf_string(buf, pos)
            return tid, name
        pos += 4
    raise ValueError("Sweep section contains no trace entry")


def _parse_trace_section(buf: bytes, start: int, end: int) -> List[Tuple[int, str]]:
    """Return list of (trace_id, trace_name) in declaration order."""
    traces: List[Tuple[int, str]] = []
    pos = start + 8
    if pos + 8 <= end and _U32.unpack_from(buf, pos)[0] == 0x16:
        pos += 8
    while pos + 12 <= end:
        code = _U32.unpack_from(buf, pos)[0]
        if code != 0x10:
            pos += 4
            continue
        pos += 4
        tid, pos = _read_u32(buf, pos)
        name, pos = _read_psf_string(buf, pos)
        traces.append((tid, name))
        if pos + 4 <= end:
            pos += 4  # skip type_ref
    return traces


def _parse_single_op_section4(buf: bytes, start: int, end: int) -> Dict[str, float]:
    """
    Parse a single-operating-point DC PSF section 4.

    Spectre emits this layout for one-point DC analyses (which is what
    ``alter ... ; dc_imgN dc oppoint=rawfile`` produces — each named
    ``dc_imgN`` is one operating point, not a sweep).  In that case
    there is no SWEEP section (2) or TRACE section (3), and section 4
    stores the values inline. Each record is::

        0x00000010   (4 bytes)   record start marker
        type_id      (4 bytes)   monotonically increasing record id
        N            (4 bytes)   name length
        name         (N bytes, 4-byte padded)
        type_ref     (4 bytes)   reference into the TYPES section
        value        (8 bytes)   IEEE 754 big-endian double
        0x00000021   (4 bytes)   attributes block start (optional)
        ...                       'units' + unit string + padding

    The ``type_ref`` field is NOT a fixed literal — earlier versions of
    this parser hardcoded it to ``0x16``/``0x1a`` (the values observed in
    the very first analysis of a multi-analysis deck), which made
    everything after ``dc_img0`` fail to parse.  Type IDs are assigned
    sequentially across the whole simulation, so by ``dc_img1`` the
    type-refs have already advanced to ``0x63b`` or higher.  The fix is
    simple: don't validate the type-ref value, just consume it and read
    the 8-byte double that always follows.

    Returns ``{trace_name: scalar_value}``.
    """
    out: Dict[str, float] = {}
    pos = start + 8  # skip section header
    while pos + 4 <= end:
        code = _U32.unpack_from(buf, pos)[0]
        if code != 0x10:
            pos += 4
            continue
        record_pos = pos
        pos += 4  # consume 0x10 record-start marker
        if pos + 8 > end:
            break
        _record_id = _U32.unpack_from(buf, pos)[0]; pos += 4
        name_len   = _U32.unpack_from(buf, pos)[0]; pos += 4
        if name_len == 0 or name_len > 256 or pos + name_len > end:
            pos = record_pos + 4
            continue
        name = buf[pos: pos + name_len].decode("ascii", errors="replace")
        pos += name_len
        # 4-byte alignment for the name field
        pos += (-name_len) & 3
        # type-ref (4 bytes) — value varies per simulation, don't check it
        if pos + 4 > end:
            break
        type_ref = _U32.unpack_from(buf, pos)[0]
        # Defensive: refuse if the "type-ref" looks like a stray record
        # marker or attribute marker — those would mean the record is
        # malformed and we'd misalign the double read below.
        if type_ref in (0x10, 0x21, 0x15):
            pos = record_pos + 4
            continue
        pos += 4
        if pos + 8 > end:
            break
        (value,) = _F64_BE.unpack_from(buf, pos); pos += 8
        out[name] = value
        # Optional units attribute block follows; outer loop finds next 0x10.
    return out




def _parse_value_section(
    buf: bytes,
    start: int,
    end: int,
    n_traces_per_point: int,
) -> np.ndarray:
    """Decode the flat value stream into a (n_points, n_traces_per_point) array."""
    payload_start = start + 8
    payload = buf[payload_start:end]
    rec_size = 12
    n_records = len(payload) // rec_size
    if n_records % n_traces_per_point != 0:
        raise ValueError(
            f"VALUE section record count ({n_records}) is not a multiple of "
            f"traces-per-point ({n_traces_per_point}); file may be truncated."
        )
    n_points = n_records // n_traces_per_point
    dtype = np.dtype([("code", ">u4"), ("tid", ">u4"), ("val", ">f4")])
    arr = np.frombuffer(payload[: n_records * rec_size], dtype=dtype)
    if not np.all(arr["code"] == 0x10):
        raise ValueError("VALUE section contains unexpected record codes")
    return arr["val"].astype(np.float64, copy=False).reshape(n_points, n_traces_per_point)


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------
class PsfDcReader:
    """
    Read a Cadence Spectre DC binary PSF file and expose the result as a
    NumPy matrix + column names.

    Attributes
    ----------
    columns : list[str]
        Column order, with the sweep variable first.
    matrix : np.ndarray of shape (n_points, len(columns))
    sweep_name : str
    sweep_id   : int
    """

    def __init__(self, psf_path: str | Path):
        self.psf_path = Path(psf_path)
        buf = self.psf_path.read_bytes()

        entries, index_start = _parse_trailer(buf)
        entries_sorted = sorted(entries, key=lambda e: e[0])
        section_bounds: Dict[int, Tuple[int, int]] = {}
        for i, (sec_num, off) in enumerate(entries_sorted):
            end_off = entries_sorted[i + 1][1] if i + 1 < len(entries_sorted) else index_start
            section_bounds[sec_num] = (off, end_off)

        # Single-operating-point DC files have only sections 0, 1, and 4
        # (no SWEEP=2 or TRACE=3). This is what altergroup-driven analyses
        # produce — each named ``dc_imgN`` is one op-point, not a sweep.
        # Swept DC analyses have all five sections.
        value_start, value_end = section_bounds[4]
        is_single_op = (2 not in section_bounds) or (3 not in section_bounds)

        if is_single_op:
            # Parse the inline (name -> double) records out of section 4
            name_to_val = _parse_single_op_section4(buf, value_start, value_end)
            if not name_to_val:
                raise ValueError(
                    f"PSF {self.psf_path} has no swept axis and no decodable "
                    "single-op records — file may be corrupt or empty."
                )
            # Synthesize a "sweep" axis with a single phantom point so the
            # (matrix, columns) API stays uniform across single-op and swept.
            names = list(name_to_val.keys())
            self.sweep_name = "_op_"
            self.sweep_id   = -1
            self.columns    = [self.sweep_name, *names]
            self.matrix     = np.empty((1, 1 + len(names)), dtype=np.float64)
            self.matrix[0, 0]  = 0.0
            for j, n in enumerate(names, start=1):
                self.matrix[0, j] = name_to_val[n]
            return

        # Swept DC: original 5-section parser path
        type_start, type_end   = section_bounds[1]
        sweep_start, sweep_end = section_bounds[2]
        trace_start, trace_end = section_bounds[3]

        # Parse metadata
        _ = _parse_type_section(buf, type_start, type_end)
        sweep_id, sweep_name = _parse_sweep_section(buf, sweep_start, sweep_end)
        traces = _parse_trace_section(buf, trace_start, trace_end)

        # Determine column order by sampling the first sweep point
        first_tid = _U32.unpack_from(buf, value_start + 8 + 4)[0]
        if first_tid == sweep_id:
            column_order = [sweep_name, *(n for _, n in traces)]
        else:
            n_per = 1 + len(traces)
            first_ids = [
                _U32.unpack_from(buf, value_start + 8 + i * 12 + 4)[0]
                for i in range(n_per)
            ]
            id_to_name = {sweep_id: sweep_name, **{t: n for t, n in traces}}
            column_order = [id_to_name[i] for i in first_ids]

        self.columns    = column_order
        self.sweep_name = sweep_name
        self.sweep_id   = sweep_id
        self.matrix     = _parse_value_section(buf, value_start, value_end, len(column_order))

    @property
    def n_points(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_traces(self) -> int:
        """Number of non-sweep traces (excludes the sweep variable column)."""
        return self.matrix.shape[1] - 1

    def column(self, name: str) -> np.ndarray:
        """Get a 1-D array of values for a named trace."""
        try:
            j = self.columns.index(name)
        except ValueError:
            raise KeyError(f"No such trace '{name}'. Available: {self.columns}")
        return self.matrix[:, j]

    def save_csv(self, csv_path: str | Path) -> None:
        with Path(csv_path).open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            for row in self.matrix:
                w.writerow(f"{v:.8g}" for v in row)


# ---------------------------------------------------------------------------
# Convenience: file -> CSV in one call
# ---------------------------------------------------------------------------
def psf_dc_to_csv(psf_path: str | Path, csv_path: str | Path) -> Dict:
    """
    Convert a Cadence Spectre DC binary PSF file to a CSV file.
    Returns metadata with `columns`, `n_points`, `n_traces`, `output`.
    """
    r = PsfDcReader(psf_path)
    r.save_csv(csv_path)
    return {
        "columns":  r.columns,
        "n_points": r.n_points,
        "n_traces": r.n_traces,
        "output":   str(csv_path),
    }


# ---------------------------------------------------------------------------
# Reducer: pick the relevant operating point(s) from a sweep
# ---------------------------------------------------------------------------
def settled_dc_value(
    matrix: np.ndarray,
    columns: List[str],
    target_net: str,
    reducer: str = "last",
) -> float:
    """
    Reduce a multi-point DC sweep for `target_net` into a single scalar.

    'last'   : take the value at the final sweep point.
    'mean'   : average across all sweep points.
    'median' : median across all sweep points (robust to convergence wobble).
    """
    j = columns.index(target_net)
    vals = matrix[:, j]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    r = reducer.lower()
    if r == "last":   return float(vals[-1])
    if r == "mean":   return float(np.mean(vals))
    if r == "median": return float(np.median(vals))
    raise ValueError(f"Unknown reducer '{reducer}'.")


# ---------------------------------------------------------------------------
# Folder walker (mirrors process_files_in_folder for transient, but for DC)
# ---------------------------------------------------------------------------
def _find_dc_binaries(folder_path: str | Path) -> List[Path]:
    """Locate every `*.dc` binary under folder_path (recursive)."""
    return sorted(Path(folder_path).rglob("*.dc"))


def process_folder_dc(
    folder_path: str | Path,
    ncolumn: int,
    settings: DcReadSettings = DcReadSettings(),
) -> Optional[Dict[str, List[float]]]:
    """
    Reduce all DC sweep dumps inside `folder_path` (one or more `*.dc` files)
    into per-net settled values, ordered by sweep position.

    The result is a dict identical in shape to `process_folder` from
    parse_transient.py — column 'Folder' plus IPRB{k}:in lists. Each list
    has one entry per `*.dc` file found (sorted), making it directly
    consumable by the existing downstream reducer.
    """
    folder_path = Path(folder_path)
    dc_files = _find_dc_binaries(folder_path)
    if not dc_files:
        return None

    target_names = [f"IPRB{c}:in" for c in range(ncolumn)]
    per_net: Dict[str, List[float]] = {n: [] for n in target_names}

    for dc_file in dc_files:
        try:
            reader = PsfDcReader(dc_file)
        except Exception:
            for n in target_names:
                per_net[n].append(float("nan"))
            continue
        for n in target_names:
            if n in reader.columns:
                per_net[n].append(settled_dc_value(reader.matrix, reader.columns,
                                                   n, settings.reducer))
            else:
                per_net[n].append(float("nan"))

    return {"Folder": folder_path.name, **per_net}