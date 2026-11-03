"""
SCS testbench rendering for SPICECiM-X v3.

Two flavors of testbench:

1. Transient — the existing ``_golden_input.scs`` template, parameterized
   with a PWL voltage source per BL.  The PWL encodes M images of activations
   serially in time; one Spectre run per subarray-branch covers all M.

2. DC — a new ``_golden_input_dc.scs`` template that runs a parametric DC
   sweep across image index.  Each BL voltage is selected from a per-image
   parameter table by the sweep variable; one Spectre run per subarray-branch
   covers all M operating points (Option A from the design memo).

The PWL renderer plus the testbench substituter are both stateless string
builders — multiple subarrays can be rendered concurrently with no shared
state.
"""

from __future__ import annotations

import os
import re
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# TRANSIENT: PWL source generation
# ---------------------------------------------------------------------------
def render_pwl_sources_text(
    V: np.ndarray,
    pulse_width: float,
    trise: float,
    tfall: float,
    scol: int,
    *,
    name: str = "VI",
    node: str = "IN",
    ref: str = "0",
) -> Tuple[str, int, float]:
    """
    Build the PWL source text for a voltage matrix ``V`` of shape
    ``(M, cols)``.  One ``VI{col}`` vsource per column; each emits M
    rectangular pulses sequentially.  ``name``/``node``/``ref`` rename the
    source, its positive node and its reference node (the loaded-RSN column
    terminations use ``VSL{col} (SLT{col} VSS)``).

    Each pulse contributes the 5-point pattern:

        (t,            0)
        (t+pw,         0)
        (t+pw+trise,   V)
        (t+2pw,        V)
        (t+2pw+tfall,  0)
        next_t = t + 3*pw + trise + tfall

    So the per-image window length is ``3*pw + trise + tfall``.

    Returns ``(text, M, total_time)`` where M is the number of pulses
    written per source.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got shape {V.shape}")

    M, cols = V.shape
    out: List[str] = []
    total_time = 0.0
    for col in range(cols):
        time_start = 0.0
        waveform: List[Tuple[float, float]] = []
        for v in V[:, col]:
            waveform.extend([
                (time_start,                          0.0),
                (time_start + pulse_width,            0.0),
                (time_start + pulse_width + trise,    float(v)),
                (time_start + 2 * pulse_width,        float(v)),
                (time_start + 2 * pulse_width + tfall, 0.0),
            ])
            time_start += 3 * pulse_width + trise + tfall
        total_time = time_start  # final = end-of-last-pulse
        waveform_str = " ".join(f"{t} {v}" for t, v in waveform)
        out.append(
            f"{name}{col} ({node}{col % scol} {ref}) vsource "
            f"wave=[{waveform_str}] pwl_period={total_time} type=pwl\n"
        )
    return "".join(out), M, total_time


def generate_pwl_sources(
    csv_path: str, outpath: str,
    pulse_width: float, trise: float, tfall: float,
    scol: int,
) -> Tuple[int, float]:
    """
    Read a voltage CSV (first column = index, first row = header) and emit
    the corresponding PWL SCS file at ``outpath``.

    Returns ``(M, total_time)`` — the per-source pulse count and the total
    waveform duration in seconds.

    Backward-compatible with the v2 ``generate_pwl_sources`` signature.
    """
    df = pd.read_csv(csv_path, index_col=0)
    V = df.to_numpy(dtype=np.float64)
    text, M, t_total = render_pwl_sources_text(V, pulse_width, trise, tfall, scol)
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        f.write(text)
    return M, t_total


# ---------------------------------------------------------------------------
# DC: parameter-table source generation (Option A — parametric sweep)
# ---------------------------------------------------------------------------
def render_dc_sources_text(
    V: np.ndarray,
    sweep_var: str = "img_idx",     # kept for API back-compat; no longer used
    *,
    scol: int = 16,
) -> Tuple[str, int]:
    """
    Build a DC source deck where each BL gets one ``vsource dc=vi{col}`` whose
    value is bound to a netlist parameter.  Returns (source_text, M).

    The returned ``source_text`` declares 16 parameters (``vi0=0 vi1=0 ...``)
    and 16 vsources of the form::

        VI0 (IN0 0) vsource dc=vi0 type=dc
        VI1 (IN1 0) vsource dc=vi1 type=dc
        ...

    Per-image values are bound via ``altergroup`` blocks emitted by
    ``render_dc_altergroups`` and run between successive ``dc`` analyses —
    this is the documented Spectre mechanism for parametric per-source
    sweeps (see Spectre Reference Manual, page 56, "Alter Group (altergroup)").

    Earlier versions of this code emitted ``dc=table(...)`` or ``dc=pwl(...)``
    expressions, but neither is a valid Spectre expression function. The
    legal expression functions are: log/ln/log10/exp/sqrt/min/max/abs/pow/
    int/floor/ceil/fmod/sgn/sign/sin/cos/tan/asin/acos/atan/atan2/hypot/
    sinh/cosh/tanh/asinh/acosh/atanh (see manual, page 475-477).

    Returns
    -------
    text : str
        Parameter declarations + vsource lines, in the order above.
    M : int
        Number of images (rows of V); needed by the caller to emit the
        matching number of altergroup blocks.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got shape {V.shape}")
    M, cols = V.shape

    # 1) Parameter declarations on one line
    param_decls = " ".join(f"vi{c}=0" for c in range(cols))

    # 2) One VI vsource per column, bound to the matching parameter
    src_lines: List[str] = []
    for col in range(cols):
        src_lines.append(
            f"VI{col} (IN{col % scol} 0) vsource dc=vi{col} type=dc\n"
        )

    text = f"parameters {param_decls}\n\n" + "".join(src_lines)
    return text, M


def render_dc_altergroups(
    V: np.ndarray,
    *,
    analysis_kwargs: str = "oppoint=rawfile maxiters=150 maxsteps=10000",
) -> str:
    """
    Emit a ``paramset`` block of per-image vi{c} values plus a ``sweep``
    analysis that runs ONE inner DC operating-point analysis across all M
    paramset rows.

    This is the canonical Spectre mechanism for "one elaboration, M
    Newton solves" — documented in the Spectre Reference Manual at:

      * page 509 — "Parameter Set - Block of Data (paramset)"
      * page 406 — "Sweep Analysis (sweep)" — example::

            swp sweep param=temp values=[-50 0 50 100 125] {
                oppoint dc oppoint=logfile
            }

    For our problem the paramset rows hold the per-image VI source values,
    so each sweep iteration realises one image's bias.  Spectre rolls all
    M operating points into a single swept-DC PSF file (the same 5-section
    layout the v2 ``dc.dc`` reference had), so the original swept-DC
    branch of ``PsfDcReader`` reads it as ``(n_points=M, columns)``.

    Compared to the previous alter-statement chain — M ``alter`` blocks +
    M named ``dc`` analyses, M elaborations of the parameter table — this
    cuts the per-deck cost dramatically (the M elaborations are the
    expensive part; the Newton solves themselves are milliseconds each).

    The function name is kept for back-compat with callers; what it emits
    is a single paramset + a single sweep wrapper rather than a chain.

    Parameters
    ----------
    V
        (M, cols) array of per-image voltages.  Row i column c is the
        value applied to ``vi{c}`` for image i.
    analysis_kwargs
        Kwargs string for the inner ``dc`` analysis line.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got shape {V.shape}")
    M, cols = V.shape

    param_names = " ".join(f"vi{c}" for c in range(cols))
    rows = [" ".join(f"{float(V[i, c]):.8g}" for c in range(cols))
            for i in range(M)]

    block = (
        "images paramset {\n"
        f"  {param_names}\n"
        + "".join(f"  {r}\n" for r in rows)
        + "}\n"
        "\n"
        "dc_sweep sweep paramset=images {\n"
        f"  dc_op dc {analysis_kwargs}\n"
        "}\n"
    )
    return block


# ---------------------------------------------------------------------------
# Common: pad the drive matrix to whole subarrays
# ---------------------------------------------------------------------------
def pad_rows_to_subarrays(V: np.ndarray, subarray_rows: int) -> np.ndarray:
    """Zero-pad the columns of ``V`` (M, k) up to a multiple of
    ``subarray_rows``, so the last subarray of a layer gets a 0 V source on
    every IN pin instead of leaving the unused ones floating.  The padded
    rows hold zero weights, and a 0 V input draws no current, so the result
    is unchanged — only the deck is well-defined."""
    k = V.shape[1]
    k_pad = -(-k // subarray_rows) * subarray_rows
    if k_pad == k:
        return V
    return np.hstack([V, np.zeros((V.shape[0], k_pad - k), dtype=V.dtype)])


# ---------------------------------------------------------------------------
# Common: extract BL slice for a given subarray row
# ---------------------------------------------------------------------------
def slice_pwl_sources_for_subarray(
    pwl_text: str,
    subarray_row: int,
    subarray_rows: int,
) -> str:
    """
    Pick the ``subarray_rows`` consecutive PWL lines that drive *this*
    subarray's BLs from the full PWL deck.

    The full deck has one VI{col} per column, ordered 0 .. total_cols-1.
    The subarray at row index ``subarray_row`` consumes lines
    ``[subarray_row*subarray_rows : (subarray_row+1)*subarray_rows]``.
    """
    lines = pwl_text.splitlines(keepends=True)
    s = subarray_row * subarray_rows
    e = s + subarray_rows
    return "".join(lines[s:e])


def array_indices_from_npy(filename: str) -> Tuple[int, int]:
    """Parse ``Array_{row}_{col}.npy`` into ``(row, col)``."""
    m = re.search(r"Array_(\d+)_(\d+)\.npy", os.path.basename(filename))
    if not m:
        raise ValueError(f"Cannot parse subarray indices from '{filename}'")
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Testbench rendering
# ---------------------------------------------------------------------------
def render_testbench_transient(
    template_path: str,
    output_path: str,
    *,
    sim_time: float,
    input_sources: str,
    device_model_path: str,
    transistor_model: str,
    process_corner: str,
    vdd: float, vss: float, temperature: float,
    iprobe_save_list: Sequence[str],
    tnom: float = 27.0,
    maxstep: float = 0.0,
) -> None:
    """
    Render the transient testbench by substituting placeholders in
    ``_golden_input.scs``.  ``maxstep`` > 0 caps the transient time step
    (``[SIM] tran_maxstep``).

    Placeholders consumed:
        <sim_time>, <input_source>, <dev_mod>, <transistor_mod>,
        <pcorner>, <vdd>, <vss>, <temp>, <csnames>
    """
    with open(template_path, "r") as f:
        content = f.read()

    csnames = " ".join(iprobe_save_list)
    subs = {
        "<sim_time>":        str(sim_time),
        "<input_source>":    input_sources,
        "<transistor_mod>":  transistor_model,
        "<pcorner>":         process_corner,
        "<vdd>":             str(vdd),
        "<vss>":             str(vss),
        "<temp>":            str(temperature),
        "<tnom>":            str(tnom),
        "<csnames>":         csnames,
    }
    for k, v in subs.items():
        content = content.replace(k, v)
    # config-drive tnom even if the golden deck still has a literal tnom=NN
    content = re.sub(r"tnom\s*=\s*[0-9.eE+-]+", f"tnom={tnom}", content)
    if maxstep and maxstep > 0:
        content = re.sub(r"^(tran\s+tran\s+stop=\S+)", rf"\1 maxstep={maxstep:g}",
                         content, count=1, flags=re.M)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


def render_testbench_dc(
    template_path: str,
    output_path: str,
    *,
    n_images: int,
    input_sources: str,
    altergroups: str,
    device_model_path: str,
    transistor_model: str,
    process_corner: str,
    vdd: float, vss: float, temperature: float,
    iprobe_save_list: Sequence[str],
    sweep_var: str = "img_idx",          # kept for back-compat, unused
    tnom: float = 27.0,
) -> None:
    """
    Render the DC testbench by substituting placeholders in
    ``_golden_input_dc.scs``.

    Placeholders consumed:
        <input_source>, <altergroups>, <transistor_mod>, <pcorner>,
        <vdd>, <vss>, <temp>, <csnames>

    The deck runs M named DC analyses (``dc_img0`` .. ``dc_img{M-1}``),
    each preceded by an altergroup that binds the per-image vi0..vi{cs-1}
    parameter values.  Each named analysis writes its own ``dc_img{i}.dc``
    file under the psf/ output directory.
    """
    with open(template_path, "r") as f:
        content = f.read()

    csnames = " ".join(iprobe_save_list)
    subs = {
        "<input_source>":    input_sources,
        "<altergroups>":     altergroups,
        "<transistor_mod>":  transistor_model,
        "<pcorner>":         process_corner,
        "<vdd>":             str(vdd),
        "<vss>":             str(vss),
        "<temp>":            str(temperature),
        "<tnom>":            str(tnom),
        "<csnames>":         csnames,
    }
    for k, v in subs.items():
        content = content.replace(k, v)
    # config-drive tnom even if the golden deck still has a literal tnom=NN
    content = re.sub(r"tnom\s*=\s*[0-9.eE+-]+", f"tnom={tnom}", content)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)