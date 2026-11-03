"""
Parse Cadence Spectre transient psfascii dumps for SPICECiM-X.

Replaces dc.py (the file misleadingly named after DC analysis but actually
parsing transient outputs).

Key fix vs the old implementation
---------------------------------
The old `max_abs_value_finder` returned the sample with the largest |value|
inside each time window.  In a PWL waveform that rises in `trise`, plateaus
for `~pulse_width`, and falls in `tfall`, the peak |I| often lands on a
displacement-current spike at an edge rather than on the settled MVM current
at the plateau.

`settled_value_finder` instead reduces the samples that lie inside a
configurable fraction of each window — by default the middle [0.40, 0.65]
of the plateau — using a median (robust) / mean / last-sample reducer.

Public API
----------
read_psfascii_tran(file_path)
    Returns (params, values) NumPy arrays — the flat token/value stream.

build_time_windows(params, values, start_time, increment)
    Returns list of (window_start_idx, window_end_idx) covering the dump.

settled_value(params, values, start_idx, end_idx, target_net,
              window_lo, window_hi, reducer)
    Returns the settled current/voltage for `target_net` in one window.

process_folder(folder_path, settings, ncolumn)
    Walks a single subarray's psf/tran.tran.tran and returns the per-window
    settled values for every IPRB<k>:in net.

process_files_in_folder(base_path, settings, output_csv_path, ncolumn)
    Drop-in replacement for the old `process_files_in_folder`.

process_power_energy_in_folder(base_path, output_csv_path, power_label)
    Drop-in replacement for the old power/energy reducer.  Unchanged from
    the original except for clearer ordering and explicit np.trapezoid use.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Settings — keep this struct small so the parser stays trivially picklable
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransientReadSettings:
    """Knobs for windowing and reducing transient samples."""
    start_time:        float = 0.0
    increment:         float = 3e-9      # one MVM window
    settling_lo:       float = 0.40
    settling_hi:       float = 0.65
    reducer:           str   = "median"  # "median" | "mean" | "last"
    # Set by the orchestrator if it wants concurrency capped
    max_workers:       Optional[int] = None


# ---------------------------------------------------------------------------
# Low-level psfascii reader
# ---------------------------------------------------------------------------
def read_psfascii_tran(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a Spectre psfascii transient dump (`tran.tran.tran`) and return two
    flat arrays preserving the file's original order:
        params: 1-D array of token strings (e.g. '"time"', '"IPRB0:in"', ...)
        values: 1-D array of floats (NaN where parse failed)

    The file is structured as a header followed by repeated (token, value)
    pairs under a `VALUE` section.  Everything after the VALUE marker is
    read until EOF.
    """
    params: List[str] = []
    values: List[float] = []

    with open(file_path, "r") as f:
        header_passed = False
        for line in f:
            if not header_passed:
                if "VALUE" in line:
                    header_passed = True
                continue
            line = line.strip().lstrip("(").rstrip(")")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            params.append(parts[0])
            try:
                values.append(float(parts[1]))
            except ValueError:
                values.append(np.nan)

    return np.array(params), np.array(values)


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
def build_time_windows(
    params: np.ndarray,
    values: np.ndarray,
    start_time: float,
    increment: float,
) -> List[Tuple[int, int]]:
    """
    Slice the flat stream into MVM windows of length `increment`.
    Each window covers [t_k, t_{k+1}) where t_k = start_time + k*increment.

    Returns a list of (window_start_idx, window_end_idx) — inclusive at both
    ends — indexing into the *flat* params/values arrays.  Index boundaries
    are anchored on '"time"' tokens so that all per-net samples that share a
    time-step land in the same window.
    """
    time_mask = (params == '"time"')
    time_idxs = np.nonzero(time_mask)[0]
    if time_idxs.size == 0:
        return []

    time_vals = values[time_idxs]
    windows: List[Tuple[int, int]] = []
    t0 = start_time
    while True:
        j0 = np.searchsorted(time_vals, t0, side="left")
        if j0 >= len(time_vals):
            break
        idx0 = int(time_idxs[j0])

        t1 = t0 + increment
        j1 = np.searchsorted(time_vals, t1, side="left")
        if j1 >= len(time_vals):
            windows.append((idx0, int(time_idxs[-1])))
            break
        windows.append((idx0, int(time_idxs[j1])))
        t0 = t1
    return windows


# ---------------------------------------------------------------------------
# Settled-value extraction  (★ correctness fix ★)
# ---------------------------------------------------------------------------
def _reduce(arr: np.ndarray, how: str) -> float:
    """Reduce a 1-D array of samples to a scalar using the given reducer."""
    if arr.size == 0:
        return float("nan")
    how = how.lower()
    if how == "median":
        return float(np.median(arr))
    if how == "mean":
        return float(np.mean(arr))
    if how == "last":
        return float(arr[-1])
    raise ValueError(f"Unknown reducer '{how}'; use 'median' | 'mean' | 'last'.")


def settled_value(
    params: np.ndarray,
    values: np.ndarray,
    start_idx: int,
    end_idx: int,
    target_net: str,
    window_lo: float,
    window_hi: float,
    reducer: str,
) -> Optional[float]:
    """
    Return the *settled* sample for `target_net` over the slice
    [start_idx, end_idx], using only samples whose time falls in the
    fractional window [window_lo, window_hi] of [t_start, t_end].

    Why a fractional window?  The PWL pulse rises during `trise`, plateaus
    for `pulse_width`, then falls during `tfall`.  Spectre may also emit
    displacement-current spikes at the edges.  By default we use the middle
    of the plateau (~40-65% of the window), which:
      * avoids edge spikes,
      * captures the genuine quasi-steady current the IMC math expects,
      * works for either current or voltage probes.

    Returns None if no eligible samples exist in this window for this net.
    """
    seg_params = params[start_idx : end_idx + 1]
    seg_vals   = values[start_idx : end_idx + 1]

    # Locate this window's local time vector and probe samples.
    t_mask    = (seg_params == '"time"')
    net_mask  = (seg_params == f'"{target_net}"')
    if not np.any(net_mask):
        return None

    t_samples = seg_vals[t_mask]
    p_samples = seg_vals[net_mask]
    if t_samples.size == 0 or p_samples.size == 0:
        return None

    # In psfascii, time tokens and probe tokens appear interleaved at every
    # time step.  When the simulator emits them in lockstep we can pair by
    # nearest-time.  Search-sorted is more robust against skipped probes
    # than positional zipping.
    if t_samples.size != p_samples.size:
        # Fall back to positional pairing on the smaller count
        m = min(t_samples.size, p_samples.size)
        t_samples = t_samples[:m]
        p_samples = p_samples[:m]

    t_start = float(t_samples[0])
    t_end   = float(t_samples[-1])
    if t_end <= t_start:
        # All samples at same time — return their reduction.
        return _reduce(p_samples, reducer)

    t_lo = t_start + window_lo * (t_end - t_start)
    t_hi = t_start + window_hi * (t_end - t_start)

    inside = (t_samples >= t_lo) & (t_samples <= t_hi)
    if not np.any(inside):
        # Window too narrow for this dump — fall back to all samples.
        return _reduce(p_samples, reducer)

    return _reduce(p_samples[inside], reducer)


def settle_time_in_window(
    params: np.ndarray,
    values: np.ndarray,
    start_idx: int,
    end_idx: int,
    target_net: str,
    settle_band: float = 0.02,
    window_lo: float = 0.40,
    window_hi: float = 0.65,
) -> Optional[float]:
    """Time (s) for ``target_net`` to settle within ``settle_band`` of its
    steady-state value, measured from the start of this MVM window.

    This is the physical, corner/supply/temperature-accurate analog-MVM
    latency: the transient deck already runs at the configured corner, so the
    RC/edge settling of the bit-line current carries the corner directly.
    Defined as the time of the LAST sample outside the tolerance band around
    the plateau value (i.e. when the waveform enters and stays in band).
    """
    seg_params = params[start_idx : end_idx + 1]
    seg_vals   = values[start_idx : end_idx + 1]
    t_mask   = (seg_params == '"time"')
    net_mask = (seg_params == f'"{target_net}"')
    if not np.any(net_mask):
        return None
    t = seg_vals[t_mask].astype(float)
    p = seg_vals[net_mask].astype(float)
    if t.size < 3 or p.size < 3:
        return None
    m = min(t.size, p.size)
    t, p = t[:m], p[:m]
    t0 = t[0]

    t_lo = t0 + window_lo * (t[-1] - t0)
    t_hi = t0 + window_hi * (t[-1] - t0)
    inside = (t >= t_lo) & (t <= t_hi)
    settled = float(np.median(p[inside])) if np.any(inside) else float(np.median(p))
    if not np.isfinite(settled) or abs(settled) < 1e-18:
        return None

    tol = settle_band * abs(settled)
    out = np.abs(p - settled) > tol
    if not np.any(out):
        return 0.0
    t_last_out = float(t[np.nonzero(out)[0][-1]])
    return max(0.0, t_last_out - t0)


def collect_worst_settle_time(
    search_dir: str,
    settings: "TransientReadSettings",
    settle_band: float = 0.02,
) -> Dict[str, float]:
    """Walk ``search_dir`` for transient dumps (``**/psf/tran.tran.tran``) and
    return the worst-case (and mean) analog MVM settle time across every
    IPRB<k>:in net and window.  ``analog_settle_s`` feeds estimate_latency
    (via measured_settle.json).  Empty dict when no waveforms are found —
    callers then fall back to the node-scaled / fixed clock.
    """
    worst = 0.0
    acc: List[float] = []
    for root, _dirs, files in os.walk(search_dir):
        if "tran.tran.tran" not in files:
            continue
        path = os.path.join(root, "tran.tran.tran")
        try:
            params, values = read_psfascii_tran(path)
        except Exception:
            continue
        windows = build_time_windows(params, values,
                                     settings.start_time, settings.increment)
        if not windows:
            continue
        col = 0
        while True:
            net = f"IPRB{col}:in"
            if not np.any(params == f'"{net}"'):
                break
            for st, en in windows:
                ts = settle_time_in_window(
                    params, values, st, en, net, settle_band=settle_band,
                    window_lo=settings.settling_lo, window_hi=settings.settling_hi)
                if ts is not None and np.isfinite(ts):
                    worst = max(worst, ts)
                    acc.append(ts)
            col += 1
    if not acc:
        return {}
    return {"analog_settle_s": worst,
            "analog_settle_mean_s": float(np.mean(acc)),
            "n_settle_samples": len(acc)}


# ---------------------------------------------------------------------------
# Per-folder reducer (one subarray, one branch)
# ---------------------------------------------------------------------------
def process_folder(
    folder_path: str,
    settings: TransientReadSettings,
    ncolumn: int,
) -> Optional[Dict[str, List[Optional[float]]]]:
    """
    Reduce a single subarray's transient dump into per-window settled values.

    Walks `folder_path/psf/tran.tran.tran`, slices it into MVM windows, and
    for each IPRB<k>:in net returns the settled current per window.

    Returns a dict suitable for DataFrame construction, or None if the dump
    is missing or no windows could be built.
    """
    target_path = os.path.join(folder_path, "psf", "tran.tran.tran")
    if not os.path.exists(target_path):
        return None

    params, values = read_psfascii_tran(target_path)
    windows = build_time_windows(params, values,
                                 settings.start_time, settings.increment)
    if not windows:
        return None
    n_windows = len(windows)

    result: Dict[str, List[Optional[float]]] = {
        "Folder": os.path.basename(folder_path)
    }
    for col in range(ncolumn):
        net = f"IPRB{col}:in"
        lst: List[Optional[float]] = []
        for s, e in windows:
            v = settled_value(params, values, s, e, net,
                              settings.settling_lo, settings.settling_hi,
                              settings.reducer)
            lst.append(v if v is not None else np.nan)
        # Defensive pad (should never happen)
        if len(lst) != n_windows:
            lst = (lst + [np.nan] * n_windows)[:n_windows]
        result[net] = lst

    return result


# ---------------------------------------------------------------------------
# Top-level reducer over a whole layer-tag folder
# ---------------------------------------------------------------------------
def process_files_in_folder(
    base_path: str,
    output_csv_path: str,
    ncolumn: int,
    settings: TransientReadSettings = TransientReadSettings(),
) -> None:
    """
    Parallel reducer over every subarray under `base_path`.

    Writes a CSV with columns:
        Folder, IPRB0:in, IPRB1:in, ..., IPRB{ncolumn-1}:in

    Each non-Folder cell is a Python-list-of-floats (one per MVM window),
    rendered as a string by pandas.  This is the format consumed by
    `dp.py::process_crossbar_csv_to_image_csv` (and its v3 replacement).
    """
    folders = sorted({
        os.path.join(r, d)
        for r, dirs, _ in os.walk(base_path)
        for d in dirs
    })

    columns = ["Folder"] + [f"IPRB{i}:in" for i in range(ncolumn)]

    if not folders:
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        return

    data: List[Dict[str, List[Optional[float]]]] = []
    max_workers = settings.max_workers  # None = let executor pick
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(process_folder, fld, settings, ncolumn)
                   for fld in folders]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                data.append(res)

    data.sort(key=lambda d: d.get("Folder", ""))
    df = pd.DataFrame(data)
    for c in columns:
        if c not in df.columns:
            df[c] = [[] for _ in range(len(df))]
    df = df[columns]
    df.to_csv(output_csv_path, index=False)


# ---------------------------------------------------------------------------
# Power / energy reducer (port of the original, with np.trapezoid)
# ---------------------------------------------------------------------------
def process_power_energy_in_folder(
    base_path: str,
    output_csv_path: str,
    power_label: str = ":pwr",
) -> None:
    """
    Walk every immediate subfolder of `base_path`, locate all
    `*/psf/tran.tran.tran` dumps, integrate the `:pwr` waveform to get
    energy and average power, and emit:

        <output_csv_path base>_pwr.csv   Folder, PWR, PWR_STATIC (+ Total_pwr)
        <output_csv_path base>_eng.csv   Folder, ENG, ENG_STATIC, ENG_DYNAMIC
                                         (+ Total_eng)

    ENG_STATIC is the deck's leakage / DC-path floor (power drawn while the
    inputs are idle) integrated over the run; ENG_DYNAMIC = ENG - ENG_STATIC is
    the MVM conduction that actually does the computation.  Reporting them
    separately keeps a leaky standard-cell library from masquerading as
    crossbar compute energy.

    Folders named `psf`, `PE<n>`, `Tile<n>` are skipped (they're container
    levels, not leaf subarrays).  Totals are element-wise sums across rows
    whose Folder starts with `Array_`.
    """
    base, _ = os.path.splitext(output_csv_path)
    pwr_csv_path = f"{base}_pwr.csv"
    eng_csv_path = f"{base}_eng.csv"

    folders_all = sorted({
        os.path.join(r, d)
        for r, dirs, _ in os.walk(base_path)
        for d in dirs
    })

    def _keep(p: str) -> bool:
        bn = os.path.basename(p)
        if bn == "psf":                       return False
        if re.fullmatch(r"PE\d+", bn):        return False
        if re.fullmatch(r"Tile\d+", bn):      return False
        return True

    folders = [f for f in folders_all if _keep(f)]
    if not folders:
        pd.DataFrame(columns=["Folder", "PWR"]).to_csv(pwr_csv_path, index=False)
        pd.DataFrame(columns=["Folder", "ENG"]).to_csv(eng_csv_path, index=False)
        return

    def _avgp_energy(tran_path: str) -> Optional[Tuple[float, float]]:
        params, values = read_psfascii_tran(tran_path)
        return energy_from_arrays(params, values, power_label)

    pwr_rows: List[Dict[str, List]] = []
    eng_rows: List[Dict[str, List]] = []
    for fld in folders:
        tran_files: List[str] = []
        for r, _, files in os.walk(fld):
            if "tran.tran.tran" in files and os.path.basename(r) == "psf":
                tran_files.append(os.path.join(r, "tran.tran.tran"))
        tran_files.sort()

        avg_p_list: List[float] = []
        eng_list:   List[float] = []
        pstat_list: List[float] = []
        estat_list: List[float] = []
        edyn_list:  List[float] = []
        for tf in tran_files:
            r = _avgp_energy(tf)
            if r is None:
                avg_p_list.append(np.nan); eng_list.append(np.nan)
                pstat_list.append(np.nan); estat_list.append(np.nan)
                edyn_list.append(np.nan)
            else:
                avg_p_list.append(r[0]);  eng_list.append(r[1])
                pstat_list.append(r[2]);  estat_list.append(r[3])
                edyn_list.append(r[4])
        bn = os.path.basename(fld)
        pwr_rows.append({"Folder": bn, "PWR": avg_p_list,
                         "PWR_STATIC": pstat_list})
        eng_rows.append({"Folder": bn, "ENG": eng_list,
                         "ENG_STATIC": estat_list, "ENG_DYNAMIC": edyn_list})
    _write_pwr_eng(pwr_rows, eng_rows, pwr_csv_path, eng_csv_path)


def _write_pwr_eng(pwr_rows, eng_rows, pwr_csv_path, eng_csv_path) -> None:
    """Per-folder rows -> <base>_pwr.csv / <base>_eng.csv with the Total rows."""
    pwr_rows.sort(key=lambda d: d["Folder"])
    eng_rows.sort(key=lambda d: d["Folder"])
    df_pwr = pd.DataFrame(pwr_rows, columns=["Folder", "PWR", "PWR_STATIC"])
    df_eng = pd.DataFrame(eng_rows,
                          columns=["Folder", "ENG", "ENG_STATIC", "ENG_DYNAMIC"])

    def _elem_sum(series: pd.Series) -> List[float]:
        max_len = max((len(x) if isinstance(x, list) else 0) for x in series)
        if max_len == 0:
            return []
        mat = np.full((len(series), max_len), np.nan, dtype=float)
        for i, x in enumerate(series):
            if isinstance(x, list) and x:
                k = min(len(x), max_len)
                mat[i, :k] = x[:k]
        return list(np.nansum(mat, axis=0))

    mp = df_pwr["Folder"].astype(str).str.startswith("Array_")
    me = df_eng["Folder"].astype(str).str.startswith("Array_")
    df_pwr = pd.concat(
        [df_pwr, pd.DataFrame([{"Folder": "Total_pwr",
                                "PWR": _elem_sum(df_pwr.loc[mp, "PWR"]),
                                "PWR_STATIC": _elem_sum(df_pwr.loc[mp, "PWR_STATIC"])}])],
        ignore_index=True,
    )
    df_eng = pd.concat(
        [df_eng, pd.DataFrame([{"Folder": "Total_eng",
                                "ENG": _elem_sum(df_eng.loc[me, "ENG"]),
                                "ENG_STATIC": _elem_sum(df_eng.loc[me, "ENG_STATIC"]),
                                "ENG_DYNAMIC": _elem_sum(df_eng.loc[me, "ENG_DYNAMIC"])}])],
        ignore_index=True,
    )
    df_pwr.to_csv(pwr_csv_path, index=False)
    df_eng.to_csv(eng_csv_path, index=False)


def energy_from_arrays(params: np.ndarray, values: np.ndarray,
                       power_label: str = ":pwr"):
    """(avg power, energy, static power floor, static energy, dynamic energy)
    of one transient dump already read by ``read_psfascii_tran``; None when it
    holds no power waveform.  Shared by the per-layer reducer and the
    transient backend's per-job post-processor."""
    if params.size == 0:
        return None
    t_mask = (params == '"time"')
    p_mask = (params == f'"{power_label}"')
    if not (np.any(t_mask) and np.any(p_mask)):
        return None
    t = values[t_mask]; t = t[np.isfinite(t)]
    p = values[p_mask]; p = p[np.isfinite(p)]
    n = min(t.size, p.size)
    if n < 2:
        return None
    t, p = t[:n], p[:n]
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t)
        t = t[order]; p = p[order]
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return None
    # np.trapezoid is the post-1.20 name for np.trapz
    try:
        eng = float(np.trapezoid(p, t))
    except AttributeError:
        eng = float(np.trapz(p, t))
    # --- static (leakage / DC path) vs dynamic (MVM conduction) split ---
    # `pwr=total` integrates EVERY source in the deck, so it also counts the
    # standing current of the BL drivers / decoders / access transistors.
    # In a leaky library (e.g. low-Vt 40nm cells) that floor can be ~100x
    # the MVM conduction, and attributing it to "crossbar" energy makes the
    # number meaningless and non-comparable across nodes.
    #
    # STATIC = the deck's DC leakage floor: the power drawn while the inputs
    #   are idle (all BL PWLs at 0 V, so there is NO MVM current).  It is a
    #   constant DC draw, so its energy is P_floor * duration.  We measure
    #   P_floor as the minimum of a smoothed waveform (robust to numerical
    #   spikes) rather than a percentile, so it is the true baseline the
    #   pulses ride on -- and it is subtracted OVER THE WHOLE DECK, which
    #   removes the leakage that keeps flowing during the pulses too.
    # DYNAMIC = the ReRAM conduction that does the MVM: the energy of the
    #   power ABOVE that baseline, integral of max(P - P_floor, 0) dt.
    #
    # Note: dynamic MVM energy is ReRAM conduction (integral of V*I on the
    # same physical device at every node), NOT a CV^2 logic term -- so it
    # does NOT shrink with the technology node.  A smaller node only lowers
    # the access-transistor R_on, which lets slightly MORE current flow, so
    # dynamic is ~node-independent (marginally higher at smaller nodes).
    # What scales down with the node is peripheral (DAC/ADC) energy, which
    # is modelled separately.  Only the STATIC floor grows at smaller nodes.
    pabs = np.abs(p)
    # P_floor = the DC leakage baseline = power when the inputs are idle
    # (no MVM current).  The most robust estimate that does NOT get pulled
    # down by pulse edges is the MEDIAN of the low-power samples: take the
    # lowest tercile of |P| and use its median.  For a clean crossbar this
    # is ~0; for a leaky library it is the standing leakage the pulses ride
    # on.  (Median, not min, so a single low outlier can't zero the floor;
    # lowest-tercile, so the pulse samples don't inflate it.)
    lo = np.sort(pabs)[: max(1, pabs.size // 3)]
    p_floor = float(np.median(lo))
    eng_static = p_floor * duration          # DC leakage over the whole run
    try:
        eng_dyn = float(np.trapezoid(np.maximum(pabs - p_floor, 0.0), t))
    except AttributeError:
        eng_dyn = float(np.trapz(np.maximum(pabs - p_floor, 0.0), t))
    return eng / duration, eng, p_floor, eng_static, eng_dyn
