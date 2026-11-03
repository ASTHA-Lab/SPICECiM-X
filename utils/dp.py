import pandas as pd
import ast
from collections import defaultdict
from typing import List, Tuple

import ast, re
import numpy as np
# ... your other imports ...

def _safe_eval_list_cell(cell):
    """
    Robustly parse a CSV cell that should contain a Python-list-like string.
    Handles 'nan' tokens and various minor corruptions.
    Returns a Python list of floats (with np.nan where appropriate).
    """
    # Already a list?
    if isinstance(cell, list):
        return [np.nan if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in cell]

    # Numbers (not lists) – coerce to 1-element list
    if isinstance(cell, (int, float)) and not np.isnan(cell):
        return [float(cell)]
    if cell is None:
        return []

    # Strings
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []

    # If it's not bracketed like a list, try to parse as single float
    if not (s.startswith("[") and s.endswith("]")):
        try:
            return [float(s)]
        except Exception:
            # Fallback: grab any floats in the string
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            return [float(x) for x in nums] if nums else []

    # Sanitize tokens that literal_eval can't handle
    # Replace bare nan/NaN/NAN with None
    s_sane = re.sub(r'\bnan\b', 'None', s, flags=re.IGNORECASE)

    try:
        out = ast.literal_eval(s_sane)
        if not isinstance(out, (list, tuple)):
            out = [out]
    except Exception:
        # Fallback: extract floats by regex, preserve positions we can
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        return [float(x) for x in nums] if nums else []

    # Convert Nones back to np.nan; coerce everything to float
    parsed = []
    for v in out:
        if v is None:
            parsed.append(np.nan)
        else:
            try:
                parsed.append(float(v))
            except Exception:
                parsed.append(np.nan)
    return parsed


def current_to_voltage(
    summed_currents: List[List[float]],
    vmax: float
) -> List[List[float]]:
    """
    Convert each summed current (per flattened column) into a voltage in [0, vmax].
    - Negative currents → 0 V
    - Positive currents are linearly mapped so that imin → 0, imax → vmax,
      where imin/imax are the min/max over all non‑negative currents.
    """
    # flatten, pick non‑negative
    all_vals = [c for col in summed_currents for c in col]
    non_neg = [c for c in all_vals if c >= 0]
    if not non_neg:
        raise ValueError("No non‑negative currents found for scaling.")
    imin, imax = min(non_neg), max(non_neg)

    voltages: List[List[float]] = []
    for col in summed_currents:
        col_v = []
        for c in col:
            if c < 0:
                v = 0.0
            else:
                if imax == imin:
                    # all currents identical → max out
                    v = vmax
                else:
                    v = ( (c - imin) / (imax - imin) ) * vmax
            col_v.append(v)
        voltages.append(col_v)
    return voltages

def process_crossbar_csv_to_image_csv(
    input_path: str,
    output_path: str,
    vmax: float = 1.0
) -> List[List[float]]:
    """
    1) Reads the input CSV (Folder + time‑series lists),
    2) computes element‑wise (pos – neg),
    3) sums across all NN‑rows → `summed` currents,
    4) converts to voltages via `current_to_voltage(..., vmax)`,
    5) writes final CSV with rows=Image_0…Image_{L-1}, cols=0…M-1.
    Returns the voltage data: List of M lists of length L.
    """
    df = pd.read_csv(input_path)
    cur_cols = [c for c in df.columns if c != 'Folder']

    # parse into data[row][col]['pos'/'neg']
    data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        _, r_str, c_str, sign = row['Folder'].split('_')
        r, c = int(r_str), int(c_str)
        arr = []
        for col in cur_cols:
            #lst = ast.literal_eval(row[col])
            lst = _safe_eval_list_cell(row[col])
            cleaned = [x for x in lst if x is not None]
            arr.append(cleaned)
        data[r][c][sign] = arr

    # dimensions
    n_rows = max(data.keys()) + 1
    n_cols = max(next(iter(data.values())).keys()) + 1
    M = n_cols * len(cur_cols)   # total flattened columns

    # compute flattened per‑row
    flattened: List[List[List[float]]] = []
    for r in range(n_rows):
        row_flat: List[List[float]] = []
        for c in range(n_cols):
            pos = data[r][c].get('pos')
            neg = data[r][c].get('neg')
            if pos is None or neg is None:
                raise ValueError(f"Missing pos/neg for row {r}, col {c}")
            for p, n in zip(pos, neg):
                if len(p) != len(n):
                    print("Warning: Mismatched lengths in a time‑series")
                    # Truncate both to the same (shorter) length
                    min_len = min(len(p), len(n))
                    p = p[:min_len]
                    n = n[:min_len]
                row_flat.append([pi - ni for pi, ni in zip(p, n)])
        flattened.append(row_flat)

    # sum across rows, element‑wise → summed[j][t]
    L = len(flattened[0][0])
    summed: List[List[float]] = []
    for j in range(M):
        summed.append([
            sum(flattened[r][j][t] for r in range(n_rows))
            for t in range(L)
        ])

    # convert currents → voltages
    voltages = summed

    # build and write final CSV
    rows = []
    for t in range(L):
        row = {'Image_name': f'f{t}'}
        for j in range(M):
            row[str(j)] = voltages[j][t]
        rows.append(row)
    out_df = pd.DataFrame(rows, columns=['Image_name'] + [str(j) for j in range(M)])
    out_df.to_csv(output_path, index=False)

    return voltages

    

