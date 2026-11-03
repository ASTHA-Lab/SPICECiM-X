"""
Hardware-accurate IMC matrix multiply for SPICECiM-X v3.

The backend programs TWO physical crossbars (W+ and W-) carrying the
positive and negative parts of B, sharing the same baseline conductance,
and returns the *differential* current ``I_diff = I+ − I-``.  In that
subtraction, three of the four affine expansion terms cancel as
common-mode noise; only ``term2 = bc·J@B`` survives:

    G+_ij = c·max(B_ij, 0) + g0     (g0 = HRS baseline shared with G-)
    G-_ij = c·max(-B_ij, 0) + g0
    G+ − G− = c·B                                       (algebraic identity)
    I_diff = V @ (G+ − G−)  with V = aA + bJ
           = ac·A@B + bc·J@B

Recovery is therefore:

    A @ B  =  (I_diff − bc·J@B) / (ac)

The four-term expansion ``(aA + bJ) @ (cB + dJ)`` applies to *single-ended*
readout where no companion subtraction is performed — see
``compute_offset_terms(..., differential=False)``.  The in-repo SPICE
backends are all differential.  Using single-ended de-mapping on
differential current adds a constant offset of order ``bd·k / (ac)`` to
every element of C; for typical ReRAM windows this constant is ~ 60–800
times the natural range of A@B and collapses every input to the same
output regardless of the input (this was the v3 accuracy bug — predictions
locked to a single class).

Polarity disambiguation
=======================

After ``I_diff`` is loaded back from the simulator, we compare both
``I_diff`` and ``-I_diff`` against the differential ideal current
``I_ideal = ac·A@B + bc·J@B``.  The sign with smaller mean-absolute-error
wins.  Orientation is cached per tag so subsequent calls with the same
tag don't re-decide.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Affine bookkeeping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AffineScales:
    """Bundle of the four affine scalars used by imc_matmul.

    V = a·A + b
    G = c·B + d
    """
    a: float
    b: float
    c: float
    d: float


def derive_affine(
    A: np.ndarray,
    B: np.ndarray,
    vmin: float,
    vmax: float,
    Gmin: float,
    Gmax: float,
    *,
    eps: float = 1e-12,
) -> AffineScales:
    """
    Solve for (a, b, c, d) such that:
        a · A.min() + b = vmin
        a · A.max() + b = vmax
        c · B.min() + d = Gmin
        c · B.max() + d = Gmax

    When A is degenerate (all-equal entries), ``a`` defaults to 1.

    The differential mapping programs ``G± = c·max(±B, 0) + Gmin``, so the
    largest conductance is ``c·max|B| + Gmin``.  For a mixed-sign B,
    ``max|B| <= Bmax - Bmin`` and the window is respected as-is; for a
    one-signed B the denominator is widened to ``max|B|`` so G never exceeds
    Gmax.  A zero B gives ``c = (Gmax - Gmin)`` (any finite value works: the
    product is zero) instead of the old 1.0 S, which overflowed the window.
    """
    Amax, Amin = float(A.max()), float(A.min())
    Bmax, Bmin = float(B.max()), float(B.min())
    denA = Amax - Amin
    denB = max(Bmax - Bmin, abs(Bmax), abs(Bmin))

    a = (vmax - vmin) / denA if abs(denA) > eps else 1.0
    b = vmin - a * Amin
    c = (Gmax - Gmin) / denB if denB > eps else (Gmax - Gmin)
    d = Gmin - c * Bmin
    return AffineScales(a=a, b=b, c=c, d=d)


def compute_offset_terms(
    A: np.ndarray,
    B: np.ndarray,
    sc: AffineScales,
    *,
    differential: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Build the per-element offset matrices that arise from expanding
    ``(aA + bJ) @ (cB + dJ)`` and decide which subset has to be
    subtracted from the measured current before dividing by ``ac``.

    Differential readout (default).  The SPICE backend programs two
    crossbars (W+, W-) sharing the SAME baseline conductance and returns
    ``I_diff = I+ − I-``.  The common-mode contributions of the baseline
    (term1 = ad·A@J and term3 = bd·k·J) cancel in the subtraction; only
    ``term2 = bc·J@B`` survives.  Algebraically::

        G+_ij = c·max(B_ij, 0) + g0,    G-_ij = c·max(-B_ij, 0) + g0
        G+ − G−   =  c·B                                          (identity)
        I_diff    =  V @ (G+ − G-)  =  c·(aX + bJ)@B
                  =  ac·X@B + bc·J@B

    Single-ended readout (``differential=False``).  Only used when the
    backend measures one polarity at a time with no companion
    subtraction; sum then includes all four terms.

    Returns
    -------
    dict with keys ``'term1'``, ``'term2'``, ``'term3'`` (always populated
    for diagnostics) and ``'sum'`` — the matrix to subtract from the
    measured current.  All matrices have shape ``(m, n)``.
    """
    m, k = A.shape
    n = B.shape[1]
    one_m = np.ones((m, 1))
    one_k = np.ones((k, 1))
    one_n = np.ones((n, 1))

    term1 = sc.a * sc.d * (A @ one_k) @ one_n.T          # ad · (A · 1_k) · 1_n^T
    term2 = sc.b * sc.c * one_m @ (one_k.T @ B)          # bc · 1_m · (1_k^T · B)
    term3 = sc.b * sc.d * k * (one_m @ one_n.T)          # bd · k · J_{m×n}

    if differential:
        offset_sum = term2
    else:
        offset_sum = term1 + term2 + term3

    return {"term1": term1, "term2": term2, "term3": term3,
            "sum":   offset_sum}


def ideal_current(
    A: np.ndarray, B: np.ndarray, sc: AffineScales,
    *,
    differential: bool = True,
) -> np.ndarray:
    """
    Predicted SPICE current ``I = V @ G``, expanded.

    Differential (default):  ``I_diff = ac·A@B + bc·J@B``.
    Single-ended:            ``I_se   = (aA + bJ) @ (cB + dJ)`` (all 4 terms).
    """
    return sc.a * sc.c * (A @ B) + compute_offset_terms(
        A, B, sc, differential=differential)["sum"]


# ---------------------------------------------------------------------------
# Polarity (sign) cache — shared across imc_matmul calls
# ---------------------------------------------------------------------------
class PolarityCache:
    """Per-tag {+1, -1} orientation, decided once and reused.

    The cache is per-instance so different ViT forwards don't pollute
    each other (the v2 implementation used a module-level dict).
    """

    def __init__(self):
        self._table: Dict[str, int] = {}

    def decide(
        self,
        tag: str,
        I_spice: np.ndarray,
        I_ideal_full: np.ndarray,
    ) -> int:
        """
        Return the orientation for ``tag``: +1 keeps I_spice as-is,
        -1 flips its sign.  Compares against the differential ideal.

        Non-finite entries (failed subarrays) are ignored.  If nothing finite
        is left, +1 is returned and *not* cached, so a later healthy call
        decides the orientation instead of inheriting a NaN-driven flip.
        """
        if tag in self._table:
            return self._table[tag]
        ok = np.isfinite(I_spice) & np.isfinite(I_ideal_full)
        if not np.any(ok):
            return +1
        err_pos = float(np.mean(np.abs(I_spice[ok] - I_ideal_full[ok])))
        err_neg = float(np.mean(np.abs(-I_spice[ok] - I_ideal_full[ok])))
        orient = +1 if err_pos <= err_neg else -1
        self._table[tag] = orient
        return orient

    def get(self, tag: str) -> Optional[int]:
        return self._table.get(tag)

    def clear(self) -> None:
        self._table.clear()


# ---------------------------------------------------------------------------
# I/O helpers (small wrappers; kept here so the module is self-contained)
# ---------------------------------------------------------------------------
def load_2d_csv_as_array(path: str) -> np.ndarray:
    """
    Load a CSV whose first row is a header and first column is an index,
    returning only the inner numeric 2-D block as a float64 NumPy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError(f"CSV too small to contain header/index: {df.shape}")
    return df.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce") \
        .to_numpy(dtype=np.float64, copy=False)


def save_np_2d(path: str, X: np.ndarray) -> None:
    """Save a 1-D or 2-D NumPy array as a CSV with row index + f0..fN-1 headers."""
    import csv
    if X.ndim == 1:
        X = X[None, :]
    X2 = X.reshape(X.shape[0], -1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row"] + [f"f{j}" for j in range(X2.shape[1])])
        for i in range(X2.shape[0]):
            w.writerow([i] + X2[i].tolist())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def imc_matmul(
    A: np.ndarray,
    B: np.ndarray,
    *,
    Gmax: float,
    Gmin: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    sim_backend: Callable[..., np.ndarray],
    polarity_cache: PolarityCache,
    tia_apply: Callable[[np.ndarray, str], Tuple[np.ndarray, np.ndarray, dict]],
    tag: str = "",
    bname: str = "",
    dump_dir: Optional[str] = None,
    qbit: Optional[int] = None,
) -> np.ndarray:
    """
    Hardware-accurate IMC matrix multiply with full affine de-mapping.

    Parameters
    ----------
    A, B
        Software matrices of shapes ``(m, k)`` and ``(k, n)``.  The result
        approximates ``A @ B`` to within hardware non-idealities + ADC error.
    Gmax, Gmin
        ReRAM conductance window in Siemens.
    vmin, vmax
        Input voltage range (Volts) for the activation encoding.
    sim_backend
        Callable that takes (A, B, scales, tag, bname, dump_dir) and returns
        the SPICE-measured differential current matrix ``I_SPICE`` of shape
        ``(m, n)``.  The backend encapsulates all weight distribution,
        Verilog-A emission, Spectre invocation, and per-window reduction.
        This indirection is what lets us swap transient/DC/hybrid backends.
    polarity_cache
        Shared PolarityCache instance.  One per ViT forward is typical.
    tia_apply
        Callable that takes ``(I_SPICE, tag)`` and returns
        ``(V_quant, I_hat, meta)``.  Encapsulates the realistic TIA+ADC.
    tag, bname, dump_dir
        Passed through to sim_backend and used as CSV stems.  ``bname/tag``
        is the layer key for the polarity cache and the TIA calibration, so
        the same tag in different blocks (``h0_Q`` in block 0 and block 1)
        gets its own orientation and its own ADC range.
    qbit
        If set, B is symmetric-int-quantized to ``qbit`` bits before mapping.

    Returns
    -------
    C : np.ndarray of shape (m, n)
        The hardware-emulated approximation of ``A @ B``.

    Notes
    -----
    Only ``term2 = bc·J@B`` is subtracted: with differential readout the
    other offset terms cancel in ``I+ − I−`` (see the module docstring).
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0], \
        f"shape mismatch: A={A.shape}, B={B.shape}"

    B = B.astype(np.float64, copy=False)
    A = A.astype(np.float64, copy=False)

    # --- 1) Optional PTQ on B ---
    if qbit is not None and qbit > 0:
        levels_pos = 2 ** (qbit - 1) - 1
        max_abs = float(np.max(np.abs(B)))
        if max_abs > 1e-12:
            scale = levels_pos / max_abs
            q_int = np.clip(np.round(B * scale), -levels_pos, levels_pos)
            B = (q_int / scale).astype(np.float64)
        else:
            B = np.zeros_like(B, dtype=np.float64)

    # --- 2) Derive affine and offset terms (DIFFERENTIAL: term2 only) ---
    sc      = derive_affine(A, B, vmin, vmax, Gmin, Gmax)
    offsets = compute_offset_terms(A, B, sc)

    # --- 3) Dump bookkeeping files (numeric domain) ---
    if dump_dir is not None and tag:
        I_ideal = ideal_current(A, B, sc)
        save_np_2d(os.path.join(dump_dir, f"{tag}_A.csv"), A)
        save_np_2d(os.path.join(dump_dir, f"{tag}_B.csv"), B)
        save_np_2d(os.path.join(dump_dir, f"{tag}_I_ideal.csv"), I_ideal)
        save_np_2d(os.path.join(dump_dir, f"{tag}_offset_sum.csv"), offsets["sum"])

    # --- 4) Run SPICE through the backend ---
    I_SPICE = sim_backend(
        A=A, B=B, scales=sc,
        tag=tag, bname=bname, dump_dir=dump_dir,
    )

    # Defensive shape alignment (some flows produce extra trailing columns)
    m, _ = A.shape
    _, n = B.shape
    if I_SPICE.shape[0] > m: I_SPICE = I_SPICE[:m, :]
    if I_SPICE.shape[1] > n: I_SPICE = I_SPICE[:, :n]
    if I_SPICE.shape != (m, n):
        raise ValueError(
            f"[{tag}] SPICE shape {I_SPICE.shape} != expected {(m, n)}"
        )

    # --- 5) Polarity decision against the DIFFERENTIAL ideal ---
    # I_ideal_diff = ac·A@B + bc·J@B  (only term2; the other three cancel
    # in the differential subtraction the backend performs internally).
    layer_key = f"{bname}/{tag}" if bname else tag
    I_ideal_diff = sc.a * sc.c * (A @ B) + offsets["sum"]
    orient = polarity_cache.decide(layer_key, I_SPICE, I_ideal_diff)
    if orient < 0:
        I_SPICE = -I_SPICE

    # --- 6) TIA + ADC pass (caller-supplied; manages per-layer calibration) ---
    V_quant, I_hat, tia_meta = tia_apply(I_SPICE, layer_key)

    # --- 7) Full affine de-mapping ---
    C = (I_hat - offsets["sum"]) / (sc.a * sc.c)

    # --- 8) Diagnostics ---
    if dump_dir is not None and tag:
        save_np_2d(os.path.join(dump_dir, f"{tag}_I_SPICE.csv"), I_SPICE)
        save_np_2d(os.path.join(dump_dir, f"{tag}_I_hat.csv"),   I_hat)
        save_np_2d(os.path.join(dump_dir, f"{tag}_C.csv"),       C)

    # Print one-line summary so the user can spot drift mid-run
    Y_ref = A @ B
    if not np.all(np.isfinite(C)):
        print(f"[{bname}/{tag}] WARNING: {int(np.sum(~np.isfinite(C)))} non-finite "
              f"output(s) (failed SPICE jobs, sim.on_job_failure=nan)")
    rmse  = float(np.sqrt(np.nanmean((C - Y_ref) ** 2)))
    rng   = float(np.max(Y_ref) - np.min(Y_ref) + 1e-12)
    print(f"[{bname}/{tag}] orient={orient:+d}  "
          f"|C-A@B| rmse={rmse:.3e} ({100*rmse/rng:.2f}% of range)  "
          f"k_nom={tia_meta.get('k_nom', float('nan')):.3e}")

    return C
