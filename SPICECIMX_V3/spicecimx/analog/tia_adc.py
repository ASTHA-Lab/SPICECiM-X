"""
TIA + ADC modeling and per-tag calibration for SPICECiM-X.

Moved verbatim from utils.py.  No semantic changes — only:
  * imports cleaned up,
  * `TIACalibrator.save` no longer crashes when the directory is "."
    (the old version called `os.makedirs(os.path.dirname(path))` which
    raises if `path` has no directory component).
  * a single-line summary appears in the module docstring instead of
    being scattered as comments.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Simple round-trip (ideal TIA + ideal ADC), kept for parity with old API
# ---------------------------------------------------------------------------
def tia_adc_roundtrip(
    I_diff: np.ndarray,
    Vmin: float, Vmax: float,
    Nbits: int,
    Imin: Optional[float] = None,
    Imax: Optional[float] = None,
    Vfs: Optional[float] = None,
    add_qerror: bool = False,
    rng: Optional[np.random.Generator] = None,
):
    """
    Ideal round-trip: I -> V (affine) -> ADC quantize -> DAC view -> I_hat.
    Returns (V_quant, I_hat, (k, Imin, Imax, Vfs)).
    """
    m, n = I_diff.shape
    if Imin is None: Imin = float(I_diff.min())
    if Imax is None: Imax = float(I_diff.max())
    if Vfs  is None: Vfs  = Vmax

    eps = 1e-12
    Ispan = max(Imax - Imin, eps)
    Vspan = max(Vmax - Vmin, eps)

    k = Vspan / Ispan
    V_tia = Vmin + k * (I_diff - Imin)
    V_adc_in = np.clip(V_tia, 0.0, Vfs)

    L = (1 << Nbits) - 1
    D = np.rint((V_adc_in / Vfs) * L).astype(np.int32)
    D = np.clip(D, 0, L)
    V_quant = (D.astype(np.float64) / L) * Vfs

    if add_qerror:
        if rng is None:
            rng = np.random.default_rng()
        lsb = Vfs / L
        V_quant = V_quant + rng.uniform(-0.5 * lsb, 0.5 * lsb, size=V_quant.shape)

    I_hat = Imin + (V_quant - Vmin) / k
    return V_quant, I_hat, (k, Imin, Imax, Vfs)


# ---------------------------------------------------------------------------
# Realistic TIA + ADC (mismatch, noise, headroom, soft-saturation)
# ---------------------------------------------------------------------------
def tia_adc_realistic(
    I_diff: np.ndarray,
    Vmin: float, Vmax: float,
    Nbits: int,
    Imin: Optional[float] = None,
    Imax: Optional[float] = None,
    Vfs: Optional[float] = None,
    add_qerror: bool = False,
    rng: Optional[np.random.Generator] = None,
    *,
    use_batch_calib: bool = True,
    headroom: float = 1.0,
    clip_mode: str = "fs",
    stochastic_rounding: bool = False,
    sigma_v: float = 0.0,
    k_sigma_rel: float = 0.0,
    o_sigma_abs: float = 0.0,
    apply_mismatch_per: str = "col",
    nonlin: Optional[str] = None,
    nonlin_gain: float = 3.0,
    invert_with: str = "nominal",
    return_debug: bool = False,
):
    """
    Realistic TIA + ADC model.

    Parameters
    ----------
    I_diff : (m, n) differential current matrix (after pos-neg & row-sum).
    Vmin, Vmax : TIA output window; the calibrated current span [Imin, Imax]
        is mapped onto it.
    Nbits : ADC resolution.
    Imin, Imax : current calibration range. Required if use_batch_calib=False.
    Vfs : ADC full-scale voltage (default ``Vmax``).  Codes span [0, Vfs].
    use_batch_calib : if True, derive Imin/Imax from the current batch.
    headroom : multiplier on the full scale (``Vfs_eff = Vfs * headroom``);
        values < 1 shrink the ADC range.  Leave at 1.0 unless you mean it.
    clip_mode : 'fs' clips to [0, Vfs]; 'range' clips to [Vmin, Vmax].

    Non-finite currents (failed SPICE jobs) propagate as NaN in ``V_quant``
    and ``I_hat`` instead of being quantised to a rail code.
    stochastic_rounding : if True, use floor(x*L + U[0,1)) instead of rint.
    sigma_v : Gaussian voltage noise (volts rms) added before ADC.
    k_sigma_rel, o_sigma_abs : per-column TIA gain/offset mismatch stddev.
    apply_mismatch_per : 'col' | 'row' | 'none' — where to apply mismatch.
    nonlin : None or 'tanh' — soft-saturation model.
    nonlin_gain : sharpness of the tanh soft-clip.
    invert_with : 'nominal' or 'mismatch' — what k/offset to assume on inversion.

    Returns
    -------
    V_quant : quantized voltage (post-DAC view).
    I_hat   : current reconstructed from V_quant by inverse TIA.
    meta    : dict of diagnostics (k_nom, Imin, Imax, Vfs, codes_used, ...)
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = I_diff.shape
    bad = ~np.isfinite(I_diff)
    if bad.any():
        I_diff = np.where(bad, 0.0, I_diff)

    # --- Calibration ---
    if use_batch_calib:
        good = I_diff[~bad] if (~bad).any() else np.zeros(1)
        Imin_eff = float(good.min()) if Imin is None else float(Imin)
        Imax_eff = float(good.max()) if Imax is None else float(Imax)
    else:
        if Imin is None or Imax is None:
            raise ValueError("Fixed calibration requested: please pass Imin and Imax.")
        Imin_eff, Imax_eff = float(Imin), float(Imax)

    eps = 1e-12
    Ispan = max(Imax_eff - Imin_eff, eps)
    Vspan = max(Vmax - Vmin, eps)
    k_nom = Vspan / Ispan

    # --- Forward: TIA with mismatch ---
    if apply_mismatch_per == "col":
        k_vec = np.full((1, n), k_nom)
        if k_sigma_rel > 0.0:
            k_vec = k_vec * (1.0 + rng.normal(0.0, k_sigma_rel, size=(1, n)))
        o_vec = np.zeros((1, n))
        if o_sigma_abs > 0.0:
            o_vec = o_vec + rng.normal(0.0, o_sigma_abs, size=(1, n))
        V_lin = Vmin + k_vec * (I_diff - Imin_eff) + o_vec
        k_used, o_used = k_vec, o_vec
    elif apply_mismatch_per == "row":
        k_vec = np.full((m, 1), k_nom)
        if k_sigma_rel > 0.0:
            k_vec = k_vec * (1.0 + rng.normal(0.0, k_sigma_rel, size=(m, 1)))
        o_vec = np.zeros((m, 1))
        if o_sigma_abs > 0.0:
            o_vec = o_vec + rng.normal(0.0, o_sigma_abs, size=(m, 1))
        V_lin = Vmin + k_vec * (I_diff - Imin_eff) + o_vec
        k_used, o_used = k_vec, o_vec
    else:
        V_lin = Vmin + k_nom * (I_diff - Imin_eff)
        k_used, o_used = k_nom, 0.0

    # --- Optional soft nonlinearity ---
    if nonlin is not None and nonlin.lower() == "tanh":
        Vmid = 0.5 * (Vmin + Vmax)
        z = (V_lin - Vmid) / Vspan
        V_tia = Vmid + Vspan * 0.5 * np.tanh(nonlin_gain * z)
        V_tia = np.clip(V_tia, Vmin, Vmax)
    else:
        V_tia = V_lin

    # --- Pre-ADC noise ---
    if sigma_v > 0.0:
        V_tia = V_tia + rng.normal(0.0, sigma_v, size=V_tia.shape)

    # --- Headroom / Vfs ---
    if Vfs is None:
        Vfs_eff = (Vmax - 0.0) * headroom if clip_mode == "fs" else Vspan * headroom
    else:
        Vfs_eff = float(Vfs) * headroom

    # --- Clip ---
    if clip_mode == "fs":
        V_adc_in = np.clip(V_tia, 0.0, Vfs_eff)
        x = V_adc_in / max(Vfs_eff, eps)
    elif clip_mode == "range":
        V_adc_in = np.clip(V_tia, Vmin, Vmin + Vfs_eff)
        x = (V_adc_in - Vmin) / max(Vfs_eff, eps)
    else:
        raise ValueError("clip_mode must be 'fs' or 'range'.")

    # --- Quantize ---
    L = (1 << Nbits) - 1
    x = np.clip(x, 0.0, 1.0)
    if stochastic_rounding:
        D = np.floor(x * L + rng.random(size=x.shape)).astype(np.int32)
    else:
        D = np.rint(x * L).astype(np.int32)
    D = np.clip(D, 0, L)

    # --- DAC view ---
    V_quant_unit = D.astype(np.float64) / L
    V_quant = V_quant_unit * Vfs_eff if clip_mode == "fs" else Vmin + V_quant_unit * Vfs_eff

    if add_qerror:
        lsb = Vfs_eff / L
        V_quant = V_quant + rng.uniform(-0.5 * lsb, 0.5 * lsb, size=V_quant.shape)

    # --- Inverse TIA ---
    if invert_with == "mismatch" and apply_mismatch_per in ("col", "row") and nonlin is None:
        I_hat = Imin_eff + (V_quant - Vmin - o_used) / np.maximum(k_used, eps)
    else:
        I_hat = Imin_eff + (V_quant - Vmin) / max(k_nom, eps)

    if bad.any():
        V_quant = np.where(bad, np.nan, V_quant)
        I_hat = np.where(bad, np.nan, I_hat)

    # --- Diagnostics ---
    used_codes = int(np.unique(D).size)
    rail_frac  = float(np.mean((D == 0) | (D == L)))
    max_abs_I_err = float(np.max(np.abs(
        I_hat - (Imin_eff + (V_tia - Vmin) / max(k_nom, eps))
    )))

    meta = {
        "k_nom": float(k_nom),
        "Imin": float(Imin_eff),
        "Imax": float(Imax_eff),
        "Vfs":  float(Vfs_eff),
        "codes_used": used_codes,
        "rail_frac":  rail_frac,
        "max_abs_I_err_vs_nominal": max_abs_I_err,
        "Nbits": int(Nbits),
        "clip_mode": clip_mode,
        "headroom": float(headroom),
        "stochastic_rounding": bool(stochastic_rounding),
        "sigma_v": float(sigma_v),
        "k_sigma_rel": float(k_sigma_rel),
        "o_sigma_abs": float(o_sigma_abs),
        "apply_mismatch_per": apply_mismatch_per,
        "nonlin": nonlin,
        "nonlin_gain": float(nonlin_gain),
        "invert_with": invert_with,
    }
    if return_debug:
        meta["codes"] = D
    return V_quant, I_hat, meta


# ---------------------------------------------------------------------------
# Per-tag calibrator (Imin/Imax tracking across runs)
# ---------------------------------------------------------------------------
class TIACalibrator:
    """
    Tracks the Imin/Imax calibration range per layer key, using configurable
    percentiles to clip outliers (default 0.1 / 99.9).

    ``freeze_after`` models a calibrate-then-run ADC: the range of each layer
    is the union over its first ``freeze_after`` calls (one call per image in
    per-image mode, one per batch with ``--batch_all``) and is then fixed for
    every later call.  ``freeze_after = 0`` keeps widening the range forever
    (the pre-v1.1 behaviour, which made results depend on image order).
    """

    def __init__(self, p_lo: float = 0.1, p_hi: float = 99.9,
                 freeze_after: int = 0):
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.freeze_after = int(freeze_after)
        self.table: dict = {}

    def is_frozen(self, tag: str) -> bool:
        e = self.table.get(tag)
        return bool(e and self.freeze_after > 0 and e.get("n", 0) >= self.freeze_after)

    def update(self, tag: str, I_diff: np.ndarray) -> None:
        if self.is_frozen(tag):
            return
        # Ignore non-finite entries so a partially-failed batch (one
        # subarray-branch didn't converge) doesn't poison the calibration.
        arr = I_diff[np.isfinite(I_diff)]
        if arr.size == 0:
            return
        lo = float(np.percentile(arr, self.p_lo))
        hi = float(np.percentile(arr, self.p_hi))
        if tag in self.table:
            e = self.table[tag]
            e["Imin"] = min(e["Imin"], lo)
            e["Imax"] = max(e["Imax"], hi)
            e["n"] = e.get("n", 0) + 1
        else:
            self.table[tag] = {"Imin": lo, "Imax": hi, "n": 1}

    def get(self, tag: str) -> Tuple[float, float]:
        if tag not in self.table:
            raise KeyError(
                f"No usable calibration for tag '{tag}' (no finite currents yet). "
                f"Run calibrator.update(tag, I_diff) first or load from file."
            )
        return self.table[tag]["Imin"], self.table[tag]["Imax"]

    def save(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:  # only create if path has a directory component
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.table, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            self.table = json.load(f)