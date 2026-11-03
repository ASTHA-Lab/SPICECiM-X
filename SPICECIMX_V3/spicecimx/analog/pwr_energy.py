#!/usr/bin/env python3
"""
================================================================================
  pwr_energy.py — Power / Energy / Latency / Efficiency report for SPICECiM-X v3
================================================================================

This is the v3 consolidation of the standalone ``pwr_eng.py`` from the
reservoir-computing branch, adapted to the ViT-IMC framework.  It produces ONE
report combining, for a whole inference:

    1. CROSSBAR energy/power   — integrated from the per-layer ``*_pwr.csv`` /
       ``*_eng.csv`` files that the TRANSIENT SPICE flow writes (via
       ``parse_transient.process_power_energy_in_folder``; the testbench saves
       ``pwr=total``).  This is the analog VMM core only and is available in
       transient mode only (DC / FastDC have no time domain, hence no energy).

    2. PERIPHERAL energy/power — DAC, ADC and TIA, estimated from physically
       grounded models using the operation counts an inference records in
       ``<dump>/peripheral_ops.json`` (written by OpLedger during the forward
       pass).  The SPICE flow does not model these converters; this fills the
       gap so a future *physical* DAC/ADC/TIA netlist (see config switches)
       can later replace the estimate without changing this file.

    3. LATENCY (cycle-accurate)— from a configured clock, number of DAC/ADC
       units and ADC architecture, the number of clock cycles for data
       conversion (DAC + ADC) plus the MAC/VMM settling cycles in the IMC,
       giving total inference latency and latency/MAC.

    4. SYSTEM totals: energy, average power, TOPS/W, GOPS, fJ/MAC.

  ── Energy models (citations) ───────────────────────────────────────────────
  ADC : two selectable models, both rooted in B. Murmann's ADC Performance
        Survey (github.com/bmurmann/ADC-survey):
          * 'walden'    — Walden figure-of-merit.  E_conv = FOMW * 2^ENOB.
                          Survey medians: ~30 fJ/conv-step overall, ~22 fJ for
                          SAR (the dominant IMC-readout architecture); SOTA
                          SAR ~1.3 fJ.  Set adc_fom_fJ_per_convstep per node.
          * 'accelergy' — the "best-possible ADC" envelope from the Accelergy
                          ADC plug-in (github.com/ASTHA-Lab/accelergy-adc-plug-in,
                          T. Andrulis & R. Chen), built by regression on the
                          same survey.  Energy from the Schreier FOM:
                          E_conv = 1 / (2 * 10^((FOMS - SNDR)/10)), with FOMS
                          bounded by ENOB and frequency and corrected for tech
                          node.  This model is tech-node aware (it strongly
                          penalises older nodes), so prefer it when you care
                          about the actual node rather than a survey-wide median.
  DAC : either a direct per-conversion energy, or a switched-capacitor model
          E = activity * C * Vref^2 * 2^bits.
  TIA : static-bias model, E = I_bias * V_sup * t_active.

  All model parameters live in the [PERIPHERALS] and [LATENCY] sections of
  config.ini, so numbers can be tuned per technology node without editing code.

  Usage (standalone):
    python -m spicecimx.analog.pwr_energy --dump ./imc_traces/<batch> --config ./config.ini
  Usage (in-framework): infer.py calls generate_report(...) automatically when
    [PERIPHERALS] generate_energy_report = true.
================================================================================
"""
from __future__ import annotations

import os
import re
import shutil
import json
import math
import argparse
import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 0 — Operation ledger (written during the forward pass)           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
class OpLedger:
    """
    Accumulates per-matmul operation counts across a whole inference so the
    energy and latency models have an exact workload manifest.

    For each ``imc_matmul(A (m,k), B (k,n))`` the per-readout counts are
    (same convention as the reservoir-computing branch's manifest):

        DAC : m * k         one conversion per input activation per vector
        TIA : m * n         one transimpedance read per output column per vector
        ADC : m * n         one digitisation per output column per vector
        MAC : m * k * n     k multiply-accumulates per output, n outputs

    All counts are summed over every matmul in the network and also retained
    per-layer for the report breakdown.  ``n_bit_slices`` multiplies the
    DAC/ADC/TIA work when the hardware digitises each weight bit-slice
    separately (config ``adc_per_slice``); the differential pos/neg pair is a
    crossbar-internal detail and is NOT double-counted here.
    """

    def __init__(self, n_bit_slices: int = 1):
        self.n_bit_slices = int(max(1, n_bit_slices))
        self.layers: List[Dict] = []

    def record(self, tag: str, m: int, k: int, n: int) -> None:
        self.layers.append({
            "tag": str(tag), "m": int(m), "k": int(k), "n": int(n),
            "dac": int(m * k), "tia": int(m * n),
            "adc": int(m * n), "mac": int(m * k * n),
        })

    # -- aggregate accessors ------------------------------------------------
    def _sum(self, key: str) -> int:
        return int(sum(l[key] for l in self.layers))

    @property
    def total_mac(self) -> int: return self._sum("mac")
    @property
    def total_dac(self) -> int: return self._sum("dac")
    @property
    def total_adc(self) -> int: return self._sum("adc")
    @property
    def total_tia(self) -> int: return self._sum("tia")
    @property
    def m_total(self) -> int:   return int(sum(l["m"] for l in self.layers))

    def per_inference(self, n_inferences: int, batched: bool) -> "OpLedger":
        """Return a ledger holding ONE inference's work.

        Per-image runs record the network ``n_inferences`` times (keep the
        first pass); ``--batch_all`` runs record every layer once with the
        images stacked into ``m`` (divide ``m`` by the image count).
        """
        n = max(1, int(n_inferences))
        out = OpLedger(self.n_bit_slices)
        if n == 1:
            out.layers = list(self.layers)
        elif batched:
            for l in self.layers:
                out.record(l["tag"], l["m"] // n, l["k"], l["n"])
        else:
            per = len(self.layers) // n
            out.layers = list(self.layers[:per])
        return out

    def write_manifest(self, dump_dir: str, dac_bits: int, adc_bits: int,
                       *, crossbar_inferences: int = 1) -> str:
        """Serialise to ``<dump_dir>/peripheral_ops.json`` (schema is a superset
        of the reservoir branch's so the standalone reader stays compatible).

        Write a per-inference ledger (see :meth:`per_inference`);
        ``crossbar_inferences`` is how many inferences the transient crossbar
        waveforms cover (the batch size with --batch_all, else 1) so the
        report can normalise the SPICE crossbar energy the same way."""
        os.makedirs(dump_dir, exist_ok=True)
        n_slices = self.n_bit_slices
        manifest = {
            "tag": "vit_imc_inference",
            "shapes": {
                "m_input_vectors": self.m_total,      # total input vectors across layers
                "k_input_dim": (self.layers[0]["k"] if self.layers else 0),
                "n_outputs": (self.layers[-1]["n"] if self.layers else 0),
                "n_bit_slices": n_slices,
                "n_matmuls": len(self.layers),
            },
            "counts": {
                "dac_conversions": self.total_dac,
                "tia_conversions": self.total_tia,
                "adc_conversions": self.total_adc,
                "adc_conversions_per_slice": int(self.total_adc * n_slices),
                "mac_operations": self.total_mac,
            },
            "resolution_bits": {"dac": int(dac_bits), "adc": int(adc_bits)},
            "per_inference": True,
            "crossbar_inferences": int(max(1, crossbar_inferences)),
            "per_layer": self.layers,
            "notes": "Counts assume analog shift-add then a single ADC/TIA per "
                     "output column. adc_conversions_per_slice applies when the "
                     "hardware digitises each weight bit-slice separately.",
        }
        out = os.path.join(dump_dir, "peripheral_ops.json")
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        return out


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 1 — CROSSBAR power/energy CSV reduction                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _unwrap_wrappers(s: str) -> str:
    if s is None:
        return ""
    text = str(s).strip()
    while text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    for _ in range(3):
        text = re.sub(r"(?:np\.)?(?:float|int)\d+\(\s*([^\)]*?)\s*\)", r"\1", text)
        text = re.sub(r"(?:np\.)?array\(\s*([^\)]*?)\s*\)", r"\1", text)
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
    return text


def _extract_numeric(text: str) -> Optional[float]:
    cleaned = _unwrap_wrappers(text)
    m = _NUM_RE.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _read_csv_as_str_df(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, header=None, dtype=str, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            return None


def _extract_total_from_csv(path: str, key: str) -> Optional[float]:
    """Find the numeric associated with `key` ('Total_pwr' / 'Total_eng')."""
    df = _read_csv_as_str_df(path)
    if df is not None:
        where = (df == key)
        rows, cols = where.values.nonzero()
        if len(rows):
            r, c = rows[0], cols[0]
            for cc in range(c + 1, df.shape[1]):
                val = _extract_numeric(df.iat[r, cc])
                if val is not None:
                    return val
            if r + 1 < df.shape[0]:
                val = _extract_numeric(df.iat[r + 1, c])
                if val is not None:
                    return val
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell = df.iat[r, c]
                if cell is None:
                    continue
                s = str(cell)
                if key in s:
                    if c + 1 < df.shape[1]:
                        val = _extract_numeric(df.iat[r, c + 1])
                        if val is not None:
                            return val
                    after = s.split(key, 1)[1]
                    val = _extract_numeric(after)
                    if val is not None:
                        return val
    try:
        txt = Path(path).read_text(errors="ignore")
        idx = txt.find(key)
        if idx != -1:
            after = txt[idx + len(key):]
            val = _extract_numeric(after)
            if val is not None:
                return val
    except Exception:
        pass
    return None


_SUFFIX_RE = re.compile(r"(?i)_(eng|pwr)\s*(?:\(\d+\))?\.csv$")


def _parse_layer_and_kind(filename: str) -> Optional[Tuple[str, str]]:
    m = _SUFFIX_RE.search(filename)
    if not m:
        return None
    return filename[:m.start()], m.group(1).lower()


def _extract_total_column(path: str, row_key: str, col_name: str) -> Optional[float]:
    """Value of `col_name` on the `row_key` row (e.g. ENG_STATIC on Total_eng).
    Returns None when the column is absent (CSVs written by an older build)."""
    df = _read_csv_as_str_df(path)
    if df is None or df.empty:
        return None
    hdr = [str(x).strip() for x in df.iloc[0].tolist()]
    if col_name not in hdr:
        return None
    c = hdr.index(col_name)
    where = (df == row_key)
    rows, _cols = where.values.nonzero()
    if not len(rows):
        return None
    return _extract_numeric(df.iat[rows[0], c])


def collect_crossbar_power_energy(folder_path: str) -> Dict[str, Dict[str, float]]:
    """Scan for *_pwr*.csv / *_eng*.csv and sum Total_pwr / Total_eng per layer.
    Returns {layer: {"power_W": float, "energy_J": float}}."""
    records: Dict[str, Dict[str, Dict[str, float]]] = {}
    for root, _, files in os.walk(folder_path):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            parsed = _parse_layer_and_kind(fn)
            if not parsed:
                continue
            layer, kind = parsed
            full = str(Path(root) / fn)
            if kind == "pwr":
                val = _extract_total_from_csv(full, "Total_pwr")
                if val is not None:
                    rec = records.setdefault(layer, {"P": {"sum": 0.0, "n": 0},
                                                     "E": {"sum": 0.0, "n": 0}})
                    rec["P"]["sum"] += val
                    rec["P"]["n"] += 1
            elif kind == "eng":
                val = _extract_total_from_csv(full, "Total_eng")
                if val is not None:
                    rec = records.setdefault(layer, {"P": {"sum": 0.0, "n": 0},
                                                     "E": {"sum": 0.0, "n": 0}})
                    rec["E"]["sum"] += val
                    rec["E"]["n"] += 1
                    # leakage / DC-floor split (absent in CSVs from old builds)
                    st = _extract_total_column(full, "Total_eng", "ENG_STATIC")
                    dy = _extract_total_column(full, "Total_eng", "ENG_DYNAMIC")
                    if st is not None:
                        rec.setdefault("Es", {"sum": 0.0, "n": 0})
                        rec["Es"]["sum"] += st; rec["Es"]["n"] += 1
                    if dy is not None:
                        rec.setdefault("Ed", {"sum": 0.0, "n": 0})
                        rec["Ed"]["sum"] += dy; rec["Ed"]["n"] += 1
    out: Dict[str, Dict[str, float]] = {}
    for layer, rec in records.items():
        out[layer] = {
            "power_W":  rec["P"]["sum"] if rec["P"]["n"] else float("nan"),
            "energy_J": rec["E"]["sum"] if rec["E"]["n"] else float("nan"),
            "energy_static_J":  rec.get("Es", {}).get("sum", float("nan")),
            "energy_dynamic_J": rec.get("Ed", {}).get("sum", float("nan")),
        }
    return out


_BLOCK_PREFIX_RE = re.compile(r"^(block_\d+)_(.+)$")


def reduce_crossbar_csvs(hwsimdb_dir: str, dump_dir: str = "") -> None:
    """Walk each layer subdir of an hwsimdb batch and emit its *_pwr/_eng CSVs
    by integrating the transient ``:pwr`` waveforms.  No-op if the transient
    reducer isn't importable (e.g. DC-only install).

    When ``dump_dir`` is given, each layer's *_pwr.csv / *_eng.csv is also
    copied into the dump alongside the other dumped intermediates, organised
    per transformer block:

        <dump_dir>/crossbar_power/block_00/h0_Q_pwr.csv
        <dump_dir>/crossbar_power/block_00/h0_Q_eng.csv
        <dump_dir>/crossbar_power/HEAD_pwr.csv          (non-block layers)
    """
    try:
        from spicecimx.io.parse_transient import process_power_energy_in_folder
    except Exception as e:  # pragma: no cover - import guard
        print(f"[pwr] crossbar reducer unavailable ({e}); "
              f"skipping crossbar energy.")
        return
    if not os.path.isdir(hwsimdb_dir):
        return
    for tag in sorted(os.listdir(hwsimdb_dir)):
        layer_dir = os.path.join(hwsimdb_dir, tag)
        if not os.path.isdir(layer_dir):
            continue
        out_stub = os.path.join(layer_dir, f"{tag}.csv")
        eng_csv = os.path.join(layer_dir, f"{tag}_eng.csv")
        has_raw = any("tran.tran.tran" in files
                      for _r, _d, files in os.walk(layer_dir))
        if has_raw:
            try:
                process_power_energy_in_folder(layer_dir, out_stub)
            except Exception as e:
                print(f"[pwr] reduce failed for {tag}: {type(e).__name__}: {e}")
                continue
        elif os.path.exists(eng_csv):
            # Layer was already reduced and its raw data pruned
            # ([SIM] hwsimdb_retention) — reuse the kept CSVs.
            pass
        else:
            print(f"[pwr] {tag}: no waveforms and no CSVs; skipping.")
            continue
        if dump_dir:
            m = _BLOCK_PREFIX_RE.match(tag)
            sub = os.path.join(dump_dir, "crossbar_power",
                               m.group(1)) if m else                   os.path.join(dump_dir, "crossbar_power")
            name = m.group(2) if m else tag
            try:
                os.makedirs(sub, exist_ok=True)
                base, _ = os.path.splitext(out_stub)
                for kind in ("pwr", "eng"):
                    src = f"{base}_{kind}.csv"
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(sub, f"{name}_{kind}.csv"))
            except Exception as e:
                print(f"[pwr] dump copy failed for {tag}: "
                      f"{type(e).__name__}: {e}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 2 — Peripheral (DAC / ADC / TIA) energy models                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# --- Accelergy "best-possible ADC" fitted model (model.yaml, MIT-licensed) ---
# Coefficients are log-base-e in frequency (Hz) and tech node (nm).  Energy
# is returned in Joules/conversion.  Source: Accelergy ADC plug-in build over
# Murmann's survey; reproduced here so the framework runs offline.
_ACC_FOMS_INTERCEPT = 213.7806687348118
_ACC_FOMS_FREQ_SLOPE = -2.8719030399529957
_ACC_FOMS_MAX_BY_ENOB = [
    128.3068287398817, 131.52734025085925, 134.7478517618368, 137.96836327281437,
    141.18887478379193, 144.4093862947695, 147.62989780574708, 150.85040931672464,
    154.0709208277022, 157.29143233867975, 160.5119438496573, 163.73245536063487,
    166.95296687161243, 170.17347838259, 173.39398989356755, 176.61450140454514,
    179.83501291552267, 183.05552442650026, 186.27603593747781, 189.49654744845537,
    192.71705895943293,
]
_ACC_TECH_INTERCEPT = 26.0496215686518
_ACC_TECH_SLOPE = 1.8800369206815004
_ACC_ENOB_SLOPE = -2.8291256050594527
_ACC_ENRG_RESIDUAL = -1.4702022133784405


def _bits2sndr(bits: float) -> float:
    """SNDR (dB) for an ideal `bits`-bit converter (Schreier/accelergy form)."""
    return bits * 20.0 * math.log10(2.0) + 10.0 * math.log10(1.5)


def adc_energy_accelergy(enob: float, tech_nm: float, freq_hz: float) -> float:
    """Best-possible ADC energy/conversion (Joules) from the Accelergy model."""
    enob = max(float(enob), 1.0)
    lf = math.log(max(freq_hz, 1.0))
    lt = math.log(max(tech_nm, 1.0))
    foms = _ACC_FOMS_INTERCEPT + _ACC_FOMS_FREQ_SLOPE * lf
    idx = max(min(math.ceil(enob), len(_ACC_FOMS_MAX_BY_ENOB) - 1), 0)
    foms = min(foms, _ACC_FOMS_MAX_BY_ENOB[idx])
    e_pJ = 1.0 / ((10 ** ((foms - _bits2sndr(enob)) / 10.0)) * 2.0)
    e_pJ *= math.exp(_ACC_TECH_INTERCEPT + _ACC_TECH_SLOPE * lt
                     + _ACC_ENOB_SLOPE * math.log(enob) + _ACC_ENRG_RESIDUAL)
    return e_pJ * 1e-12  # pJ/op -> J/op


def adc_energy_walden(fom_fJ: float, enob: float) -> float:
    """Walden-FOM ADC energy/conversion (Joules): E = FOMW * 2^ENOB."""
    return fom_fJ * 1e-15 * (2.0 ** enob)


def adc_energy_per_conv(model: Dict, adc_bits: int) -> float:
    """Dispatch to the configured ADC energy model.  Returns J/conversion.

    Corner/supply scaling: a SAR ADC's energy is dominated by CDAC switching
    + comparator, both CV^2, so the modelled value scales with (V/Vnom)^2
    (``_scale_v2``) when operating-point tracking is on.  Node awareness:
    'accelergy' is intrinsically node-aware (tech_node_nm); 'walden' uses a
    survey-median FOM that is NOT node-aware — enable energy_node_scaling
    (or switch to accelergy) for node-realistic ADC energy.
    """
    enob = model["adc_enob"] if model["adc_enob"] is not None else adc_bits
    if str(model.get("adc_energy_model", "walden")).lower() == "accelergy":
        e = adc_energy_accelergy(enob, model["tech_node_nm"], model["rate_Hz"])
    else:
        e = adc_energy_walden(model["adc_fom_fJ"], enob)
    return e * model.get("_scale_v2", 1.0)


def dac_energy_per_conv(model: Dict, dac_bits: int) -> float:
    """DAC energy/conversion (J).  Direct value if given, else switched-cap
    E = activity * C * Vref^2 * 2^bits.  The CDAC reference is derived from
    the supply, so the result scales with the operating point as (V/Vnom)^2
    when [PERIPHERALS] track_operating_point is on (``_scale_v2``)."""
    if model["dac_energy_pJ"] is not None and model["dac_energy_pJ"] > 0:
        return model["dac_energy_pJ"] * 1e-12 * model.get("_scale_v2", 1.0)
    C = model["dac_cap_fF"] * 1e-15
    Vref = model["dac_vref_V"]
    activity = model["dac_activity"]
    return activity * C * (Vref ** 2) * (2.0 ** dac_bits) * model.get("_scale_v2", 1.0)


def tia_energy_per_conv(model: Dict) -> float:
    """TIA energy/conversion (J) from static bias: E = I_bias * V_sup * t_active.
    t_active defaults to one clock/sample period."""
    t_active = model["tia_tactive_s"]
    if t_active is None or t_active <= 0:
        t_active = 1.0 / model["rate_Hz"] if model["rate_Hz"] > 0 else 0.0
    # Static bias E = I*V*t: linear in the supply -> _scale_v1 at corners.
    return (model["tia_ibias_A"] * model["tia_vsup_V"] * t_active
            * model.get("_scale_v1", 1.0))


def subtractor_energy_per_conv(model: Dict) -> float:
    """Subtractor (TIA-output differencing) energy/conversion (J).  Direct
    ``subtractor_energy_pJ_per_op`` wins if set; else a switched-cap estimate
    ``C * Vdd^2`` for one difference operation."""
    if model.get("subtractor_energy_pJ") is not None and model["subtractor_energy_pJ"] > 0:
        return model["subtractor_energy_pJ"] * 1e-12 * model.get("_scale_v2", 1.0)
    return (model["subtractor_cap_fF"] * 1e-15 * (model["subtractor_vdd_V"] ** 2)
            * model.get("_scale_v2", 1.0))


def _mux_energy_per_routed(model: Dict, n_select_stages: int) -> float:
    """Energy (J) to route one signal through the converter-sharing mux tree.
    Direct ``mux_energy_pJ_per_op`` wins if set (>0); otherwise a transmission-
    gate model: stages * C_tg * Vdd^2, where stages = depth of the W:U mux tree."""
    if model.get("mux_energy_pJ") is not None and model["mux_energy_pJ"] > 0:
        return model["mux_energy_pJ"] * 1e-12 * model.get("_scale_v2", 1.0)
    return (max(1, n_select_stages) * model["mux_cap_fF"] * 1e-15
            * (model["mux_vdd_V"] ** 2) * model.get("_scale_v2", 1.0))


def estimate_mux_energy(manifest: Dict, model: Dict, lat: Dict) -> Dict:
    """
    Energy of the DAC/ADC-sharing multiplexers.

    A mux is only present when a converter is shared, i.e. when the number of
    physical DACs (ADCs) is smaller than a layer's input (output) width — then
    each converter feeds a ``ceil(width/units):1`` mux and every conversion is
    routed through ``ceil(log2(ceil(width/units)))`` tree stages.  When a layer
    has at least one converter per I/O line, no mux energy is charged for it.

    Requires the ``per_layer`` breakdown in the manifest (written by OpLedger);
    without it, mux energy can't be attributed per layer and is reported zero.
    """
    per_layer = manifest.get("per_layer") or []
    num_dac = max(1, int(lat["num_dac"]))
    num_adc = max(1, int(lat["num_adc"]))
    n_slices = (int(manifest["shapes"].get("n_bit_slices", 1))
                if model["adc_per_slice"] else 1)

    n_dac_mux = n_adc_mux = 0
    E_dac_mux = E_adc_mux = 0.0
    for L in per_layer:
        m, k, n = int(L["m"]), int(L["k"]), int(L["n"])
        if num_dac < k:                                  # DACs shared -> mux
            stages = max(1, math.ceil(math.log2(math.ceil(k / num_dac))))
            ops = m * k
            n_dac_mux += ops
            E_dac_mux += ops * _mux_energy_per_routed(model, stages)
        if num_adc < n:                                  # ADCs shared -> mux
            stages = max(1, math.ceil(math.log2(math.ceil(n / num_adc))))
            ops = m * n * n_slices
            n_adc_mux += ops
            E_adc_mux += ops * _mux_energy_per_routed(model, stages)

    return {
        "n_dac_mux_ops": n_dac_mux, "n_adc_mux_ops": n_adc_mux,
        "n_ops": n_dac_mux + n_adc_mux,
        "energy_dac_J": E_dac_mux, "energy_adc_J": E_adc_mux,
        "energy_J": E_dac_mux + E_adc_mux,
        "shared_dac": n_dac_mux > 0, "shared_adc": n_adc_mux > 0,
    }


def estimate_peripheral_energy(manifest: Dict, model: Dict,
                               measured: Optional[Dict] = None) -> Dict:
    """Combine op counts with per-op energy models -> total DAC/ADC/TIA energy.

    ``measured`` (from peripheral_chain's measured_converter_energy.json) overrides
    the modelled per-conversion energy for any hardware-mode converter, e.g.
    ``{"adc": {"e_per_conv_J": 4.4e-11, "source": "spice"}}``.  Roles absent from
    ``measured`` keep their model energy."""
    measured = measured or {}
    counts = manifest["counts"]
    res = manifest.get("resolution_bits", {})
    dac_bits = int(res.get("dac") or 8)
    adc_bits = int(res.get("adc") or 8)

    n_dac = counts["dac_conversions"]
    if model["adc_per_slice"]:
        n_adc = counts["adc_conversions_per_slice"]
        n_tia = counts["adc_conversions_per_slice"]
    else:
        n_adc = counts["adc_conversions"]
        n_tia = counts["tia_conversions"]

    def _pick(role: str, modelled: float) -> Tuple[float, str]:
        m = measured.get(role)
        if m and isinstance(m.get("e_per_conv_J"), (int, float)) \
                and m["e_per_conv_J"] == m["e_per_conv_J"]:
            return float(m["e_per_conv_J"]), m.get("source", "spice")
        return modelled, "model"

    e_dac_op, dac_src = _pick("dac", dac_energy_per_conv(model, dac_bits))
    e_adc_op, adc_src = _pick("adc", adc_energy_per_conv(model, adc_bits))

    # TIA: in a differential readout each output uses a +column and a -column
    # TIA.  The measured per-op energy (from peripheral_chain) is already the
    # differential *pair* energy/output; the modelled per-TIA energy is doubled
    # when differential so both paths report energy/output.  n_tia stays = the
    # per-output conversion count.
    differential = bool(model.get("differential_columns", True))
    e_tia_single = tia_energy_per_conv(model)
    e_tia_model = e_tia_single * (2.0 if differential else 1.0)
    e_tia_op, tia_src = _pick("tia", e_tia_model)

    # SUBTRACTOR: one per output, only present in a differential readout.
    n_sub = n_adc if differential else 0
    e_sub_op, sub_src = _pick("subtractor", subtractor_energy_per_conv(model))
    e_sub_total = n_sub * e_sub_op if differential else 0.0

    return {
        "dac": {"n_ops": n_dac, "e_per_op_J": e_dac_op,
                "energy_J": n_dac * e_dac_op, "bits": dac_bits, "src": dac_src},
        "adc": {"n_ops": n_adc, "e_per_op_J": e_adc_op,
                "energy_J": n_adc * e_adc_op, "bits": adc_bits, "src": adc_src,
                "model": model.get("adc_energy_model", "walden"),
                "enob": (model["adc_enob"] if model["adc_enob"] is not None
                         else adc_bits)},
        "tia": {"n_ops": n_tia, "e_per_op_J": e_tia_op,
                "energy_J": n_tia * e_tia_op, "src": tia_src,
                "differential": differential},
        "subtractor": {"n_ops": n_sub, "e_per_op_J": e_sub_op,
                       "energy_J": e_sub_total, "src": sub_src,
                       "present": differential},
        "total_peripheral_energy_J": (n_dac * e_dac_op + n_adc * e_adc_op
                                      + n_tia * e_tia_op + e_sub_total),
    }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 3 — Cycle-accurate latency model                                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def adc_cycles_for_type(adc_type: str, adc_bits: int, osr: int = 0) -> int:
    """Clock cycles per ADC conversion as a function of architecture.

    sar        : one bit-decision per cycle + 1 sample cycle  -> bits + 1
    flash      : fully parallel                                -> 1
    pipeline   : 1 conversion/cycle throughput (latency hidden)-> 1
    sigma_delta: oversampling                                  -> osr (or 2^bits)
    integrating/single_slope/ss : ramp counts the full code   -> 2^bits
    """
    t = (adc_type or "sar").strip().lower()
    if t in ("flash",):
        return 1
    if t in ("pipeline", "pipelined", "pipe"):
        return 1
    if t in ("sigma_delta", "sigmadelta", "sd", "delta_sigma"):
        return int(osr) if osr and osr > 0 else int(2 ** adc_bits)
    if t in ("integrating", "single_slope", "ss", "ramp"):
        return int(2 ** adc_bits)
    # default: SAR
    return int(adc_bits) + 1


def _stage_cycle_times(lat: Dict, measured: Dict = None) -> Dict:
    """Physical per-cycle time for each latency stage (DAC / MAC / ADC).

    The legacy model used one fixed clock period for every stage, making
    latency independent of technology, corner and temperature.  With
    ``periph_delay_model = node_scaled`` each stage gets a physical time:

      * digital converters (DAC/ADC):
            t = t_clk × (node/ref_node)^e × corner_factor × temp_factor
        — the per-node peripheral-latency approach of NeuroSim (FO4 ≈ linear
        in gate length), with the corner drive-strength spread (ff ≈ 0.85×,
        ss ≈ 1.30×, configurable) and ≈ +0.1 %/°C mobility slowdown.
      * analog MVM settle: the SPICE-measured settle time when available
        (``measured['analog_settle_s']`` from the transient waveforms — it is
        corner/supply/temperature-accurate by construction), else the scaled
        clock.

    ``fixed`` + no measurement reproduces the legacy numbers exactly.
    """
    from spicecimx.analog import tech_scale as _ts
    f_hz = lat["clock_freq_hz"]
    t_clk = 1.0 / f_hz if f_hz > 0 else float("nan")

    model = str(lat.get("periph_delay_model", "fixed")).strip().lower()
    if model == "node_scaled":
        scale = _ts.delay_scale(
            lat.get("tech_node_nm", 0.0), lat.get("process_corner", "tt"),
            lat.get("temperature_C", 27.0),
            ref_node_nm=lat.get("ref_node_nm", 0.0),
            node_exponent=lat.get("node_delay_exponent", 1.0),
            corner_factors={"ff": lat.get("corner_delay_ff", 0.85),
                            "ss": lat.get("corner_delay_ss", 1.30)},
            temp_coeff_per_C=lat.get("temp_delay_coeff_per_C", 0.001),
        )
    else:
        scale = 1.0
    t_periph = t_clk * scale

    t_mac = t_periph
    src = "clock" if scale == 1.0 else "node_scaled"
    if measured and measured.get("analog_settle_s", 0) and measured["analog_settle_s"] > 0:
        t_mac = float(measured["analog_settle_s"])
        src = "spice_transient"
    return {"t_clk": t_clk, "delay_scale": scale,
            "t_dac_cyc": t_periph, "t_adc_cyc": t_periph, "t_mac_cyc": t_mac,
            "mac_time_source": src}


def estimate_latency(manifest: Dict, lat: Dict, measured: Dict = None) -> Dict:
    """
    Cycle-accurate latency for one inference.

    Per crossbar readout the pipeline is DAC -> analog VMM settle -> ADC.  When
    the number of physical DACs (ADCs) is fewer than a layer's input (output)
    width, the converters are time-shared through a multiplexer, so each input
    vector takes ``ceil(width / units)`` conversion waves; the optional
    ``mux_cycles`` is added per shared wave for the mux settle.  The MAC/VMM
    cycles serialize the subarray tiles over ``vmm_parallel_subarrays`` spatial
    units.  Counts are computed **per layer** (from the manifest ``per_layer``
    breakdown) and summed, so the ceil rounding from converter sharing is exact;
    an aggregate fallback is used if no per-layer data is present.

    Total (serial) latency = DAC conv cycles + MAC/VMM cycles + ADC conv cycles
    (exactly the spec: data-conversion cycles + MAC cycles).  A pipelined
    estimate that overlaps the three stages is also reported.
    """
    counts = manifest["counts"]
    n_mac = counts["mac_operations"]
    per_layer = manifest.get("per_layer")
    adc_per_slice = bool(lat.get("adc_per_slice"))
    n_slices = (int(manifest["shapes"].get("n_bit_slices", 1))
                if adc_per_slice else 1)

    f_hz = lat["clock_freq_hz"]
    t_clk = 1.0 / f_hz if f_hz > 0 else float("nan")

    num_dac = max(1, int(lat["num_dac"]))
    num_adc = max(1, int(lat["num_adc"]))
    dac_cpc = max(1, int(lat["dac_cycles_per_conv"]))
    adc_cpc = (int(lat["adc_cycles_per_conv"]) if lat["adc_cycles_per_conv"] > 0
               else adc_cycles_for_type(lat["adc_type"], lat["adc_bits"],
                                        lat.get("adc_osr", 0)))
    mac_cpv = max(1, int(lat["mac_cycles_per_vector"]))
    mux_cyc = max(0, int(lat.get("mux_cycles", 0)))

    sa_r = max(1, int(lat["subarray_rows"]))
    sa_c = max(1, int(lat["subarray_cols"]))
    P = lat.get("vmm_parallel_subarrays", 0)
    if not P or P <= 0:
        P = max(1, int(lat["pe_rows"]) * int(lat["pe_cols"])
                * int(lat["tile_rows"]) * int(lat["tile_cols"]))

    cyc_dac = cyc_adc = cyc_mac = 0
    n_vmm_eval = 0
    shared_dac = shared_adc = False

    if per_layer:
        for L in per_layer:
            m, k, n = int(L["m"]), int(L["k"]), int(L["n"])
            # DAC: ceil(k/num_dac) waves/vector; +mux settle when shared
            w_dac = math.ceil(k / num_dac)
            if w_dac > 1:
                shared_dac = True
            cyc_dac += m * w_dac * (dac_cpc + (mux_cyc if w_dac > 1 else 0))
            # ADC: per-slice digitisations share num_adc through a mux
            adc_per_vec = n * n_slices
            w_adc = math.ceil(adc_per_vec / num_adc)
            if num_adc < adc_per_vec:
                shared_adc = True
            cyc_adc += m * w_adc * (adc_cpc + (mux_cyc if w_adc > 1 else 0))
            # MAC/VMM: tile (k x n) into subarrays, serialize over P spatial units
            num_sa = math.ceil(k / sa_r) * math.ceil(n / sa_c)
            w_mac = math.ceil(num_sa / P)
            cyc_mac += m * w_mac * mac_cpv
            n_vmm_eval += m * num_sa
    else:
        # Aggregate fallback (old manifests without per_layer): ceil on totals.
        n_dac_conv = counts["dac_conversions"]
        n_adc_conv = (counts["adc_conversions_per_slice"]
                      if adc_per_slice else counts["adc_conversions"])
        sa_area = sa_r * sa_c
        n_vmm_eval = math.ceil(n_mac / sa_area)
        cyc_dac = math.ceil(n_dac_conv / num_dac) * dac_cpc
        cyc_adc = math.ceil(n_adc_conv / num_adc) * adc_cpc
        cyc_mac = math.ceil(n_vmm_eval / P) * mac_cpv

    cyc_total_serial = cyc_dac + cyc_mac + cyc_adc

    # Physical per-stage times: node/corner/temperature-aware digital
    # converter timing + SPICE-measured analog settle when available.
    times = _stage_cycle_times(lat, measured)
    lat_dac_s = cyc_dac * times["t_dac_cyc"]
    lat_mac_s = cyc_mac * times["t_mac_cyc"]
    lat_adc_s = cyc_adc * times["t_adc_cyc"]
    lat_total_s = lat_dac_s + lat_mac_s + lat_adc_s
    lat_per_mac_s = lat_total_s / n_mac if n_mac > 0 else float("nan")

    return {
        "t_clk_s": t_clk,
        "delay_scale": times["delay_scale"],
        "t_dac_cycle_s": times["t_dac_cyc"],
        "t_adc_cycle_s": times["t_adc_cyc"],
        "t_mac_cycle_s": times["t_mac_cyc"],
        "mac_time_source": times["mac_time_source"],
        "latency_s": {"dac": lat_dac_s, "mac": lat_mac_s, "adc": lat_adc_s},
        "clock_freq_hz": f_hz,
        "adc_type": lat["adc_type"],
        "adc_cycles_per_conv": adc_cpc,
        "dac_cycles_per_conv": dac_cpc,
        "mac_cycles_per_vector": mac_cpv,
        "mux_cycles": mux_cyc,
        "num_dac": num_dac, "num_adc": num_adc,
        "dac_shared": shared_dac, "adc_shared": shared_adc,
        "vmm_parallel_subarrays": P,
        "cycles": {"dac": cyc_dac, "mac": cyc_mac, "adc": cyc_adc,
                   "total_serial": cyc_total_serial},
        "n_vmm_eval": n_vmm_eval,
        "latency_total_s": lat_total_s,
        "latency_per_mac_s": lat_per_mac_s,
    }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 4 — config -> model dict                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def _cfg_get(cp, section, key, cast, fallback):
    if not cp.has_section(section) or not cp.has_option(section, key):
        return fallback
    raw = cp.get(section, key).strip()
    if raw == "":
        return fallback
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return fallback


def _cfg_bool(cp, section, key, fallback=False):
    raw = _cfg_get(cp, section, key, str, None)
    if raw is None:
        return fallback
    return raw.strip().lower() in ("1", "true", "yes", "on", "y", "t")


def _cfg_get_path(config_path: str, sec: str, key: str, default: float) -> float:
    """One-off float read from a config file path (report-driver convenience)."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    cp.read(config_path)
    return _cfg_get(cp, sec, key, float, default)


def load_models(config_path: str) -> Tuple[Dict, Dict]:
    """Read [PERIPHERALS], [LATENCY] (+ [SIM]/[HW] for shared knobs).
    Returns (peripheral_model, latency_model) plain dicts."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    cp.read(config_path)

    # adc_bits lives in [PERIPHERALS] (older configs: [SIM])
    adc_bits = int(_cfg_get(cp, "PERIPHERALS", "adc_bits", float,
                            _cfg_get(cp, "SIM", "adc_bits", float, 8.0)))
    enob = _cfg_get(cp, "PERIPHERALS", "adc_enob", float, -1.0)
    rate = _cfg_get(cp, "LATENCY", "clock_freq_hz", float,
                    _cfg_get(cp, "PERIPHERALS", "conversion_rate_Hz", float, 1e9))

    model = {
        "tech_node_nm": _cfg_get(cp, "PERIPHERALS", "tech_node_nm", float, 180.0),
        "adc_energy_model": _cfg_get(cp, "PERIPHERALS", "adc_energy_model", str, "walden"),
        "adc_fom_fJ": _cfg_get(cp, "PERIPHERALS", "adc_fom_fJ_per_convstep", float, 22.5),
        "adc_enob": (None if enob is None or enob < 0 else enob),
        "dac_energy_pJ": (lambda v: None if v is None or v < 0 else v)(
            _cfg_get(cp, "PERIPHERALS", "dac_energy_pJ_per_conv", float, -1.0)),
        "dac_cap_fF": _cfg_get(cp, "PERIPHERALS", "dac_cap_fF", float, 500.0),
        "dac_vref_V": _cfg_get(cp, "PERIPHERALS", "dac_vref_V", float, 0.2),
        "dac_activity": _cfg_get(cp, "PERIPHERALS", "dac_switching_activity", float, 1.0/3.0),
        "tia_ibias_A": _cfg_get(cp, "PERIPHERALS", "tia_bias_current_A", float, 20e-6),
        "tia_vsup_V": _cfg_get(cp, "PERIPHERALS", "tia_supply_V", float, 1.8),
        "tia_tactive_s": (lambda v: None if v is None or v < 0 else v)(
            _cfg_get(cp, "PERIPHERALS", "tia_active_time_s", float, -1.0)),
        "rate_Hz": rate,
        "adc_per_slice": _cfg_bool(cp, "PERIPHERALS", "adc_per_slice", False),
        # ── multiplexer (converter sharing) ──
        "mux_energy_pJ": (lambda v: None if v is None or v < 0 else v)(
            _cfg_get(cp, "PERIPHERALS", "mux_energy_pJ_per_op", float, -1.0)),
        "mux_cap_fF": _cfg_get(cp, "PERIPHERALS", "mux_cap_fF", float, 2.0),
        "mux_vdd_V": _cfg_get(cp, "PERIPHERALS", "mux_vdd_V", float,
                              _cfg_get(cp, "PERIPHERALS", "tia_supply_V", float, 1.8)),
        # ── subtractor (separate stage between TIA and ADC) ──
        "differential_columns": _cfg_bool(cp, "PERIPHERALS", "differential_columns", True),
        "subtractor_energy_pJ": (lambda v: None if v is None or v < 0 else v)(
            _cfg_get(cp, "PERIPHERALS", "subtractor_energy_pJ_per_op", float, -1.0)),
        "subtractor_cap_fF": _cfg_get(cp, "PERIPHERALS", "subtractor_cap_fF", float, 10.0),
        "subtractor_vdd_V": _cfg_get(cp, "PERIPHERALS", "subtractor_vdd_V", float,
                                     _cfg_get(cp, "PERIPHERALS", "tia_supply_V", float, 1.8)),
    }

    lat = {
        "clock_freq_hz": rate,
        "adc_bits": adc_bits,
        "adc_type": _cfg_get(cp, "LATENCY", "adc_type", str, "sar"),
        "adc_cycles_per_conv": int(_cfg_get(cp, "LATENCY", "adc_cycles_per_conv", float, 0.0)),
        "dac_cycles_per_conv": int(_cfg_get(cp, "LATENCY", "dac_cycles_per_conv", float, 1.0)),
        "mac_cycles_per_vector": int(_cfg_get(cp, "LATENCY", "mac_cycles_per_vector", float, 1.0)),
        "num_dac": int(_cfg_get(cp, "LATENCY", "num_dac", float, 16.0)),
        "num_adc": int(_cfg_get(cp, "LATENCY", "num_adc", float, 16.0)),
        "vmm_parallel_subarrays": int(_cfg_get(cp, "LATENCY", "vmm_parallel_subarrays", float, 0.0)),
        "adc_osr": int(_cfg_get(cp, "LATENCY", "adc_osr", float, 0.0)),
        "mux_cycles": int(_cfg_get(cp, "LATENCY", "mux_cycles", float, 0.0)),
        "adc_per_slice": model["adc_per_slice"],
        # geometry for VMM parallelism (from [HW])
        "subarray_rows": int(_cfg_get(cp, "HW", "subarray_rows", float, 16.0)),
        "subarray_cols": int(_cfg_get(cp, "HW", "subarray_cols", float, 16.0)),
        "pe_rows": int(_cfg_get(cp, "HW", "pe_rows", float, 2.0)),
        "pe_cols": int(_cfg_get(cp, "HW", "pe_cols", float, 2.0)),
        "tile_rows": int(_cfg_get(cp, "HW", "tile_rows", float, 2.0)),
        "tile_cols": int(_cfg_get(cp, "HW", "tile_cols", float, 2.0)),
        # --- technology / corner aware timing ([LATENCY], opt-in) ---
        # periph_delay_model: 'fixed' (legacy, one clock for everything) or
        # 'node_scaled' (digital converter cycle time scales with the node and
        # the process corner / temperature; analog MVM term prefers the
        # SPICE-measured settle time).
        "periph_delay_model": _cfg_get(cp, "LATENCY", "periph_delay_model", str, "fixed"),
        "tech_node_nm": _cfg_get(cp, "PERIPHERALS", "tech_node_nm", float, 180.0),
        "ref_node_nm": _cfg_get(cp, "LATENCY", "ref_node_nm", float, 0.0),
        "node_delay_exponent": _cfg_get(cp, "LATENCY", "node_delay_exponent", float, 1.0),
        "corner_delay_ff": _cfg_get(cp, "LATENCY", "corner_delay_ff", float, 0.85),
        "corner_delay_ss": _cfg_get(cp, "LATENCY", "corner_delay_ss", float, 1.30),
        "temp_delay_coeff_per_C": _cfg_get(cp, "LATENCY", "temp_delay_coeff_per_C", float, 0.001),
        "settle_band": _cfg_get(cp, "LATENCY", "settle_band", float, 0.02),
        # operating point (shared with the energy scaling below)
        "process_corner": _cfg_get(cp, "SIM", "process_corner", str, "tt"),
        "vdd_V": _cfg_get(cp, "SIM", "vdd", float, 0.0),
        "temperature_C": _cfg_get(cp, "SIM", "temperature", float, 27.0),
    }

    # ── Operating-point / technology scaling of the peripheral models ──────
    # [PERIPHERALS] track_operating_point = true makes every modelled per-op
    # energy follow the corner supply: dynamic (CV^2) blocks scale (V/Vnom)^2,
    # static-bias blocks (TIA) scale (V/Vnom).  V = [SIM] vdd (the corner
    # supply already configured per run); Vnom = nominal_vdd_V, defaulting to
    # the per-node industry table in analog.tech_scale.  energy_node_scaling
    # additionally projects the constants from energy_ref_node_nm to the
    # current node (C ∝ node, V ∝ per-node nominal supply) so one reference
    # config covers 180 nm → 3 nm.  Both default OFF (legacy output).
    from spicecimx.analog import tech_scale as _ts
    track = str(_cfg_get(cp, "PERIPHERALS", "track_operating_point", str, "false")
                ).strip().lower() in ("1", "true", "yes", "on")
    v_nom_cfg = _cfg_get(cp, "PERIPHERALS", "nominal_vdd_V", float, 0.0)
    node_scaling = str(_cfg_get(cp, "PERIPHERALS", "energy_node_scaling", str, "false")
                       ).strip().lower() in ("1", "true", "yes", "on")
    e_ref_node = _cfg_get(cp, "PERIPHERALS", "energy_ref_node_nm", float, 0.0)
    node = model["tech_node_nm"]
    corner = lat["process_corner"]
    scale_v2 = scale_v1 = 1.0
    if track:
        v_nom = v_nom_cfg if v_nom_cfg > 0 else _ts.vdd_nominal(node)
        v_op = _ts.corner_vdd(node, corner, vdd_override=lat["vdd_V"],
                              nominal_override=v_nom)
        if v_nom and v_nom > 0:
            scale_v1 = v_op / v_nom
            scale_v2 = scale_v1 ** 2
    if node_scaling and e_ref_node > 0:
        scale_v2 *= _ts.cap_scale(node, e_ref_node)
        v_nom_node, v_nom_ref = _ts.vdd_nominal(node), _ts.vdd_nominal(e_ref_node)
        if v_nom_node > 0 and v_nom_ref > 0:
            scale_v2 *= (v_nom_node / v_nom_ref) ** 2
            scale_v1 *= (v_nom_node / v_nom_ref)
    model["_scale_v2"] = scale_v2
    model["_scale_v1"] = scale_v1
    model["_track_operating_point"] = track
    model["_energy_node_scaling"] = node_scaling
    return model, lat


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  PART 5 — report assembly                                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def _fmt_energy(j: Optional[float]) -> str:
    if j is None or j != j:
        return "   n/a"
    for unit, scale in [("J", 1), ("mJ", 1e-3), ("uJ", 1e-6),
                        ("nJ", 1e-9), ("pJ", 1e-12), ("fJ", 1e-15)]:
        if abs(j) >= scale:
            return f"{j/scale:9.3f} {unit}"
    return f"{j/1e-15:9.3f} fJ"


def _fmt_power(w: Optional[float]) -> str:
    if w is None or w != w:
        return "   n/a"
    for unit, scale in [("W", 1), ("mW", 1e-3), ("uW", 1e-6),
                        ("nW", 1e-9), ("pW", 1e-12)]:
        if abs(w) >= scale:
            return f"{w/scale:9.3f} {unit}"
    return f"{w/1e-12:9.3f} pW"


def _fmt_time(s: Optional[float]) -> str:
    if s is None or s != s:
        return "   n/a"
    for unit, scale in [("s", 1), ("ms", 1e-3), ("us", 1e-6),
                        ("ns", 1e-9), ("ps", 1e-12)]:
        if abs(s) >= scale:
            return f"{s/scale:9.3f} {unit}"
    return f"{s/1e-12:9.3f} ps"


def generate_report(dump_dir: str, config_path: str,
                    hwsimdb_dir: Optional[str] = None,
                    output_name: str = "power_energy_latency_report.csv") -> str:
    """Build the consolidated power/energy/latency report.  Reads the op
    manifest from ``dump_dir``, the crossbar CSVs from ``hwsimdb_dir`` (falls
    back to ``dump_dir``), and the models from ``config_path``."""
    dump = Path(dump_dir)
    manifest_path = dump / "peripheral_ops.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found.  Run inference with an OpLedger first "
            f"(infer.py writes it automatically).")
    with open(manifest_path) as f:
        manifest = json.load(f)

    model, lat = load_models(config_path)
    xbar_dir = hwsimdb_dir or dump_dir

    # 1) crossbar.  Two layouts are supported:
    #    (a) single transient image (legacy): the layer folders live directly
    #        under xbar_dir -> sum their energies.
    #    (b) multi-image auto mode: xbar_dir contains one subdir per transient
    #        image, listed in transient_images.json -> the reported crossbar
    #        energy/power is the AVERAGE across the images (each image is a
    #        full inference; averaging estimates the expected per-inference
    #        cost over the input distribution).
    crossbar_images = {}
    marker = Path(xbar_dir) / "transient_images.json"
    if marker.exists():
        try:
            with open(marker) as f:
                info = json.load(f)
        except Exception:
            info = {}
        e_vals, p_vals, s_vals, d_vals = [], [], [], []

        def _fsum(c, key):
            return sum(v.get(key, float("nan")) for v in c.values()
                       if v.get(key, float("nan")) == v.get(key, float("nan")))
        for sub in info.get("subdirs", []):
            c = collect_crossbar_power_energy(str(Path(xbar_dir) / sub))
            if not c:
                continue
            e_i, p_i = _fsum(c, "energy_J"), _fsum(c, "power_W")
            crossbar_images[sub] = {"energy_J": e_i, "power_W": p_i,
                                    "image": info.get("images", {}).get(sub, "")}
            e_vals.append(e_i)
            p_vals.append(p_i)
            s_vals.append(_fsum(c, "energy_static_J"))
            d_vals.append(_fsum(c, "energy_dynamic_J"))
        _avg = lambda L: (sum(L) / len(L)) if L else float("nan")
        crossbar = {"__avg_of_images__": {
            "energy_J": _avg(e_vals), "power_W": _avg(p_vals),
            "energy_static_J": _avg(s_vals), "energy_dynamic_J": _avg(d_vals),
        }} if e_vals else {}
        E_crossbar = crossbar["__avg_of_images__"]["energy_J"] if crossbar else float("nan")
        P_crossbar = crossbar["__avg_of_images__"]["power_W"] if crossbar else float("nan")
    else:
        crossbar = collect_crossbar_power_energy(xbar_dir)
        E_crossbar = sum(v["energy_J"] for v in crossbar.values()
                         if v["energy_J"] == v["energy_J"])
        P_crossbar = sum(v["power_W"] for v in crossbar.values()
                         if v["power_W"] == v["power_W"])
        if not crossbar:
            E_crossbar = float("nan")
            P_crossbar = float("nan")
    E_xbar_static = sum(v.get("energy_static_J", float("nan"))
                        for v in crossbar.values()
                        if v.get("energy_static_J") == v.get("energy_static_J"))
    E_xbar_dyn = sum(v.get("energy_dynamic_J", float("nan"))
                     for v in crossbar.values()
                     if v.get("energy_dynamic_J") == v.get("energy_dynamic_J"))
    # Per-inference normalisation: a --batch_all transient run simulates the
    # whole batch in one waveform; the manifest counts are already per
    # inference (OpLedger.per_inference).
    n_xb = int(manifest.get("crossbar_inferences", 1) or 1)
    if n_xb > 1:
        E_crossbar /= n_xb
        E_xbar_static /= n_xb
        E_xbar_dyn /= n_xb

    # 2) peripheral energy (DAC / ADC / TIA) — measured SPICE energy (from
    #    peripheral_chain, hardware-mode converters) overrides the model.
    measured = {}
    meas_path = dump / "measured_converter_energy.json"
    if meas_path.exists():
        try:
            with open(meas_path) as f:
                measured = json.load(f)
        except Exception:
            measured = {}
    periph = estimate_peripheral_energy(manifest, model, measured=measured)
    # 2b) multiplexer energy (only when converters are shared, num<width)
    mux = estimate_mux_energy(manifest, model, lat)
    E_periph = periph["total_peripheral_energy_J"] + mux["energy_J"]

    # 3) latency — feed the SPICE-measured analog settle time (corner/tech/
    #    temperature accurate) when the transient waveforms are available.
    measured_lat = {}
    settle_path = dump / "measured_settle.json"
    if settle_path.exists():
        try:
            with open(settle_path) as f:
                measured_lat = json.load(f)
        except Exception:
            measured_lat = {}
    if not measured_lat:
        try:
            from spicecimx.io.parse_transient import (
                collect_worst_settle_time, TransientReadSettings)
            settings = TransientReadSettings(
                start_time=_cfg_get_path(config_path, "SIM", "settling_start_time", 0.0),
                settling_lo=_cfg_get_path(config_path, "SIM", "settling_window_lo", 0.40),
                settling_hi=_cfg_get_path(config_path, "SIM", "settling_window_hi", 0.65),
            )
            measured_lat = collect_worst_settle_time(
                str(xbar_dir), settings, lat.get("settle_band", 0.02))
            if measured_lat:
                with open(settle_path, "w") as f:
                    json.dump(measured_lat, f, indent=2)
        except Exception:
            measured_lat = {}
    # 3b) tile interconnect (arch.loading / arch.noc), when the run wrote it:
    #     the shared lines' Elmore constant lengthens every MVM settle, the
    #     NoC moves the activations between the tiles and the global buffer.
    ic_rep = {}
    ic_path = dump / "interconnect_report.json"
    if ic_path.exists():
        try:
            with open(ic_path) as f:
                ic_rep = json.load(f)
        except Exception:
            ic_rep = {}
    tau_line = float(ic_rep.get("loading", {}).get("tau_line_max_s", 0.0) or 0.0)
    if tau_line > 0:
        band = max(1e-6, float(lat.get("settle_band", 0.02)))
        measured_lat = dict(measured_lat or {})
        base = measured_lat.get("analog_settle_s") or 0.0
        if not base:                     # no waveforms: start from the clock
            base = _stage_cycle_times(lat, {})["t_mac_cyc"]
        measured_lat["line_settle_s"] = tau_line * math.log(1.0 / band)
        measured_lat["analog_settle_s"] = base + measured_lat["line_settle_s"]
    latres = estimate_latency(manifest, lat, measured=measured_lat)

    n_ic = int(ic_rep.get("crossbar_inferences", 1) or 1)
    E_lines = float(ic_rep.get("loading", {}).get("line_energy_J", {})
                    .get("total_J", 0.0) or 0.0) / n_ic
    noc = ic_rep.get("noc", {}) or {}
    E_noc = float(noc.get("energy_J", 0.0) or 0.0)
    if noc.get("latency_s"):
        t_noc = float(noc["fill_latency_s"] if ic_rep.get("noc_overlap")
                      and "fill_latency_s" in noc else noc["latency_s"])
        latres["latency_s"]["noc"] = t_noc
        latres["latency_total_s"] += t_noc
        n_mac_ = manifest["counts"]["mac_operations"]
        latres["latency_per_mac_s"] = (latres["latency_total_s"] / n_mac_
                                       if n_mac_ > 0 else float("nan"))
    E_ic = E_lines + E_noc

    # 4) system
    E_total = (0.0 if E_crossbar != E_crossbar else E_crossbar) + E_periph + E_ic
    n_mac = manifest["counts"]["mac_operations"]
    n_ops = 2 * n_mac
    fJ_per_mac = (E_total / n_mac * 1e15) if n_mac > 0 else float("nan")
    tops_per_w = (n_ops / E_total) / 1e12 if E_total > 0 else float("nan")

    lat_s = latres["latency_total_s"]
    gops = (n_ops / lat_s) / 1e9 if lat_s and lat_s == lat_s and lat_s > 0 else float("nan")
    P_total_time = E_total / lat_s if lat_s and lat_s > 0 else float("nan")

    # ── console ──
    L = "=" * 74
    print(L)
    print("  SPICECiM-X v3 — POWER / ENERGY / LATENCY / EFFICIENCY REPORT")
    print(L)
    print(f"  Dump dir   : {os.path.abspath(dump_dir)}")
    print(f"  Tech node  : {model['tech_node_nm']:.0f} nm")
    sh = manifest["shapes"]
    print(f"  Workload   : {sh.get('n_matmuls','?')} matmuls, "
          f"{sh['m_input_vectors']} input vectors, "
          f"{sh.get('n_bit_slices',1)} bit-slice(s)")
    print(f"  Clock      : {latres['clock_freq_hz']:.3e} Hz "
          f"(T_clk = {_fmt_time(latres['t_clk_s']).strip()})")
    print("-" * 74)
    print("  OPERATION COUNTS")
    print(f"    DAC conversions : {periph['dac']['n_ops']:>14,d}  @ {periph['dac']['bits']}b")
    print(f"    ADC conversions : {periph['adc']['n_ops']:>14,d}  @ {periph['adc']['bits']}b "
          f"(ENOB {periph['adc']['enob']}, model={periph['adc']['model']})")
    print(f"    TIA conversions : {periph['tia']['n_ops']:>14,d}")
    print(f"    MUX routings    : {mux['n_ops']:>14,d}  "
          f"(DAC-share={mux['shared_dac']}, ADC-share={mux['shared_adc']})")
    print(f"    MAC operations  : {n_mac:>14,d}")
    print("-" * 74)
    print("  ENERGY / OPERATION (model)")
    print(f"    DAC : {_fmt_energy(periph['dac']['e_per_op_J'])} / conv "
          f"[{periph['dac'].get('src','model')}]")
    print(f"    ADC : {_fmt_energy(periph['adc']['e_per_op_J'])} / conv "
          f"[{periph['adc'].get('src','model')}]", end="")
    if periph['adc'].get('src') == 'spice':
        print("  (measured from ADC subckt)")
    elif periph['adc']['model'] == "walden":
        print(f"  (Walden FOM {model['adc_fom_fJ']:.1f} fJ/step; node set by FOM)")
    else:
        print(f"  (Accelergy best-possible, auto-scaled to {model['tech_node_nm']:.0f}nm)")
    print(f"    TIA : {_fmt_energy(periph['tia']['e_per_op_J'])} / conv "
          f"[{periph['tia'].get('src','model')}]"
          + ("  (differential pair)" if periph['tia'].get('differential') else ""))
    if periph['subtractor'].get('present'):
        print(f"    SUB : {_fmt_energy(periph['subtractor']['e_per_op_J'])} / conv "
              f"[{periph['subtractor'].get('src','model')}]")
    print("-" * 74)
    print("  ENERGY BREAKDOWN")
    print(f"    Crossbar (SPICE): {_fmt_energy(E_crossbar)}"
          + ("   [transient only]" if E_crossbar != E_crossbar else ""))
    print(f"    DAC             : {_fmt_energy(periph['dac']['energy_J'])}")
    print(f"    TIA             : {_fmt_energy(periph['tia']['energy_J'])}")
    if periph['subtractor'].get('present'):
        print(f"    SUBTRACTOR      : {_fmt_energy(periph['subtractor']['energy_J'])}")
    print(f"    ADC             : {_fmt_energy(periph['adc']['energy_J'])}")
    print(f"    MUX (sharing)   : {_fmt_energy(mux['energy_J'])}")
    print(f"    Peripheral total: {_fmt_energy(E_periph)}")
    if ic_rep:
        print(f"    Tile lines      : {_fmt_energy(E_lines)}   [shared bus/column wires]")
        print(f"    NoC             : {_fmt_energy(E_noc)}   "
              f"[{noc.get('model', 'none')}]")
    print(f"    SYSTEM TOTAL    : {_fmt_energy(E_total)}")
    if E_total > 0 and E_crossbar == E_crossbar:
        print(f"      (crossbar {100*E_crossbar/E_total:4.1f}%  |  "
              f"peripheral {100*E_periph/E_total:4.1f}%"
              + (f"  |  interconnect {100*E_ic/E_total:4.1f}%" if ic_rep else "")
              + ")")
    print("-" * 74)
    print("  LATENCY (cycle-accurate)")
    cyc = latres["cycles"]
    print(f"    ADC architecture: {latres['adc_type']}  "
          f"({latres['adc_cycles_per_conv']} cyc/conv, {latres['num_adc']} ADCs)")
    print(f"    DAC             : {cyc['dac']:>14,d} cycles  ({latres['num_dac']} DACs"
          + (", muxed" if latres.get('dac_shared') else "") + ")")
    print(f"    MAC / VMM       : {cyc['mac']:>14,d} cycles  "
          f"({latres['vmm_parallel_subarrays']} subarrays in parallel)")
    print(f"    ADC             : {cyc['adc']:>14,d} cycles  ({latres['num_adc']} ADCs"
          + (", muxed" if latres.get('adc_shared') else "") + ")")
    if "noc" in latres["latency_s"]:
        print(f"    NoC             : {_fmt_time(latres['latency_s']['noc']).strip()}"
              + ("  (overlapped: pipeline fill only)" if ic_rep.get("noc_overlap") else ""))
    if measured_lat.get("line_settle_s"):
        print(f"    (MAC settle includes {_fmt_time(measured_lat['line_settle_s']).strip()} "
              f"of shared-line RC)")
    print(f"    TOTAL (serial)  : {cyc['total_serial']:>14,d} cycles "
          f"= {_fmt_time(lat_s).strip()}")
    print(f"    Latency / MAC   : {_fmt_time(latres['latency_per_mac_s']).strip()}")
    print("-" * 74)
    print("  EFFICIENCY")
    print(f"    Energy / MAC    : {fJ_per_mac:9.3f} fJ/MAC")
    print(f"    Throughput      : {gops:9.3f} GOPS")
    print(f"    Efficiency      : {tops_per_w:9.3f} TOPS/W")
    print(L)
    print("  ADC energy: Murmann ADC Survey (Walden FOM) / Accelergy ADC plug-in")
    print("  (Schreier-FOM best-possible). DAC switched-cap, TIA static bias.")
    print("  Tune [PERIPHERALS] / [LATENCY] per technology node.")
    print(L)

    # ── CSV ──
    rows = [
        {"Component": "Crossbar", "Ops": "", "E_per_op_J": "",
         "Energy_J": E_crossbar, "Power_W": P_crossbar, "Cycles": cyc["mac"]},
        {"Component": "DAC", "Ops": periph["dac"]["n_ops"],
         "E_per_op_J": periph["dac"]["e_per_op_J"],
         "Energy_J": periph["dac"]["energy_J"], "Power_W": "", "Cycles": cyc["dac"]},
        {"Component": "ADC", "Ops": periph["adc"]["n_ops"],
         "E_per_op_J": periph["adc"]["e_per_op_J"],
         "Energy_J": periph["adc"]["energy_J"], "Power_W": "", "Cycles": cyc["adc"]},
        {"Component": "TIA", "Ops": periph["tia"]["n_ops"],
         "E_per_op_J": periph["tia"]["e_per_op_J"],
         "Energy_J": periph["tia"]["energy_J"], "Power_W": "", "Cycles": ""},
        {"Component": "SUBTRACTOR", "Ops": periph["subtractor"]["n_ops"],
         "E_per_op_J": periph["subtractor"]["e_per_op_J"],
         "Energy_J": periph["subtractor"]["energy_J"], "Power_W": "", "Cycles": ""},
        {"Component": "MUX", "Ops": mux["n_ops"], "E_per_op_J": "",
         "Energy_J": mux["energy_J"], "Power_W": "", "Cycles": ""},
        {"Component": "Tile_lines", "Ops": "", "E_per_op_J": "",
         "Energy_J": E_lines, "Power_W": "", "Cycles": ""},
        {"Component": "NoC", "Ops": noc.get("bits", ""), "E_per_op_J": "",
         "Energy_J": E_noc, "Power_W": "",
         "Cycles": latres["latency_s"].get("noc", "")},
        {"Component": "Peripheral_total", "Ops": "", "E_per_op_J": "",
         "Energy_J": E_periph, "Power_W": "", "Cycles": ""},
        {"Component": "System_total", "Ops": n_mac, "E_per_op_J": "",
         "Energy_J": E_total, "Power_W": P_total_time,
         "Cycles": cyc["total_serial"]},
        {"Component": "Latency_total_s", "Ops": "", "E_per_op_J": "",
         "Energy_J": "", "Power_W": "", "Cycles": lat_s},
        {"Component": "Latency_per_MAC_s", "Ops": "", "E_per_op_J": "",
         "Energy_J": "", "Power_W": "", "Cycles": latres["latency_per_mac_s"]},
        {"Component": "fJ_per_MAC", "Ops": "", "E_per_op_J": "",
         "Energy_J": fJ_per_mac, "Power_W": "", "Cycles": ""},
        {"Component": "GOPS", "Ops": "", "E_per_op_J": "",
         "Energy_J": gops, "Power_W": "", "Cycles": ""},
        {"Component": "TOPS_per_W", "Ops": "", "E_per_op_J": "",
         "Energy_J": tops_per_w, "Power_W": "", "Cycles": ""},
    ]
    out_path = str(dump / output_name)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    if E_xbar_static and E_crossbar and E_crossbar == E_crossbar:
        frac = E_xbar_static / E_crossbar if E_crossbar else 0.0
        print(f"    Crossbar static  :  {E_xbar_static*1e9:9.3f} nJ  "
              f"(leakage / DC floor = {frac*100:.1f}% of crossbar)")
        print(f"    Crossbar dynamic :  {E_xbar_dyn*1e9:9.3f} nJ  (MVM conduction)")
        if frac > 0.5:
            print("    [warn] The crossbar decks are dominated by STATIC current: "
                  "this is the\n           standard-cell library's leakage "
                  "(BL drivers / decoders / access\n           transistors), not "
                  "compute. Compare nodes on the DYNAMIC term, and\n           "
                  "consider a standard-Vt (not low-Vt) cell library for the array.")

    detail = {
        "energy_J_crossbar_static": E_xbar_static,
        "energy_J_crossbar_dynamic": E_xbar_dyn,
        "crossbar_transient_images": crossbar_images,
        "crossbar_n_images_averaged": len(crossbar_images) if crossbar_images else 1,
        "tech_node_nm": model["tech_node_nm"],
        "shapes": manifest["shapes"],
        "energy_J": {
            "crossbar": E_crossbar, "dac": periph["dac"]["energy_J"],
            "adc": periph["adc"]["energy_J"], "tia": periph["tia"]["energy_J"],
            "subtractor": periph["subtractor"]["energy_J"],
            "mux": mux["energy_J"],
            "peripheral_total": E_periph,
            "tile_lines": E_lines, "noc": E_noc, "interconnect_total": E_ic,
            "system_total": E_total,
        },
        "interconnect": ({
            "crossbar_ideal_rsn_J": (float(ic_rep.get("loading", {})
                                           .get("E_deck_ideal_J", float("nan")))
                                     / n_ic),
            "crossbar_loaded_rsn_J": (float(ic_rep.get("loading", {})
                                            .get("E_deck_loaded_J", float("nan")))
                                      / n_ic),
            "line_energy_J": {k: v / n_ic for k, v in ic_rep.get("loading", {})
                              .get("line_energy_J", {}).items()},
            "line_settle_s": measured_lat.get("line_settle_s", 0.0),
            "noc_model": noc.get("model", "none"),
            "noc_mesh": noc.get("mesh", {}),
            "noc_latency_s": latres["latency_s"].get("noc", 0.0),
            "noc_mean_hops": noc.get("mean_hops", 0.0),
        } if ic_rep else {}),
        "mux": mux,
        "latency": latres,
        "efficiency": {"fJ_per_MAC": fJ_per_mac, "GOPS": gops,
                       "TOPS_per_W": tops_per_w},
        "adc_energy_model": model["adc_energy_model"],
        "model_params": model,
    }
    with open(str(dump / "power_energy_latency_report.json"), "w") as f:
        json.dump(detail, f, indent=2, default=str)

    print(f"\n  Report written to: {out_path}")
    return out_path


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Power/energy/latency/TOPS-W report for SPICECiM-X v3.")
    ap.add_argument("--dump", "-d", required=True,
                    help="Dump dir containing peripheral_ops.json (the per-batch "
                         "imc_traces/<batch> folder).")
    ap.add_argument("--hwsimdb", default=None,
                    help="hwsimdb/<batch> dir with the *_pwr/_eng CSVs "
                         "(default: same as --dump).")
    ap.add_argument("--config", "-c", default="./config.ini")
    ap.add_argument("--out", "-o", default="power_energy_latency_report.csv")
    args = ap.parse_args(argv)
    generate_report(args.dump, args.config, hwsimdb_dir=args.hwsimdb,
                    output_name=args.out)


if __name__ == "__main__":
    main()