'''

Required:

--> module load cadence/2025
--> conda activate spicecim
--> python reram_imc_infer.py --batch_all

Batch-safe version: all working directories (hwsimdb, tmp, imc_traces,
calibration files, simulation artifacts) are scoped by a unique batch ID
derived from --images_dir so that multiple concurrent runs never collide.

'''




import os, sys, glob, math, csv, argparse, json, shutil
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import pandas as pd
import time

import torch
import torch.nn as nn
import torchvision.transforms as T

from utils.utils import *

from utils.dc import process_files_in_folder, process_power_energy_in_folder
from utils.dp import process_crossbar_csv_to_image_csv

# ================================
# CONFIGURATION
# ================================

PW = 1e-9
Trise = 1e-12
Tfall = 1e-12
process_corner = "TT"
VDD = 1.8
VSS = 0
Temperature = 25


synaptic_array_size= (16,16)
golden_input_path = "./supports/_golden_input.scs"
#base_dir = weights_distribution
stdcell_file_path = "./supports/standard_cell.netlist"
device_model_path = "./supports/reram_cell.v"
transistor_model = "/path/to/gpdk180_v3.3/models/spectre/gpdk.scs"  # set to your PDK

Tox = 5.0e-9
Tref = 4.7249e-9
Ik1 = 6.14e-5
Vref = 0.43
Tmax = 4.9e-9
Tmin = 3.3e-9
res_val = 0
cap_val = 0

qbit = None
if qbit:
    print(f"[PTQ] {qbit}-bit quantized block used.")

alpha = Ik1 / Vref
Gmax, Gmin = conductance_calc(alpha, Tox, Tref, Tmax, Tmin)

start_time_tick = 0
increment = 3e-9    # 3*PW + Trise + Tfall

# Choose ADC settings
adc_bits = 16               # e.g., 8, 10, 12
Vmin_TIA, Vmax_TIA = (0, 1)  # typically match your analog design
Vmax_adc= 0.95               # or a bit smaller for headroom
enable_qerror= False


cal = TIACalibrator(p_lo=0.1, p_hi=99.9)

#cal.load("./supports/tia_calibration.json")

# ================================
# BATCH-SAFE DIRECTORY MANAGEMENT
# ================================
# These globals are set once per run in infer_images() based on --images_dir.
# Every function that previously used hardcoded "./hwsimdb", "./tmp", etc.
# now reads from these variables instead.

BATCH_ID       = ""          # e.g. "t1", "t2", ... derived from --images_dir
HWSIMDB_DIR    = "./hwsimdb"
TMP_DIR        = "./tmp"
WORK_DIR       = "."         # cwd-relative root for sim artifacts


def _init_batch_dirs(images_dir: str):
    """
    Derive a unique batch ID from the images_dir path and set all
    working-directory globals so that concurrent runs never collide.
    """
    global BATCH_ID, HWSIMDB_DIR, TMP_DIR, WORK_DIR

    # Use the leaf folder name as the batch ID (e.g. "t1", "t42")
    BATCH_ID = os.path.basename(os.path.normpath(images_dir))
    if not BATCH_ID:
        BATCH_ID = "default"

    HWSIMDB_DIR = os.path.join("./hwsimdb", BATCH_ID)
    TMP_DIR     = os.path.join("./tmp", BATCH_ID)
    WORK_DIR    = "."

    print(f"[batch] ID={BATCH_ID}  hwsimdb={HWSIMDB_DIR}  tmp={TMP_DIR}")


# ================================
# Defaults from user's environment
# ================================
DEFAULT_PROJECT_DIR = "."
DEFAULT_IMAGES_DIR  = "./Test_Image"

# -----------------------------
# Project imports (same folder)
# -----------------------------
def add_project(project_dir: str):
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

def load_project(project_dir: str):
    add_project(project_dir)
    from utils.config import load_cfg
    from utils.main import build_model
    from utils.vit_reram import ViT_ReRAM
    return load_cfg, build_model, ViT_ReRAM

# -----------------------------
# Helpers
# -----------------------------
def _print_tag_progress(tag: str, done: int, total: int, width: int = 30):
    if total <= 0:
        return
    filled = int(width * done / total)
    bar = "█" * filled + "-" * (width - filled)
    # single-line, in-place progress per {tag}
    print(f"\r[{tag}] [{bar}] {done}/{total}", end="", flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv_matrix(path: str, M: np.ndarray, header: bool = True):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["row"] + [f"f{j}" for j in range(M.shape[1])])
        for i in range(M.shape[0]):
            w.writerow([i] + M[i].tolist())

def save_torch_2d(path: str, X: torch.Tensor):
    X = X.detach().float().cpu()
    if X.dim() == 1:
        X = X.unsqueeze(0)
    X2 = X.reshape(X.shape[0], -1).numpy()
    save_csv_matrix(path, X2)

def save_np_2d(path: str, X: np.ndarray):
    if X.ndim == 1:
        X = X[None, :]
    X2 = X.reshape(X.shape[0], -1)
    save_csv_matrix(path, X2)

def save_shapes_json(path: str, d: Dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

# -----------------------------
# ReRAM crossbar primitives
# -----------------------------
def hw_params_from_cfg(hw_cfg: dict):
    Tox  = float(hw_cfg["Tox"])
    Tref = float(hw_cfg["Tref"])
    Ik1  = float(hw_cfg["Ik1"])
    Vref = float(hw_cfg["Vref"])
    Tmax = float(hw_cfg["Tmax"])
    Tmin = float(hw_cfg["Tmin"])
    alpha = Ik1 / Vref
    Gmax = alpha * np.exp(-((Tox - Tmax) / (Tox - Tref)))
    Gmin = alpha * np.exp(-((Tox - Tmin) / (Tox - Tref)))
    return alpha, Gmax, Gmin

def voltage_convert(X: np.ndarray, vmin: float = 0.0, vmax: float = 1.0):
    xmin = X.min()
    xmax = X.max()
    if xmax - xmin < 1e-12:
        V = np.full_like(X, fill_value=vmin, dtype=np.float64)
    else:
        V = vmin + (X - xmin) * (vmax - vmin) / (xmax - xmin)
    return V, xmin, xmax

import re

def first_int_or_default(s, default=float('inf')):
    m = re.search(r'\d+', s)
    return int(m.group()) if m else default



def second_field_first_int_or_default(s, default=float('inf')):
    parts = s.split('_', 2)
    chunk = parts[1] if len(parts) > 1 else ''
    return first_int_or_default(chunk, default)


def load_2d_csv_as_array(path: str) -> np.ndarray:
    """
    Load a CSV whose first row is a header and first column is an index,
    returning ONLY the inner numeric 2D block as a float64 NumPy array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    # Don't let pandas auto-use the first row as header; we'll drop it ourselves
    df = pd.read_csv(path, header=None)

    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError(f"CSV too small to contain header/index: {df.shape}")

    # Drop first row and first column
    data_df = df.iloc[1:, 1:]

    # Coerce any stray text/whitespace to numeric; non-numeric turns into NaN
    data_df = data_df.apply(pd.to_numeric, errors="coerce")

    # Return as float64 (FP64)
    return data_df.to_numpy(dtype=np.float64, copy=False)


# --- Polarity orientation cache (per-layer/tag) ---
_ORIENTATION_CACHE = {}  # maps tag -> +1 (pos-neg) or -1 (neg-pos)

def _pick_orientation(I_pos, I_neg, A, B, DeltaV, DeltaG, tag):
    """
    Decide whether the correct reconstruction is (pos - neg) or (neg - pos)
    by comparing against the ideal current computed from software matrices.

    I_pos, I_neg: 2D arrays of summed crossbar currents [T, Dout] in Amps
    A:            input activations matrix [T, Din]  (software)
    B:            weight matrix [Din, Dout]          (software)
    DeltaV:       scalar (V) used when generating PWL inputs
    DeltaG:       scalar (S) used when mapping weights to conductance
    tag:          string layer identifier ("Q", "K", "V", "W0", "FFN", "HEAD", etc.)
    """
    # If we've already decided an orientation for this tag, reuse it.
    if tag in _ORIENTATION_CACHE:
        return _ORIENTATION_CACHE[tag]

    # Ideal IMC current in Amps (what Spice should produce in perfect hardware):
    #   I_ideal = (ΔV * A) @ (ΔG * B)
    # Shapes: A[T, Din] @ B[Din, Dout] -> [T, Dout]
    I_ideal = (DeltaV * A) @ (DeltaG * B)

    # Two candidate reconstructions from Spice
    I_spice_a = I_pos - I_neg      # assume probes wired so "pos - neg" is correct
    I_spice_b = I_neg - I_pos      # opposite polarity

    # Mean absolute error against the ideal current
    err_a = float(np.mean(np.abs(I_spice_a - I_ideal)))
    err_b = float(np.mean(np.abs(I_spice_b - I_ideal)))

    # Choose the orientation with smaller error
    orientation = +1 if err_a <= err_b else -1
    _ORIENTATION_CACHE[tag] = orientation

    # Debug print 1
    #print(f"[POLARITY] {tag}: err(pos-neg)={err_a:.4e}, err(neg-pos)={err_b:.4e}, "
          #f"chosen={'pos-neg' if orientation==1 else 'neg-pos'}")

    return orientation


def imc_matmul(A: np.ndarray, B: np.ndarray, Gmax: float, Gmin: float,
               dump_dir: str = None, tag: str = "",
               vmin: float = 0.0, vmax: float = 1.0, bname: str = ""):
    """
    Hardware-accurate IMC matmul with optional PTQ.

    A: [m, k]  activations
    B: [k, n]  full-precision weights from PyTorch

    If qbit is not None, we first quantize+dequantize B to B_q and then use
    B_q everywhere (scaling, crossbar mapping, and reconstruction).

    NOTE: Uses HWSIMDB_DIR and TMP_DIR globals for batch-safe file isolation.
    """
    global qbit
    assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]

    # -------------------------------
    # 1) PTQ: quantize B in NUMERIC domain
    # -------------------------------
    B = B.astype(np.float64, copy=False)
    if qbit is not None and qbit > 0:
        levels_pos = 2 ** (qbit - 1) - 1  # e.g., 127 for 8-bit, 32767 for 16-bit
        max_abs = np.max(np.abs(B))
        if max_abs > 1e-12:
            scale = levels_pos / max_abs
            q_int = np.round(B * scale)
            q_int = np.clip(q_int, -levels_pos, levels_pos)
            # dequantize back to float
            B_q = (q_int / scale).astype(np.float64)
        else:
            B_q = np.zeros_like(B, dtype=np.float64)

        print(f"[PTQ] Using {qbit}-bit symmetric quantization for {tag}")
        B = B_q  # from here on, EVERYTHING uses the quantized weights

    # -------------------------------
    # 2) Derive affine scales A->V and B->G using (possibly quantized) B
    # -------------------------------
    eps = 1e-12

    Amax, Amin = A.max(), A.min()
    Bmax, Bmin = B.max(), B.min()

    denA = Amax - Amin
    denB = Bmax - Bmin

    a = (vmax - vmin) / denA if abs(denA) > eps else 1.0
    b = vmin - a * Amin

    c = (Gmax - Gmin) / denB if abs(denB) > eps else 1.0
    d = Gmin - c * Bmin

    V = a * A + b
    G = c * B + d

    I = V @ G  # ideal current (for reference / dumping)

    # Helpers
    m, k = A.shape
    k2, n = B.shape
    assert k == k2

    one_m = np.ones((m, 1))
    one_k = np.ones((k, 1))
    one_n = np.ones((n, 1))

    # Terms to subtract (same algebra as before)
    term1 = a * d * (A @ one_k) @ one_n.T          # = ad * (A @ J_{k×n})
    term2 = b * c * one_m @ (one_k.T @ B)          # = bc * (J_{m×k} @ B)
    term3 = b * d * k * (one_m @ one_n.T)          # = bd * k * J_{m×n}

    # -------------------------------
    # 3) Dump numeric-side matrices and generate PWL
    # -------------------------------
    if dump_dir is not None and tag:
        save_np_2d(os.path.join(dump_dir, f"{tag}_A.csv"), A)
        save_np_2d(os.path.join(dump_dir, f"{tag}_B.csv"), B)
        save_np_2d(os.path.join(dump_dir, f"{tag}_V.csv"), V)
        save_np_2d(os.path.join(dump_dir, f"{tag}_G.csv"), G)
        save_np_2d(os.path.join(dump_dir, f"{tag}_I.csv"), I)
        save_np_2d(os.path.join(dump_dir, f"{tag}_Y.csv"),
                   (I - term1 - term2 - term3) / (a * c))
        save_shapes_json(os.path.join(dump_dir, f"{tag}_shapes.json"),
                         {"A": list(A.shape), "B": list(B.shape),
                          "V": list(V.shape), "G": list(G.shape),
                          "I": list(I.shape),
                          "DeltaV": float(a), "DeltaG": float(c),
                          "vmin": float(vmin), "vmax": float(vmax)})

    total_rows, sim_time = generate_pwl_sources(
        os.path.join(dump_dir, f"{tag}_V.csv"),
        os.path.join(dump_dir, f"{tag}_V_pwl.scs"),
        PW, Trise, Tfall, synaptic_array_size[0]
    )

    # ── BATCH-SAFE: use HWSIMDB_DIR instead of hardcoded "./hwsimdb" ──
    summary = distribute_weights(
        layer_name=f"{tag}",
        layer_weights=B,
        synaptic_array_size=synaptic_array_size,
        pe_size=(2, 2),
        tile_size=(2, 2),
        base_dir=HWSIMDB_DIR
    )

    layer_dir = os.path.join(HWSIMDB_DIR, tag)
    pwl_source = os.path.join(dump_dir, f"{tag}_V_pwl.scs")

    total_npy = sum(
        1
        for _root, _dirs, _files in os.walk(layer_dir, topdown=True)
        for _f in _files
        if _f.endswith(".npy")
    ) * 2
    processed_npy = 0
    ltag = f"[{bname}]-->[{tag}]"
    print(f"{ltag} starting simulations...", flush=True)

    # -------------------------------
    # 4) For each .npy block: map quantized B-block to conductances (continuous),
    #    run SPICE for pos/neg
    # -------------------------------
    for root, dirs, files in os.walk(layer_dir, topdown=True):
        dirs.sort(key=first_int_or_default)
        files.sort(key=second_field_first_int_or_default)

        for file in files:
            if file.endswith(".npy"):
                full_path = os.path.join(root, file)
                weights = np.load(full_path)         # already quantized B-block

                # Use continuous mapping with the SAME 'c' as above.
                Tpos, Tneg = convert_to_conductance(weights, c, Gmin, Tmin, Tox, Tref, alpha)

                weight_mapping(Tpos, Tneg, device_model_path, res_val, cap_val, tmp_dir=TMP_DIR)
                infile_noext = os.path.splitext(full_path)[0]

                # ── BATCH-SAFE: use TMP_DIR instead of hardcoded "./tmp" ──
                siminput_path_pos = os.path.join(os.getcwd(), TMP_DIR, "input_pos.scs")
                siminput_path_neg = os.path.join(os.getcwd(), TMP_DIR, "input_neg.scs")

                processed_npy += 1
                _print_tag_progress(ltag, processed_npy, total_npy)

                run_string = setup_simulation_args_spectrex(siminput_path_pos, infile_noext, '_pos')
                map_primary_input(siminput_path_pos, siminput_path_neg, full_path,
                                  pwl_source, sim_time, synaptic_array_size,
                                  golden_input_path, transistor_model, tmp_dir=TMP_DIR)
                run_sim(run_string, infile_noext)

                processed_npy += 1
                _print_tag_progress(ltag, processed_npy, total_npy)

                run_string = setup_simulation_args_spectrex(siminput_path_neg, infile_noext, '_neg')
                map_primary_input(siminput_path_pos, siminput_path_neg, full_path,
                                  pwl_source, sim_time, synaptic_array_size,
                                  golden_input_path, transistor_model, tmp_dir=TMP_DIR)
                run_sim(run_string, infile_noext)

    process_files_in_folder(
        layer_dir,
        start_time_tick,
        increment,
        f'{dump_dir}/output_currents_{tag}.csv',
        synaptic_array_size[1]
    )

    process_power_energy_in_folder(layer_dir, f'{dump_dir}/_{tag}.csv')

    out_path = f'{dump_dir}/output_currents_{tag}.csv'
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        process_crossbar_csv_to_image_csv(
            out_path, f'{dump_dir}/output_currents_summed_{tag}.csv', vmax
        )
    else:
        print(f"[warn] Skipping {tag}: {out_path} is empty.")

    if total_npy > 0:
        print()

    I_SPICE = load_2d_csv_as_array(f'{dump_dir}/output_currents_summed_{tag}.csv')

    # Align SPICE output to (m, n)
    Ismax, Ismin = I_SPICE.max(), I_SPICE.min()
    m, k = A.shape
    n = B.shape[1]

    if I_SPICE.shape[0] > m:
        I_SPICE = I_SPICE[:m, :]
    if I_SPICE.shape[1] > n:
        I_SPICE = I_SPICE[:, :n]

    if I_SPICE.shape != (m, n):
        raise ValueError(f"[{tag}] SPICE shape {I_SPICE.shape} doesn't match expected {(m, n)}")

    # -------------------------------
    # 5) Polarity fix (same B, now quantized) & TIA+ADC
    # -------------------------------
    Bmax, Bmin = B.max(), B.min()
    denB = Bmax - Bmin
    DeltaG_eff = (Gmax - Gmin) / denB if abs(denB) > 1e-12 else 1.0

    one_m = np.ones((m, 1))
    one_k = np.ones((k, 1))
    term2_only = b * DeltaG_eff * (one_m @ (one_k.T @ B))

    mae_pos = float(np.mean(np.abs((I_SPICE - term2_only) - (a * A) @ (DeltaG_eff * B))))
    mae_neg = float(np.mean(np.abs((-I_SPICE - term2_only) - (a * A) @ (DeltaG_eff * B))))
    if mae_neg < mae_pos:
        I_SPICE = -I_SPICE

    cal.update(tag, I_SPICE)

    # S_B = J @ B
    one_m = np.ones((A.shape[0], 1))
    one_k = np.ones((A.shape[1], 1))
    S_B   = one_m @ (one_k.T @ B)

    # TIA+ADC
    # ── BATCH-SAFE: use TMP_DIR for calibration file ──
    tia_cal_path = os.path.join(TMP_DIR, "tia_calibration.json")
    try:
        Imin_fixed, Imax_fixed = cal.get(tag)
    except KeyError:
        cal.update(tag, I_SPICE)                 # compute from this run
        Imin_fixed, Imax_fixed = cal.get(tag)
        cal.save(tia_cal_path)
    Imin_fixed, Imax_fixed = cal.get(tag)
    Vq, I_hat, meta = tia_adc_realistic(
        I_SPICE, Vmin_TIA, Vmax_TIA, adc_bits,
        Imin=Imin_fixed, Imax=Imax_fixed,
        use_batch_calib=False,            #make it True
        headroom=0.95,
        add_qerror=False, stochastic_rounding=False,
        k_sigma_rel=0, o_sigma_abs=0, apply_mismatch_per="col",
        nonlin=None,
        sigma_v=0,
        invert_with="nominal"
    )

    # Final numeric result
    C_SPICE = (I_hat - (b * c) * S_B) / (a * c)
    #C_SPICE = (I_SPICE - - term1 (b * c) * S_B - term3) / (a * c)

    delta_I = np.max(np.abs(I_hat - I_SPICE))
    delta_C = np.max(np.abs((I_SPICE - (b * c) * S_B) / (a * c) - C_SPICE))
    print(f"[{tag}] max |I_hat - I_SPICE| = {delta_I:.3e}")
    print(f"[{tag}] max |C_curr - C_TIAADC| = {delta_C:.3e}")

    return C_SPICE

# -----------------------------
# Math helpers
# -----------------------------
def layernorm_np(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    Xn = (X - mu) / np.sqrt(var + eps)
    return Xn * gamma + beta

def relu_np(X: np.ndarray):
    return np.maximum(X, 0.0)

def softmax_lastdim_np(X: np.ndarray):
    X_shift = X - X.max(axis=-1, keepdims=True)
    e = np.exp(X_shift)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

# -----------------------------
# ViT weights extraction
# -----------------------------
def extract_vit_weights(model) -> dict:
    W = {}
    # Patch projection Conv2d -> linearized
    W_patch = model.patch_proj.weight.detach().cpu().numpy()  # [D, C, ps, ps]
    D, C, ps, _ = W_patch.shape
    W_patch_lin = W_patch.reshape(D, C * ps * ps).transpose(1, 0)  # (C*ps*ps, D)
    W["patch_proj"] = W_patch_lin

    # Positional & class tokens
    W["pos_embed"] = model.pos_embed.detach().cpu().numpy() if model.pos_embed is not None else None
    W["use_class_token"] = model.cls_token is not None
    W["cls_token"] = model.cls_token.detach().cpu().numpy() if model.cls_token is not None else None

    # Transformer blocks
    blocks = []
    for blk in model.blocks:
        bd = {}
        bd["WQ"] = [lin.weight.detach().cpu().numpy().transpose(1, 0) for lin in blk.mha.WQ]
        bd["WK"] = [lin.weight.detach().cpu().numpy().transpose(1, 0) for lin in blk.mha.WK]
        bd["WV"] = [lin.weight.detach().cpu().numpy().transpose(1, 0) for lin in blk.mha.WV]
        bd["WO"] = blk.mha.WO.weight.detach().cpu().numpy().transpose(1, 0)
        bd["ln1_gamma"] = blk.ln1.weight.detach().cpu().numpy()
        bd["ln1_beta"]  = blk.ln1.bias.detach().cpu().numpy()
        bd["ln2_gamma"] = blk.ln2.weight.detach().cpu().numpy()
        bd["ln2_beta"]  = blk.ln2.bias.detach().cpu().numpy()
        bd["W1"] = blk.mlp.fc1.weight.detach().cpu().numpy().transpose(1, 0)
        bd["b1"] = blk.mlp.fc1.bias.detach().cpu().numpy()
        bd["W2"] = blk.mlp.fc2.weight.detach().cpu().numpy().transpose(1, 0)
        bd["b2"] = blk.mlp.fc2.bias.detach().cpu().numpy()
        blocks.append(bd)
    W["blocks"] = blocks

    # Final norm + head
    W["final_ln_gamma"] = model.norm.weight.detach().cpu().numpy()
    W["final_ln_beta"]  = model.norm.bias.detach().cpu().numpy()
    W["head_W"] = model.head.weight.detach().cpu().numpy().transpose(1, 0)  # (D, C)
    W["head_b"] = model.head.bias.detach().cpu().numpy()
    return W

# -----------------------------
# End-to-end IMC ViT forward
# -----------------------------
def vit_forward_imc_numpy(img_tensor: torch.Tensor,
                          model,
                          hw_cfg: dict,
                          dumps_dir: str,
                          output_classes: int = None) -> Tuple[np.ndarray, dict]:
    """
    img_tensor: torch tensor [1,C,H,W] already preprocessed like training
    Returns (logits_np, extras)
    """
    model.eval()
    with torch.no_grad():
        logits_torch = model(img_tensor).detach().cpu().numpy()

    _, Gmax, Gmin = hw_params_from_cfg(hw_cfg)
    W = extract_vit_weights(model)
    D_model = W["final_ln_gamma"].shape[0]

    # --- Patchify via model op for bit-exactness
    with torch.no_grad():
        x_grid = model.patch_proj(img_tensor)              # [1, D, H/ps, W/ps]
        save_torch_2d(os.path.join(dumps_dir, "01_patch_grid.csv"), x_grid)
        x = x_grid.flatten(2).transpose(1, 2)              # [1, Np, D]
        save_torch_2d(os.path.join(dumps_dir, "02_patch_tokens.csv"), x)

    # Class token
    if W["use_class_token"]:
        cls = torch.from_numpy(W["cls_token"]).to(x.device)  # [1,1,D]
        x = torch.cat([cls, x], dim=1)                       # [1,N,D]
        save_torch_2d(os.path.join(dumps_dir, "03_with_cls.csv"), x)

    # Positional
    if W["pos_embed"] is not None:
        pos = torch.from_numpy(W["pos_embed"][:, :x.size(1), :]).to(x.device)
    else:
        from vit_reram import SinusoidalPositionalEmbedding
        posmod = SinusoidalPositionalEmbedding(D_model, x.size(1)).to(x.device)
        pos = posmod(x)
    save_torch_2d(os.path.join(dumps_dir, "05_positional_embed.csv"), pos)

    # tokens + pos
    x = (x + pos).detach()
    save_torch_2d(os.path.join(dumps_dir, "06_tokens_plus_pos.csv"), x)

    # ----- Voltage convert ONLY once for initial tokens -----
    X_np = x.squeeze(0).contiguous().cpu().numpy()  # [N,D]
    V_np, xmin, xmax = voltage_convert(X_np, vmin=0.0, vmax=1.0)
    save_np_2d(os.path.join(dumps_dir, "07_Vin_tokens.csv"), V_np)

    # ----- Transformer blocks -----
    blocks = W["blocks"]
    tokens = X_np    # numeric domain copy
    # IMPORTANT: For numerically exact parity, we DO NOT feed the voltage-scaled Vin into IMC math.
    # Vin is saved for hardware inspection only. We feed original numeric 'tokens' to IMC matmuls.

    

    for bi, bd in enumerate(blocks):
        blk_dir = os.path.join(dumps_dir, f"block_{bi:02d}")
        ensure_dir(blk_dir)

        # Save per-head weight matrices
        for hi, (WQ, WK, WV) in enumerate(zip(bd["WQ"], bd["WK"], bd["WV"])):
            save_np_2d(os.path.join(blk_dir, f"h{hi}_WQ.csv"), WQ)
            save_np_2d(os.path.join(blk_dir, f"h{hi}_WK.csv"), WK)
            save_np_2d(os.path.join(blk_dir, f"h{hi}_WV.csv"), WV)

        # Q/K/V via IMC
        heads_ctx = []
        for hi in range(len(bd["WQ"])):
            Q = imc_matmul(tokens, bd["WQ"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_Q", bname=f"block_{bi:02d}")
            K = imc_matmul(tokens, bd["WK"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_K", bname=f"block_{bi:02d}")
            Vh = imc_matmul(tokens, bd["WV"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_V",bname=f"block_{bi:02d}")

            # Attention (numeric)
            S = (Q @ K.T) / math.sqrt(bd["WQ"][hi].shape[1])        # [N,N]
            P = softmax_lastdim_np(S)                                # [N,N]
            O = P @ Vh                                              # [N,d_v]

            save_np_2d(os.path.join(blk_dir, f"h{hi}_scores.csv"), S)
            save_np_2d(os.path.join(blk_dir, f"h{hi}_probs.csv"), P)
            save_np_2d(os.path.join(blk_dir, f"h{hi}_context.csv"), O)
            heads_ctx.append(O)

        O_concat = np.concatenate(heads_ctx, axis=-1)               # [N, H*d_v]
        save_np_2d(os.path.join(blk_dir, "attn_concat.csv"), O_concat)
        save_np_2d(os.path.join(blk_dir, "WO.csv"), bd["WO"])

        
        # WO via IMC
        O_proj = imc_matmul(O_concat, bd["WO"], Gmax, Gmin, dump_dir=blk_dir, tag="WO", bname=f"block_{bi:02d}")
        save_np_2d(os.path.join(blk_dir, "attn_out.csv"), O_proj)

        # Residual + LN1
        X1 = layernorm_np(tokens + O_proj, bd["ln1_gamma"], bd["ln1_beta"])
        save_np_2d(os.path.join(blk_dir, "after_ln1.csv"), X1)

        # FFN
        save_np_2d(os.path.join(blk_dir, "W1.csv"), bd["W1"])
        H1 = imc_matmul(X1, bd["W1"], Gmax, Gmin, dump_dir=blk_dir, tag="FFN_W1", bname=f"block_{bi:02d}") + bd["b1"][None, :]
        save_np_2d(os.path.join(blk_dir, "ffn_fc1.csv"), H1)

        A1 = relu_np(H1)
        save_np_2d(os.path.join(blk_dir, "ffn_relu.csv"), A1)

        save_np_2d(os.path.join(blk_dir, "W2.csv"), bd["W2"])
        H2 = imc_matmul(A1, bd["W2"], Gmax, Gmin, dump_dir=blk_dir, tag="FFN_W2", bname=f"block_{bi:02d}") + bd["b2"][None, :]
        save_np_2d(os.path.join(blk_dir, "ffn_fc2.csv"), H2)

        tokens = layernorm_np(X1 + H2, bd["ln2_gamma"], bd["ln2_beta"])
        save_np_2d(os.path.join(blk_dir, "after_ln2.csv"), tokens)

        # For next layer IMC: use numeric tokens directly (linear IMC + exact scaling)

    # Final norm
    Xf = layernorm_np(tokens, W["final_ln_gamma"], W["final_ln_beta"])
    save_np_2d(os.path.join(dumps_dir, "98_final_norm.csv"), Xf)

    # Pool
    if W["use_class_token"]:
        pooled = Xf[0:1, :]   # [1,D]
    else:
        pooled = Xf.mean(axis=0, keepdims=True)
    save_np_2d(os.path.join(dumps_dir, "99_pooled.csv"), pooled)

    # Head via IMC
    save_np_2d(os.path.join(dumps_dir, "head_W.csv"), W["head_W"])
    logits_proj = imc_matmul(pooled, W["head_W"], Gmax, Gmin,
                             dump_dir=dumps_dir, tag="HEAD", bname="CL")

    # If SPICE produced extra columns (e.g., 16), keep only the first K
    if output_classes is not None:
        if output_classes > logits_proj.shape[1]:
            raise ValueError(f"--output_classes ({output_classes}) > HEAD columns ({logits_proj.shape[1]})")
        logits_proj = logits_proj[:, :output_classes]
        head_b = W["head_b"][:output_classes]
    else:
        head_b = W["head_b"]

    logits_np = logits_proj + head_b[None, :]

    save_np_2d(os.path.join(dumps_dir, "logits_imc.csv"), logits_np)
    save_np_2d(os.path.join(dumps_dir, "logits_torch.csv"), logits_torch)

    return logits_np, {"torch_logits": logits_torch}

def vit_forward_imc_numpy_batched(img_tensors: List[torch.Tensor],
                                  model,
                                  hw_cfg: dict,
                                  dumps_dir: str,
                                  output_classes: int = None):
    """
    Time-multiplex ALL images through each IMC matmul (single SPICE run per matmul),
    then split the outputs back per image. Fully robust to unequal token counts.
    Returns:
        logits_list: [np.ndarray of shape (1,C), ...] length == #images
        extras: {"torch_logits": numpy [B,C]}
    """
    model.eval()
    device = next(model.parameters()).device

    # ---------- Software reference logits (true batch) ----------
    with torch.no_grad():
        X = torch.cat([x.to(device) for x in img_tensors], dim=0)  # [B,C,H,W]
        torch_logits = model(X).detach().cpu().numpy()

    # ---------- Build per-image tokens (numeric domain) ----------
    with torch.no_grad():
        x_grid = model.patch_proj(X)                 # [B, D, H/ps, W/ps]
        x      = x_grid.flatten(2).transpose(1, 2)   # [B, N, D]
        if getattr(model, "cls_token", None) is not None:
            cls = model.cls_token.expand(x.size(0), -1, -1)  # [B,1,D]
            x   = torch.cat([cls, x], dim=1)                 # [B,N+1,D]

        if getattr(model, "pos_embed", None) is not None:
            pos = model.pos_embed[:, :x.size(1), :].expand_as(x)
            x = x + pos

    tokens_np = x.detach().cpu().numpy()  # [B,Ni,D] (Ni may vary per image)
    Bsz = tokens_np.shape[0]

    # list of [Ni, D]
    tokens_list = [tokens_np[i] for i in range(Bsz)]
    row_counts  = [t.shape[0] for t in tokens_list]      # [N0, N1, ...]
    total_rows  = int(sum(row_counts))
    cum_edges   = np.cumsum(row_counts)[:-1]             # split edges

    def split_by_counts(arr_2d):
        """
        Split [sum(Ni), M] into [ [N0,M], [N1,M], ... ] using row_counts.
        Tolerate a single trailing extra row (partial final slot) by trimming it.
        """
        rows = arr_2d.shape[0]
        if len(row_counts) == 1:
            if rows == row_counts[0]:
                return [arr_2d]
            if rows == row_counts[0] + 1:
                return [arr_2d[:row_counts[0], :]]
            raise ValueError(f"Expected {row_counts[0]} rows back from IMC, got {rows}")

        total_rows = int(sum(row_counts))
        if rows == total_rows:
            return np.split(arr_2d, np.cumsum(row_counts)[:-1], axis=0)
        if rows == total_rows + 1:
            arr_2d = arr_2d[:total_rows, :]
            return np.split(arr_2d, np.cumsum(row_counts)[:-1], axis=0)

        raise ValueError(f"Expected {total_rows} rows back from IMC, got {rows}")


    # Hardware scaling
    _, Gmax, Gmin = hw_params_from_cfg(hw_cfg)

    # ---------- Pull weights ----------
    W = extract_vit_weights(model)
    blocks = W["blocks"]

    # Work on a per-image list of token matrices, transform block by block
    for bi, bd in enumerate(blocks):
        blk_dir = os.path.join(dumps_dir, "batch", f"block_{bi:02d}")
        ensure_dir(blk_dir)

        # ---- Multi-head Attention: Q/K/V (batched matmuls) ----
        heads_ctx_per_img = [[] for _ in range(Bsz)]

        for hi in range(len(bd["WQ"])):
            A_batch = np.vstack(tokens_list)

            Q_batch = imc_matmul(A_batch, bd["WQ"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_Q", bname=f"block_{bi:02d}")
            K_batch = imc_matmul(A_batch, bd["WK"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_K", bname=f"block_{bi:02d}")
            V_batch = imc_matmul(A_batch, bd["WV"][hi], Gmax, Gmin, dump_dir=blk_dir, tag=f"h{hi}_V", bname=f"block_{bi:02d}")

            Q_chunks = split_by_counts(Q_batch)
            K_chunks = split_by_counts(K_batch)
            V_chunks = split_by_counts(V_batch)

            for b in range(Bsz):
                Qb, Kb, Vb = Q_chunks[b], K_chunks[b], V_chunks[b]
                d_k = Qb.shape[1]
                Sb = (Qb @ Kb.T) / math.sqrt(d_k)
                Pb = softmax_lastdim_np(Sb)
                if Pb.shape[1] != Vb.shape[0]:
                    raise ValueError(f"Head {hi}, block {bi}: softmax rows {Pb.shape} mismatch V rows {Vb.shape}")
                Ob = Pb @ Vb
                heads_ctx_per_img[b].append(Ob)

        O_concat_list = [np.concatenate(heads_ctx_per_img[b], axis=-1) for b in range(Bsz)]
        A_WO = np.vstack(O_concat_list)
        Oproj_batch = imc_matmul(A_WO, bd["WO"], Gmax, Gmin, dump_dir=blk_dir, tag="WO", bname=f"block_{bi:02d}")
        Oproj_chunks = split_by_counts(Oproj_batch)

        X1_list = [
            layernorm_np(tokens_list[b] + Oproj_chunks[b], bd["ln1_gamma"], bd["ln1_beta"])
            for b in range(Bsz)
        ]

        A_W1 = np.vstack(X1_list)
        H1_batch = imc_matmul(A_W1, bd["W1"], Gmax, Gmin, dump_dir=blk_dir, tag="FFN_W1", bname=f"block_{bi:02d}")
        H1_chunks = split_by_counts(H1_batch)
        A1_list = [relu_np(H1_chunks[b] + bd["b1"][None, :]) for b in range(Bsz)]

        A_W2 = np.vstack(A1_list)
        H2_batch = imc_matmul(A_W2, bd["W2"], Gmax, Gmin, dump_dir=blk_dir, tag="FFN_W2", bname=f"block_{bi:02d}")
        H2_chunks = split_by_counts(H2_batch)
        tokens_list = [
            layernorm_np(
                X1_list[b] + (H2_chunks[b] + bd["b2"][None, :]),
                bd["ln2_gamma"], bd["ln2_beta"]
            )
            for b in range(Bsz)
        ]

    Xf_list = [layernorm_np(tokens_list[b], W["final_ln_gamma"], W["final_ln_beta"]) for b in range(Bsz)]
    if W.get("use_class_token", True):
        pooled_list = [Xf[0:1, :] for Xf in Xf_list]
    else:
        pooled_list = [Xf.mean(axis=0, keepdims=True) for Xf in Xf_list]

    A_head = np.vstack(pooled_list)
    head_dir = os.path.join(dumps_dir, "batch")
    logits_proj = imc_matmul(A_head, W["head_W"], Gmax, Gmin, dump_dir=head_dir, tag="HEAD", bname=f"CL")

    if output_classes is not None:
        if output_classes > logits_proj.shape[1]:
            raise ValueError(f"--output_classes ({output_classes}) > HEAD columns ({logits_proj.shape[1]})")
        logits_proj = logits_proj[:, :output_classes]
        head_b = W["head_b"][:output_classes]
    else:
        head_b = W["head_b"]

    logits_proj = logits_proj + head_b[None, :]
    logits_list = [logits_proj[b:b+1, :] for b in range(Bsz)]

    if any(r <= 0 for r in row_counts):
        raise ValueError(f"Empty token list for some images: row_counts={row_counts}")

    return logits_list, {"torch_logits": torch_logits}





# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("ViT ReRAM-IMC Inference (matches PyTorch numerics)")
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT_DIR,
                   help="Path to project root containing main.py/config.py/vit_reram.py")
    p.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR,
                   help="Directory of images to run")
    p.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    p.add_argument("--dump_root", type=str, default="imc_traces",
                   help="Where to save CSV dumps per image (inside project dir if relative)")
    # HW block
    p.add_argument("--Tox", type=float, default=5.0e-9)
    p.add_argument("--Tref", type=float, default=4.7249e-9)
    p.add_argument("--Ik1", type=float, default=6.14e-5)
    p.add_argument("--Vref", type=float, default=0.43)
    p.add_argument("--Tmax", type=float, default=4.9e-9)
    p.add_argument("--Tmin", type=float, default=3.3e-9)
    p.add_argument("--output_classes", type=int, default=None,
                help="Trim final SPICE logits to this many classes (take first columns).")
    p.add_argument("--batch_all", action="store_true",
               help="Time-multiplex all images through each IMC matmul (single SPICE run per matmul).")


    return p.parse_args()

def find_checkpoint(project_dir: str, override: str = ""):
    if override and os.path.exists(override):
        return override
    preferred = [
        os.path.join(project_dir, "outputs", "best.pt"),
        os.path.join(project_dir, "outputs", "last.pt"),
    ]
    for p in preferred:
        if os.path.exists(p):
            return p
    cands = []
    for ext in ("*.pt", "*.pth"):
        cands += glob.glob(os.path.join(project_dir, "**", ext), recursive=True)
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0] if cands else ""

def infer_images():
    args = parse_args()

    # ── BATCH-SAFE: initialize isolated directories from --images_dir ──
    _init_batch_dirs(args.images_dir)

    # Variables: use batch-scoped HWSIMDB_DIR instead of hardcoded "./hwsimdb"
    base_dir = HWSIMDB_DIR

    # Ensure distribution directory is set up
    def setup_distribution_directory(target_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)

    setup_distribution_directory(base_dir)
    setup_distribution_directory(TMP_DIR)

    # ── BATCH-SAFE: clean sim artifacts only for THIS batch's directories ──
    # The old code did glob.glob on "./" which would collide across runs.
    # Now we only clean artifacts that are inside our scoped dirs.
    for pattern in ("*.ahdlSimDB", "*.srf", "*.ic", "*.fc"):
        for f in glob.glob(os.path.join(HWSIMDB_DIR, "**", pattern), recursive=True):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

    top_netlist = generate_subarray(synaptic_array_size[0], synaptic_array_size[1], stdcell_file_path)
    # ── BATCH-SAFE: write seed_netlist into scoped TMP_DIR ──
    seed_netlist_path = os.path.join(TMP_DIR, "seed_netlist.scs")
    with open(seed_netlist_path, 'w') as file:
        file.write(top_netlist)


    load_cfg, build_model, ViT_ReRAM = load_project(args.project)

    # Load cfg + device
    cfg = load_cfg(os.path.join(args.project, "config.ini"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Infer channel count like training
    def infer_in_channels(dataset_name: str, channels: str):
        if channels != "auto": return int(channels)
        ds = dataset_name.lower()
        if ds in ("mnist", "fashion", "fashion_mnist", "mnist-fashion", "fashionmnist"):
            return 1
        elif ds == "cifar10":
            return 3
        else:
            raise ValueError(f"Unknown dataset for auto channels: {dataset_name}")
    in_channels = infer_in_channels(cfg.data.dataset, cfg.data.channels)
    model = build_model(cfg, in_channels=in_channels, num_classes=cfg.model.num_classes).to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = find_checkpoint(args.project, args.checkpoint)
    assert ckpt_path and os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    print(f"[info] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[warn] Missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:print(f"[warn] Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    H, W = cfg.data.img_height, cfg.data.img_width
    ops = [T.Resize((H, W))]
    if in_channels == 1:
        ops += [T.Grayscale(num_output_channels=1)]
    else:
        # Enforce 3ch: identity for RGB, expands grayscale-saved images.
        ops += [T.Lambda(lambda im: im.convert("RGB"))]
    ops += [T.ToTensor()]
    preprocess = T.Compose(ops)

    # HW params dict
    hw_cfg = {
        "Tox": args.Tox, "Tref": args.Tref, "Ik1": args.Ik1, "Vref": args.Vref,
        "Tmax": args.Tmax, "Tmin": args.Tmin,
    }
    out_classes = args.output_classes or cfg.model.num_classes


    # Collect images
    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        img_files.extend(glob.glob(os.path.join(args.images_dir, "**", ext), recursive=True))
    img_files.sort()
    assert img_files, f"No images found under: {args.images_dir}"

    # ── BATCH-SAFE: scope dump_root by BATCH_ID ──
    dump_root = args.dump_root
    if not os.path.isabs(dump_root):
        dump_root = os.path.join(args.project, dump_root)
    # Append batch ID so each run gets its own trace directory
    dump_root = os.path.join(dump_root, BATCH_ID)
    ensure_dir(dump_root)

    softmax = nn.Softmax(dim=1)

    if args.batch_all:
        # Preprocess all images → list of tensors
        imgs = []
        names = []
        for fpath in img_files:
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception as e:
                print(f"{os.path.basename(fpath)} --> Skipped (cannot open): {e}")
                continue
            x = preprocess(img).unsqueeze(0)  # [1,C,H,W]
            imgs.append(x)
            names.append(os.path.basename(fpath))

        # Single batched IMC forward
        logits_list, extras = vit_forward_imc_numpy_batched(
            img_tensors=imgs, model=model, hw_cfg=hw_cfg,
            dumps_dir=dump_root, output_classes=out_classes
        )
        torch_logits = extras["torch_logits"]  # [B,C] numpy

        # ── BATCH-SAFE: write report into scoped TMP_DIR ──
        report_path = os.path.join(TMP_DIR, "Final_report.txt")
        report_lines = []

        # Per-image reporting, preserving your CSVs for compatibility
        for i, name in enumerate(names):
            base = os.path.splitext(name)[0]
            dump_dir = os.path.join(dump_root, base)
            ensure_dir(dump_dir)

            probs_torch = softmax(torch.from_numpy(torch_logits[i:i+1])).numpy()
            probs_imc   = softmax(torch.from_numpy(logits_list[i])).numpy()

            pred_torch = int(np.argmax(probs_torch, axis=1)[0])
            pred_imc   = int(np.argmax(probs_imc, axis=1)[0])
            p_torch    = float(np.max(probs_torch))
            p_imc      = float(np.max(probs_imc))

            save_np_2d(os.path.join(dump_dir, "softmax_torch.csv"), probs_torch)
            save_np_2d(os.path.join(dump_dir, "softmax_imc.csv"),   probs_imc)
            save_np_2d(os.path.join(dump_dir, "logits_imc.csv"),    logits_list[i])
            save_np_2d(os.path.join(dump_dir, "logits_torch.csv"),  torch_logits[i:i+1])

            prediction = f"{name} | torch: {pred_torch} (p={p_torch:.3f}) | imc: {pred_imc} (p={p_imc:.3f}) | match={pred_torch==pred_imc}"
            print(prediction)
            report_lines.append(prediction)

        with open(report_path, 'w') as file:
            file.write("\n".join(report_lines))

    else:
        # === existing single-image loop, unchanged except for scoped paths ===
        device = next(model.parameters()).device
        for fpath in img_files:
            name = os.path.basename(fpath)
            base = os.path.splitext(name)[0]
            dump_dir = os.path.join(dump_root, base)
            ensure_dir(dump_dir)

            # Open & preprocess
            try:
                img = Image.open(fpath).convert("RGB")
            except Exception as e:
                print(f"{name} --> Skipped (cannot open): {e}")
                continue
            x = preprocess(img).unsqueeze(0)  # [1,C,H,W]

            # Torch forward (reference)
            with torch.no_grad():
                torch_logits_np = model(x.to(device)).detach().cpu().numpy()

            # IMC forward (single image)
            res = vit_forward_imc_numpy(
                img_tensor=x, model=model, hw_cfg=hw_cfg,
                dumps_dir=dump_dir, output_classes=out_classes
            )
            if isinstance(res, tuple):
                imc_logits_np = res[0]
            else:
                imc_logits_np = res

            # Softmax & predictions
            probs_torch = softmax(torch.from_numpy(torch_logits_np)).numpy()
            probs_imc   = softmax(torch.from_numpy(imc_logits_np)).numpy()

            pred_torch = int(np.argmax(probs_torch, axis=1)[0])
            pred_imc   = int(np.argmax(probs_imc, axis=1)[0])
            p_torch    = float(np.max(probs_torch))
            p_imc      = float(np.max(probs_imc))

            # Persist per-image CSVs
            save_np_2d(os.path.join(dump_dir, "softmax_torch.csv"), probs_torch)
            save_np_2d(os.path.join(dump_dir, "softmax_imc.csv"),   probs_imc)
            save_np_2d(os.path.join(dump_dir, "logits_imc.csv"),    imc_logits_np)
            save_np_2d(os.path.join(dump_dir, "logits_torch.csv"),  torch_logits_np)

            prediction=f"{name} | torch: {pred_torch} (p={p_torch:.3f}) | imc: {pred_imc} (p={p_imc:.3f}) | match={pred_torch==pred_imc}"
            print(prediction)
            # ── BATCH-SAFE: write report into scoped TMP_DIR ──
            report_path = os.path.join(TMP_DIR, "Final_report.txt")
            with open(report_path, 'a') as file:
                file.write(prediction + "\n")


if __name__ == "__main__":
    _t0 = time.time()
    infer_images()
    _elapsed = time.time() - _t0
    # human-friendly final runtime report
    hrs = int(_elapsed // 3600)
    mins = int((_elapsed % 3600) // 60)
    secs = _elapsed % 60
    if hrs > 0:
        print(f"[done] Total runtime: {hrs}h {mins}m {secs:.1f}s")
    elif mins > 0:
        print(f"[done] Total runtime: {mins}m {secs:.1f}s")
    else:
        print(f"[done] Total runtime: {secs:.1f}s")