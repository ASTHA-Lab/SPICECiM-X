"""
PyTorch ↔ numpy bridge for SPICECiM-X v3.

Extracts the trained weights from a ``ViT_ReRAM`` model into a plain
dict-of-numpy structure that the v3 ``vit_forward_imc`` orchestrator can
walk without holding any torch references during the SPICE phase.

Mirrors v2's ``extract_vit_weights`` byte-for-byte (so the v2 trained
checkpoints load identically), but lives inside the spicecimx package so
that downstream code doesn't have to do path gymnastics with
``sys.path.insert(project_dir)`` the way v2's ``reram_imc_infer.py``
does.
"""
from __future__ import annotations

import os
import sys
import glob
import importlib
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dynamic project import — mirrors v2's load_project()
# ---------------------------------------------------------------------------
def load_project_modules(project_dir: str):
    """
    Make the user's trainer project importable, then return
    (load_cfg, build_model, ViT_ReRAM).

    The project layout is the same as v2's:
        <project_dir>/
            config.py        (load_cfg)
            main.py          (build_model)
            vit_reram.py     (ViT_ReRAM)
            utils/           (or with utils.{config,main,vit_reram})

    We support both "flat" (top-level) and v2-style "utils.*" layouts.
    """
    project_dir = os.path.abspath(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    # Try v2-style utils.* first, then flat layout
    load_cfg = build_model = ViT_ReRAM = None

    for mod_path, sym in (
        ("utils.config", "load_cfg"),
        ("config", "load_cfg"),
    ):
        try:
            m = importlib.import_module(mod_path)
            load_cfg = getattr(m, sym)
            break
        except (ImportError, AttributeError):
            continue

    for mod_path, sym in (
        ("utils.main", "build_model"),
        ("main", "build_model"),
    ):
        try:
            m = importlib.import_module(mod_path)
            build_model = getattr(m, sym)
            break
        except (ImportError, AttributeError):
            continue

    for mod_path, sym in (
        ("utils.vit_reram", "ViT_ReRAM"),
        ("vit_reram", "ViT_ReRAM"),
    ):
        try:
            m = importlib.import_module(mod_path)
            ViT_ReRAM = getattr(m, sym)
            break
        except (ImportError, AttributeError):
            continue

    missing = [n for n, v in (("load_cfg", load_cfg),
                              ("build_model", build_model),
                              ("ViT_ReRAM", ViT_ReRAM)) if v is None]
    if missing:
        raise ImportError(
            f"Could not import {missing} from project_dir={project_dir!r}. "
            f"Expected either utils.* or top-level config.py / main.py / vit_reram.py."
        )
    return load_cfg, build_model, ViT_ReRAM


def find_checkpoint(project_dir: str, override: str = "",
                    out_dir: str = "") -> str:
    """
    Locate a trained checkpoint.  Priority order:
        1. explicit --checkpoint override — must exist, else FileNotFoundError
        2. <out_dir>/best.pt, <out_dir>/last.pt   — the trainer's configured
           [SAVE] out_dir (resolved against project_dir when relative), so
           changing out_dir in config.ini is honoured WITHOUT --checkpoint
        3. <project>/outputs/best.pt, <project>/outputs/last.pt  (legacy)

    Returns "" when nothing is found.  There is deliberately no "newest *.pt
    anywhere" fallback: it could silently pick a checkpoint from another
    experiment.
    """
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"--checkpoint {override!r} does not exist")
        return override
    preferred = []
    if out_dir:
        od = out_dir if os.path.isabs(out_dir) else os.path.join(project_dir, out_dir)
        preferred += [os.path.join(od, "best.pt"), os.path.join(od, "last.pt")]
    preferred += [
        os.path.join(project_dir, "outputs", "best.pt"),
        os.path.join(project_dir, "outputs", "last.pt"),
    ]
    for p in preferred:
        if os.path.exists(p):
            return p
    return ""


def load_state_dict_file(path: str) -> Dict[str, Any]:
    """Load a checkpoint's state dict on CPU, unwrapping ``{"state_dict": ...}``
    and the ``module.`` (DataParallel) / ``_orig_mod.`` (torch.compile) key
    prefixes."""
    import torch
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Weight extraction — port of v2's extract_vit_weights, kept byte-identical
# ---------------------------------------------------------------------------
def extract_vit_weights(model) -> Dict[str, Any]:
    """
    Convert a ``ViT_ReRAM`` torch model into a dict of numpy arrays
    suitable for the v3 IMC forward pass.

    Returned dict keys (same as v2):
        patch_proj           (C*ps*ps, D)
        pos_embed            (1, N, D) or None
        use_class_token      bool
        cls_token            (1, 1, D) or None
        blocks               list of per-block dicts:
            WQ, WK, WV       list[H] of (D, d_k) / (D, d_v) ndarrays
            WO               (H*d_v, D)
            ln1_gamma, ln1_beta, ln2_gamma, ln2_beta   (D,)
            W1               (D, d_ff)
            b1               (d_ff,)
            W2               (d_ff, D)
            b2               (D,)
        final_ln_gamma, final_ln_beta   (D,)
        head_W               (D, num_classes)
        head_b               (num_classes,)
    """
    W: Dict[str, Any] = {}

    # Patch projection Conv2d → linearised
    W_patch = model.patch_proj.weight.detach().cpu().numpy()       # [D, C, ps, ps]
    D, C, ps, _ = W_patch.shape
    W["patch_proj"] = W_patch.reshape(D, C * ps * ps).transpose(1, 0)

    # Positional & class tokens
    W["pos_embed"]       = (model.pos_embed.detach().cpu().numpy()
                            if model.pos_embed is not None else None)
    W["use_class_token"] = model.cls_token is not None
    W["cls_token"]       = (model.cls_token.detach().cpu().numpy()
                            if model.cls_token is not None else None)

    # Transformer blocks
    blocks: List[Dict[str, Any]] = []
    for blk in model.blocks:
        bd: Dict[str, Any] = {}
        bd["WQ"] = [lin.weight.detach().cpu().numpy().transpose(1, 0)
                    for lin in blk.mha.WQ]
        bd["WK"] = [lin.weight.detach().cpu().numpy().transpose(1, 0)
                    for lin in blk.mha.WK]
        bd["WV"] = [lin.weight.detach().cpu().numpy().transpose(1, 0)
                    for lin in blk.mha.WV]
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
    W["head_W"] = model.head.weight.detach().cpu().numpy().transpose(1, 0)
    W["head_b"] = model.head.bias.detach().cpu().numpy()

    return W


# ---------------------------------------------------------------------------
# Image preprocessing — preserves v2 behaviour
# ---------------------------------------------------------------------------
def build_preprocess(cfg, in_channels: int = 0) -> Any:
    """
    Build a torchvision transform matching the trainer's config so that
    inference inputs are pre-processed identically to training.

    ``in_channels`` is the channel count the model was built with.  The
    transform ALWAYS enforces it (1 -> Grayscale, 3 -> convert("RGB")), even
    when [DATA] channels = auto — test images are often saved as RGB PNGs
    regardless of the source dataset, so skipping the channel op (the old
    behaviour for "auto") fed 3-channel tensors into 1-channel models.
    When ``in_channels`` is 0 it is resolved from the config (explicit int,
    or the same MNIST/Fashion->1 else 3 heuristic used to build the model).
    """
    import torchvision.transforms as T

    img_size = (cfg.data.img_height, cfg.data.img_width)
    ops: List[Any] = [T.Resize(img_size)]
    ch = int(in_channels)
    if ch <= 0:
        channels = cfg.data.channels
        if isinstance(channels, str) and channels != "auto":
            ch = int(channels)
        elif isinstance(channels, int):
            ch = channels
        else:  # auto: mirror the model-building heuristic
            name = str(getattr(cfg.data, "dataset", "")).lower()
            ch = 1 if ("mnist" in name or "fashion" in name) else 3
    if ch == 1:
        ops.append(T.Grayscale(num_output_channels=1))
    else:
        # NOT Grayscale(3): that converts colour images to grayscale.
        ops.append(T.Lambda(lambda im: im.convert("RGB")))
    ops.append(T.ToTensor())
    return T.Compose(ops)