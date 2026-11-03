"""
End-to-end ViT IMC forward pass for SPICECiM-X v3.

This module is the v3 equivalent of v2's ``vit_forward_imc_numpy`` and
``vit_forward_imc_numpy_batched``: it walks the ViT computation in numpy
and delegates every linear layer to the v3 ``imc_matmul`` kernel, which
in turn dispatches to the configured SPICE backend (transient / DC /
hybrid).

The math is preserved bit-for-bit from v2 (so existing trained
checkpoints reproduce v2-style numerics in the soft layers) — what
changes is the inner ``imc_matmul`` (full three-term affine de-mapping,
parallel SPICE, proper settled-current extraction).
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Soft numerical layers — identical to v2
# ---------------------------------------------------------------------------
def _layernorm_np(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    Xh = (X - mu) / np.sqrt(var + eps)
    return Xh * gamma + beta


def _relu_np(X: np.ndarray) -> np.ndarray:
    return np.maximum(X, 0.0)


def _softmax_lastdim_np(X: np.ndarray) -> np.ndarray:
    Xs = X - X.max(axis=-1, keepdims=True)
    e = np.exp(Xs)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Lightweight CSV dumping (only used when dump_dir is provided)
# ---------------------------------------------------------------------------
def _ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def _save_np_2d(path: str, X: np.ndarray) -> None:
    if X.ndim == 1:
        X = X[None, :]
    X2 = X.reshape(X.shape[0], -1)
    _ensure_dir(os.path.dirname(path))
    # Use numpy savetxt for speed; one row per matrix row
    np.savetxt(path, X2, delimiter=",", fmt="%.10g")


# ---------------------------------------------------------------------------
# Single-image forward pass
# ---------------------------------------------------------------------------
def vit_forward_imc(
    img_tensor,           # torch.Tensor [1, C, H, W]
    model,                # ViT_ReRAM
    *,
    Gmax:           float,
    Gmin:           float,
    imc_kernel,           # callable from spicecimx.core.imc_matmul
    sim_backend,          # callable produced by spicecimx.sim.make_backend
    polarity_cache,       # spicecimx.core.imc_matmul.PolarityCache
    tia_apply,            # callable (I, tag) -> (V_quant, I_hat, meta)
    dumps_dir:      str,
    output_classes: Optional[int] = None,
    save_intermediates: bool = True,
    weights_dict:   Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run one image through the IMC ViT.

    Parameters
    ----------
    img_tensor
        [1, C, H, W] torch tensor, preprocessed exactly like training.
    model
        Trained ``ViT_ReRAM`` (used for patch projection and a reference
        torch forward).
    Gmax, Gmin
        ReRAM conductance window.
    imc_kernel
        ``spicecimx.core.imc_matmul.imc_matmul`` callable.
    sim_backend, polarity_cache, tia_apply
        Plumbing for ``imc_kernel`` — built once per inference run.
    dumps_dir
        Per-image directory for CSV diagnostics.
    output_classes
        Trim final logits to first K columns (matches v2).
    save_intermediates
        Set False to skip CSV dumps for speed during sweeps.
    weights_dict
        Pre-extracted weight dict from ``extract_vit_weights``.  If None,
        we extract here (slower if called many times).

    Returns
    -------
    logits_np : (1, num_classes)
    extras    : dict with at least ``torch_logits``
    """
    import torch
    from spicecimx.io.torch_bridge import extract_vit_weights

    model.eval()
    with torch.no_grad():
        logits_torch = model(img_tensor).detach().cpu().numpy()

    if weights_dict is None:
        weights_dict = extract_vit_weights(model)
    W = weights_dict
    D_model = W["final_ln_gamma"].shape[0]

    # Patchify via the model op for bit-exactness with training
    with torch.no_grad():
        x_grid = model.patch_proj(img_tensor)              # [1, D, H/ps, W/ps]
        if save_intermediates and dumps_dir:
            _save_np_2d(os.path.join(dumps_dir, "01_patch_grid.csv"),
                        x_grid.detach().cpu().numpy().reshape(x_grid.shape[0], -1))
        x = x_grid.flatten(2).transpose(1, 2)              # [1, Np, D]
        if save_intermediates and dumps_dir:
            _save_np_2d(os.path.join(dumps_dir, "02_patch_tokens.csv"),
                        x.detach().cpu().numpy().reshape(-1, x.shape[-1]))

    # Class token
    if W["use_class_token"]:
        cls = torch.from_numpy(W["cls_token"]).to(x.device)
        x = torch.cat([cls, x], dim=1)                     # [1, N, D]

    # Positional embedding
    if W["pos_embed"] is not None:
        pos = torch.from_numpy(W["pos_embed"][:, :x.size(1), :]).to(x.device)
    else:
        # Sinusoidal fallback — import lazily to avoid forcing torch in tests
        try:
            from vit_reram import SinusoidalPositionalEmbedding
        except ImportError:
            from utils.vit_reram import SinusoidalPositionalEmbedding
        posmod = SinusoidalPositionalEmbedding(D_model, x.size(1)).to(x.device)
        pos = posmod(x)

    x = (x + pos).detach()

    # Convert to numpy for the IMC walk
    X_np = x.squeeze(0).contiguous().cpu().numpy()         # [N, D]
    tokens = X_np

    if save_intermediates and dumps_dir:
        _save_np_2d(os.path.join(dumps_dir, "06_tokens_plus_pos.csv"), tokens)

    # ── Transformer blocks ────────────────────────────────────────────────
    for bi, bd in enumerate(W["blocks"]):
        blk_dir = os.path.join(dumps_dir, f"block_{bi:02d}") if dumps_dir else ""
        if blk_dir:
            _ensure_dir(blk_dir)

        # Q / K / V per head
        heads_ctx: List[np.ndarray] = []
        for hi in range(len(bd["WQ"])):
            Q  = imc_kernel(tokens, bd["WQ"][hi], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag=f"h{hi}_Q",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)
            K  = imc_kernel(tokens, bd["WK"][hi], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag=f"h{hi}_K",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)
            Vh = imc_kernel(tokens, bd["WV"][hi], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag=f"h{hi}_V",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)

            S = (Q @ K.T) / math.sqrt(bd["WQ"][hi].shape[1])
            P = _softmax_lastdim_np(S)
            O = P @ Vh

            if save_intermediates and blk_dir:
                _save_np_2d(os.path.join(blk_dir, f"h{hi}_scores.csv"), S)
                _save_np_2d(os.path.join(blk_dir, f"h{hi}_probs.csv"),  P)
                _save_np_2d(os.path.join(blk_dir, f"h{hi}_context.csv"), O)
            heads_ctx.append(O)

        O_concat = np.concatenate(heads_ctx, axis=-1)
        # WO projection
        O_proj = imc_kernel(O_concat, bd["WO"], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag="WO",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)

        X1 = _layernorm_np(tokens + O_proj, bd["ln1_gamma"], bd["ln1_beta"])

        # FFN
        H1 = imc_kernel(X1, bd["W1"], Gmax=Gmax, Gmin=Gmin,
                        sim_backend=sim_backend, polarity_cache=polarity_cache,
                        tia_apply=tia_apply, tag="FFN_W1",
                        bname=f"block_{bi:02d}", dump_dir=blk_dir) + bd["b1"][None, :]
        A1 = _relu_np(H1)
        H2 = imc_kernel(A1, bd["W2"], Gmax=Gmax, Gmin=Gmin,
                        sim_backend=sim_backend, polarity_cache=polarity_cache,
                        tia_apply=tia_apply, tag="FFN_W2",
                        bname=f"block_{bi:02d}", dump_dir=blk_dir) + bd["b2"][None, :]

        tokens = _layernorm_np(X1 + H2, bd["ln2_gamma"], bd["ln2_beta"])

        if save_intermediates and blk_dir:
            _save_np_2d(os.path.join(blk_dir, "after_ln2.csv"), tokens)

    # Final norm + pool
    Xf = _layernorm_np(tokens, W["final_ln_gamma"], W["final_ln_beta"])
    pooled = Xf[0:1, :] if W["use_class_token"] else Xf.mean(axis=0, keepdims=True)

    # Classification head via IMC
    logits_proj = imc_kernel(pooled, W["head_W"], Gmax=Gmax, Gmin=Gmin,
                             sim_backend=sim_backend, polarity_cache=polarity_cache,
                             tia_apply=tia_apply, tag="HEAD",
                             bname="CL", dump_dir=dumps_dir)

    if output_classes is not None:
        if output_classes > logits_proj.shape[1]:
            raise ValueError(
                f"output_classes ({output_classes}) > HEAD columns ({logits_proj.shape[1]})"
            )
        logits_proj = logits_proj[:, :output_classes]
        head_b = W["head_b"][:output_classes]
    else:
        head_b = W["head_b"]

    logits_np = logits_proj + head_b[None, :]

    if save_intermediates and dumps_dir:
        _save_np_2d(os.path.join(dumps_dir, "logits_imc.csv"),   logits_np)
        _save_np_2d(os.path.join(dumps_dir, "logits_torch.csv"), logits_torch)

    return logits_np, {"torch_logits": logits_torch}


# ---------------------------------------------------------------------------
# Batched forward pass (time-multiplexed) — one SPICE run per matmul
# ---------------------------------------------------------------------------
def vit_forward_imc_batched(
    img_tensors:    List,       # list of torch.Tensor [1, C, H, W]
    model,
    *,
    Gmax:           float,
    Gmin:           float,
    imc_kernel,
    sim_backend,
    polarity_cache,
    tia_apply,
    dumps_dir:      str,
    output_classes: Optional[int] = None,
    save_intermediates: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Time-multiplex all images through each IMC matmul: a single SPICE run
    per weight matrix covers all images by stacking their token activations
    row-wise.  This is much cheaper than running one full pass per image
    when the dataset is small (e.g. validation sweeps).

    Returns
    -------
    logits_list : list of (1, num_classes) per image
    extras      : dict with batched ``torch_logits`` of shape (B, num_classes)
    """
    import torch
    from spicecimx.io.torch_bridge import extract_vit_weights

    model.eval()
    Bsz = len(img_tensors)
    with torch.no_grad():
        torch_logits = np.vstack([
            model(img).detach().cpu().numpy() for img in img_tensors
        ])

    W = extract_vit_weights(model)
    D_model = W["final_ln_gamma"].shape[0]

    # Patchify per image so we can keep per-image row counts (varying with
    # use_class_token + img sizes).
    patch_token_lists: List[np.ndarray] = []
    for img in img_tensors:
        with torch.no_grad():
            xg = model.patch_proj(img)
            xt = xg.flatten(2).transpose(1, 2)
            if W["use_class_token"]:
                cls = torch.from_numpy(W["cls_token"]).to(xt.device)
                xt = torch.cat([cls, xt], dim=1)
            if W["pos_embed"] is not None:
                pos = torch.from_numpy(W["pos_embed"][:, :xt.size(1), :]).to(xt.device)
                xt = xt + pos
            else:
                try:
                    from vit_reram import SinusoidalPositionalEmbedding
                except ImportError:
                    from utils.vit_reram import SinusoidalPositionalEmbedding
                posmod = SinusoidalPositionalEmbedding(D_model, xt.size(1)).to(xt.device)
                xt = xt + posmod(xt)
        patch_token_lists.append(xt.squeeze(0).detach().cpu().numpy())

    row_counts = [t.shape[0] for t in patch_token_lists]

    def _split_by_counts(M: np.ndarray) -> List[np.ndarray]:
        out, off = [], 0
        for c in row_counts:
            out.append(M[off:off + c])
            off += c
        return out

    tokens_list = patch_token_lists  # numeric domain copy

    for bi, bd in enumerate(W["blocks"]):
        blk_dir = os.path.join(dumps_dir, "batch", f"block_{bi:02d}")
        _ensure_dir(blk_dir)

        # Stack all images' tokens row-wise, run one IMC matmul per (Q,K,V,WO,W1,W2)
        A = np.vstack(tokens_list)

        heads_out: List[List[np.ndarray]] = []  # per-head, per-image O matrices
        for hi in range(len(bd["WQ"])):
            Q_all  = imc_kernel(A, bd["WQ"][hi], Gmax=Gmax, Gmin=Gmin,
                                sim_backend=sim_backend, polarity_cache=polarity_cache,
                                tia_apply=tia_apply, tag=f"h{hi}_Q",
                                bname=f"block_{bi:02d}", dump_dir=blk_dir)
            K_all  = imc_kernel(A, bd["WK"][hi], Gmax=Gmax, Gmin=Gmin,
                                sim_backend=sim_backend, polarity_cache=polarity_cache,
                                tia_apply=tia_apply, tag=f"h{hi}_K",
                                bname=f"block_{bi:02d}", dump_dir=blk_dir)
            V_all  = imc_kernel(A, bd["WV"][hi], Gmax=Gmax, Gmin=Gmin,
                                sim_backend=sim_backend, polarity_cache=polarity_cache,
                                tia_apply=tia_apply, tag=f"h{hi}_V",
                                bname=f"block_{bi:02d}", dump_dir=blk_dir)
            Q_split = _split_by_counts(Q_all)
            K_split = _split_by_counts(K_all)
            V_split = _split_by_counts(V_all)

            per_img_O: List[np.ndarray] = []
            for b in range(Bsz):
                S = Q_split[b] @ K_split[b].T / math.sqrt(bd["WQ"][hi].shape[1])
                P = _softmax_lastdim_np(S)
                per_img_O.append(P @ V_split[b])
            heads_out.append(per_img_O)

        # Concat heads per image, then WO via IMC
        O_concat_list = [
            np.concatenate([heads_out[hi][b] for hi in range(len(bd["WQ"]))], axis=-1)
            for b in range(Bsz)
        ]
        A_O = np.vstack(O_concat_list)
        O_proj_all = imc_kernel(A_O, bd["WO"], Gmax=Gmax, Gmin=Gmin,
                                sim_backend=sim_backend, polarity_cache=polarity_cache,
                                tia_apply=tia_apply, tag="WO",
                                bname=f"block_{bi:02d}", dump_dir=blk_dir)
        O_proj_split = _split_by_counts(O_proj_all)

        X1_list = [
            _layernorm_np(tokens_list[b] + O_proj_split[b], bd["ln1_gamma"], bd["ln1_beta"])
            for b in range(Bsz)
        ]

        A_W1 = np.vstack(X1_list)
        H1_all = imc_kernel(A_W1, bd["W1"], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag="FFN_W1",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)
        H1_split = _split_by_counts(H1_all)
        A1_list = [_relu_np(H1_split[b] + bd["b1"][None, :]) for b in range(Bsz)]

        A_W2 = np.vstack(A1_list)
        H2_all = imc_kernel(A_W2, bd["W2"], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag="FFN_W2",
                            bname=f"block_{bi:02d}", dump_dir=blk_dir)
        H2_split = _split_by_counts(H2_all)

        tokens_list = [
            _layernorm_np(
                X1_list[b] + H2_split[b] + bd["b2"][None, :],
                bd["ln2_gamma"], bd["ln2_beta"],
            )
            for b in range(Bsz)
        ]

    # Final norm + pool, then head
    Xf_list = [_layernorm_np(t, W["final_ln_gamma"], W["final_ln_beta"])
               for t in tokens_list]
    if W["use_class_token"]:
        pooled_list = [Xf[0:1, :] for Xf in Xf_list]
    else:
        pooled_list = [Xf.mean(axis=0, keepdims=True) for Xf in Xf_list]

    A_head = np.vstack(pooled_list)
    head_dir = os.path.join(dumps_dir, "batch")
    _ensure_dir(head_dir)
    logits_all = imc_kernel(A_head, W["head_W"], Gmax=Gmax, Gmin=Gmin,
                            sim_backend=sim_backend, polarity_cache=polarity_cache,
                            tia_apply=tia_apply, tag="HEAD",
                            bname="CL", dump_dir=head_dir)
    if output_classes is not None:
        logits_all = logits_all[:, :output_classes]
        head_b = W["head_b"][:output_classes]
    else:
        head_b = W["head_b"]
    logits_all = logits_all + head_b[None, :]
    logits_list = [logits_all[b:b + 1, :] for b in range(Bsz)]

    return logits_list, {"torch_logits": torch_logits}
