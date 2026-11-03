import math
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        # odd dim: ceil(dim/2) sin columns but only floor(dim/2) cos columns
        pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class MultiHeadAttentionReRAM(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int = 0, d_v: int = 0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0 or (d_k > 0 and d_v > 0), "d_model must be divisible by n_heads if d_k/d_v not given"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k > 0 else d_model // n_heads
        self.d_v = d_v if d_v > 0 else d_model // n_heads
        self.WQ = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.WK = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.WV = nn.ModuleList([nn.Linear(d_model, self.d_v, bias=False) for _ in range(n_heads)])
        self.WO = nn.Linear(n_heads * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        heads_out = []
        for h in range(self.n_heads):
            Q = self.WQ[h](x)
            K = self.WK[h](x)
            V = self.WV[h](x)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask==0, float("-inf"))
            P = torch.softmax(scores, dim=-1)
            P = self.dropout(P)
            O = torch.matmul(P, V)
            heads_out.append(O)
        O_concat = torch.cat(heads_out, dim=-1)
        out = self.WO(O_concat)
        out = self.dropout(out)
        return out
    def export_weights(self, prefix: str, out_dir: str) -> Dict[str, str]:
        import os, numpy as np
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for name, mlist in (("WQ", self.WQ), ("WK", self.WK), ("WV", self.WV)):
            for h, lin in enumerate(mlist):
                arr = lin.weight.detach().cpu().numpy().T
                fn = f"{prefix}_{name}_h{h}.npy"
                np.save(os.path.join(out_dir, fn), arr)
                paths[f"{prefix}_{name}_h{h}"] = fn
        WO_arr = self.WO.weight.detach().cpu().numpy().T
        fn = f"{prefix}_WO.npy"
        np.save(os.path.join(out_dir, fn), WO_arr)
        paths[f"{prefix}_WO"] = fn
        return paths

class MLP_ReRAM(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    def export_weights(self, prefix: str, out_dir: str):
        import os, numpy as np
        os.makedirs(out_dir, exist_ok=True)
        W1 = self.fc1.weight.detach().cpu().numpy().T
        b1 = self.fc1.bias.detach().cpu().numpy()
        W2 = self.fc2.weight.detach().cpu().numpy().T
        b2 = self.fc2.bias.detach().cpu().numpy()
        np.save(os.path.join(out_dir, f"{prefix}_W1.npy"), W1)
        np.save(os.path.join(out_dir, f"{prefix}_b1.npy"), b1)
        np.save(os.path.join(out_dir, f"{prefix}_W2.npy"), W2)
        np.save(os.path.join(out_dir, f"{prefix}_b2.npy"), b2)
        return {
            f"{prefix}_W1": f"{prefix}_W1.npy",
            f"{prefix}_b1": f"{prefix}_b1.npy",
            f"{prefix}_W2": f"{prefix}_W2.npy",
            f"{prefix}_b2": f"{prefix}_b2.npy",
        }

class TransformerBlockReRAM(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttentionReRAM(d_model, n_heads, d_k, d_v, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = MLP_ReRAM(d_model, d_ff, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        z = self.mha(x, mask=mask)
        x = self.ln1(x + z)
        z2 = self.mlp(x)
        x = self.ln2(x + z2)
        return x
    def export_weights(self, layer_idx: int, out_dir: str):
        paths = {}
        p_mha = self.mha.export_weights(prefix=f"layer{layer_idx}_attn", out_dir=out_dir)
        paths.update(p_mha)
        p_mlp = self.mlp.export_weights(prefix=f"layer{layer_idx}_ffn", out_dir=out_dir)
        paths.update(p_mlp)
        import os, numpy as np
        os.makedirs(out_dir, exist_ok=True)
        for i, ln in enumerate([self.ln1, self.ln2], start=1):
            gamma = ln.weight.detach().cpu().numpy()
            beta  = ln.bias.detach().cpu().numpy()
            np.save(os.path.join(out_dir, f"layer{layer_idx}_ln{i}_gamma.npy"), gamma)
            np.save(os.path.join(out_dir, f"layer{layer_idx}_ln{i}_beta.npy"), beta)
            paths[f"layer{layer_idx}_ln{i}_gamma"] = f"layer{layer_idx}_ln{i}_gamma.npy"
            paths[f"layer{layer_idx}_ln{i}_beta"]  = f"layer{layer_idx}_ln{i}_beta.npy"
        return paths

class ViT_ReRAM(nn.Module):
    def __init__(self, img_size: Tuple[int,int], in_channels: int, patch_size: int, d_model: int, depth: int, n_heads: int, d_k: int, d_v: int, mlp_ratio: float, num_classes: int, dropout: float, use_class_token: bool, pos_embed: str = "learnable"):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.use_class_token = use_class_token

        self.patch_proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False)

        N = self.num_patches + (1 if use_class_token else 0)
        if pos_embed.lower() == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, N, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.positional = None
        else:
            self.pos_embed = None
            self.positional = SinusoidalPositionalEmbedding(d_model, N)

        if use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.dropout = nn.Dropout(dropout)

        d_ff = int(d_model * mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlockReRAM(d_model, n_heads, d_k if d_k>0 else d_model//n_heads, d_v if d_v>0 else d_model//n_heads, d_ff, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)
        if self.use_class_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        else:
            x = x + self.positional(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.use_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        logits = self.head(x)
        return logits

    def export_all_weights(self, out_dir: str):
        import os, numpy as np
        os.makedirs(out_dir, exist_ok=True)
        W_patch = self.patch_proj.weight.detach().cpu().numpy()
        ps = W_patch.shape[2]
        W_patch_lin = W_patch.reshape(W_patch.shape[0], -1).T
        np.save(os.path.join(out_dir, "patch_proj.npy"), W_patch_lin)
        if self.pos_embed is not None:
            np.save(os.path.join(out_dir, "pos_embed.npy"), self.pos_embed.detach().cpu().numpy())
        if self.cls_token is not None:
            np.save(os.path.join(out_dir, "cls_token.npy"), self.cls_token.detach().cpu().numpy())
        paths = {"patch_proj": "patch_proj.npy"}
        for i, blk in enumerate(self.blocks):
            p = blk.export_weights(layer_idx=i, out_dir=out_dir)
            paths.update(p)
        np.save(os.path.join(out_dir, "final_ln_gamma.npy"), self.norm.weight.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "final_ln_beta.npy"),  self.norm.bias.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "head_W.npy"),         self.head.weight.detach().cpu().numpy().T)
        np.save(os.path.join(out_dir, "head_b.npy"),         self.head.bias.detach().cpu().numpy())
        paths.update({
            "final_ln_gamma": "final_ln_gamma.npy",
            "final_ln_beta": "final_ln_beta.npy",
            "head_W": "head_W.npy",
            "head_b": "head_b.npy",
        })
        return paths
