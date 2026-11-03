import os, math, argparse, torch
import torch.nn as nn
import torch.optim as optim

from utils.config import load_cfg
from utils.data import prepare_dataloaders
from utils.vit_reram import ViT_ReRAM
from utils.train import train_one_epoch, validate

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_model(cfg, in_channels, num_classes):
    img_size = (cfg.data.img_height, cfg.data.img_width)
    d_k = cfg.model.d_k if cfg.model.d_k > 0 else cfg.model.d_model // cfg.model.n_heads
    d_v = cfg.model.d_v if cfg.model.d_v > 0 else cfg.model.d_model // cfg.model.n_heads
    model = ViT_ReRAM(
        img_size=img_size,
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        n_heads=cfg.model.n_heads,
        d_k=d_k, d_v=d_v,
        mlp_ratio=cfg.model.mlp_ratio,
        num_classes=num_classes,
        dropout=cfg.model.dropout,
        use_class_token=cfg.model.use_class_token,
        pos_embed=cfg.model.pos_embed,
    )
    return model

def build_scheduler(optimizer, name, warmup_epochs, total_epochs):
    """Per-epoch LR schedule: linear warmup over ``warmup_epochs``, then
    'none' (constant) or 'cosine' (decay to 0 at the last epoch)."""
    name = (name or "none").strip().lower()
    if name not in ("none", "constant", "cosine"):
        raise ValueError(f"[TRAIN] lr_scheduler must be 'none'|'cosine', got '{name}'")
    warmup_epochs = max(0, int(warmup_epochs))

    def factor(e):  # e = number of completed epochs (0-based)
        if e < warmup_epochs:
            return (e + 1) / warmup_epochs
        if name == "cosine":
            span = max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(e - warmup_epochs, span) / span))
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

def main(args):
    cfg = load_cfg(args.config)
    set_seed(cfg.train.seed)
    if cfg.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.train.device)
    print("Device:", device)

    train_loader, val_loader, num_classes, inferred_channels = prepare_dataloaders(
        cfg.data.dataset, cfg.data.data_root, (cfg.data.img_height, cfg.data.img_width),
        cfg.data.channels, cfg.train.batch_size, cfg.data.num_workers, cfg.data.val_split, cfg.data.augment
    )
    C = inferred_channels if cfg.data.channels == "auto" else int(cfg.data.channels)
    if cfg.model.num_classes != num_classes:
        print(f"[warn] Overriding num_classes from cfg ({cfg.model.num_classes}) -> dataset ({num_classes})")
        cfg.model.num_classes = num_classes

    model = build_model(cfg, in_channels=C, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas)
    scheduler = build_scheduler(optimizer, cfg.train.lr_scheduler, cfg.train.warmup_epochs, cfg.train.epochs)
    # AMP only when requested AND on CUDA; scaler=None makes train_one_epoch run fp32.
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_acc = -1.0
    best_path = os.path.join(cfg.save.out_dir, "best.pt")
    os.makedirs(cfg.save.out_dir, exist_ok=True)

    for epoch in range(1, cfg.train.epochs+1):
        print(f"\n=== Epoch {epoch}/{cfg.train.epochs} (lr {optimizer.param_groups[0]['lr']:.3e}) ===")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, log_interval=cfg.train.log_interval)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Train: loss {tr_loss:.4f} acc {tr_acc:.4f}")
        print(f"    Val: loss {va_loss:.4f} acc {va_acc:.4f}")
        if cfg.save.save_best and va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  [saved best.pt @ acc={best_acc:.4f}]")

    torch.save(model.state_dict(), os.path.join(cfg.save.out_dir, "last.pt"))
    print("Saved last.pt")

    if cfg.save.save_weights:
        weights_dir = os.path.join(cfg.save.out_dir, "weights")
        # Inference loads best.pt, so export the weights of that checkpoint.
        src = "last.pt"
        if cfg.save.save_best and os.path.isfile(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            src = f"best.pt (val acc={best_acc:.4f})"
        model.export_all_weights(weights_dir)
        print(f"Saved per-layer weights from {src} -> {weights_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.ini")
    args = p.parse_args()
    main(args)
