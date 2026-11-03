import torch
import torch.nn as nn

def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, log_interval=50):
    # Mixed precision only when an *enabled* GradScaler is passed and the
    # device is CUDA; otherwise plain fp32 (a disabled scaler means amp=false).
    use_amp = (scaler is not None and scaler.is_enabled()
               and torch.device(device).type == "cuda")
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        acc = accuracy_top1(logits.detach(), y)
        running_loss += loss.item() * x.size(0)
        running_acc += acc * x.size(0)
        count += x.size(0)
        if step % log_interval == 0:
            print(f"  step {step:4d}  loss {running_loss/count:.4f}  acc {running_acc/count:.4f}")
    return running_loss / count, running_acc / count

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        total_loss += loss.item() * x.size(0)
        total_acc += acc * x.size(0)
        count += x.size(0)
    return total_loss / count, total_acc / count
