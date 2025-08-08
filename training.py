import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sgnNet import SGN
from dataset import Seq2PointWindows
import matplotlib.pyplot as plt



def train_one_epoch(model, loader, opt, device, pos_weight= 5.0,
                    w_reg = 1.0, w_cls = 0.5, w_gate = 0.5):
    model.train()
    total = 0.0

    pw = torch.tensor([pos_weight], device=device)

    for x,y,s in loader:
        x = x.to(device)
        y = y.to(device)
        s = s.to(device).float().clamp(0, 1)

        opt.zero_grad()
        power, cls_logits, reg = model(x)

        # ensure shapes
        power = power.view_as(y)
        reg   = reg.view_as(y)
        s  = s.view_as(cls_logits)

        #compute loss
        loss_reg = F.l1_loss(reg,y)
        loss_gate = F.l1_loss(power,y)
        bce = F.binary_cross_entropy_with_logits(cls_logits, s, pos_weight=pw)
        loss = w_reg * loss_reg + w_cls * bce + w_gate * loss_gate
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)




@torch.no_grad()
def evaluate_mae(model, loader, device, target_scale=None):
    model.eval()
    total_mae = 0.0
    n = 0
    for x,y,_ in loader:
        x, y = x.to(device), y.to(device)
        power, _, _ = model(x)
        power = power.view_as(y)
        total_mae += torch.abs(power - y).sum().item()
        n += y.numel()
    mae = total_mae / n
    if target_scale is not None:
        mae *= (target_scale + 1e-6)
    return mae




@torch.no_grad()
def evaluate_loss(model, loader, device, pos_weight=5.0, w_reg=1.0, w_cls=0.5, w_gate=0.5):
    model.eval()
    total = 0.0
    pw = torch.tensor([pos_weight], device=device)
    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device).float().clamp(0, 1)
        power, cls_logits, reg = model(x)
        power = power.view_as(y)
        reg   = reg.view_as(y)
        s     = s.view_as(cls_logits)

        loss_reg  = F.l1_loss(reg, y)
        loss_gate = F.l1_loss(power, y)
        bce = F.binary_cross_entropy_with_logits(cls_logits, s, pos_weight=pw)
        loss = w_reg * loss_reg + w_cls * bce + w_gate * loss_gate
        total += loss.item() * x.size(0)

    return total / len(loader.dataset)





class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0, restore_best=True, ckpt_path="sgn_best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.ckpt_path = ckpt_path
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val_metric):
        improved = (self.best - val_metric) > self.min_delta
        if improved:
            self.best = val_metric
            self.bad_epochs = 0
            return False  # don't stop
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience





def main_train(
    mains_train, target_train,
    mains_val, target_val,
    win_len=512, batch_size=128, lr=1e-3, epochs=50, kind="tcn",
    patience=6, min_delta=0.0, ckpt_path="sgn_best.pt",
    use_scheduler=True, plot=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGN(in_ch=1, hid=64, kind=kind, out_len=1).to(device)

    # Datasets share train stats for consistent scaling
    ds_tr = Seq2PointWindows(mains_train, target_train, win_len=win_len, stride=32, train=True)
    ds_va = Seq2PointWindows(mains_val, target_val, win_len=win_len, stride=1, train=False,
                             mains_mean=ds_tr.mains_mean, mains_std=ds_tr.mains_std,
                             target_scale=ds_tr.target_scale)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2) if use_scheduler else None
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best=True, ckpt_path=ckpt_path)

    train_losses, val_losses, val_maes = [], [], []

    for ep in range(1, epochs + 1):
        # Train epoch (scaled loss)
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        train_losses.append(tr_loss)

        # Validate: loss (scaled) + MAE in watts
        val_loss = evaluate_loss(model, dl_va, device)
        va_mae   = evaluate_mae(model, dl_va, device, target_scale=ds_tr.target_scale)
        val_losses.append(val_loss)
        val_maes.append(va_mae)

        if sched is not None:
            sched.step(va_mae)  # keep scheduler on the primary metric you care about

        print(f"Epoch {ep:03d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE (W): {va_mae:.2f}")

        # Save best + early stop check (still keying on MAE in W)
        if va_mae < stopper.best - 1e-12:
            torch.save({
                "model": model.state_dict(),
                "stats": {
                    "mains_mean": ds_tr.mains_mean,
                    "mains_std": ds_tr.mains_std,
                    "target_scale": ds_tr.target_scale,
                    "win_len": win_len
                }
            }, ckpt_path)

        if stopper.step(va_mae):
            print(f"Early stopping at epoch {ep} (no improvement for {stopper.patience} epochs).")
            break

    # Restore best weights
    if stopper.restore_best:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Restored best model (Val MAE best: {stopper.best:.4f} W)")

    print(f"Best Val MAE: {stopper.best:.4f} W")

    # Plot: loss vs loss
    if plot:
        plt.figure(figsize=(5, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_maes": val_maes,
        "best_val_mae": stopper.best,
        "ckpt_path": ckpt_path,
        "model": model,
        "stats": {
            "mains_mean": ds_tr.mains_mean,
            "mains_std": ds_tr.mains_std,
            "target_scale": ds_tr.target_scale,
            "win_len": win_len
        }
    }
