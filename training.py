# training.py
import math
from itertools import cycle
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from sgnNet import SGN
from dataset import Seq2PointWindows
from loss import sgn_loss

# -------------------------
# Utils
# -------------------------
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

class EarlyStopper:
    def __init__(self, patience=6, min_delta=0.0, restore_best=True, ckpt_path="sgn_best.pt"):
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
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience


# -------------------------
# Evaluation helpers
# -------------------------
@torch.no_grad()
def evaluate_mae(model, loader, device, target_scale=None):
    model.eval()
    total_mae, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        power, _, _ = model(x)
        power = power.view_as(y)
        total_mae += torch.abs(power - y).sum().item()
        n += y.numel()
    mae = total_mae / max(1, n)
    if target_scale is not None:
        mae *= (target_scale + 1e-6)
    return mae

@torch.no_grad()
def evaluate_mae_on_off(model, loader, device, target_scale=None, on_threshold=1e-6):
    model.eval()
    abs_all = abs_on = abs_off = 0.0
    n_all = n_on = n_off = 0
    for x, y, s in loader:
        x = x.to(device); y = y.to(device).squeeze(-1); s = s.to(device).squeeze(-1)
        power, _, _ = model(x)
        pred = power.squeeze(-1)
        err = (pred - y).abs()
        abs_all += err.sum().item(); n_all += y.numel()
        on_mask = (s > on_threshold); off_mask = ~on_mask
        if on_mask.any():  abs_on  += err[on_mask].sum().item();  n_on  += on_mask.sum().item()
        if off_mask.any(): abs_off += err[off_mask].sum().item(); n_off += off_mask.sum().item()

    scale = (target_scale + 1e-6) if target_scale is not None else 1.0
    return (abs_all / max(1, n_all)) * scale, \
           (abs_on  / max(1, n_on )) * scale, \
           (abs_off / max(1, n_off)) * scale

@torch.no_grad()
def evaluate_loss_masked(
    model, loader, device,
    alpha_on=1.0, alpha_off=0.02, beta_cls=1.5,
    delta_huber=0.05, focal_gamma=2.0, pos_weight=3.0, alpha_reg_raw=1.0
):
    model.eval()
    total = 0.0
    n = 0
    for x, y, s in loader:
        x = x.to(device); y = y.to(device).squeeze(-1); s = s.to(device).squeeze(-1)
        power, cls_logits, reg = model(x)
        tau = float(getattr(model, "gate_tau", 1.0))
        logits_t = (cls_logits / tau).squeeze(-1)
        loss, _ = sgn_loss(
            power.squeeze(-1), logits_t, reg.squeeze(-1),
            y, s,
            delta_huber=delta_huber,
            alpha_on=alpha_on, alpha_reg_raw=alpha_reg_raw, alpha_off=alpha_off,
            beta_cls=beta_cls, focal_gamma=focal_gamma, pos_weight=pos_weight
        )
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(1, n)



@torch.no_grad()
def evaluate_loss_mode(
    model, loader, device, *,
    mode: str,                       # "reg" | "cls" | "all"
    alpha_on, alpha_off, alpha_reg_raw,
    beta_cls, delta_huber, focal_gamma, pos_weight
):
    """
    Phase-matched validator:
      - mode="reg":   regression-only (beta_cls=0)
      - mode="cls":   classification-only (reg weights 0, delta_huber=0)
      - mode="all":   full composite (same as evaluate_loss_masked default)
    Returns per-sample mean loss.
    """
    if mode == "reg":
        a_on, a_off, a_raw, b_cls, d_h = alpha_on, alpha_off, alpha_reg_raw, 0.0, delta_huber
    elif mode == "cls":
        a_on, a_off, a_raw, b_cls, d_h = 0.0, 0.0, 0.0, 1.0, 0.0
    elif mode == "all":
        a_on, a_off, a_raw, b_cls, d_h = alpha_on, alpha_off, alpha_reg_raw, beta_cls, delta_huber
    else:
        raise ValueError("mode must be 'reg', 'cls', or 'all'")

    return evaluate_loss_masked(
        model, loader, device,
        alpha_on=a_on, alpha_off=a_off, alpha_reg_raw=a_raw,
        beta_cls=b_cls, delta_huber=d_h,
        focal_gamma=focal_gamma, pos_weight=pos_weight
    )




# -------------------------
# Loaders for staged training
# -------------------------
def make_stage_loaders(
    mains_train, target_train,
    mains_val, target_val,
    *,
    win_len=512,
    # NEW: separate strides (defaults keep backward-compat)
    stride_reg=32,
    stride_cls=32,
    batch_size=128,
    on_sample_prob_reg=0.7,
    on_threshold=20.0, off_threshold=10.0, min_hold=1,
    # CLS emphasis knobs (None → unbiased sampling, 0.0 edges → off)
    cls_on_sample_prob: float | None = None,
    edge_sample_prob_cls: float = 0.0,
):
    # REG: ON-rich
    ds_tr_reg = Seq2PointWindows(
        mains_train, target_train, win_len=win_len, stride=stride_reg, train=True,
        balance_sampling=True, on_sample_prob=on_sample_prob_reg,
        on_threshold=on_threshold, off_threshold=off_threshold, min_hold=min_hold,
        edge_sample_prob=0.0,   # reg: no edge emphasis
    )

    # CLS: either unbiased or OFF-tilted + edges
    if cls_on_sample_prob is None:
        ds_tr_cls = Seq2PointWindows(
            mains_train, target_train, win_len=win_len, stride=stride_cls, train=True,
            balance_sampling=False,
            on_threshold=on_threshold, off_threshold=off_threshold, min_hold=min_hold,
            edge_sample_prob=edge_sample_prob_cls,
        )
    else:
        ds_tr_cls = Seq2PointWindows(
            mains_train, target_train, win_len=win_len, stride=stride_cls, train=True,
            balance_sampling=True, on_sample_prob=float(cls_on_sample_prob),
            on_threshold=on_threshold, off_threshold=off_threshold, min_hold=min_hold,
            edge_sample_prob=edge_sample_prob_cls,
        )

    # VAL: unbiased, share train stats
    ds_va = Seq2PointWindows(
        mains_val, target_val, win_len=win_len, stride=1, train=False,
        mains_mean=ds_tr_reg.mains_mean, mains_std=ds_tr_reg.mains_std,
        target_scale=ds_tr_reg.target_scale,
        on_threshold=on_threshold, off_threshold=off_threshold, min_hold=min_hold,
        edge_sample_prob=0.0,
    )

    dl_tr_reg = DataLoader(ds_tr_reg, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    dl_tr_cls = DataLoader(ds_tr_cls, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    dl_va     = DataLoader(ds_va,     batch_size=batch_size, shuffle=False, num_workers=0)
    return (ds_tr_reg, ds_tr_cls, ds_va, dl_tr_reg, dl_tr_cls, dl_va)


# -------------------------
# Phase 1: Regressor pretrain (ON-rich batches only)
# -------------------------
def pretrain_regressor_one_epoch(
    model, dl_tr_reg, opt, device,
    delta_huber, focal_gamma,
    alpha_on=1.0, alpha_reg_raw=1.0, alpha_off=0.02
):
    model.train()
    set_requires_grad(model.head_cls, False)
    set_requires_grad(model.head_reg, True)
    set_requires_grad(model.backbone, True)

    total = 0.0; n = 0
    for x, y, s in dl_tr_reg:
        x = x.to(device); y = y.to(device).squeeze(-1); s = s.to(device).squeeze(-1)
        power, cls_logits, reg = model(x)
        tau = float(getattr(model, "gate_tau", 1.0))
        logits_t = (cls_logits / tau).squeeze(-1)

        loss, _ = sgn_loss(
            power.squeeze(-1), logits_t, reg.squeeze(-1),
            y, s,
            delta_huber=delta_huber,
            alpha_on=alpha_on, alpha_reg_raw=alpha_reg_raw, alpha_off=alpha_off,
            beta_cls=0.0, focal_gamma=focal_gamma, pos_weight=None
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0); n += x.size(0)

    set_requires_grad(model.head_cls, True)
    return total / max(1, n)


# -------------------------
# Phase 2: Classifier pretrain (unbiased or OFF-tilted + edges)
# -------------------------
def pretrain_classifier_one_epoch(
    model, dl_tr_cls, opt, device,
    pos_weight, focal_gamma
):
    model.train()
    set_requires_grad(model.head_cls, True)
    set_requires_grad(model.head_reg, False)
    set_requires_grad(model.backbone, True)

    total = 0.0; n = 0
    for x, y, s in dl_tr_cls:
        x = x.to(device); y = y.to(device).squeeze(-1); s = s.to(device).squeeze(-1)
        power, cls_logits, reg = model(x)
        tau = float(getattr(model, "gate_tau", 1.0))
        logits_t = (cls_logits / tau).squeeze(-1)

        loss, _ = sgn_loss(
            power.squeeze(-1), logits_t, reg.squeeze(-1),
            y, s,
            delta_huber=0.0,
            alpha_on=0.0, alpha_reg_raw=0.0, alpha_off=0.0,
            beta_cls=1.0, focal_gamma=focal_gamma, pos_weight=pos_weight
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * x.size(0); n += x.size(0)

    set_requires_grad(model.head_reg, True)
    return total / max(1, n)


# -------------------------
# Phase 3: Joint training (two batches per step → one loss)
# -------------------------
def joint_train_one_epoch_composite(
    model, dl_tr_reg, dl_tr_cls, opt, device,
    delta_huber, pos_weight, focal_gamma,
    alpha_on=1.0, alpha_reg_raw=1.0, alpha_off=0.02, beta_cls=1.5
):
    """
    Train with the SAME composite objective used in validation:
    loss = reg(on/off) + alpha_reg_raw*raw_reg + beta_cls*cls
    on BOTH the reg batch and the cls batch; then combine sample-weighted.
    This keeps the classifier learning (beta_cls>0) and aligns train/val formulas.
    """
    model.train()
    # all heads & backbone learn in joint
    for p in model.parameters():
        p.requires_grad = True

    total, nsamp = 0.0, 0

    for (x_r, y_r, s_r), (x_c, y_c, s_c) in zip(cycle(dl_tr_reg), dl_tr_cls):
        # Merge the two mini-batches (keeps your sampling biases but one forward/backward)
        x = torch.cat([x_r, x_c], dim=0).to(device)
        y = torch.cat([y_r, y_c], dim=0).to(device).squeeze(-1)
        s = torch.cat([s_r, s_c], dim=0).to(device).squeeze(-1)

        power, cls_logits, reg = model(x)
        tau = float(getattr(model, "gate_tau", 1.0))
        logits_t = (cls_logits / tau).squeeze(-1)

        # FULL composite – same as validation
        loss, _ = sgn_loss(
            power.squeeze(-1), logits_t, reg.squeeze(-1),
            y, s,
            delta_huber=delta_huber,
            alpha_on=alpha_on, alpha_reg_raw=alpha_reg_raw, alpha_off=alpha_off,
            beta_cls=beta_cls, focal_gamma=focal_gamma, pos_weight=pos_weight
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        bs = x.size(0)
        total  += loss.item() * bs   # per-sample accumulator
        nsamp  += bs

    # Return per-sample mean, same reduction as evaluate_loss_masked
    return total / max(1, nsamp)



# -------------------------
# Top-level driver
# -------------------------
def main_train_staged(
    mains_train, target_train,
    mains_val, target_val,
    *,
    win_len=512,
    # Back-compat: if 'stride' is given, it applies to both unless overridden
    stride: int | None = None,
    stride_reg: int | None = None,
    stride_cls: int | None = None,
    batch_size=128,
    lr=1e-3,
    epochs_reg=5, epochs_cls=5, epochs_joint=50,
    on_sample_prob_reg=0.7,
    kind="tcn", patience=8, min_delta=0.0, ckpt_path="sgn_best.pt",
    delta_huber=0.05, focal_gamma=2.0, pos_weight=3.0,
    alpha_on=1.0, alpha_reg_raw=1.0, alpha_off=0.02, beta_cls=1.5,
    use_scheduler=False, plot=True,
    reg_es_patience=2, reg_es_min_delta=0.0,   # monitor Val MAE@ON
    cls_es_patience=2, cls_es_min_delta=0.0,   # monitor Val MAE@OFF
    on_threshold=20.0, off_threshold=10.0, min_hold=1,
    # CLS emphasis
    cls_on_sample_prob: float | None = None,
    edge_sample_prob_cls: float = 0.0,
):
   

    # Resolve strides with backward compatibility
    if stride_reg is None and stride_cls is None and stride is not None:
        stride_reg = stride
        stride_cls = stride
    if stride_reg is None: stride_reg = 32
    if stride_cls is None: stride_cls = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGN(in_ch=1, hid=64, kind=kind, out_len=1).to(device)

    ds_tr_reg, ds_tr_cls, ds_va, dl_tr_reg, dl_tr_cls, dl_va = make_stage_loaders(
        mains_train, target_train, mains_val, target_val,
        win_len=win_len,
        stride_reg=stride_reg,
        stride_cls=stride_cls,
        batch_size=batch_size,
        on_sample_prob_reg=on_sample_prob_reg,
        on_threshold=on_threshold, off_threshold=off_threshold, min_hold=min_hold,
        cls_on_sample_prob=cls_on_sample_prob,
        edge_sample_prob_cls=edge_sample_prob_cls,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    stopper = EarlyStopper(patience=patience, min_delta=min_delta, restore_best=True, ckpt_path=ckpt_path)
    train_hist, val_hist, mae_hist = [], [], []

    def eval_all():
        va_loss = evaluate_loss_masked(
            model, dl_va, device,
            alpha_on=alpha_on, alpha_off=alpha_off, beta_cls=beta_cls,
            delta_huber=delta_huber, focal_gamma=focal_gamma, pos_weight=pos_weight, alpha_reg_raw=alpha_reg_raw
        )
        va_mae = evaluate_mae(model, dl_va, device, target_scale=ds_tr_reg.target_scale)
        return va_loss, va_mae

    # ---- Phase 1: Regressor pretrain (ES on Val MAE@ON) ----
    best_reg_metric = float("inf")
    best_reg_state = None
    bad_reg = 0

    for ep in range(1, epochs_reg + 1):
        tr = pretrain_regressor_one_epoch(
            model, dl_tr_reg, opt, device,
            delta_huber=delta_huber, focal_gamma=focal_gamma,
            alpha_on=alpha_on, alpha_reg_raw=alpha_reg_raw, alpha_off=alpha_off
        )
        # evaluate split MAE
        _, va_mae_on, va_mae_off = evaluate_mae_on_off(
            model, dl_va, device, target_scale=ds_tr_reg.target_scale
        )
        va_loss, va_mae = eval_all()

        # Phase-matched: regression-only validation loss (no CLS term)
        val_loss_regonly = evaluate_loss_mode(
            model, dl_va, device, mode="reg",
            alpha_on=alpha_on, alpha_off=alpha_off, alpha_reg_raw=alpha_reg_raw,
            beta_cls=beta_cls, delta_huber=delta_huber,
            focal_gamma=focal_gamma, pos_weight=pos_weight
)


        print(
        f"[REG] Epoch {ep:03d} | Train {tr:.4f} | "
        f"ValLoss(REG-only) {val_loss_regonly:.4f} | ValLoss(all) {va_loss:.4f} | "
        f"ValMAE {va_mae:.2f} | MAE@ON {va_mae_on:.2f} | MAE@OFF {va_mae_off:.2f}"
        )


        improved = (best_reg_metric - va_mae_on) > reg_es_min_delta
        if improved:
            best_reg_metric = va_mae_on
            best_reg_state = copy.deepcopy(model.state_dict())
            bad_reg = 0
        else:
            bad_reg += 1
            if bad_reg > reg_es_patience:
                print(f"Early stopping REG at epoch {ep} (no Val MAE@ON improvement for {reg_es_patience} epochs).")
                break

    if best_reg_state is not None:
        model.load_state_dict(best_reg_state)
        print(f"Restored best REG weights (Val MAE@ON: {best_reg_metric:.2f} W)")

    # ---- Phase 2: Classifier pretrain (ES on Val MAE@OFF) ----
    best_cls_metric = float("inf")
    best_cls_state = None
    bad_cls = 0

    for ep in range(1, epochs_cls + 1):
        tr = pretrain_classifier_one_epoch(
            model, dl_tr_cls, opt, device,
            pos_weight=pos_weight, focal_gamma=focal_gamma
        )
        # evaluate split MAE
        _, va_mae_on, va_mae_off = evaluate_mae_on_off(
            model, dl_va, device, target_scale=ds_tr_reg.target_scale
        )
        va_loss, va_mae = eval_all()

        val_loss_clsonly = evaluate_loss_mode(
            model, dl_va, device, mode="cls",
            alpha_on=alpha_on, alpha_off=alpha_off, alpha_reg_raw=alpha_reg_raw,
            beta_cls=beta_cls, delta_huber=delta_huber,  # delta ignored in cls mode
            focal_gamma=focal_gamma, pos_weight=pos_weight
        )

        print(
            f"[CLS] Epoch {ep:03d} | Train {tr:.4f} | "
            f"ValLoss(CLS-only) {val_loss_clsonly:.4f} | ValLoss(all) {va_loss:.4f} | "
            f"ValMAE {va_mae:.2f} | MAE@ON {va_mae_on:.2f} | MAE@OFF {va_mae_off:.2f}"
        )


        improved = (best_cls_metric - va_mae_off) > cls_es_min_delta
        if improved:
            best_cls_metric = va_mae_off
            best_cls_state = copy.deepcopy(model.state_dict())
            bad_cls = 0
        else:
            bad_cls += 1
            if bad_cls > cls_es_patience:
                print(f"Early stopping CLS at epoch {ep} (no Val MAE@OFF improvement for {cls_es_patience} epochs).")
                break

    if best_cls_state is not None:
        model.load_state_dict(best_cls_state)
        print(f"Restored best CLS weights (Val MAE@OFF: {best_cls_metric:.2f} W)")

    # ---- Phase 3: Joint training (early stop on overall Val MAE) ----
    for ep in range(1, epochs_joint + 1):
        tr = joint_train_one_epoch_composite(
            model, dl_tr_reg, dl_tr_cls, opt, device,
            delta_huber=delta_huber, pos_weight=pos_weight, focal_gamma=focal_gamma,
            alpha_on=alpha_on, alpha_reg_raw=alpha_reg_raw, alpha_off=alpha_off, beta_cls=beta_cls
        )
        va_loss, va_mae = eval_all()
        train_hist.append(tr); val_hist.append(va_loss); mae_hist.append(va_mae)

        if va_mae < stopper.best - 1e-12:
            torch.save({
                "model": model.state_dict(),
                "stats": {
                    "mains_mean": ds_tr_reg.mains_mean,
                    "mains_std": ds_tr_reg.mains_std,
                    "target_scale": ds_tr_reg.target_scale,
                    "win_len": win_len
                }
            }, ckpt_path)

        print(f"[JOINT] Epoch {ep:03d} | Train {tr:.4f} | ValLoss {va_loss:.4f} | ValMAE {va_mae:.2f}")
        if stopper.step(va_mae):
            print(f"Early stopping at epoch {ep}."); break

    if stopper.restore_best:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Restored best model (Val MAE best: {stopper.best:.4f} W)")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(train_hist, label="Train (joint)")
        plt.plot(val_hist,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training (joint) vs Validation Loss")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    return {
        "train_losses": train_hist,
        "val_losses": val_hist,
        "val_maes": mae_hist,
        "best_val_mae": stopper.best,
        "ckpt_path": ckpt_path,
        "model": model,
        "stats": {
            "mains_mean": ds_tr_reg.mains_mean,
            "mains_std": ds_tr_reg.mains_std,
            "target_scale": ds_tr_reg.target_scale,
            "win_len": win_len,
            "gate_tau": float(getattr(model, "gate_tau", 1.0))
        }
    }
