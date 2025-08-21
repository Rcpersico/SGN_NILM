# plot.py
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(results, enable=True):
    """
    results: dict returned by main_train_staged (must contain 'train_losses' and 'val_losses')
    """
    if not enable:
        return
    tr = results.get("train_losses", [])
    vl = results.get("val_losses", [])
    if len(tr) == 0 and len(vl) == 0:
        return
    plt.figure(figsize=(6,4))
    if tr: plt.plot(tr, label="Train (joint)")
    if vl: plt.plot(vl, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training (joint) vs Validation Loss")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def plot_reg_vs_true(y_true, reg_w, N=None, show=True):
    if not show: return
    if N is None: N = len(y_true)
    N = min(N, len(y_true), len(reg_w))
    plt.figure(figsize=(12,4))
    plt.plot(y_true[:N], label="True", linewidth=1.0, alpha=0.9)
    plt.plot(reg_w[:N], label="Regression (pre-gate, W)", linewidth=1.0)
    plt.xlabel("Timestep"); plt.ylabel("Power (W)")
    plt.title("Regression vs True (test slice)")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_soft(y_true, power_soft_w, N=None, show=True):
    if not show: return
    if N is None: N = len(y_true)
    N = min(N, len(y_true), len(power_soft_w))
    plt.figure(figsize=(12,4))
    plt.plot(y_true[:N], label="True", linewidth=1.0, alpha=0.9)
    plt.plot(power_soft_w[:N], label="Pred (Soft-gated)", linewidth=1.0)
    plt.xlabel("Timestep"); plt.ylabel("Power (W)")
    plt.title("Soft-gated Prediction vs True (test slice)")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_hard(y_true, power_hard_w, gate_hard, on_threshold=15.0, N=None, show=True):
    if not show: return
    if N is None: N = len(y_true)
    N = min(N, len(y_true), len(power_hard_w), len(gate_hard))
    true_on = (y_true[:N] > on_threshold).astype(float)

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(y_true[:N], label="True", color="black", linewidth=1.0)
    ax1.plot(power_hard_w[:N], label="Pred (Hard-gated)", linewidth=1.0)
    ax1.set_ylabel("Power (W)")
    ax1.set_title("Hard-gated Prediction vs True (test slice)")
    ax1.legend()

    ax2.plot(true_on, label="True ON/OFF", color="black", linestyle="--")
    ax2.plot(gate_hard[:N], label="Gate {0,1}", linewidth=1.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ON / Gate")
    ax2.legend()
    plt.tight_layout(); plt.show()
