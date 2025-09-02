# plot.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def _effective_len(total_len: int, N: Optional[int], offset: int) -> int:
    """
    Compute the number of points to plot given total length, requested N, and offset.
    Ensures we don't exceed array bounds and never return negative values.
    """
    if offset < 0:
        offset = 0
    if offset >= total_len:
        return 0
    max_len_from_offset = total_len - offset
    if N is None:
        return max_len_from_offset
    return max(0, min(N, max_len_from_offset))


def plot_training_curves(results, enable: bool = True):
    """
    results: dict returned by training that contains 'train_losses' and 'val_losses'
    """
    if not enable:
        return
    tr = results.get("train_losses", [])
    vl = results.get("val_losses", [])
    if len(tr) == 0 and len(vl) == 0:
        return

    plt.figure(figsize=(6, 4))
    if tr:
        plt.plot(tr, label="Train (joint)")
    if vl:
        plt.plot(vl, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training (joint) vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reg_vs_true(y_true, reg_w, N: Optional[int] = None, offset: int = 0, show: bool = True):
    if not show:
        return
    n = _effective_len(min(len(y_true), len(reg_w)), N, offset)
    if n <= 0:
        print("[plot_reg_vs_true] Nothing to plot: check offset/N vs array lengths.")
        return

    ys_true = y_true[offset:offset + n]
    ys_reg = reg_w[offset:offset + n]

    plt.figure(figsize=(12, 4))
    plt.plot(ys_true, label="True", linewidth=1.0, alpha=0.9)
    plt.plot(ys_reg, label="Regression (pre-gate, W)", linewidth=1.0)
    plt.xlabel("Timestep")
    plt.ylabel("Power (W)")
    plt.title(f"Regression vs True (slice, offset={offset}, len={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_soft(y_true, power_soft_w, N: Optional[int] = None, offset: int = 0, show: bool = True):
    if not show:
        return
    n = _effective_len(min(len(y_true), len(power_soft_w)), N, offset)
    if n <= 0:
        print("[plot_soft] Nothing to plot: check offset/N vs array lengths.")
        return

    ys_true = y_true[offset:offset + n]
    ys_soft = power_soft_w[offset:offset + n]

    plt.figure(figsize=(12, 4))
    plt.plot(ys_true, label="True", linewidth=1.0, alpha=0.9)
    plt.plot(ys_soft, label="Pred (Soft-gated)", linewidth=1.0)
    plt.xlabel("Timestep")
    plt.ylabel("Power (W)")
    plt.title(f"Soft-gated Prediction vs True (slice, offset={offset}, len={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hard(
    y_true,
    power_hard_w,
    gate_hard,
    on_threshold: float = 15.0,
    N: Optional[int] = None,
    offset: int = 0,
    show: bool = True
):
    if not show:
        return
    n = _effective_len(min(len(y_true), len(power_hard_w), len(gate_hard)), N, offset)
    if n <= 0:
        print("[plot_hard] Nothing to plot: check offset/N vs array lengths.")
        return

    ys_true = y_true[offset:offset + n]
    ys_pred = power_hard_w[offset:offset + n]
    ys_gate = gate_hard[offset:offset + n]
    true_on = (ys_true > on_threshold).astype(float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(ys_true, label="True", color="black", linewidth=1.0)
    ax1.plot(
        ys_pred,
        label="Pred (Hard-gated)",
        color="blue",
        linewidth=1.0,
        linestyle=":"
    )
    ax1.set_ylabel("Power (W)")
    ax1.set_title(f"Hard-gated Prediction vs True (slice, offset={offset}, len={n})")
    ax1.legend()

    ax2.plot(true_on, label="True ON/OFF", color="black", linestyle="--")
    ax2.plot(ys_gate, label="Gate {0,1}", linewidth=1.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("ON / Gate")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_mains(mains_w, N: Optional[int] = None, offset: int = 0, show: bool = True):
    """
    Plot the aggregate mains signal alone.
    """
    if not show:
        return
    n = _effective_len(len(mains_w), N, offset)
    if n <= 0:
        print("[plot_mains] Nothing to plot: check offset/N vs array lengths.")
        return

    ys = mains_w[offset:offset + n]

    plt.figure(figsize=(12, 4))
    plt.plot(ys, label="Mains (aggregate)", linewidth=1.0, alpha=0.9)
    plt.xlabel("Timestep")
    plt.ylabel("Power (W)")
    plt.title(f"Aggregate Mains Signal (slice, offset={offset}, len={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()
