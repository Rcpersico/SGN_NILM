import torch
import numpy as np


@torch.no_grad()
def infer_seq2point_timeline(model, mains, stats, device):
    model.eval()
    mains = mains.astype("float32")

    x = (mains - stats["mains_mean"]) / (stats["mains_std"] + 1e-6)
    L = stats["win_len"]
    half = L // 2
    T = len(x)
    y_hat = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    for c in range(half, T - (L - half) + 1):
        seg = x[c-half:c-half+L][None, None, :]           # [1,1,L]
        seg_t = torch.from_numpy(seg).to(device)
        power, cls, reg = model(seg_t)                    
        pred = power.squeeze().item()
        y_hat[c] += pred
        counts[c] += 1.0

    counts[counts == 0] = 1.0
    y_hat = y_hat / counts
    y_hat = y_hat * (stats["target_scale"] + 1e-6)        # back to watts
    return y_hat





@torch.no_grad()
def infer_seq2point_timeline_all(model, mains, stats, device):
    """
    Returns arrays length T:
      power_w : gated power in watts
      reg_w   : regression head (pre-gate) in watts
      cls_p   : ON probability (0..1)
    """
    model.eval()
    x = mains.astype("float32")
    x = (x - stats["mains_mean"]) / (stats["mains_std"] + 1e-6)

    L = stats["win_len"]; half = L // 2; T = len(x)

    power_sum = np.zeros(T, dtype=np.float32)
    reg_sum   = np.zeros(T, dtype=np.float32)
    prob_sum  = np.zeros(T, dtype=np.float32)
    counts    = np.zeros(T, dtype=np.float32)

    for c in range(half, T - (L - half) + 1):
        seg = x[c-half:c-half+L][None, None, :]   # [1,1,L]
        seg_t = torch.from_numpy(seg).to(device)

        power, cls_logits, reg = model(seg_t)     # your model returns (power, logits, reg)
        p  = torch.sigmoid(cls_logits).squeeze().item()
        pw = power.squeeze().item()
        rw = reg.squeeze().item()

        power_sum[c] += pw
        reg_sum[c]   += rw
        prob_sum[c]  += p
        counts[c]    += 1.0

    counts[counts == 0] = 1.0
    power = power_sum / counts
    reg   = reg_sum   / counts
    prob  = prob_sum  / counts

    # unscale power-like outputs back to watts
    scale = stats["target_scale"] + 1e-6
    power_w = power * scale
    reg_w   = reg   * scale

    return power_w, reg_w, prob