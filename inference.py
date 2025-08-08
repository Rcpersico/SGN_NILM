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
