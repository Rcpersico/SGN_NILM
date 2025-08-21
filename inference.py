import torch
import numpy as np

def smape(a, f):
    denom = (np.abs(a) + np.abs(f)).clip(1e-6, None)
    return (100.0 / len(a)) * np.sum(np.abs(f - a) / denom)

@torch.no_grad()
def infer_seq2point_timeline(model, mains, stats, device):
    model.eval()
    mains = mains.astype("float32")
    x = (mains - stats["mains_mean"]) / (stats["mains_std"] + 1e-6)
    L = stats["win_len"]; half = L // 2; T = len(x)
    y_hat = np.zeros(T, dtype=np.float32); counts = np.zeros(T, dtype=np.float32)

    for c in range(half, T - (L - half) + 1):
        seg = x[c-half:c-half+L][None, None, :]
        seg_t = torch.from_numpy(seg).to(device)
        power, _, _ = model(seg_t)
        pred = power.squeeze().detach().cpu().item()
        y_hat[c] += pred; counts[c] += 1.0

    counts[counts == 0] = 1.0
    y_hat = y_hat / counts
    y_hat = y_hat * (stats["target_scale"] + 1e-6)
    return y_hat

@torch.no_grad()
def infer_seq2point_timeline_all(model, mains, stats, device, gate_tau=0.75):
    model.eval()
    x = mains.astype("float32")
    x = (x - stats["mains_mean"]) / (stats["mains_std"] + 1e-6)

    L = stats["win_len"]; half = L // 2; T = len(x)
    power_sum = np.zeros(T, dtype=np.float32)
    reg_sum   = np.zeros(T, dtype=np.float32)
    prob_sum  = np.zeros(T, dtype=np.float32)
    counts    = np.zeros(T, dtype=np.float32)

    for c in range(half, T - (L - half) + 1):
        seg = x[c-half:c-half+L][None, None, :]
        seg_t = torch.from_numpy(seg).to(device)
        power, cls_logits, reg = model(seg_t)
        p  = torch.sigmoid(cls_logits / gate_tau).squeeze().detach().cpu().item()
        pw = power.squeeze().detach().cpu().item()
        rw = reg.squeeze().detach().cpu().item()
        power_sum[c] += pw; reg_sum[c] += rw; prob_sum[c] += p; counts[c] += 1.0

    counts[counts == 0] = 1.0
    power = power_sum / counts
    reg   = reg_sum   / counts
    prob  = prob_sum  / counts

    scale = stats["target_scale"] + 1e-6
    power_w = power * scale
    reg_w   = reg   * scale
    return power_w, reg_w, prob

def hysteresis_gate(prob, t_on=0.65, t_off=0.45, min_hold=3):
    T = prob.shape[0]
    gate = np.zeros(T, dtype=np.uint8)
    state = 0; hold = 0
    for t in range(T):
        p = prob[t]
        if state == 0:
            if p >= t_on:
                state = 1; hold = min_hold
        else:
            if hold > 0:
                hold -= 1
            elif p <= t_off:
                state = 0
        gate[t] = state
    return gate

def apply_hard_gate(reg_w, prob, *, t_on=0.65, t_off=0.45, min_hold=3, gate_floor=0.0):
    gate_h = hysteresis_gate(prob, t_on=t_on, t_off=t_off, min_hold=min_hold).astype(np.float32)
    gate_eff = gate_floor + (1.0 - gate_floor) * gate_h
    power_hard_w = reg_w * gate_eff
    return gate_h.astype(np.uint8), power_hard_w

@torch.no_grad()
def infer_seq2point_timeline_all_with_hard(
    model, mains, stats, device,
    *, gate_tau=0.75, t_on=0.65, t_off=0.45, min_hold=3, gate_floor=0.0
):
    power_w, reg_w, prob = infer_seq2point_timeline_all(model, mains, stats, device, gate_tau=gate_tau)
    gate_hard, power_hard_w = apply_hard_gate(reg_w, prob, t_on=t_on, t_off=t_off, min_hold=min_hold, gate_floor=gate_floor)
    return power_w, reg_w, prob, gate_hard, power_hard_w

def tune_hysteresis_for_mae(reg_w, prob, y_true, *, min_hold_choices=(2,3,4,5),
                            t_on_grid=None, t_off_grid=None, gate_floor=0.0):
    if t_on_grid is None:
        t_on_grid = np.linspace(0.55, 0.85, 7)
    if t_off_grid is None:
        t_off_grid = np.linspace(0.15, 0.55, 9)

    best = (float("inf"), 0.65, 0.45, 3)
    for t_on in t_on_grid:
        for t_off in t_off_grid:
            if t_off >= t_on:
                continue
            for mh in min_hold_choices:
                gate_h, power_hard = apply_hard_gate(reg_w, prob, t_on=t_on, t_off=t_off, min_hold=mh, gate_floor=gate_floor)
                mae = float(np.mean(np.abs(power_hard - y_true)))
                if mae < best[0]:
                    best = (mae, float(t_on), float(t_off), int(mh))
    _, b_on, b_off, b_mh = best
    return b_on, b_off, b_mh, best[0]
