import torch
import torch.nn.functional as F

__all__ = ["focal_bce_with_logits", "delta_from_watts", "sgn_loss"]

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, eps=1e-6):
    """
    Focal BCE for binary classification.
    logits [...], targets [...] in {0,1}
    pos_weight: scalar to upweight positives (e.g., neg/pos ratio)
    """
    p = torch.sigmoid(logits)
    if pos_weight is None:
        bce = -(targets * torch.log(p + eps) + (1 - targets) * torch.log(1 - p + eps))
    else:
        bce = -(pos_weight * targets * torch.log(p + eps) +
                (1 - targets) * torch.log(1 - p + eps))
    pt = targets * p + (1 - targets) * (1 - p)
    focal = (1 - pt).pow(gamma) * bce
    return focal.mean()

def delta_from_watts(delta_watts: float, target_scale: float) -> float:
    return float(delta_watts) / (float(target_scale) + 1e-6)

def _ensure_flat(*tensors):
    flat = []
    for t in tensors:
        flat.append(None if t is None else t.reshape(-1))
    return flat

def sgn_loss(
    power_pred,      # gated = reg * gate, [B] or [B,L]
    cls_logits,      # logits, [B] or [B,L]
    reg_pred,        # raw regression >=0, [B] or [B,L]
    y_true_power,    # target (scaled), [B] or [B,L]
    y_true_on,       # {0,1}, [B] or [B,L]
    *,
    delta_huber=0.05,
    alpha_on=1.0,        # ON regression on gated power
    alpha_reg_raw=1.0,   # ON regression on raw reg head (purity)
    alpha_off=0.02,      # OFF leakage penalty on gated power (tiny)
    beta_cls=1.5,        # classification weight
    focal_gamma=2.0,
    pos_weight=None
):
    power_pred, cls_logits, reg_pred, y_true_power, y_true_on = _ensure_flat(
        power_pred, cls_logits, reg_pred, y_true_power, y_true_on
    )

    on_mask  = (y_true_on > 0.5)
    off_mask = ~on_mask

    if on_mask.any():
        reg_on = F.smooth_l1_loss(
            power_pred[on_mask], y_true_power[on_mask], beta=delta_huber
        )
        reg_on_raw = F.smooth_l1_loss(
            reg_pred[on_mask], y_true_power[on_mask], beta=delta_huber
        )
    else:
        reg_on = power_pred.new_tensor(0.0)
        reg_on_raw = power_pred.new_tensor(0.0)

    if off_mask.any():
        reg_off = power_pred[off_mask].abs().mean()
    else:
        reg_off = power_pred.new_tensor(0.0)

    cls_loss = focal_bce_with_logits(cls_logits, y_true_on.float(),
                                     gamma=focal_gamma, pos_weight=pos_weight)

    total = alpha_on * reg_on + alpha_reg_raw * reg_on_raw + alpha_off * reg_off + beta_cls * cls_loss
    logs = {
        "reg_on": reg_on.detach(),
        "reg_on_raw": reg_on_raw.detach(),
        "reg_off": reg_off.detach(),
        "cls": cls_loss.detach(),
        "total": total.detach()
    }
    return total, logs
