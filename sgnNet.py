import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def auto_tcn_dilations(win_len: int) -> tuple[int, ...]:
    if win_len < 8:
        return (1,)  # smallest sensible set
    max_target = max(1, win_len // 8)        # e.g., 64→8, 256→32, 1024→128
    # largest power of two <= max_target
    d = 1
    while (d << 1) <= max_target:
        d <<= 1
    # build (1, 2, 4, ..., d)
    L = int(math.log2(d)) + 1
    return tuple(1 << i for i in range(L))



# -------------------------
# Blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, c, k=3, p=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=k//2)
        self.conv2 = nn.Conv1d(c, c, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(c)
        self.bn2 = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = F.relu(self.bn2(self.conv2(y)))
        return x + y


class TCNBlock(nn.Module):
    def __init__(self, c, k=3, d=1, p=0.1, causal=False):
        super().__init__()
        assert k % 2 == 1, "use odd k for centered padding"
        self.causal = causal
        self.left_pad = d * (k - 1)
        same_pad = d * (k - 1) // 2
        self.conv1 = nn.Conv1d(c, c, k, dilation=d, padding=0 if causal else same_pad)
        self.conv2 = nn.Conv1d(c, c, k, dilation=d, padding=0 if causal else same_pad)
        self.bn1 = nn.BatchNorm1d(c)
        self.bn2 = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        y = F.pad(x, (self.left_pad, 0)) if self.causal else x
        y = F.relu(self.bn1(self.conv1(y)))
        y = self.dropout(y)
        y = F.pad(y, (self.left_pad, 0)) if self.causal else y
        y = F.relu(self.bn2(self.conv2(y)))
        return x + y


# -------------------------
# Backbone
# -------------------------
class Backbone(nn.Module):
    def __init__(self, in_ch=1, hid=64, depth=6, kind="tcn", k=3, p=0.1,
                 dilations=(1, 2, 4, 8, 16, 32), causal=False):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, hid, kernel_size=3, padding=1)
        blocks = []
        if kind == "tcn":
            if dilations is None:
                dilations = tuple(2**i for i in range(depth))
            for d in dilations:
                blocks.append(TCNBlock(hid, k=k, d=d, p=p, causal=causal))
        elif kind == "cnn":
            for _ in range(depth):
                blocks.append(ConvBlock(hid, k=k, p=p))
        else:
            raise ValueError(f"Unknown kind: {kind}")
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = b(h)
        return h


class SGN(nn.Module):
    def __init__(self, in_ch=1, hid=64, kind="tcn", k=3, p=0.2,
                 dilations="auto", out_len=1, win_len=None,
                 gate_tau=0.75, gate_floor=0.05, use_calib=True, causal=False):
        super().__init__()
        if kind == "tcn":
            if dilations in (None, "auto"):
                if win_len is None:
                    # fallback to previous default if no window length provided
                    dilations = (1, 2, 4, 8, 16, 32)
                else:
                    dilations = auto_tcn_dilations(int(win_len))
            depth = len(dilations)
        else:
            depth = 6  # cnn default

        self.out_len = out_len
        self.gate_tau = gate_tau
        self.gate_floor = gate_floor

        self.backbone = Backbone(
            in_ch=in_ch, hid=hid,
            depth=depth, kind=kind, k=k, p=p,
            dilations=dilations if kind == "tcn" else None,
            causal=causal
        )
        self.head_reg = nn.Conv1d(hid, 1, kernel_size=1)
        self.head_cls = nn.Conv1d(hid, 1, kernel_size=1)

        self.use_calib = use_calib
        if use_calib:
            self.calib_gain = nn.Parameter(torch.tensor(1.0))
            self.calib_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = self.backbone(x)
        mid = h.size(-1) // 2
        h = h[:, :, mid:mid+1]  # seq2point center

        reg = self.head_reg(h).squeeze(1)
        reg = F.softplus(reg)

        cls_logits = self.head_cls(h).squeeze(1)
        cls_prob = torch.sigmoid(cls_logits / self.gate_tau)

        gate = self.gate_floor + (1.0 - self.gate_floor) * cls_prob
        power = reg * gate

        if self.use_calib:
            power = self.calib_gain * power + self.calib_bias

        return power, cls_logits, reg
