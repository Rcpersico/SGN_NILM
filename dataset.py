import numpy as np
import torch
from torch.utils.data import Dataset



def make_onoff_labels(target_watts, on_threshold=20.0, off_threshold=10.0, min_hold=1):
    on = np.zeros_like(target_watts, dtype=np.uint8)
    state = 0
    hold = 0
    for i, p in enumerate(target_watts):
        if state == 0:
            if p >= on_threshold:
                state = 1
                hold = min_hold
        else:
            if p <= off_threshold and hold <= 0:
                state = 0
            else:
                hold -= 1
        on[i] = state
    return on



class Seq2PointWindows(Dataset):
    def __init__(self, mains, target, win_len=512, stride=32,
                 mains_mean=None, mains_std=None, target_scale=None, train=True):
        assert len(mains) == len(target), "Mains and target must have the same length"
        self.mains = mains.astype(np.float32)
        self.target = target.astype(np.float32)
        self.win_len = win_len
        self.train = train

        # normalization (fit on train; reuse stats for val/test)
        self.mains_mean = float(np.mean(self.mains)) if mains_mean is None else float(mains_mean)
        self.mains_std  = float(np.std(self.mains) + 1e-6) if mains_std is None else float(mains_std)
        self.target_scale = float(np.percentile(np.abs(self.target), 99)) if target_scale is None else float(target_scale)

        self.mains  = (self.mains - self.mains_mean) / self.mains_std
        self.target = self.target / (self.target_scale + 1e-6)

        # ON/OFF labels built from unscaled watts (reconstruct)
        self.onoff = make_onoff_labels(self.target * (self.target_scale + 1e-6))

        half = win_len // 2
        valid = np.arange(half, len(self.mains) - (win_len - half) + 1)

        if train:
            self.centers = valid
            self.stride = stride
        else:
            self.centers = valid
            self.stride = 1

    def __len__(self):
        if self.train:
            # approximate number of samples per epoch
            return max(1, len(self.centers) // self.stride)
        return len(self.centers)

    def __getitem__(self, idx):
        if self.train:
            # random center position each call
            c = np.random.choice(self.centers)
        else:
            c = self.centers[idx]

        L = self.win_len
        half = L // 2
        start = c - half
        end = start + L

        x = self.mains[start:end]               # [L]
        y = self.target[c]                      # scalar (seq2point)
        s = self.onoff[c]                       # 0/1

        return (
            torch.from_numpy(x[None, :]),                       # [1, L], float32
            torch.tensor([y], dtype=torch.float32),             # [1]
            torch.tensor([s], dtype=torch.float32),             # [1]
        )
