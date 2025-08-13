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
    def __init__(self,
                 mains, target,
                 win_len=512, stride=32,
                 mains_mean=None, mains_std=None, target_scale=None,
                 train=True,
                 on_threshold=20.0, off_threshold=10.0, min_hold=1,
                 balance_sampling=True, on_sample_prob=0.5):
        assert len(mains) == len(target)
        self.mains = mains.astype(np.float32)
        self.target = target.astype(np.float32)
        self.win_len = win_len
        self.train = train
        self.balance_sampling = bool(balance_sampling) and bool(train)
        self.on_sample_prob = float(on_sample_prob)

        self.mains_mean = float(np.mean(self.mains)) if mains_mean is None else float(mains_mean)
        self.mains_std  = float(np.std(self.mains) + 1e-6) if mains_std is None else float(mains_std)
        self.target_scale = float(np.percentile(np.abs(self.target), 99)) if target_scale is None else float(target_scale)

        self.mains  = (self.mains - self.mains_mean) / self.mains_std
        self.target = self.target / (self.target_scale + 1e-6)

        target_watts = self.target * (self.target_scale + 1e-6)
        self.onoff = make_onoff_labels(
            target_watts,
            on_threshold=on_threshold,
            off_threshold=off_threshold,
            min_hold=min_hold
        )

        half = win_len // 2
        valid = np.arange(half, len(self.mains) - (win_len - half) + 1)
        self.centers = valid
        self.stride = stride if train else 1

        if self.balance_sampling:
            on_mask = (self.onoff[self.centers] == 1)
            self.on_centers  = self.centers[on_mask]
            self.off_centers = self.centers[~on_mask]
            if len(self.on_centers) == 0:
                self.on_centers = self.off_centers
            if len(self.off_centers) == 0:
                self.off_centers = self.on_centers

    def __len__(self):
        if self.train:
            return max(1, len(self.centers) // self.stride)
        return len(self.centers)

    def _sample_center(self):
        if self.balance_sampling and len(self.on_centers) > 0 and len(self.off_centers) > 0:
            if np.random.rand() < self.on_sample_prob:
                return int(np.random.choice(self.on_centers))
            else:
                return int(np.random.choice(self.off_centers))
        return int(np.random.choice(self.centers))

    def __getitem__(self, idx):
        c = self._sample_center() if self.train else int(self.centers[idx])
        L = self.win_len
        half = L // 2
        start = c - half
        end = start + L

        x = self.mains[start:end]
        y = self.target[c]
        s = self.onoff[c]

        return (
            torch.from_numpy(x[None, :]),
            torch.tensor([y], dtype=torch.float32),
            torch.tensor([s], dtype=torch.float32),
        )

    @property
    def on_rate(self) -> float:
        return float(np.mean(self.onoff))
