import numpy as np
import torch
from torch.utils.data import Dataset

def make_onoff_labels(target_watts, on_threshold=20.0, off_threshold=10.0, min_hold=1):
    on = np.zeros_like(target_watts, dtype=np.uint8)
    state, hold = 0, 0
    for i, p in enumerate(target_watts):
        if state == 0:
            if p >= on_threshold:
                state, hold = 1, min_hold
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
                 balance_sampling=True, on_sample_prob=0.5,
                 # NEW: fraction of batches to draw from transition (ON↔OFF) edges
                 edge_sample_prob=0.0,
                 # === NEW ROBUST-SCALE & EDGE OPTIONS (safe defaults) ===
                 target_scale_floor=50.0,                 # minimum watts for scale to avoid blow-ups
                 target_scale_pos_quantile=95.0,          # percentile of positives to use when available
                 target_scale_global_quantile=99.9,       # fallback percentile of all values
                 min_pos_for_scale=50,                    # min #positive samples required to trust pos-quantile
                 edge_dilate=0,                           # ±k-center dilation around label transitions
                 fail_on_empty_on_pool=True               # raise if ON pool is empty when balance_sampling=True
                 ):
        assert len(mains) == len(target)
        self.train = bool(train)
        self.win_len = int(win_len)
        self.stride = int(stride) if self.train else 1
        self.balance_sampling = bool(balance_sampling) and self.train
        self.on_sample_prob = float(on_sample_prob)
        self.edge_sample_prob = float(np.clip(edge_sample_prob, 0.0, 1.0))
        self.edge_dilate = int(max(0, edge_dilate))
        self.fail_on_empty_on_pool = bool(fail_on_empty_on_pool)

        # ---- Raw arrays (float32) ----
        self.mains = mains.astype(np.float32)
        raw_target = target.astype(np.float32)

        # ---- Standardize mains ----
        self.mains_mean = float(np.mean(self.mains)) if mains_mean is None else float(mains_mean)
        self.mains_std  = float(np.std(self.mains) + 1e-6) if mains_std is None else float(mains_std)
        mains_n = (self.mains - self.mains_mean) / self.mains_std

        # ---- Robust target scale for sparse channels ----
        if target_scale is None:
            # Choose "positive" mask: > max(off_threshold, 0.5*on_threshold)
            pos_cut = max(off_threshold, 0.5 * on_threshold)
            pos_mask = raw_target > pos_cut
            if np.sum(pos_mask) >= int(min_pos_for_scale):
                ts = float(np.percentile(np.abs(raw_target[pos_mask]), float(target_scale_pos_quantile)))
            else:
                ts = float(np.percentile(np.abs(raw_target), float(target_scale_global_quantile)))
            ts = max(ts, float(target_scale_floor))
        else:
            ts = float(target_scale)

        self.target_scale = ts
        target_n = raw_target / (self.target_scale + 1e-6)

        # ---- Labels on WATTS (use raw_target, not normalized) ----
        self.onoff = make_onoff_labels(raw_target,
                                       on_threshold=on_threshold,
                                       off_threshold=off_threshold,
                                       min_hold=min_hold)

        # ---- Build valid centers for seq2point ----
        half = self.win_len // 2
        valid = np.arange(half, len(self.mains) - (self.win_len - half) + 1)
        self.centers = valid

        # ---- Keep normalized arrays for training ----
        self.mains = mains_n
        self.target = target_n

        # ---- Pools for class-balanced sampling (REG or optional CLS OFF-tilt) ----
        if self.balance_sampling:
            on_mask_centers = (self.onoff[self.centers] == 1)
            self.on_centers  = self.centers[on_mask_centers]
            self.off_centers = self.centers[~on_mask_centers]

            if self.fail_on_empty_on_pool and len(self.on_centers) == 0:
                raise ValueError(
                    "Seq2PointWindows: No ON centers in TRAIN split with current thresholds. "
                    "Adjust split/thresholds or disable balance_sampling."
                )
            # Fallbacks only if not failing
            if not self.fail_on_empty_on_pool:
                if len(self.on_centers) == 0:
                    self.on_centers = self.off_centers
                if len(self.off_centers) == 0:
                    self.off_centers = self.on_centers

        # ---- Transition/edge pool for CLS emphasis (with optional dilation) ----
        change = np.zeros_like(self.onoff, dtype=bool)
        change[1:]  |= (self.onoff[1:]  != self.onoff[:-1])
        change[:-1] |= (self.onoff[:-1] != self.onoff[1:])

        if self.edge_dilate > 0:
            edge = change.astype(np.uint8)
            k = self.edge_dilate
            kernel = np.ones(2 * k + 1, dtype=np.uint8)
            dil = np.convolve(edge, kernel, mode='same') > 0
            edge_mask = dil
        else:
            edge_mask = change

        self.edge_centers = self.centers[edge_mask[self.centers]]

    def __len__(self):
        if self.train:
            return max(1, len(self.centers) // self.stride)
        return len(self.centers)

    def _sample_center(self):
        # 1) Class-balanced sampling (REG or CLS OFF-tilt) if enabled & pools exist
        if self.balance_sampling:
            has_on  = hasattr(self, "on_centers")  and len(self.on_centers)  > 0
            has_off = hasattr(self, "off_centers") and len(self.off_centers) > 0
            if has_on and has_off:
                if np.random.rand() < self.on_sample_prob:
                    return int(np.random.choice(self.on_centers))
                else:
                    return int(np.random.choice(self.off_centers))

        # 2) Otherwise, allow edge oversampling for CLS
        if self.train and self.edge_sample_prob > 0.0 and len(self.edge_centers) > 0:
            if np.random.rand() < self.edge_sample_prob:
                return int(np.random.choice(self.edge_centers))

        # 3) Plain unbiased
        return int(np.random.choice(self.centers))

    def __getitem__(self, idx):
        c = self._sample_center() if self.train else int(self.centers[idx])
        L = self.win_len; half = L // 2
        start = c - half; end = start + L
        x = self.mains[start:end]
        y = self.target[c]
        s = self.onoff[c]
        return (
            torch.from_numpy(x[None, :]),
            torch.tensor([y], dtype=torch.float32),
            torch.tensor([s], dtype=torch.float32),
        )

    # === Diagnostics ===
    @property
    def on_rate(self) -> float:
        """ON-rate computed over centers (what training actually sees)."""
        return float(np.mean(self.onoff[self.centers])) if len(self.centers) > 0 else 0.0

    @property
    def n_centers(self) -> int:
        return int(len(self.centers))

    @property
    def n_on_centers(self) -> int:
        return int(np.sum(self.onoff[self.centers] == 1)) if len(self.centers) > 0 else 0

    @property
    def n_edge_centers(self) -> int:
        return int(len(self.edge_centers))
