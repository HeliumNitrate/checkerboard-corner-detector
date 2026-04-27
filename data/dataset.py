"""
dataset.py
==========
PyTorch Dataset that generates synthetic checkerboard samples on-the-fly.
Each index maps to a deterministic RNG seed so that the same index always
produces the same sample (reproducible validation).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthesize import generate_sample


class CheckerboardDataset(Dataset):
    """
    On-the-fly synthetic dataset.  No disk storage required.

    Parameters
    ----------
    length   : virtual dataset size (affects epoch length, not diversity)
    img_size : spatial resolution of generated images
    seed     : base RNG seed; sample i uses seed = base + i
    """

    def __init__(self, length: int = 10_000, img_size: int = 256, seed: int = 0):
        self.length   = length
        self.img_size = img_size
        self.seed     = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        rng    = np.random.default_rng(self.seed + idx)
        sample = generate_sample(img_size=self.img_size, rng=rng)

        # (H, W) uint8  →  (3, H, W) float32 in [0, 1]  (repeat grayscale → RGB)
        img = torch.from_numpy(sample['image']).float() / 255.0
        img = img.unsqueeze(0).expand(3, -1, -1).clone()

        # (H, W) float32  →  (1, H, W)
        hmap = torch.from_numpy(sample['heatmap']).float().unsqueeze(0)

        # (n_rows, n_cols, 2) float64 — variable shape, kept as numpy for collation
        return img, hmap, sample['corners']
