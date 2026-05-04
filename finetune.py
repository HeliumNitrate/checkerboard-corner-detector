"""
finetune.py
===========
Gerçek kamera görüntüleriyle CNN'i fine-tune eder.

- Gerçek görüntüler: corners.npz'den yüklenir
- Sentetik veri: %50 oranında karıştırılır (catastrophic forgetting önlemi)
- Düşük LR: 1e-5 (pretrained ağırlıkları korumak için)

Kullanım:
    cd checkerboard-corner-detector
    python finetune.py

    python finetune.py --epochs 20 --lr 1e-5 --real_ratio 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, gaussian_filter
from torch.utils.data import DataLoader, Dataset

from data.dataset import CheckerboardDataset
from model.detector import CornerDetector
from train import focal_bce, corner_pr

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_DIR     = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH     = os.path.join(REPO_DIR, '..', 'OpticalDistortionLab',
                             'datasets', 'corners.npz')
CKPT_IN      = os.path.join(REPO_DIR, 'checkpoints', 'model.pt')
CKPT_OUT     = os.path.join(REPO_DIR, 'checkpoints', 'finetuned.pt')

IMG_SIZE     = 256


# ─── Real-image dataset ───────────────────────────────────────────────────────

def _make_heatmap(corners_rc: np.ndarray, H: int, W: int, sigma: float = 1.5) -> np.ndarray:
    """corners_rc: (N, 2) array of (row, col) positions."""
    impulse = np.zeros((H, W), dtype=np.float32)
    for r, c in corners_rc:
        if np.isnan(r) or np.isnan(c):
            continue
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < H and 0 <= ci < W:
            impulse[ri, ci] = 1.0
    hmap = gaussian_filter(impulse, sigma=sigma)
    mx   = hmap.max()
    return hmap / mx if mx > 0 else hmap


def _augment_real(img: np.ndarray) -> np.ndarray:
    """Hafif photometric augmentation (uint8 grayscale)."""
    rng = np.random.default_rng()
    # Brightness / contrast
    alpha = rng.uniform(0.8, 1.2)
    beta  = rng.uniform(-20, 20)
    img   = np.clip(alpha * img.astype(float) + beta, 0, 255).astype(np.uint8)
    # Gaussian blur (50% prob)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # Gaussian noise
    noise = rng.normal(0, rng.uniform(0, 8), img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class RealCornersDataset(Dataset):
    """
    Gerçek kamera görüntülerinden oluşan dataset.
    corners.npz'deki her görüntüyü IMG_SIZE x IMG_SIZE'a yeniden boyutlandırır
    ve köşeleri buna göre ölçekler.
    """

    def __init__(self, npz_path: str, img_size: int = 256, augment: bool = True):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"corners.npz bulunamadı: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        n    = int(data["n_images"][0])

        self.samples  = []
        self.img_size = img_size
        self.augment  = augment

        for i in range(n):
            path     = str(data[f"path_{i}"][0])
            corners  = data[f"corners_{i}"]   # (N, 2) in OpenCV (x, y) format
            img_size_orig = data[f"img_size_{i}"]  # (w, h)
            self.samples.append((path, corners, img_size_orig))

        print(f"RealCornersDataset: {len(self.samples)} görüntü yüklendi.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, corners_xy, img_size_orig = self.samples[idx]
        orig_w, orig_h = int(img_size_orig[0]), int(img_size_orig[1])
        S = self.img_size

        # Görüntü yükle ve yeniden boyutlandır
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((S, S), dtype=np.uint8)
        else:
            img = cv2.resize(img, (S, S))

        if self.augment:
            img = _augment_real(img)

        # Köşe koordinatlarını ölçekle: OpenCV (x,y) → normalize → (row,col)
        scale_x = S / orig_w
        scale_y = S / orig_h
        x_scaled = corners_xy[:, 0] * scale_x   # col
        y_scaled = corners_xy[:, 1] * scale_y   # row
        corners_rc = np.stack([y_scaled, x_scaled], axis=1)  # (N, 2) row,col

        hmap = _make_heatmap(corners_rc, S, S, sigma=1.5)

        # Tensor'a çevir
        img_t  = torch.from_numpy(img).float() / 255.0
        img_t  = img_t.unsqueeze(0).expand(3, -1, -1).clone()
        hmap_t = torch.from_numpy(hmap).float().unsqueeze(0)

        return img_t, hmap_t, corners_rc


def _collate(batch):
    imgs    = torch.stack([b[0] for b in batch])
    hmaps   = torch.stack([b[1] for b in batch])
    corners = [b[2] for b in batch]
    return imgs, hmaps, corners


# ─── Mixed DataLoader ─────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    """
    Gerçek ve sentetik veriyi karıştırır.
    real_ratio: her batch'te gerçek görüntü oranı (0-1)
    """

    def __init__(self, real_ds: RealCornersDataset,
                 synth_ds: CheckerboardDataset,
                 length: int,
                 real_ratio: float = 0.5):
        self.real_ds    = real_ds
        self.synth_ds   = synth_ds
        self.length     = length
        self.real_ratio = real_ratio
        self.rng        = np.random.default_rng(42)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.rng.random() < self.real_ratio:
            i = int(self.rng.integers(0, len(self.real_ds)))
            return self.real_ds[i]
        else:
            i = int(self.rng.integers(0, len(self.synth_ds)))
            return self.synth_ds[i]


# ─── Fine-tuning loop ─────────────────────────────────────────────────────────

def finetune(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Model yükle
    if not os.path.exists(args.ckpt_in):
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {args.ckpt_in}\n"
            "model.pt'yi checkpoints/ klasörüne koy."
        )
    ckpt  = torch.load(args.ckpt_in, map_location=device)
    model = CornerDetector(pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Model yüklendi: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    # Dataset
    real_ds  = RealCornersDataset(args.npz, img_size=IMG_SIZE, augment=True)
    synth_ds = CheckerboardDataset(length=2000, img_size=IMG_SIZE, seed=9999)
    val_ds   = RealCornersDataset(args.npz, img_size=IMG_SIZE, augment=False)

    train_ds = MixedDataset(real_ds, synth_ds,
                            length=args.steps_per_epoch * args.batch,
                            real_ratio=args.real_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=_collate, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for img, hmap, _ in train_loader:
            img, hmap = img.to(device), hmap.to(device)
            loss = focal_bce(model(img), hmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        # ── Val (gerçek görüntülerle) ──
        model.eval()
        val_loss = 0.0
        prec_sum = rec_sum = n_batches = 0
        with torch.no_grad():
            for img, hmap, corners in val_loader:
                img, hmap = img.to(device), hmap.to(device)
                logits     = model(img)
                val_loss  += focal_bce(logits, hmap).item() * img.size(0)
                p, r       = corner_pr(logits, corners)
                prec_sum  += p
                rec_sum   += r
                n_batches += 1
        val_loss /= len(val_ds)
        prec = prec_sum / max(n_batches, 1)
        rec  = rec_sum  / max(n_batches, 1)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train {train_loss:.4f}  val {val_loss:.4f}  '
              f'prec {prec:.3f}  rec {rec:.3f}  '
              f'lr {lr:.1e}  {dt:.0f}s')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_loss': val_loss}, args.ckpt_out)
            print(f'  >> checkpoint saved  (val {best_val:.4f})')

    print(f'\nBest val loss: {best_val:.4f}')
    print(f'Model kaydedildi: {args.ckpt_out}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',          type=int,   default=20)
    p.add_argument('--batch',           type=int,   default=8)
    p.add_argument('--lr',              type=float, default=1e-5)
    p.add_argument('--real_ratio',      type=float, default=0.5,
                   help='Her adımda gerçek görüntü oranı (0-1)')
    p.add_argument('--steps_per_epoch', type=int,   default=100,
                   help='Epoch başına adım sayısı')
    p.add_argument('--npz',    type=str, default=NPZ_PATH)
    p.add_argument('--ckpt_in',  type=str, default=CKPT_IN)
    p.add_argument('--ckpt_out', type=str, default=CKPT_OUT)
    finetune(p.parse_args())
