"""
finetune.py
===========
Gerçek kamera görüntüleriyle CNN'i fine-tune eder.

Pipeline:
  gerçek görüntü (resize 256x256)
    → sentetik radyal distorsiyon uygula  (warp_checkerboard + distort_corners)
    → photometric augment
    → heatmap üret
    → eğit

- Sentetik veri: %50 oranında karıştırılır (catastrophic forgetting önlemi)
- Düşük LR: 1e-5

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
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data.dataset import CheckerboardDataset
from data.synthesize import (warp_checkerboard, sample_monotonic_params,
                              _augment as synth_augment)
from model.detector import CornerDetector
from train import focal_bce, corner_pr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'OpticalDistortionLab', 'python'))
from optical_distortion_engine.estimation.checkerboard import distort_corners

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(REPO_DIR, '..', 'OpticalDistortionLab',
                        'datasets', 'corners.npz')
CKPT_IN  = os.path.join(REPO_DIR, 'checkpoints', 'model.pt')
CKPT_OUT = os.path.join(REPO_DIR, 'checkpoints', 'finetuned.pt')

IMG_SIZE = 256


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_heatmap(corners_rc: np.ndarray, H: int, W: int,
                  sigma: float = 1.5) -> np.ndarray:
    """corners_rc: (N, 2) — (row, col); NaN → skip."""
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


def _apply_distortion(img: np.ndarray,
                      corners_rc: np.ndarray,
                      rng: np.random.Generator):
    """
    Gerçek görüntüye sentetik radyal distorsiyon uygula.

    img        : (H, W) uint8 grayscale — zaten 256×256
    corners_rc : (N, 2) (row, col)
    """
    H, W   = img.shape
    center = ((H - 1) / 2.0, (W - 1) / 2.0)
    r_half = np.sqrt(2) * (H - 1) / 2.0

    target_r_max = float(rng.uniform(0.6, 1.2))
    PSN          = target_r_max / r_half

    c, d = sample_monotonic_params(target_r_max, rng)

    # Görüntüyü warp et
    dist_img = warp_checkerboard(img, c, d, center, PSN)

    # Köşeleri dönüştür: distort_corners (n_rows, n_cols, 2) bekliyor
    corners_grid = corners_rc.reshape(-1, 1, 2)          # (N, 1, 2)
    dist_grid    = distort_corners(corners_grid, center, PSN, c, d)
    dist_rc      = dist_grid.reshape(-1, 2)              # (N, 2)

    # Frame dışına çıkanları NaN yap
    oob = ((dist_rc[:, 0] < 0) | (dist_rc[:, 0] >= H) |
           (dist_rc[:, 1] < 0) | (dist_rc[:, 1] >= W))
    dist_rc[oob] = np.nan

    return dist_img, dist_rc


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RealCornersDataset(Dataset):
    """
    Gerçek görüntüleri 256×256'ya yeniden boyutlandırır,
    sentetik distorsiyon uygular ve heatmap üretir.
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
            self.samples.append({
                "path":     str(data[f"path_{i}"][0]),
                "corners":  data[f"corners_{i}"],          # (N,2) OpenCV (x,y)
                "img_size": tuple(data[f"img_size_{i}"]),  # (w, h) original
            })

        print(f"RealCornersDataset: {len(self.samples)} görüntü")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s       = self.samples[idx]
        S       = self.img_size
        orig_w  = int(s["img_size"][0])
        orig_h  = int(s["img_size"][1])

        # Görüntü yükle → gri → 256×256
        img = cv2.imread(s["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((S, S), dtype=np.uint8)
        else:
            img = cv2.resize(img, (S, S))

        # OpenCV (x,y) → (row,col) ve 256×256 ölçekle
        cx  = s["corners"][:, 0] * (S / orig_w)
        cy  = s["corners"][:, 1] * (S / orig_h)
        rc  = np.stack([cy, cx], axis=1)   # (N, 2) row,col

        rng = np.random.default_rng()

        if self.augment:
            # Sentetik distorsiyon uygula (her seferinde farklı parametreler)
            img, rc = _apply_distortion(img, rc, rng)
            # Photometric augment
            img = synth_augment(img.astype(float), rng)

        hmap = _make_heatmap(rc, S, S, sigma=1.5)

        img_t  = torch.from_numpy(img).float() / 255.0
        img_t  = img_t.unsqueeze(0).expand(3, -1, -1).clone()
        hmap_t = torch.from_numpy(hmap).float().unsqueeze(0)

        return img_t, hmap_t, rc


def _collate(batch):
    imgs  = torch.stack([b[0] for b in batch])
    hmaps = torch.stack([b[1] for b in batch])
    cors  = [b[2] for b in batch]
    return imgs, hmaps, cors


class MixedDataset(Dataset):
    def __init__(self, real_ds, synth_ds, length, real_ratio=0.5):
        self.real_ds    = real_ds
        self.synth_ds   = synth_ds
        self.length     = length
        self.real_ratio = real_ratio
        self.rng        = np.random.default_rng(42)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.rng.random() < self.real_ratio:
            return self.real_ds[int(self.rng.integers(0, len(self.real_ds)))]
        return self.synth_ds[int(self.rng.integers(0, len(self.synth_ds)))]


# ─── Fine-tune loop ───────────────────────────────────────────────────────────

def finetune(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    if not os.path.exists(args.ckpt_in):
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {args.ckpt_in}\n"
            "model.pt dosyasını checkpoints/ klasörüne koy.")

    ckpt  = torch.load(args.ckpt_in, map_location=device)
    model = CornerDetector(pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Model yüklendi: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

    real_ds  = RealCornersDataset(args.npz, augment=True)
    synth_ds = CheckerboardDataset(length=2000, img_size=IMG_SIZE, seed=9999)
    val_ds   = RealCornersDataset(args.npz, augment=False)

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

        model.train()
        train_loss = 0.0
        for img, hmap, _ in train_loader:
            img, hmap = img.to(device), hmap.to(device)
            loss = focal_bce(model(img), hmap)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = prec_sum = rec_sum = n_b = 0.0
        with torch.no_grad():
            for img, hmap, corners in val_loader:
                img, hmap = img.to(device), hmap.to(device)
                logits    = model(img)
                val_loss += focal_bce(logits, hmap).item() * img.size(0)
                p, r      = corner_pr(logits, corners)
                prec_sum += p; rec_sum += r; n_b += 1
        val_loss /= len(val_ds)
        prec = prec_sum / max(n_b, 1)
        rec  = rec_sum  / max(n_b, 1)

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
    print(f'Model: {args.ckpt_out}')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',          type=int,   default=20)
    p.add_argument('--batch',           type=int,   default=8)
    p.add_argument('--lr',              type=float, default=1e-5)
    p.add_argument('--real_ratio',      type=float, default=0.5)
    p.add_argument('--steps_per_epoch', type=int,   default=100)
    p.add_argument('--npz',      type=str, default=NPZ_PATH)
    p.add_argument('--ckpt_in',  type=str, default=CKPT_IN)
    p.add_argument('--ckpt_out', type=str, default=CKPT_OUT)
    finetune(p.parse_args())
