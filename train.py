"""
train.py
========
Training script for the checkerboard corner detector.

Usage
-----
    python train.py                        # defaults
    python train.py --epochs 100 --batch 16 --img_size 512
    python train.py --no_pretrained        # train from scratch

Checkpoints are saved to  checkpoints/best.pt  (best validation loss).
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from torch.utils.data import DataLoader

from data.dataset import CheckerboardDataset
from model.detector import CornerDetector


def _collate(batch):
    """Stack img and hmap as tensors; keep corners as a list (variable shape)."""
    imgs    = torch.stack([b[0] for b in batch])
    hmaps   = torch.stack([b[1] for b in batch])
    corners = [b[2] for b in batch]
    return imgs, hmaps, corners


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_bce(logits: torch.Tensor,
              target: torch.Tensor,
              gamma: float = 2.0,
              alpha: float = 0.5) -> torch.Tensor:
    """
    Focal binary cross-entropy — reduces the weight of easy background
    pixels and focuses learning on corner regions.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------

def corner_pr(logits: torch.Tensor,
              corners_batch,
              nms_size: int = 7,
              threshold: float = 0.3,
              tolerance: float = 3.0) -> tuple[float, float]:
    """
    Compute mean precision and recall over a batch.

    A predicted corner is a true positive if the nearest ground-truth corner
    is within `tolerance` pixels.  Each ground-truth corner can be claimed
    by at most one prediction (greedy nearest-neighbour matching).

    Parameters
    ----------
    logits        : (B, 1, H, W) raw model output
    corners_batch : list of (n_rows, n_cols, 2) float arrays  [row, col], NaN = missing
    nms_size      : NMS window size in pixels
    threshold     : sigmoid threshold for peak detection
    tolerance     : max pixel distance to count as correct

    Returns
    -------
    mean precision, mean recall  (over samples in the batch)
    """
    heatmaps = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, H, W)
    prec_list, rec_list = [], []

    for hmap, corners in zip(heatmaps, corners_batch):
        # NMS peaks
        dilated = maximum_filter(hmap, size=nms_size)
        peaks   = np.argwhere((hmap == dilated) & (hmap > threshold)).astype(float)

        # ground-truth corners (valid only)
        gt_flat = corners.reshape(-1, 2)
        gt      = gt_flat[~np.isnan(gt_flat[:, 0])]

        if len(gt) == 0:
            continue

        if len(peaks) == 0:
            prec_list.append(0.0)
            rec_list.append(0.0)
            continue

        # greedy NN matching: peaks → gt
        matched_gt = set()
        tp = 0
        for p in peaks:
            dists = np.linalg.norm(gt - p, axis=1)
            j     = int(np.argmin(dists))
            if dists[j] <= tolerance and j not in matched_gt:
                tp += 1
                matched_gt.add(j)

        prec_list.append(tp / len(peaks))
        rec_list.append(tp / len(gt))

    if not prec_list:
        return 0.0, 0.0
    return float(np.mean(prec_list)), float(np.mean(rec_list))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    train_ds = CheckerboardDataset(length=args.n_train, img_size=args.img_size, seed=0)
    val_ds   = CheckerboardDataset(length=args.n_val,   img_size=args.img_size, seed=args.n_train)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=_collate,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=_collate,
        num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0,
    )

    model = CornerDetector(pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters : {n_params / 1e6:.2f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # --- resume from checkpoint if it exists ---
    start_epoch = 1
    best_val    = math.inf
    resume      = getattr(args, 'resume', True)
    if resume and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        best_val    = ckpt['val_loss']
        start_epoch = ckpt['epoch'] + 1
        # fast-forward scheduler to match saved epoch
        for _ in range(ckpt['epoch']):
            scheduler.step()
        print(f'Resumed from epoch {ckpt["epoch"]}  (best val {best_val:.4f})')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ---- train ----
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

        # ---- validate ----
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
                        'val_loss': val_loss}, args.save)
            print(f'  >> checkpoint saved  (val {best_val:.4f})')

    print(f'\nBest val loss: {best_val:.4f}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch',        type=int,   default=8)
    p.add_argument('--n_train',      type=int,   default=8_000)
    p.add_argument('--n_val',        type=int,   default=1_000)
    p.add_argument('--img_size',     type=int,   default=256)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--workers',      type=int,   default=4)
    p.add_argument('--save',         type=str,   default='checkpoints/best.pt')
    p.add_argument('--no_pretrained',dest='pretrained', action='store_false', default=True)
    p.add_argument('--no_resume',    dest='resume',     action='store_false', default=True)
    train(p.parse_args())
