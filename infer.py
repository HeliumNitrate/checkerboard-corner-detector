"""
infer.py
========
Inference pipeline: distorted image → corner positions → distortion constants.

Full pipeline
-------------
1. CornerDetector  : image  → heatmap
2. NMS             : heatmap → (N, 2) corner candidates
3. sort_corners_to_grid : candidates → (n_rows, n_cols, 2) grid
4. estimate_from_corners: grid → distortion constants

Real-image active-learning helper
----------------------------------
When the detected grid looks wrong, call ``review_corners`` to overlay
detected corners on the image and let the user identify bad ones.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import torch
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from optical_distortion_engine.estimation.corner_detect import sort_corners_to_grid
from optical_distortion_engine.estimation.estimator import estimate_from_corners
from model.detector import CornerDetector


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str,
               device: torch.device = None) -> CornerDetector:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = CornerDetector(pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Corner detection from heatmap
# ---------------------------------------------------------------------------

def predict_heatmap(model: CornerDetector,
                    img: np.ndarray,
                    device: torch.device = None) -> np.ndarray:
    """
    Run the detector on a single (H, W) or (H, W, C) image.
    Returns the (H, W) sigmoid heatmap as a numpy float32 array.
    """
    if device is None:
        device = next(model.parameters()).device

    if img.ndim == 3:
        img = img.mean(axis=2)

    H, W = img.shape
    # Pad to a multiple of 32 (required by EfficientNet stride pattern)
    Hp = math.ceil(H / 32) * 32
    Wp = math.ceil(W / 32) * 32
    padded = np.pad(img, ((0, Hp - H), (0, Wp - W)), mode='reflect')

    t = torch.from_numpy(padded).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).clone().to(device)

    with torch.no_grad():
        logits = model(t)

    return torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()


def heatmap_to_corners(heatmap: np.ndarray,
                        min_distance: int = 8,
                        threshold: float = 0.3) -> np.ndarray:
    """NMS on heatmap → (N, 2) corner positions (row, col)."""
    dilated   = maximum_filter(heatmap, size=min_distance)
    peaks     = (heatmap == dilated) & (heatmap > threshold)
    corners   = np.argwhere(peaks).astype(float)
    # Sort by descending response for reproducibility
    responses = heatmap[peaks]
    order     = np.argsort(-responses)
    return corners[order]


# ---------------------------------------------------------------------------
# Full estimation pipeline
# ---------------------------------------------------------------------------

def estimate_distortion(model: CornerDetector,
                        img: np.ndarray,
                        n_inner_rows: int,
                        n_inner_cols: int,
                        image_center,
                        pix_size_norm: float,
                        pol_degree,
                        min_distance: int = 8,
                        threshold: float = 0.3) -> tuple[np.ndarray, dict]:
    """
    End-to-end: distorted image → distortion constants.

    Parameters
    ----------
    model         : trained CornerDetector
    img           : (H, W) or (H, W, C) uint8 distorted checkerboard
    n_inner_rows  : expected number of inner corner rows
    n_inner_cols  : expected number of inner corner columns
    image_center  : (cy, cx) optical centre in pixels
    pix_size_norm : pixel_size / focal_length
    pol_degree    : polynomial degrees for the distortion model
    min_distance  : NMS suppression radius in pixels
    threshold     : heatmap confidence threshold

    Returns
    -------
    constants : fitted distortion coefficients
    info      : diagnostics dict (includes heatmap, corners_grid)
    """
    heatmap    = predict_heatmap(model, img)
    corners_raw = heatmap_to_corners(heatmap, min_distance, threshold)

    corners_grid = sort_corners_to_grid(
        corners_raw, n_inner_rows, n_inner_cols, image_center=image_center,
    )
    constants, info = estimate_from_corners(
        corners_grid, image_center, pix_size_norm, pol_degree,
    )
    info['heatmap']      = heatmap
    info['corners_grid'] = corners_grid
    return constants, info


# ---------------------------------------------------------------------------
# Active-learning helper for real images
# ---------------------------------------------------------------------------

def review_corners(img: np.ndarray,
                   corners_grid: np.ndarray,
                   save_path: str = None):
    """
    Overlay detected corners on the image (matplotlib).
    Valid corners in green, NaN positions shown as red crosses.
    Useful for spotting mis-assigned corners before manual correction.

    Parameters
    ----------
    img          : (H, W) or (H, W, C) image
    corners_grid : (n_rows, n_cols, 2) from sort_corners_to_grid
    save_path    : if given, save the figure instead of showing it
    """
    import matplotlib.pyplot as plt

    flat    = corners_grid.reshape(-1, 2)
    valid   = ~np.isnan(flat[:, 0])
    n_rows, n_cols = corners_grid.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(img)

    ax.scatter(flat[valid, 1],  flat[valid, 0],  s=12, c='lime',
               linewidths=0, label=f'detected ({valid.sum()})')
    ax.scatter(flat[~valid, 1], flat[~valid, 0], s=20, c='red',
               marker='x', label=f'missing ({(~valid).sum()})')
    ax.set_title(f'Corner grid  {n_rows}×{n_cols}  —  '
                 f'{valid.sum()}/{n_rows*n_cols} detected')
    ax.legend(fontsize=9)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved corner review → {save_path}')
    else:
        plt.show()
    plt.close(fig)
