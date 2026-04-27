"""
synthesize.py
=============
Synthetic training data generation for the checkerboard corner detector.

Pipeline per sample
-------------------
1. make_checkerboard(img_size, img_size, square_px)  – perfect undistorted board
2. sample_monotonic_params(r_max_norm)               – random c, d within monotonicity constraint
3. warp_checkerboard(img, c, d, center, PSN)         – backward-mapped distorted image
4. distort_corners(corners, center, PSN, c, d)       – exact distorted corner positions
5. _perspective_augment(img, corners, max_tilt=10°)  – simulates board tilt (50 % probability)
6. make_heatmap(corners, H, W)                       – Gaussian-blob label map
7. _augment(img)                                     – photometric augmentation

Perspective note
----------------
The perspective step trains the CNN to detect corners even when the board is
slightly tilted in X/Y (up to MAX_TILT_DEG degrees).  For distortion *estimation*
the user should keep the board nearly perpendicular to the optical axis (Z-tilt
only); X/Y tilt breaks the radial-symmetry assumption of estimate_from_corners.
"""

from __future__ import annotations

import io
import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from optical_distortion_engine.estimation.checkerboard import (
    make_checkerboard, distort_corners,
)


# ---------------------------------------------------------------------------
# Distortion inversion (Newton-Raphson)
# ---------------------------------------------------------------------------

def invert_distortion(r_out: np.ndarray,
                      c: np.ndarray,
                      d: np.ndarray,
                      n_iter: int = 20) -> np.ndarray:
    """
    Find r_in such that r_in + sum(c_m * r_in^d_m) = r_out.
    Initialised at r_in = r_out (identity).  Converges in < 10 iterations
    for all physically meaningful distortions.
    """
    c = np.asarray(c).ravel()
    d = np.asarray(d).ravel()
    r = r_out.copy()
    for _ in range(n_iter):
        f  = r + np.sum(c * r[:, None] ** d,           axis=1) - r_out
        fp = 1.0 + np.sum(c * d * r[:, None] ** (d - 1), axis=1)
        r -= f / np.where(np.abs(fp) > 1e-10, fp, 1e-10)
    return np.maximum(r, 0.0)


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

def warp_checkerboard(img: np.ndarray,
                      c: np.ndarray,
                      d: np.ndarray,
                      center: tuple,
                      pix_size_norm: float) -> np.ndarray:
    """
    Render distorted checkerboard via backward mapping.

    For each output pixel at distorted position r_out, invert the distortion
    to find the undistorted source position r_in, then bilinear-interpolate.

    Parameters
    ----------
    img           : (H, W) uint8 undistorted checkerboard
    c, d          : distortion constants and polynomial degrees
    center        : (cy, cx) optical centre in pixels
    pix_size_norm : pixel_size / focal_length

    Returns
    -------
    (H, W) uint8 distorted image
    """
    H, W = img.shape
    cy, cx = center

    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    dx_n = (cc - cx) * pix_size_norm
    dy_n = -(rr - cy) * pix_size_norm
    r_out_n = np.hypot(dx_n, dy_n).ravel()

    r_in_n = invert_distortion(r_out_n, c, d)

    safe = np.where(r_out_n > 1e-10, r_out_n, 1.0)
    scale = np.where(r_out_n > 1e-10, r_in_n / safe, 1.0)

    src_col = (cx + dx_n.ravel() * scale / pix_size_norm).reshape(H, W)
    src_row = (cy - dy_n.ravel() * scale / pix_size_norm).reshape(H, W)

    warped = map_coordinates(
        img.astype(float),
        [src_row.ravel(), src_col.ravel()],
        order=1, mode='constant', cval=128,
    ).reshape(H, W)
    return np.clip(warped, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def make_heatmap(corners: np.ndarray,
                 H: int, W: int,
                 sigma: float = 1.5) -> np.ndarray:
    """
    Gaussian-blob heatmap from corner grid positions.

    corners : (n_rows, n_cols, 2) float array, NaN for missing corners
    Returns : (H, W) float32 in [0, 1]
    """
    flat = corners.reshape(-1, 2)
    impulse = np.zeros((H, W), dtype=np.float32)
    for r, c in flat:
        if np.isnan(r) or np.isnan(c):
            continue
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < H and 0 <= ci < W:
            impulse[ri, ci] = 1.0
    heatmap = gaussian_filter(impulse, sigma=sigma)
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx
    return heatmap


# ---------------------------------------------------------------------------
# Distortion parameter sampling
# ---------------------------------------------------------------------------

def sample_monotonic_params(r_max_norm: float,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample (c, d) that are globally monotonic (dr_out/dr_in > 0 for all r >= 0).

    Model: r_out = r_in + c[0]*r_in^3 + c[1]*r_in^5
    Global monotonicity requires c[1] >= 9*c[0]^2 / 20  when c[0] < 0.

    Returns
    -------
    c : (2,) float array of distortion constants
    d : (2,) float array = [3., 5.]
    """
    d = np.array([3.0, 5.0])
    p = rng.random()

    if p < 0.12:                                    # ~12% identity
        return np.zeros(2), d

    if p < 0.56:                                    # ~44% barrel
        c1 = -rng.uniform(0.03, 0.50)
        c2_min = 9 * c1 ** 2 / 20                  # monotonicity floor
        c2 = c2_min * rng.uniform(1.05, 2.0)       # margin above floor
        return np.array([c1, c2]), d

    else:                                           # ~44% pincushion
        c1 = rng.uniform(0.03, 0.50)
        c2 = rng.uniform(0.0, c1 * 0.5)            # positive → always monotonic
        return np.array([c1, c2]), d


# ---------------------------------------------------------------------------
# Perspective augmentation  (simulates board tilt ≤ MAX_TILT_DEG degrees)
# ---------------------------------------------------------------------------

MAX_TILT_DEG = 10.0


def _compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    DLT homography: 4×(col,row) source/destination pairs → 3×3 matrix H.
    H maps src homogeneous coords to dst homogeneous coords.
    """
    A = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([-xs, -ys, -1,   0,    0,   0, xd*xs, xd*ys, xd])
        A.append([  0,    0,  0, -xs, -ys, -1, yd*xs, yd*ys, yd])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=float))
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]
    return H


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply H to (N,2) array of (col,row)=(x,y) coordinates.
    Returns (N,2) in the same (col,row) convention.
    """
    ones  = np.ones((len(pts), 1))
    dst_h = np.hstack([pts, ones]) @ H.T       # (N, 3)
    return dst_h[:, :2] / dst_h[:, 2:3]


def _perspective_augment(img: np.ndarray,
                          corners: np.ndarray,
                          max_angle_deg: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a random perspective transform simulating board tilt.

    Displacement model
    ------------------
    Two independent tilts are sampled (rotation around horizontal axis tilt_x
    and around vertical axis tilt_y), each bounded by
        max_disp = min(H,W) * sin(max_angle_deg) / 2.
    The four image corners are displaced as:
        TL  ←  (−tilt_y/2, −tilt_x/2)    (col, row)
        TR  ←  (+tilt_y/2, −tilt_x/2)
        BR  ←  (+tilt_y/2, +tilt_x/2)
        BL  ←  (−tilt_y/2, +tilt_x/2)
    This preserves the physical anti-symmetry of a rigid planar board.

    The same homography is applied to the image (backward mapping) and to
    the corner positions (forward mapping).  Out-of-frame corners should be
    re-filtered by the caller.
    """
    H, W = img.shape
    max_disp = min(H, W) * np.sin(np.radians(max_angle_deg)) / 2.0

    tx = rng.uniform(-max_disp, max_disp)   # horizontal-axis tilt  (row displacement)
    ty = rng.uniform(-max_disp, max_disp)   # vertical-axis tilt    (col displacement)

    # Image corner correspondences  (col, row) = (x, y)
    src = np.array([[0,   0  ],
                    [W-1, 0  ],
                    [W-1, H-1],
                    [0,   H-1]], dtype=float)
    dst = src + np.array([[-ty/2, -tx/2],
                           [+ty/2, -tx/2],
                           [+ty/2, +tx/2],
                           [-ty/2, +tx/2]], dtype=float)

    H_fwd = _compute_homography(src, dst)   # undistorted → perspective-warped
    H_inv = np.linalg.inv(H_fwd)

    # --- warp image (backward mapping: output pixel → source pixel) ---
    cc, rr = np.meshgrid(np.arange(W), np.arange(H))
    pts_xy  = np.column_stack([cc.ravel().astype(float), rr.ravel().astype(float)])
    src_xy  = _apply_homography(H_inv, pts_xy)

    warped = map_coordinates(
        img.astype(float),
        [src_xy[:, 1].reshape(H, W), src_xy[:, 0].reshape(H, W)],
        order=1, mode='constant', cval=128,
    ).reshape(H, W)
    warped_img = np.clip(warped, 0, 255).astype(np.uint8)

    # --- transform corners (forward mapping: source position → warped position) ---
    flat  = corners.reshape(-1, 2).copy()   # (N, 2) as (row, col)
    valid = ~np.isnan(flat[:, 0])
    if valid.any():
        pts_xy       = flat[valid][:, ::-1]            # (col, row)
        dst_xy       = _apply_homography(H_fwd, pts_xy)
        flat[valid]  = dst_xy[:, ::-1]                 # back to (row, col)

    return warped_img, flat.reshape(corners.shape)


# ---------------------------------------------------------------------------
# Photometric augmentation
# ---------------------------------------------------------------------------

def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric augmentation only — does not alter geometry."""
    f = img.astype(float)

    if rng.random() < 0.6:
        f = gaussian_filter(f, sigma=rng.uniform(0.3, 1.5))

    if rng.random() < 0.7:
        f = f * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)

    if rng.random() < 0.8:
        f = f + rng.normal(0, rng.uniform(0, 15), f.shape)

    if rng.random() < 0.4:
        quality = int(rng.uniform(60, 95))
        pil = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        f = np.array(Image.open(buf)).astype(float)

    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Top-level sample generator
# ---------------------------------------------------------------------------

# square_px values that give an odd number of inner corners for img_size=256
_SQUARE_OPTIONS_256 = [8, 10, 16, 20, 32]   # 256//s gives even n_squares → odd n_inner


def generate_sample(img_size: int = 256,
                    rng: np.random.Generator = None) -> dict:
    """
    Generate one synthetic (distorted_image, heatmap) training pair.

    Returns
    -------
    dict with keys:
        image    : (H, W) uint8  – distorted + augmented checkerboard
        heatmap  : (H, W) float32 in [0, 1]  – Gaussian blobs at corner positions
        corners  : (n_rows, n_cols, 2) float  – distorted corner positions (NaN if outside)
        params   : dict  – c, d, PSN, square_px used
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- checkerboard config ---
    square_px = int(rng.choice(_SQUARE_OPTIONS_256))
    img, corners = make_checkerboard(img_size, img_size, square_px)
    n_rows, n_cols = corners.shape[:2]

    # --- optical parameters ---
    center = ((img_size - 1) / 2.0, (img_size - 1) / 2.0)
    r_half_diag = float(np.sqrt(2) * (img_size - 1) / 2.0)
    target_r_max = float(rng.uniform(0.6, 1.2))
    PSN = target_r_max / r_half_diag

    # --- distortion ---
    c, d = sample_monotonic_params(target_r_max, rng)

    dist_img     = warp_checkerboard(img, c, d, center, PSN)
    dist_corners = distort_corners(corners, center, PSN, c, d)

    # --- perspective augmentation (50 % probability, ≤ MAX_TILT_DEG°) ---
    tilted = False
    if rng.random() < 0.5:
        dist_img, dist_corners = _perspective_augment(
            dist_img, dist_corners, MAX_TILT_DEG, rng,
        )
        tilted = True

    # mark corners outside the frame as NaN (after all geometric transforms)
    flat = dist_corners.reshape(-1, 2)
    oob  = ((flat[:, 0] < 0) | (flat[:, 0] >= img_size) |
            (flat[:, 1] < 0) | (flat[:, 1] >= img_size))
    flat[oob] = np.nan
    dist_corners = flat.reshape(n_rows, n_cols, 2)

    # --- heatmap ---
    sigma   = max(1.0, square_px / 10.0)
    heatmap = make_heatmap(dist_corners, img_size, img_size, sigma=sigma)

    # --- photometric augmentation ---
    dist_img = _augment(dist_img, rng)

    return {
        'image'  : dist_img,
        'heatmap': heatmap,
        'corners': dist_corners,
        'params' : {'c': c, 'd': d, 'PSN': PSN, 'square_px': square_px, 'tilted': tilted},
    }
