"""
checkerboard.py
===============
Generate synthetic checkerboard images and apply mathematical distortion
to corner positions for estimation pipeline validation.
"""

import numpy as np
from ..core.distortion_fun import distortion_fun


def make_checkerboard(height: int, width: int, square_px: int):
    """
    Generate a grayscale checkerboard and return its inner corner grid.

    Parameters
    ----------
    height, width : image dimensions in pixels
    square_px     : side length of each square in pixels

    Returns
    -------
    img          : (height, width) uint8 array
    corners_grid : (n_inner_rows, n_inner_cols, 2) float array of (row, col)
    """
    row_idx = np.arange(height)[:, None]
    col_idx = np.arange(width)[None, :]
    img = (((row_idx // square_px) + (col_idx // square_px)) % 2 * 255).astype(np.uint8)

    # Corners at square_px, 2·square_px, …, (n_squares-1)·square_px.
    # Using (height // square_px) * square_px as the upper bound ensures the
    # count is always (n_squares - 1), which is odd when n_squares is even —
    # guaranteeing a corner lands exactly on the image centre for images of
    # size n_squares * square_px + 1  (e.g. 801 px with 50 px squares → 15 corners, centre at px 400).
    corner_rows = np.arange(square_px, (height // square_px) * square_px, square_px)
    corner_cols = np.arange(square_px, (width  // square_px) * square_px, square_px)

    n_r, n_c = len(corner_rows), len(corner_cols)
    corners_grid = np.zeros((n_r, n_c, 2), dtype=float)
    for i, r in enumerate(corner_rows):
        for j, c in enumerate(corner_cols):
            corners_grid[i, j] = [r, c]

    return img, corners_grid


def distort_corners(corners_grid: np.ndarray,
                    image_center,
                    pix_size_norm: float,
                    constants,
                    pol_degree) -> np.ndarray:
    """
    Apply radial distortion to corner pixel positions (mathematical, no image).

    Used to produce ground-truth distorted corners for validation:
    generate a checkerboard, distort its corners, then run the estimator
    on those corners and compare recovered constants to the originals.

    Parameters
    ----------
    corners_grid  : (n_rows, n_cols, 2) array of (row, col) pixel positions
    image_center  : (cy, cx) optical centre in pixels
    pix_size_norm : pix_size / focal_length
    constants     : distortion coefficients
    pol_degree    : polynomial degrees

    Returns
    -------
    distorted_grid : same shape as corners_grid with distorted (row, col)
    """
    constants  = np.asarray(constants,  dtype=float).ravel()
    pol_degree = np.asarray(pol_degree, dtype=float).ravel()
    cy, cx = image_center

    flat = corners_grid.reshape(-1, 2)
    dx   = flat[:, 1] - cx
    dy   = -(flat[:, 0] - cy)          # flip to math convention (y up)

    r_in_px   = np.hypot(dx, dy)
    theta     = np.arctan2(dy, dx)

    r_in_norm  = r_in_px * pix_size_norm
    r_out_norm = distortion_fun(r_in_norm, constants, pol_degree)
    r_out_px   = r_out_norm / pix_size_norm

    safe_r  = np.where(r_in_px > 0, r_in_px, 1.0)   # avoid /0; replaced by 1.0 → scale=1
    scale   = np.where(r_in_px > 0, r_out_px / safe_r, 1.0)
    col_out = cx + dx * scale
    row_out = cy - dy * scale           # flip back to image convention

    return np.column_stack([row_out, col_out]).reshape(corners_grid.shape)
