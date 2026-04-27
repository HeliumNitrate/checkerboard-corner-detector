"""
estimator.py
============
Estimate radial distortion constants from a checkerboard corner grid.


Algorithm
---------
1. Detect inner corners in the distorted checkerboard image.
2. Compute grid spacing from the 4 unit-step cardinal neighbours of the
   grid centre (gi=±1, gj=0 and gi=0, gj=±1).  These corners are the
   closest to the optical axis, so their radial distortion is smallest
   and the spacing estimate is most accurate.
3. Extrapolate undistorted pixel distances r_in for every corner using
   the regular grid structure:
       r_in = sqrt(gi² + gj²) × spacing
4. r_out is the measured pixel distance from the image centre in the
   distorted image.
5. Solve the linear system:
       A · constants = Δr        (Δr = r_out − r_in, normalised)
       A[k, m] = r_in_norm[k] ^ pol_degree[m]
   via least squares → constants (deterministic, no iterative optimisation).
"""

from __future__ import annotations

import numpy as np
from .corner_detect import detect_corners, sort_corners_to_grid
from ..core.distortion_fun import distortion_fun as _distortion_fun


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_constants(r_in_norm:  np.ndarray,
                  r_out_norm: np.ndarray,
                  pol_degree) -> tuple[np.ndarray, dict]:
    """
    Fit distortion constants from normalised (r_in, r_out) radial pairs.

    Solves:   A · constants = Δr
    where     A[k, m] = r_in_norm[k] ^ pol_degree[m]
              Δr[k]   = r_out_norm[k] - r_in_norm[k]

    This is a direct linear least-squares problem — no iterative
    optimisation is involved.

    Parameters
    ----------
    r_in_norm  : undistorted normalised radial distances  (1-D)
    r_out_norm : distorted   normalised radial distances  (1-D)
    pol_degree : polynomial degrees to use in the model

    Returns
    -------
    constants : (len(pol_degree),) fitted coefficients
    info      : dict — residuals, rank, condition_number, n_points
    """
    pol_degree = np.asarray(pol_degree, dtype=float).ravel()
    delta_r    = r_out_norm - r_in_norm
    A          = np.column_stack([r_in_norm ** d for d in pol_degree])

    constants, residuals, rank, sv = np.linalg.lstsq(A, delta_r, rcond=None)

    return constants, {
        "residuals"       : residuals,
        "rank"            : int(rank),
        "singular_values" : sv,
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "n_points"        : len(r_in_norm),
    }


def estimate_from_corners(corners_grid: np.ndarray,
                          image_center,
                          pix_size_norm: float,
                          pol_degree) -> tuple[np.ndarray, dict]:
    """
    Estimate distortion constants from a pre-detected corner grid.

    Spacing is estimated from the 4 unit-step cardinal neighbours of the
    grid centre (gi=±1,gj=0 and gi=0,gj=±1).  These are the corners
    closest to the optical axis, so their radial distortion is smallest.
    For the estimate to be accurate, the checkerboard squares must be
    small enough that the distortion within one square is negligible:

        square_px * pix_size_norm  <<  1
        (e.g.  10 px * 0.0025 = 0.025  →  spacing error < 0.05 %)

    Parameters
    ----------
    corners_grid  : (n_rows, n_cols, 2) detected (row, col) positions.
                    Both n_rows and n_cols must be odd so the grid centre
                    aligns with an actual corner. Use make_checkerboard
                    with height = n_squares * square_px + 1.
    image_center  : (cy, cx) optical centre in pixels
    pix_size_norm : pix_size / focal_length
    pol_degree    : polynomial degrees to include in the fit

    Returns
    -------
    constants : fitted distortion coefficients
    info      : diagnostics dict (spacing_px, n_total, ...)
    """
    cy, cx         = image_center
    n_rows, n_cols = corners_grid.shape[:2]

    if n_rows % 2 == 0 or n_cols % 2 == 0:
        raise ValueError(
            f"corners_grid has shape ({n_rows}, {n_cols}). "
            "Both n_rows and n_cols must be odd so the grid centre aligns "
            "with an actual corner. Use make_checkerboard with "
            "height = n_squares * square_px + 1."
        )

    # Integer grid offsets from grid centre
    ci = (n_rows - 1) // 2
    cj = (n_cols - 1) // 2
    gi_2d = np.arange(n_rows)[:, None] - ci
    gj_2d = np.arange(n_cols)[None, :] - cj
    gi_flat = np.broadcast_to(gi_2d, (n_rows, n_cols)).ravel().copy()
    gj_flat = np.broadcast_to(gj_2d, (n_rows, n_cols)).ravel().copy()

    # Detected positions relative to image centre
    flat        = corners_grid.reshape(-1, 2)
    has_corner  = ~np.isnan(flat[:, 0])
    dx_detected = flat[:, 1] - cx
    dy_detected = -(flat[:, 0] - cy)
    r_detected  = np.hypot(dx_detected, dy_detected)

    # Spacing from unit-step cardinal neighbours (minimum r → minimum distortion)
    unit_step  = (np.abs(gi_flat) + np.abs(gj_flat)) == 1
    unit_valid = unit_step & has_corner
    if not np.any(unit_valid):
        raise ValueError(
            "No valid unit-step cardinal neighbours found. "
            "The checkerboard may be too coarse or central corners missing."
        )
    spacing = float(np.median(r_detected[unit_valid]))

    # Undistorted pixel distances from regular grid structure
    r_in_px  = np.hypot(gi_flat, gj_flat) * spacing
    r_out_px = r_detected

    # Exclude the exact grid centre (r_in = 0) and unmatched NaN positions
    valid = (r_in_px > spacing * 0.3) & has_corner

    # Keep full-size arrays (same length as gi_flat) for mask arithmetic
    r_in_norm_all  = r_in_px  * pix_size_norm   # (N,), 0 at grid centre
    r_out_norm_all = r_out_px * pix_size_norm   # (N,), NaN at missing corners

    pol_deg = np.asarray(pol_degree, dtype=float).ravel()

    # Two-pass robust fit: first pass uses all valid corners to obtain c_rough
    # (uses the full r range → well-conditioned for multi-term models), then
    # sigma-clips positions whose detected r_out deviates far from the model
    # (false positives from compressed outer squares in strong barrel images).
    if valid.sum() >= len(pol_deg) + 1:
        c_rough, _ = fit_constants(r_in_norm_all[valid], r_out_norm_all[valid], pol_deg)
        r_pred      = _distortion_fun(r_in_norm_all[valid], c_rough, pol_deg)
        residuals   = r_out_norm_all[valid] - r_pred
        mad         = float(np.median(np.abs(residuals - np.median(residuals))))
        sigma_rob   = 1.4826 * mad
        tol         = max(5.0 * sigma_rob, 5.0 * pix_size_norm)
        inlier_mask_on_valid = np.abs(residuals) < tol
        # Build inlier mask over the full-size array
        inlier      = valid.copy()
        inlier[valid] = inlier_mask_on_valid
        r_in_fit    = r_in_norm_all[inlier]
        r_out_fit   = r_out_norm_all[inlier]
    else:
        r_in_fit  = r_in_norm_all[valid]
        r_out_fit = r_out_norm_all[valid]

    constants, info = fit_constants(r_in_fit, r_out_fit, pol_deg)
    info["spacing_px"] = spacing
    info["n_total"]    = len(r_in_fit)

    return constants, info


def estimate_from_image(img: np.ndarray,
                        n_inner_rows: int,
                        n_inner_cols: int,
                        image_center,
                        pix_size_norm: float,
                        pol_degree,
                        min_distance:     int   = 10,
                        corner_threshold: float = 0.01) -> tuple[np.ndarray, dict]:
    """
    Full pipeline: detect corners from a distorted checkerboard image,
    then estimate the distortion constants.

    Parameters
    ----------
    img            : 2D or 3D distorted checkerboard image (uint8)
    n_inner_rows   : expected number of inner corner rows (must be odd)
    n_inner_cols   : expected number of inner corner columns (must be odd)
    image_center   : (cy, cx) optical centre in pixels
    pix_size_norm  : pix_size / focal_length
    pol_degree     : polynomial degrees to include in the fit
    min_distance   : minimum pixel separation for corner detection
    corner_threshold : Harris response threshold (relative to max)

    Returns
    -------
    constants : fitted distortion coefficients
    info      : diagnostics dict (includes detected corners_grid)
    """
    corners_raw  = detect_corners(img,
                                  min_distance=min_distance,
                                  threshold_rel=corner_threshold)
    corners_grid = sort_corners_to_grid(corners_raw, n_inner_rows, n_inner_cols,
                                        image_center=image_center)

    constants, info = estimate_from_corners(
        corners_grid, image_center, pix_size_norm, pol_degree
    )
    info["corners_grid"] = corners_grid
    return constants, info
