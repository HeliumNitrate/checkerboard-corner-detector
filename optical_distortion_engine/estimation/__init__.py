"""
estimation
==========
Distortion function estimation from checkerboard calibration images.

Approach
--------
A synthetic (or real) checkerboard is distorted by an unknown radial function.
Near-centre squares remain approximately undistorted, establishing a pixel/grid
scale. All corner undistorted positions are then extrapolated from the regular
grid structure. Comparing extrapolated r_in with detected r_out yields the
distortion curve, fitted via linear least squares:

    Δr = r_out − r_in = Σ c_i · r_in^d_i   →   A · constants = Δr

The solution is deterministic (one lstsq call, no iterative optimisation).

Main entry points
-----------------
estimate_from_image   — full pipeline from a distorted image
estimate_from_corners — from a pre-detected (n_rows, n_cols, 2) corner grid
fit_constants         — raw lstsq fit from (r_in_norm, r_out_norm) arrays

Synthetic calibration helpers
------------------------------
make_checkerboard     — generate a perfect checkerboard + inner corner grid
distort_corners       — apply distortion to corners mathematically (validation)
"""

from .checkerboard import make_checkerboard, distort_corners
from .estimator    import estimate_from_image, estimate_from_corners, fit_constants

__all__ = [
    "make_checkerboard",
    "distort_corners",
    "estimate_from_image",
    "estimate_from_corners",
    "fit_constants",
]
