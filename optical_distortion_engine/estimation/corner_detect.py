"""
corner_detect.py
================
Detect and sort inner checkerboard corners from a (possibly distorted) image.
Uses Harris corner detection via scipy — no OpenCV dependency.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def _harris_response(img: np.ndarray, sigma: float = 2.0, k: float = 0.05) -> np.ndarray:
    f   = img.astype(np.float64)
    Ix  = gaussian_filter(f, sigma=sigma, order=[0, 1])
    Iy  = gaussian_filter(f, sigma=sigma, order=[1, 0])
    Ixx = gaussian_filter(Ix * Ix, sigma=sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)
    det   = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    return det - k * trace ** 2


def _subpixel_refine(img: np.ndarray,
                     corners: np.ndarray,
                     half_window: int = 5,
                     sigma: float     = 1.0,
                     max_iter: int    = 20,
                     eps: float       = 0.01) -> np.ndarray:
    """
    Sub-pixel corner refinement via gradient orthogonality (OpenCV cornerSubPix algorithm).

    At a true corner every image-gradient vector is perpendicular to the
    vector from the corner to that pixel.  This condition gives a linear
    system that is solved iteratively.

    Parameters
    ----------
    img         : 2D float or uint8 image
    corners     : (N, 2) initial (row, col) positions (integer)
    half_window : radius of the search window in pixels
    sigma       : Gaussian smoothing for gradient computation
    max_iter    : maximum Newton iterations per corner
    eps         : convergence threshold in pixels

    Returns
    -------
    refined : (N, 2) sub-pixel (row, col) positions
    """
    f  = gaussian_filter(img.astype(float), sigma=sigma)
    Ix = gaussian_filter(f, sigma=sigma, order=[0, 1])
    Iy = gaussian_filter(f, sigma=sigma, order=[1, 0])

    H, W = f.shape
    refined = np.empty_like(corners, dtype=float)

    for k, (r0, c0) in enumerate(corners):
        y, x = float(r0), float(c0)

        for _ in range(max_iter):
            ri, ci = int(round(y)), int(round(x))
            r1 = max(0, ri - half_window);  r2 = min(H, ri + half_window + 1)
            c1 = max(0, ci - half_window);  c2 = min(W, ci + half_window + 1)
            if r2 - r1 < 3 or c2 - c1 < 3:
                break

            PR, PC = np.mgrid[r1:r2, c1:c2]   # (nr, nc) each
            ix = Ix[r1:r2, c1:c2].ravel()
            iy = Iy[r1:r2, c1:c2].ravel()
            px = PC.ravel().astype(float)
            py = PR.ravel().astype(float)

            A00 = (ix * ix).sum();  A01 = (ix * iy).sum();  A11 = (iy * iy).sum()
            b0  = (ix * (px * ix + py * iy)).sum()
            b1  = (iy * (px * ix + py * iy)).sum()

            det = A00 * A11 - A01 * A01
            if abs(det) < 1e-10:
                break

            x_new = (A11 * b0 - A01 * b1) / det
            y_new = (A00 * b1 - A01 * b0) / det

            if abs(x_new - x) < eps and abs(y_new - y) < eps:
                x, y = x_new, y_new
                break
            x, y = x_new, y_new

        refined[k] = [y, x]    # (row, col)

    return refined


def detect_corners(img: np.ndarray,
                   min_distance:  int   = 10,
                   threshold_rel: float = 0.01,
                   sigma:         float = 2.0,
                   subpixel:      bool  = True) -> np.ndarray:
    """
    Detect corner peaks in a grayscale checkerboard image.

    Parameters
    ----------
    img           : 2D (H, W) or 3D (H, W, C) image array
    min_distance  : minimum pixel separation between accepted corners
    threshold_rel : Harris response threshold relative to global max
    sigma         : Gaussian smoothing scale for Harris detector
    subpixel      : if True, refine each peak to sub-pixel accuracy via the
                    gradient-orthogonality method (equivalent to OpenCV
                    cornerSubPix); strongly recommended for accurate estimation

    Returns
    -------
    corners : (N, 2) float array of (row, col) positions
    """
    if img.ndim == 3:
        img = img.mean(axis=2)

    response  = _harris_response(img, sigma=sigma)
    dilated   = maximum_filter(response, size=min_distance)
    local_max = (response == dilated) & (response > threshold_rel * response.max())

    corners_int = np.argwhere(local_max).astype(float)

    if subpixel and len(corners_int) > 0:
        corners_int = _subpixel_refine(img, corners_int, half_window=int(min_distance // 2 + 1), sigma=max(0.5, sigma / 2))

    return corners_int


def sort_corners_to_grid(corners: np.ndarray,
                         n_rows: int,
                         n_cols: int,
                         image_center=None) -> np.ndarray:
    """
    Sort N detected corner positions into an (n_rows, n_cols, 2) grid.

    Uses center-out BFS with linear extrapolation so distorted images with
    non-uniform spacing and extra false-positive corners are handled correctly.
    Unmatched positions are filled with NaN.

    Parameters
    ----------
    corners      : (N, 2) array of (row, col) positions
    n_rows       : expected grid row count
    n_cols       : expected grid column count
    image_center : (row, col) hint for the grid centre; defaults to corners mean

    Returns
    -------
    grid : (n_rows, n_cols, 2); unmatched entries are NaN

    Raises
    ------
    ValueError if fewer than half the positions can be assigned
    """
    from scipy.spatial import cKDTree
    from collections import deque

    if image_center is None:
        image_center = corners.mean(axis=0)
    image_center = np.asarray(image_center, dtype=float)

    tree = cKDTree(corners)
    ci = (n_rows - 1) // 2
    cj = (n_cols - 1) // 2

    grid = np.full((n_rows, n_cols, 2), np.nan)
    used: set = set()

    # Centre corner: detected point closest to image_center
    _, c_idx = tree.query(image_center)
    grid[ci, cj] = corners[c_idx]
    used.add(int(c_idx))

    # Bootstrap spacing from nearest neighbours of the centre corner
    k = min(8, len(corners) - 1)
    nbr_d, _ = tree.query(corners[c_idx], k=k + 1)
    init_spacing = float(np.median(nbr_d[1:]))

    def _predict(i, j, di, dj):
        """Predict (row, col) of grid[i+di, j+dj] by linear extrapolation."""
        cur = grid[i, j]
        bi, bj = i - di, j - dj
        if (0 <= bi < n_rows and 0 <= bj < n_cols
                and not np.isnan(grid[bi, bj, 0])):
            return 2.0 * cur - grid[bi, bj]
        # No back reference — estimate step size from perpendicular neighbours
        sizes = []
        if di == 0:    # horizontal step → sample vertical direction
            for off in (-1, 1, -2, 2):
                pi = i + off
                if 0 <= pi < n_rows and not np.isnan(grid[pi, j, 0]):
                    sizes.append(np.linalg.norm(grid[pi, j] - cur) / abs(off))
        else:          # vertical step → sample horizontal direction
            for off in (-1, 1, -2, 2):
                pj = j + off
                if 0 <= pj < n_cols and not np.isnan(grid[i, pj, 0]):
                    sizes.append(np.linalg.norm(grid[i, pj] - cur) / abs(off))
        step = float(np.median(sizes)) if sizes else init_spacing
        return cur + np.array([di, dj], dtype=float) * step

    bfs = deque([(ci, cj)])
    queued = {(ci, cj)}

    while bfs:
        i, j = bfs.popleft()
        if np.isnan(grid[i, j, 0]):
            continue   # unmatched node — cannot extrapolate from it

        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if not (0 <= ni < n_rows and 0 <= nj < n_cols):
                continue
            if not np.isnan(grid[ni, nj, 0]):
                continue   # already matched
            if (ni, nj) in queued:
                continue

            pred     = _predict(i, j, di, dj)
            step_len = float(np.linalg.norm(pred - grid[i, j]))
            radius   = max(step_len * 0.60, 5.0)

            best_d, best_idx = np.inf, None
            for idx in tree.query_ball_point(pred, radius):
                if idx in used:
                    continue
                d = float(np.linalg.norm(corners[idx] - pred))
                if d < best_d:
                    best_d, best_idx = d, idx

            if best_idx is not None:
                grid[ni, nj] = corners[best_idx]
                used.add(best_idx)

            bfs.append((ni, nj))
            queued.add((ni, nj))

    n_assigned = int(np.sum(~np.isnan(grid[:, :, 0])))
    min_req    = max(4, n_rows * n_cols // 4)
    if n_assigned < min_req:
        raise ValueError(
            f"Only {n_assigned}/{n_rows * n_cols} grid positions assigned. "
            "Try reducing min_distance or corner_threshold."
        )
    return grid
