"""
distortion_fun.py
=================
Generalized polynomial radial distortion function.

  r_out = r_in + sum_i( constants[i] * r_in ^ pol_degree[i] )

No limitation on number of terms or polynomial degree.
Any monotonic function can be substituted here.
"""
from __future__ import annotations

import numpy as np


def distortion_fun(r_in: np.ndarray,
                   constants: list | np.ndarray,
                   pol_degree: list | np.ndarray) -> np.ndarray:
    """
    Apply generalized polynomial radial distortion.

    Parameters
    ----------
    r_in       : radial distances (any shape)
    constants  : distortion coefficients [k1, k2, ...]
    pol_degree : corresponding polynomial degrees [d1, d2, ...]

    Returns
    -------
    r_out : distorted radial distances (same shape as r_in)
    """
    constants  = np.asarray(constants,  dtype=float).ravel()
    pol_degree = np.asarray(pol_degree, dtype=float).ravel()

    if len(constants) != len(pol_degree):
        raise ValueError(
            f"constants (len={len(constants)}) and pol_degree "
            f"(len={len(pol_degree)}) must have the same length."
        )

    r_out = r_in.copy().astype(float)
    for k, d in zip(constants, pol_degree):
        r_out = r_out + k * (r_in ** d)

    return r_out
