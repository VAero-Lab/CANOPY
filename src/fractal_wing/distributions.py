"""
Station distribution functions.

Provides configurable, non-uniform spacing laws for placing branching
stations along a trunk, and also accepts arbitrary user / optimizer-defined
positions.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def distribute_stations(
    n: int,
    method: str = 'uniform',
    start: float = 0.02,
    end: float = 0.95,
    **kwargs,
) -> np.ndarray:
    """
    Generate *n* station positions along the trunk.

    Parameters
    ----------
    n : int
        Number of stations.
    method : str
        Distribution law:

        ``'uniform'``
            Equal spacing (default, current behaviour).
        ``'geometric'``
            Geometric progression.  ``ratio > 1`` clusters stations
            toward the root; ``ratio < 1`` toward the tip.
            Extra kwarg: **ratio** (float, default 1.3).
        ``'cosine'``
            Half-cosine clustering — dense at both root and tip,
            sparse in the mid-span.
        ``'power'``
            Power-law grading.  ``exponent > 1`` clusters toward the root.
            Extra kwarg: **exponent** (float, default 2.0).
        ``'bimodal'``
            Dense near root *and* tip with a sparse middle region.
            Uses tanh stretching.
            Extra kwarg: **weight** (float, default 2.0).
        ``'custom'``
            User / optimizer-supplied positions.
            Extra kwarg: **positions** (array-like of length *n*).

    start, end : float
        Range limits within [0, 1].

    Returns
    -------
    np.ndarray
        Sorted positions in [start, end].

    Examples
    --------
    >>> distribute_stations(6)
    array([0.02 , 0.206, 0.392, 0.578, 0.764, 0.95 ])

    >>> distribute_stations(6, 'geometric', ratio=2.0)
    array([0.02  , 0.05  , 0.11  , 0.23  , 0.47  , 0.95  ])

    >>> distribute_stations(5, 'custom', positions=[0.05, 0.1, 0.5, 0.8, 0.95])
    array([0.05, 0.1 , 0.5 , 0.8 , 0.95])
    """
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([0.5 * (start + end)])

    if method == 'uniform':
        return np.linspace(start, end, n)

    elif method == 'geometric':
        ratio = kwargs.get('ratio', 1.3)
        if abs(ratio - 1.0) < 1e-10:
            return np.linspace(start, end, n)
        k = n - 1  # number of intervals
        s0 = (end - start) * (ratio - 1) / (ratio**k - 1)
        spacings = s0 * ratio ** np.arange(k)
        positions = np.concatenate([[start], start + np.cumsum(spacings)])
        return positions

    elif method == 'cosine':
        # Half-cosine: dense at both ends, sparse in middle
        theta = np.linspace(0, np.pi, n)
        t = 0.5 * (1 - np.cos(theta))
        return start + (end - start) * t

    elif method == 'power':
        exponent = kwargs.get('exponent', 2.0)
        t = np.linspace(0, 1, n)
        t = t ** exponent  # exponent > 1 → denser at start
        return start + (end - start) * t

    elif method == 'bimodal':
        # tanh stretching: dense at both ends
        weight = kwargs.get('weight', 2.0)
        t = np.linspace(-1, 1, n)
        t_stretched = np.tanh(weight * t) / np.tanh(weight)
        return start + (end - start) * (t_stretched + 1) / 2

    elif method == 'custom':
        positions = np.asarray(kwargs['positions'], dtype=float)
        if len(positions) != n:
            raise ValueError(
                f"'custom' method requires {n} positions, got {len(positions)}"
            )
        return np.sort(np.clip(positions, start, end))

    else:
        raise ValueError(f"Unknown distribution method: {method!r}")
