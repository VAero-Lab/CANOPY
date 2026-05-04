"""
NACA 4-digit airfoil geometry.
"""

import numpy as np


class NACAFourDigit:
    """
    NACA 4-digit airfoil parameterisation.

    Parameters
    ----------
    s : str
        Four-digit NACA code, e.g. ``'2412'``.
    """

    def __init__(self, s: str = '2412'):
        self.M = int(s[0]) / 100
        self.P = int(s[1]) / 10
        self.T = int(s[2:4]) / 100

    # ── internal ────────────────────────────────────────────────
    def _yt(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, 1)
        t = self.T
        return (t / 0.2) * (
            0.2969 * np.sqrt(x)
            - 0.126 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    def _yc(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, 1)
        yc = np.zeros_like(x)
        if self.M == 0 or self.P == 0:
            return yc
        f = x < self.P
        yc[f] = (self.M / self.P**2) * (2 * self.P * x[f] - x[f] ** 2)
        yc[~f] = (self.M / (1 - self.P) ** 2) * (
            (1 - 2 * self.P) + 2 * self.P * x[~f] - x[~f] ** 2
        )
        return yc

    # ── public ──────────────────────────────────────────────────
    def zu(self, xi: float) -> float:
        """Upper-surface z/c at chordwise fraction *xi*."""
        x = np.atleast_1d(np.clip(xi, 0.001, 0.999)).astype(float)
        return float((self._yc(x) + self._yt(x)).item())

    def zl(self, xi: float) -> float:
        """Lower-surface z/c at chordwise fraction *xi*."""
        x = np.atleast_1d(np.clip(xi, 0.001, 0.999)).astype(float)
        return float((self._yc(x) - self._yt(x)).item())
