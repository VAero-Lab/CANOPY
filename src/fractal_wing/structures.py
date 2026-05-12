"""
Data structures for fractal tree-branch definitions.

Enhanced from v4:
  - ``SubParams.mode`` accepts a list of modes (one per recursion depth)
  - ``SubParams.min_length`` controls the minimum-segment termination threshold
  - ``BranchDef.thick_frac`` gives per-branch thickness control
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Union


# ─────────────────────────────────────────────────────────────────
# Segment
# ─────────────────────────────────────────────────────────────────

@dataclass
class Seg:
    """One structural web segment."""
    p0: np.ndarray
    p1: np.ndarray
    thick: float
    level: int
    sid: int = 0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


def segments_cross(a0, a1, b0, b1) -> bool:
    """Test if segment *a0–a1* crosses segment *b0–b1* (2-D)."""
    d1 = a1 - a0
    d2 = b1 - b0
    d3 = b0 - a0
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False  # parallel
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
    u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross
    return 0.01 < t < 0.99 and 0.01 < u < 0.99


# ─────────────────────────────────────────────────────────────────
# Sub-branch recursion parameters
# ─────────────────────────────────────────────────────────────────

@dataclass
class SubParams:
    """
    Recursion control for sub-branches.

    Parameters
    ----------
    max_depth : int
        Maximum recursion depth.
    angles : list of float
        Bifurcation angle [deg] per depth level.
    length_ratios : list of float
        Child / parent length ratio per depth.
    thickness_ratios : list of float
        Child / parent thickness ratio per depth.
    n_children : list of int
        Number of children per node per depth.
    mode : str or list of str
        Branching pattern per depth level.  Accepted values:

        - ``'monopodial'``  — continuation + symmetric side branches
        - ``'sympodial'``   — no continuation, all children fan out
        - ``'dichotomous'`` — equal binary split (children keep parent thickness)
        - ``'monochasium'`` — continuation with deflection + one alternating side branch

        If a single string is provided it is used at every depth (backward
        compatible).  If a list is provided, each entry controls the mode
        at the corresponding recursion depth; the last entry repeats for
        any deeper levels.
    min_length : float
        Minimum segment length [m] before recursion terminates.  Lower
        values allow finer branching at the expense of more segments.

    Notes
    -----
    All list parameters are indexed by recursion depth.  If the list is
    shorter than *max_depth*, the last value repeats.
    """

    max_depth: int = 3
    angles: List[float] = None
    length_ratios: List[float] = None
    thickness_ratios: List[float] = None
    n_children: List[int] = None
    mode: Union[str, List[str]] = 'sympodial'
    min_length: float = 0.04

    def __post_init__(self):
        def _pad(v, default, n):
            if v is None:
                return [default] * (n + 1)
            o = list(v)
            while len(o) <= n:
                o.append(o[-1])
            return o

        self.angles = _pad(self.angles, 30.0, self.max_depth)
        self.length_ratios = _pad(self.length_ratios, 0.55, self.max_depth)
        self.thickness_ratios = _pad(self.thickness_ratios, 0.7, self.max_depth)
        self.n_children = _pad(self.n_children, 2, self.max_depth)

        # Normalise mode to a padded list
        if isinstance(self.mode, str):
            self._modes = [self.mode] * (self.max_depth + 1)
        else:
            self._modes = list(self.mode)
            while len(self._modes) <= self.max_depth:
                self._modes.append(self._modes[-1])

    def mode_at(self, depth: int) -> str:
        """Return the branching mode for recursion *depth*."""
        i = min(depth, len(self._modes) - 1)
        return self._modes[i]

    def at(self, d: int):
        """Return ``(angle, length_ratio, thickness_ratio, n_children)`` at depth *d*."""
        i = min(d, len(self.angles) - 1)
        return (
            self.angles[i],
            self.length_ratios[i],
            self.thickness_ratios[i],
            self.n_children[i],
        )


# ─────────────────────────────────────────────────────────────────
# Branch definition
# ─────────────────────────────────────────────────────────────────

@dataclass
class BranchDef:
    """
    One side-branch emerging at a station.

    Parameters
    ----------
    angle : float
        Angle relative to trunk direction [deg].
        Small angles (±20–40°) → diagonal / spanwise branches.
        Large angles (±60–80°) → chordwise branches (rib-like).
        Positive → toward trailing edge; negative → toward leading edge.
    length : float
        Initial segment length [m].
    thick_frac : float
        Fraction of trunk thickness used as the initial branch thickness.
        Default ``0.7``.  Values close to 1.0 produce branches nearly as
        thick as the trunk; values near 0.3 produce very light branches.
    sub : SubParams
        Recursion parameters for this branch's sub-tree.
    """

    angle: float = 45.0
    length: float = 2.0
    thick_frac: float = 0.7
    sub: SubParams = None

    def __post_init__(self):
        if self.sub is None:
            self.sub = SubParams()


# ─────────────────────────────────────────────────────────────────
# Station
# ─────────────────────────────────────────────────────────────────

@dataclass
class Station:
    """
    A branching point along the trunk.

    Parameters
    ----------
    position : float
        Fraction along trunk length [0 = root, 1 = tip].
        Use values close to 0 (e.g. 0.02) for branches near the root.
    branches : list of BranchDef
        Each can have different angle, length, mode, and thickness.
    mono_defl : float
        Trunk deflection [deg] at this station (monochasium trunk mode).
    """

    position: float = 0.5
    branches: List[BranchDef] = None
    mono_defl: float = 0.0

    def __post_init__(self):
        if self.branches is None:
            self.branches = [BranchDef(45.0), BranchDef(-45.0)]


# ─────────────────────────────────────────────────────────────────
# Trunk specification
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrunkSpec:
    """
    Single trunk specification.

    Parameters
    ----------
    chord_frac : float
        Chordwise position at root [0 = front, 1 = rear of domain].
    span_cov : float
        Fraction of semi-span the trunk covers [0 … 1].
    thick : float
        Web thickness [m].
    stations : list of Station
        Where branches emerge.
    allow_crossing : bool
        If ``False``, new branches that cross existing ones are rejected.
    dir_offset : float
        Angular offset from domain centreline [deg].
    """

    chord_frac: float = 0.5
    span_cov: float = 1.0
    thick: float = 0.003
    stations: List[Station] = None
    allow_crossing: bool = True
    dir_offset: float = 0.0
