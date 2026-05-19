"""
Convenience factories for creating station configurations.

These functions combine the distribution utilities with branch/sub-parameter
construction so that common fractal patterns can be set up in one call.
"""

from __future__ import annotations

import copy
from typing import List

from .structures import SubParams, BranchDef, Station
from .distributions import distribute_stations


# ─────────────────────────────────────────────────────────────────
# Original helpers (enhanced with spacing parameter)
# ─────────────────────────────────────────────────────────────────

def make_mixed_stations(
    n_stations: int = 12,
    spacing: str = 'uniform',
    spacing_kwargs: dict | None = None,
    diag_angle: float = 35.0,
    chord_angle: float = 75.0,
    diag_length: float = 2.0,
    chord_length: float = 1.8,
    diag_sub: SubParams | None = None,
    chord_sub: SubParams | None = None,
    start: float = 0.02,
    end: float = 0.95,
) -> List[Station]:
    """
    Create stations with BOTH chordwise and diagonal branches.

    At each station:
      - Two diagonal branches (±diag_angle from trunk)
      - Two chordwise branches (±chord_angle from trunk)

    Parameters
    ----------
    n_stations : int
        Number of branching stations.
    spacing : str
        Distribution method (see :func:`distribute_stations`).
    spacing_kwargs : dict, optional
        Extra kwargs for the distribution (e.g. ``{'ratio': 1.5}``).
    diag_angle, chord_angle : float
        Branch angles [deg].
    diag_length, chord_length : float
        Initial branch lengths [m].
    diag_sub, chord_sub : SubParams, optional
        Sub-branch parameters.  Defaults are sympodial/dichotomous.
    start, end : float
        Station position range.
    """
    if diag_sub is None:
        diag_sub = SubParams(
            max_depth=2,
            angles=[30, 40],
            length_ratios=[0.5, 0.45],
            thickness_ratios=[0.7, 0.65],
            mode='sympodial',
        )
    if chord_sub is None:
        chord_sub = SubParams(
            max_depth=1,
            angles=[25],
            length_ratios=[0.45],
            thickness_ratios=[0.65],
            mode='dichotomous',
        )

    positions = distribute_stations(
        n_stations, spacing, start, end, **(spacing_kwargs or {}),
    )

    stations = []
    for p in positions:
        bl_diag = diag_length * (1.0 - 0.4 * p)
        bl_chord = chord_length * (1.0 - 0.3 * p)
        stations.append(Station(
            position=p,
            branches=[
                BranchDef(angle=diag_angle,  length=bl_diag,  sub=copy.deepcopy(diag_sub)),
                BranchDef(angle=-diag_angle, length=bl_diag,  sub=copy.deepcopy(diag_sub)),
                BranchDef(angle=chord_angle, length=bl_chord, sub=copy.deepcopy(chord_sub)),
                BranchDef(angle=-chord_angle,length=bl_chord, sub=copy.deepcopy(chord_sub)),
            ],
        ))
    return stations


def make_diagonal_only_stations(
    n_stations: int = 12,
    spacing: str = 'uniform',
    spacing_kwargs: dict | None = None,
    angle: float = 40.0,
    length: float = 2.5,
    sub: SubParams | None = None,
    start: float = 0.02,
    end: float = 0.95,
) -> List[Station]:
    """
    Stations with diagonal branches only (no chordwise).

    Parameters
    ----------
    n_stations : int
        Number of stations.
    spacing : str
        Distribution method.
    angle : float
        Branch angle [deg].
    length : float
        Initial branch length [m].
    sub : SubParams, optional
        Sub-branch parameters.
    """
    if sub is None:
        sub = SubParams(
            max_depth=3,
            angles=[25, 35, 45],
            length_ratios=[0.55, 0.5, 0.45],
            mode='sympodial',
        )

    positions = distribute_stations(
        n_stations, spacing, start, end, **(spacing_kwargs or {}),
    )

    stations = []
    for p in positions:
        bl = length * (1.0 - 0.4 * p)
        stations.append(Station(
            position=p,
            branches=[
                BranchDef(angle=angle,  length=bl, sub=copy.deepcopy(sub)),
                BranchDef(angle=-angle, length=bl, sub=copy.deepcopy(sub)),
            ],
        ))
    return stations


# ─────────────────────────────────────────────────────────────────
# New helpers
# ─────────────────────────────────────────────────────────────────

def make_zoned_stations(
    n_stations: int = 12,
    zones: List[dict] | None = None,
    spacing: str = 'uniform',
    spacing_kwargs: dict | None = None,
    start: float = 0.02,
    end: float = 0.95,
) -> List[Station]:
    """
    Create stations with **spanwise-varying** branch configurations.

    Each zone defines a span range and its own branch parameters (angles,
    lengths, modes), enabling e.g. sympodial near the root and dichotomous
    near the tip.

    Parameters
    ----------
    n_stations : int
        Total number of stations.
    zones : list of dict
        Each dict has keys:

        - ``'eta_start'``, ``'eta_end'`` : span fraction range
        - ``'diag_angle'``, ``'chord_angle'`` : branch angles [deg]
        - ``'diag_length'``, ``'chord_length'`` : initial lengths [m]
        - ``'diag_sub'``, ``'chord_sub'`` : SubParams instances

        If a station falls outside all zones, it uses the last zone's
        parameters.
    spacing : str
        Distribution method.

    Examples
    --------
    >>> zones = [
    ...     {'eta_start': 0.0, 'eta_end': 0.4,
    ...      'diag_angle': 30, 'chord_angle': 70,
    ...      'diag_length': 2.5, 'chord_length': 2.0,
    ...      'diag_sub': SubParams(mode='sympodial', max_depth=3),
    ...      'chord_sub': SubParams(mode='dichotomous', max_depth=1)},
    ...     {'eta_start': 0.4, 'eta_end': 1.0,
    ...      'diag_angle': 45, 'chord_angle': 75,
    ...      'diag_length': 1.5, 'chord_length': 1.2,
    ...      'diag_sub': SubParams(mode='monopodial', max_depth=2),
    ...      'chord_sub': SubParams(mode='sympodial', max_depth=1)},
    ... ]
    >>> stations = make_zoned_stations(14, zones=zones)
    """
    if zones is None:
        # Default: single zone covering full span
        zones = [
            {
                'eta_start': 0.0,
                'eta_end': 1.0,
                'diag_angle': 35.0,
                'chord_angle': 75.0,
                'diag_length': 2.0,
                'chord_length': 1.8,
                'diag_sub': SubParams(
                    max_depth=2, angles=[30, 40],
                    length_ratios=[0.5, 0.45],
                    thickness_ratios=[0.7, 0.65],
                    mode='sympodial',
                ),
                'chord_sub': SubParams(
                    max_depth=1, angles=[25],
                    length_ratios=[0.45],
                    thickness_ratios=[0.65],
                    mode='dichotomous',
                ),
            }
        ]

    positions = distribute_stations(
        n_stations, spacing, start, end, **(spacing_kwargs or {}),
    )

    def _find_zone(p):
        for z in zones:
            if z['eta_start'] <= p <= z['eta_end']:
                return z
        return zones[-1]

    stations = []
    for p in positions:
        z = _find_zone(p)
        da = z.get('diag_angle', 35.0)
        ca = z.get('chord_angle', 75.0)
        dl = z.get('diag_length', 2.0) * (1.0 - 0.4 * p)
        cl = z.get('chord_length', 1.8) * (1.0 - 0.3 * p)
        ds = z.get('diag_sub', SubParams(mode='sympodial'))
        cs = z.get('chord_sub', SubParams(mode='dichotomous'))
        tf = z.get('thick_frac', 0.7)

        branches = [
            BranchDef(angle=da,  length=dl, thick_frac=tf, sub=copy.deepcopy(ds)),
            BranchDef(angle=-da, length=dl, thick_frac=tf, sub=copy.deepcopy(ds)),
        ]
        if ca is not None:
            branches.extend([
                BranchDef(angle=ca,  length=cl, thick_frac=tf, sub=copy.deepcopy(cs)),
                BranchDef(angle=-ca, length=cl, thick_frac=tf, sub=copy.deepcopy(cs)),
            ])

        stations.append(Station(position=p, branches=branches))

    return stations


def make_graded_stations(
    n_stations: int = 12,
    spacing: str = 'uniform',
    spacing_kwargs: dict | None = None,
    angle_range: tuple = (25.0, 50.0),
    length_range: tuple = (2.5, 1.0),
    depth_range: tuple = (3, 1),
    mode_sequence: List[str] | None = None,
    thick_frac_range: tuple = (0.8, 0.5),
    start: float = 0.02,
    end: float = 0.95,
) -> List[Station]:
    """
    Create stations with parameters that **grade smoothly** from root to tip.

    Branch angle, length, recursion depth, thickness fraction, and optionally
    the branching mode vary linearly (or by schedule) from root values to
    tip values.

    Parameters
    ----------
    n_stations : int
        Number of stations.
    spacing : str
        Distribution method.
    angle_range : (float, float)
        (root_angle, tip_angle) in degrees.
    length_range : (float, float)
        (root_length, tip_length) in metres.
    depth_range : (int, int)
        (root_depth, tip_depth) for max recursion.
    mode_sequence : list of str, optional
        If provided, the branching mode transitions along the span.
        The list is interpolated across stations — e.g. ``['sympodial',
        'monopodial', 'dichotomous']`` would use sympodial near root,
        monopodial mid-span, and dichotomous near tip.
    thick_frac_range : (float, float)
        (root_thick_frac, tip_thick_frac).
    """
    positions = distribute_stations(
        n_stations, spacing, start, end, **(spacing_kwargs or {}),
    )

    # Normalise positions to [0, 1] for interpolation
    p_norm = (positions - positions[0]) / max(positions[-1] - positions[0], 1e-10)

    stations = []
    for i, p in enumerate(positions):
        frac = p_norm[i]

        ang = angle_range[0] + frac * (angle_range[1] - angle_range[0])
        length = length_range[0] + frac * (length_range[1] - length_range[0])
        depth = int(round(depth_range[0] + frac * (depth_range[1] - depth_range[0])))
        tf = thick_frac_range[0] + frac * (thick_frac_range[1] - thick_frac_range[0])

        # Mode selection
        if mode_sequence and len(mode_sequence) > 1:
            idx = frac * (len(mode_sequence) - 1)
            mode = mode_sequence[min(int(round(idx)), len(mode_sequence) - 1)]
        else:
            mode = (mode_sequence[0] if mode_sequence else 'sympodial')

        sub = SubParams(
            max_depth=max(depth, 1),
            angles=[ang],
            length_ratios=[0.55],
            thickness_ratios=[0.7],
            mode=mode,
        )

        bl = length * (1.0 - 0.4 * p)
        stations.append(Station(
            position=p,
            branches=[
                BranchDef(angle=ang,  length=bl, thick_frac=tf, sub=copy.deepcopy(sub)),
                BranchDef(angle=-ang, length=bl, thick_frac=tf, sub=copy.deepcopy(sub)),
            ],
        ))

    return stations

def ensure_open_te(af, cut_fraction: float = 0.01):
    """
    Ensures that an AeroShape AirfoilProfile has an open (blunt) trailing edge.
    
    If the airfoil converges to a single point at the trailing edge, this removes
    the trailing edge points and rescales the remaining coordinates so that
    the chord length remains exactly 1.0. This prevents meshing singularities
    where internal webs would degenerate into 3-sided triangles.
    
    Parameters
    ----------
    af : aeroshape.geometry.airfoils.AirfoilProfile
        The input airfoil profile.
    cut_fraction : float
        The fraction of the chord to slice off the trailing edge if it is closed.
        Default is 0.01 (1%).
        
    Returns
    -------
    aeroshape.geometry.airfoils.AirfoilProfile
        A new airfoil profile guaranteed to have an open trailing edge.
    """
    import numpy as np
    from aeroshape.geometry.airfoils import AirfoilProfile

    x = np.array(af.x)
    z = np.array(af.z)
    
    x_max = np.max(x)
    te_points = np.where(x > 0.999 * x_max)[0]
    z_te = z[te_points]
    
    if np.max(z_te) - np.min(z_te) < 1e-3 * x_max:
        le_idx = np.argmin(x)
        cut_x = x_max * (1.0 - cut_fraction)
        
        x_lower = x[:le_idx+1]
        z_lower = z[:le_idx+1]
        
        x_upper = x[le_idx:]
        z_upper = z[le_idx:]
        
        # Interpolate at exactly cut_x
        z_lower_cut = np.interp(cut_x, x_lower[::-1], z_lower[::-1])
        z_upper_cut = np.interp(cut_x, x_upper, z_upper)
        
        keep_lower = x_lower <= cut_x
        keep_upper = x_upper <= cut_x
        
        new_x_lower = x_lower[keep_lower]
        new_z_lower = z_lower[keep_lower]
        new_x_lower = np.insert(new_x_lower, 0, cut_x)
        new_z_lower = np.insert(new_z_lower, 0, z_lower_cut)
        
        new_x_upper = x_upper[keep_upper]
        new_z_upper = z_upper[keep_upper]
        new_x_upper = np.append(new_x_upper, cut_x)
        new_z_upper = np.append(new_z_upper, z_upper_cut)
        
        new_x = np.concatenate([new_x_lower[:-1], new_x_upper])
        new_z = np.concatenate([new_z_lower[:-1], new_z_upper])
        
        new_x = new_x / cut_x
        new_z = new_z / cut_x
        
        return AirfoilProfile.from_points(new_x, new_z, name=f"{af.name}_open")
        
    return af
