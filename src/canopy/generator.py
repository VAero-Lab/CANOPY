"""
Fractal tree-branch generator and conventional baseline.

Key enhancements over v4:
  - ``_grow`` uses ``sub.mode_at(depth)`` for per-depth mode switching
  - ``_grow`` uses ``sub.min_length`` instead of a hardcoded threshold
  - ``generate()`` delegates to ``_generate_trunk()`` for multi-trunk support
  - ``generate_multi()`` accumulates segments from several trunks while
    respecting crossing constraints globally
  - Branch initial thickness uses ``BranchDef.thick_frac``
"""

from __future__ import annotations

import numpy as np
from typing import List
import shapely
from shapely.geometry import LineString, Point
from shapely import set_precision
from shapely.ops import unary_union

from .aeroshape_adapter import AeroWingAdapter
from .structures import Seg, TrunkSpec, segments_cross


class TreeGenerator:
    """
    Recursive fractal tree-branch structure generator.

    Parameters
    ----------
    wing : AeroWingAdapter
        Wing geometry that defines the structural domain.
    """

    def __init__(self, wing: AeroWingAdapter):
        self.w = wing
        self.segs: List[Seg] = []
        self._id = 0
        self._allow_crossing = True

    # ── public API ──────────────────────────────────────────────

    def generate(self, spec: TrunkSpec) -> List[Seg]:
        """
        Generate segments for a single trunk specification.

        Resets internal state before generating.
        """
        self.segs = []
        self._id = 0
        self._allow_crossing = spec.allow_crossing
        self._generate_trunk(spec)
        # Compute dynamic relative tolerance based on wing dimensions
        tol = self.w.b * 1e-4
        self.segs = clean_topology(self.segs, tol)
        return self.segs

    def generate_multi(self, specs: List[TrunkSpec]) -> List[Seg]:
        """
        Generate segments from multiple trunk specifications.

        Segments accumulate across trunks, so crossing constraints
        are enforced globally.

        Parameters
        ----------
        specs : list of TrunkSpec
            Each trunk can have its own chord position, stations,
            branching modes, etc.

        Returns
        -------
        list of Seg
        """
        self.segs = []
        self._id = 0
        for spec in specs:
            self._allow_crossing = spec.allow_crossing
            self._generate_trunk(spec)
        
        tol = self.w.b * 1e-4
        self.segs = clean_topology(self.segs, tol)
        return self.segs

    def stats(self) -> dict:
        """Return ``{'n': <segment_count>, 'L': <total_length>}``."""
        if not self.segs:
            return {'n': 0, 'L': 0}
        return {
            'n': len(self.segs),
            'L': round(sum(s.length for s in self.segs), 1),
        }

    # ── internals ───────────────────────────────────────────────

    def _generate_trunk(self, spec: TrunkSpec):
        """Generate one trunk without resetting the segment list."""
        w = self.w

        # Trunk start
        xf, xr = w.xdf(0), w.xdr(0)
        x0 = xf + spec.chord_frac * (xr - xf)
        start = np.array([x0, 0.0])

        # Trunk direction and length
        trunk_a = w.cla(spec.span_cov) + np.radians(spec.dir_offset)
        y_end = spec.span_cov * w.b
        trunk_len = y_end / max(np.sin(trunk_a), 0.1)
        trunk_d = np.array([np.cos(trunk_a), np.sin(trunk_a)])

        end = start + trunk_len * trunk_d
        if not w.inside(end[0], end[1]):
            c = w.clip(start, end)
            if c is not None:
                end = c
            trunk_len = np.linalg.norm(end - start)

        # Sort stations
        stations = sorted(spec.stations or [], key=lambda s: s.position)

        # Build trunk segments between stations and spawn branches
        pts = [0.0] + [s.position for s in stations] + [1.0]
        objs = [None] + list(stations) + [None]
        cur_a = trunk_a

        for i in range(len(pts) - 1):
            t0, t1 = pts[i], pts[i + 1]
            d = np.array([np.cos(cur_a), np.sin(cur_a)])
            p0 = start + t0 * trunk_len * d
            p1 = start + t1 * trunk_len * d

            if not w.inside(p0[0], p0[1]):
                break
            if not w.inside(p1[0], p1[1]):
                c = w.clip(p0, p1)
                if c is None:
                    break
                p1 = c

            added = self._add_seg(p0, p1, spec.thick, 0, protected=spec.protect_trunk)
            if not added:
                break

            st = objs[i + 1]
            if st is not None:
                for bdef in st.branches:
                    ba = cur_a + np.radians(bdef.angle)
                    bd = np.array([np.cos(ba), np.sin(ba)])
                    self._grow(
                        p1, bd, bdef.length,
                        spec.thick * bdef.thick_frac,
                        1, bdef.sub, 0,
                    )
                cur_a += np.radians(st.mono_defl)

    def _add_seg(self, p0, p1, thick, level, protected=False) -> bool:
        """Add segment, checking for crossings if enabled.

        Parameters
        ----------
        protected : bool
            If True, the segment is always added regardless of crossings.
            Used for trunk (level 0) segments that must never be interrupted.
        """
        if not protected and not self._allow_crossing:
            for s in self.segs:
                if segments_cross(p0, p1, s.p0, s.p1):
                    return False
        sid = self._id
        self._id += 1
        self.segs.append(Seg(p0.copy(), p1.copy(), thick, level, sid))
        return True

    def _grow(self, start, direction, length, thick, level, sub, depth):
        """Recursively grow a branch."""
        if depth > sub.max_depth or length < sub.min_length:
            return
        if not self.w.inside(start[0], start[1]):
            return

        end = start + length * direction
        clipped = False
        if not self.w.inside(end[0], end[1]):
            c = self.w.clip(start, end)
            if c is None:
                return
            end = c
            clipped = True

        added = self._add_seg(start, end, thick, level)
        if not added or clipped:
            return

        ang, lr, tr, nc = sub.at(depth)
        cl = length * lr
        ct = thick * tr
        theta = np.radians(ang)
        pa = np.arctan2(direction[1], direction[0])

        # Per-depth mode switching
        mode = sub.mode_at(depth)

        if mode == 'monopodial':
            # Continuation along parent direction
            self._grow(
                end, np.array([np.cos(pa), np.sin(pa)]),
                cl, ct, level + 1, sub, depth + 1,
            )
            # Symmetric side branches
            for k in range(1, nc // 2 + 1):
                for s in (+1, -1):
                    a = pa + s * k * theta
                    self._grow(
                        end, np.array([np.cos(a), np.sin(a)]),
                        cl, ct * tr, level + 1, sub, depth + 1,
                    )

        elif mode == 'sympodial':
            if nc == 2:
                for s in (+1, -1):
                    self._grow(
                        end,
                        np.array([np.cos(pa + s * theta),
                                 np.sin(pa + s * theta)]),
                        cl, ct, level + 1, sub, depth + 1,
                    )
            else:
                for da in np.linspace(-theta, theta, nc):
                    self._grow(
                        end,
                        np.array([np.cos(pa + da), np.sin(pa + da)]),
                        cl, ct, level + 1, sub, depth + 1,
                    )

        elif mode == 'dichotomous':
            for s in (+1, -1):
                self._grow(
                    end,
                    np.array([
                        np.cos(pa + s * theta * 0.5),
                        np.sin(pa + s * theta * 0.5),
                    ]),
                    cl, thick, level + 1, sub, depth + 1,
                )

        elif mode == 'monochasium':
            sign = 1 if depth % 2 == 0 else -1
            self._grow(
                end,
                np.array([
                    np.cos(pa + np.radians(sign * 5)),
                    np.sin(pa + np.radians(sign * 5)),
                ]),
                cl, ct, level + 1, sub, depth + 1,
            )
            self._grow(
                end,
                np.array([
                    np.cos(pa + sign * theta),
                    np.sin(pa + sign * theta),
                ]),
                cl * 0.8, ct * tr, level + 1, sub, depth + 1,
            )


# ─────────────────────────────────────────────────────────────────
# Topology Cleanup
# ─────────────────────────────────────────────────────────────────

def clean_topology(segs: List[Seg], tol: float) -> List[Seg]:
    """
    Cleans up the topological graph by snapping near-miss nodes, splitting
    intersections, and culling micro-segments. Eliminates CAD brittleness.
    """
    if not segs:
        return []
        
    lines = []
    for s in segs:
        l = LineString([s.p0, s.p1])
        # Snap coordinates to a grid of size `tol`
        l_grid = set_precision(l, grid_size=tol)
        if l_grid.length >= tol:
            lines.append(l_grid)
            
    if not lines:
        return []

    # Node the graph (intersect and split automatically)
    noded = unary_union(lines)
    
    if isinstance(noded, LineString):
        out_lines = [noded]
    else:
        out_lines = list(noded.geoms)

    cleaned_segs = []
    for l in out_lines:
        if l.length < tol:
            continue
            
        mid = l.interpolate(0.5, normalized=True)
        best_seg = None
        best_dist = float('inf')
        
        for s in segs:
            orig_l = LineString([s.p0, s.p1])
            d = orig_l.distance(mid)
            if d < best_dist:
                best_dist = d
                best_seg = s
                
        if best_seg:
            c0, c1 = l.coords[0], l.coords[-1]
            cleaned_segs.append(Seg(np.array(c0), np.array(c1), best_seg.thick, best_seg.level, best_seg.sid))

    return cleaned_segs


# ─────────────────────────────────────────────────────────────────
# Conventional baseline
# ─────────────────────────────────────────────────────────────────

class ConvWB:
    """
    Conventional wingbox internal structure (spars + ribs).

    Parameters
    ----------
    w : AeroWingAdapter
        Wing geometry.
    nr : int
        Number of rib stations.
    ns : int
        Number of spanwise stiffeners (spar-to-spar).
    """

    def __init__(self, w: AeroWingAdapter, nr: int = 12, ns: int = 3):
        self.w = w
        self.nr = nr
        self.ns = ns

    def generate(self) -> List[Seg]:
        segs = []
        sid = 0
        # Spanwise stiffeners
        for f in np.linspace(self.w.sf, self.w.sr, self.ns):
            for i in range(self.nr - 1):
                e1 = i / (self.nr - 1)
                e2 = (i + 1) / (self.nr - 1)
                segs.append(Seg(
                    np.array(
                        [self.w.xle(e1) + f * self.w.c(e1), e1 * self.w.b]),
                    np.array(
                        [self.w.xle(e2) + f * self.w.c(e2), e2 * self.w.b]),
                    0.003, 0, sid,
                ))
                sid += 1
        # Ribs
        for i in range(self.nr):
            e = i / (self.nr - 1)
            y = e * self.w.b
            segs.append(Seg(
                np.array([self.w.xfs(e), y]),
                np.array([self.w.xrs(e), y]),
                0.002, 1, sid,
            ))
            sid += 1
        return segs
