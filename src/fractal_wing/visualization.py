"""
2-D and 3-D visualisation for fractal wing structures.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — side-effect import
from typing import Dict, List, Sequence

from .aeroshape_adapter import AeroWingAdapter
from .structures import Seg


class Viz:
    """
    Plotting utilities for fractal wing structures.

    Parameters
    ----------
    w : AeroWingAdapter
        Wing geometry used to draw planform boundaries.
    """

    def __init__(self, w: AeroWingAdapter):
        self.w = w

    # ── multi-panel comparison ──────────────────────────────────

    def multi(self, var: Dict[str, List[Seg]], ww: float = 5.5):
        """Side-by-side comparison of labelled segment sets."""
        n = len(var)
        fig, axes = plt.subplots(1, n, figsize=(ww * n, 7))
        if n == 1:
            axes = [axes]
        for ax, (label, br) in zip(axes, var.items()):
            self._dom(ax)
            self._br(ax, br)
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def cmp(self, a: List[Seg], b: List[Seg], ta: str, tb: str):
        """Two-panel comparison."""
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
        for ax, br, t in [(a1, a, ta), (a2, b, tb)]:
            self._dom(ax)
            self._br(ax, br)
            ax.set_title(t, fontsize=11, fontweight='bold')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # ── 3-D view ────────────────────────────────────────────────

    def view3d(self, br: List[Seg], title: str = '', elev: float = 25, azim: float = -50):
        """3-D view with translucent skin and internal webs."""
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._skin3d(ax, 0.12)
        self._web3d(ax, br)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        r = max(self.w.b, self.w.c(0)) * 0.6
        mx = self.w.xle(0.5) + self.w.c(0.5) * 0.4
        x_range = (mx + r * 0.8) - (mx - r * 0.5)
        y_range = (self.w.b + 0.5) - (-0.5)
        z_range = (r * 0.15) - (-r * 0.15)
        
        ax.set_xlim(mx - r * 0.5, mx + r * 0.8)
        ax.set_ylim(-0.5, self.w.b + 0.5)
        ax.set_zlim(-r * 0.15, r * 0.15)
        
        # Set proportional box aspect ratio (x, y, z)
        ax.set_box_aspect((x_range, y_range, z_range))
        plt.tight_layout()
        return fig

    # ── cross-section slices ────────────────────────────────────

    def xsec(self, br: List[Seg], etas: Sequence[float] | None = None):
        """Cross-section views at given span fractions."""
        if etas is None:
            etas = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
        fig, axes = plt.subplots(len(etas), 1, figsize=(10, 2.8 * len(etas)))
        if len(etas) == 1:
            axes = [axes]
        for ax, e in zip(axes, etas):
            self._xs(ax, br, e)
        plt.tight_layout()
        return fig

    # ── private helpers ─────────────────────────────────────────

    def _dom(self, ax):
        """Draw wing planform domain outline on *ax*."""
        eta = np.linspace(0, 1, 100)
        y = eta * self.w.b
        ax.plot([self.w.xle(e) for e in eta], y, 'k-', lw=1.5)
        ax.plot([self.w.xte(e) for e in eta], y, 'k-', lw=1.5)
        ax.plot([self.w.xle(0), self.w.xte(0)], [0, 0], 'k-', lw=1.5)
        ax.plot([self.w.xle(1), self.w.xte(1)], [y[-1], y[-1]], 'k-', lw=1.5)
        ax.plot([self.w.xfs(e) for e in eta], y, 'b--', lw=0.7, alpha=0.3)
        ax.plot([self.w.xrs(e) for e in eta], y, 'b--', lw=0.7, alpha=0.3)
        ax.fill_betweenx(
            y,
            [self.w.xdf(e) for e in eta],
            [self.w.xdr(e) for e in eta],
            alpha=0.04, color='green',
        )

    def _br(self, ax, segs):
        """Draw branch segments on *ax*."""
        if not segs:
            return
        mx = max(s.level for s in segs)
        for s in segs:
            t = s.level / max(mx, 1)
            ax.plot(
                [s.p0[0], s.p1[0]], [s.p0[1], s.p1[1]],
                color=plt.cm.YlOrRd(0.2 + 0.7 * t),
                lw=max(0.3, 3.0 - 2.2 * t),
                solid_capstyle='round',
            )

    def _skin3d(self, ax, al):
        ne, nc = 30, 20
        ev = np.linspace(0, 1, ne)
        Xu, Yu, Zu = [np.zeros((ne, nc)) for _ in range(3)]
        Xl, Yl, Zl = [np.zeros((ne, nc)) for _ in range(3)]
        for i, e in enumerate(ev):
            y = e * self.w.b
            ch = self.w.c(e)
            xl = self.w.xle(e)
            for j, xi in enumerate(np.linspace(self.w.sf, self.w.sr, nc)):
                x = xl + xi * ch
                zu, zl = self.w.z_at(x, y)
                Xu[i, j], Yu[i, j], Zu[i, j] = x, y, zu
                Xl[i, j], Yl[i, j], Zl[i, j] = x, y, zl
        ax.plot_surface(Xu, Yu, Zu, alpha=al, color='steelblue', edgecolor='none')
        ax.plot_surface(Xl, Yl, Zl, alpha=al, color='steelblue', edgecolor='none')

    def _web3d(self, ax, segs):
        if not segs:
            return
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mx = max(s.level for s in segs)
        polys = []
        colors = []
        for s in segs:
            t = s.level / max(mx, 1)
            c = plt.cm.YlOrRd(0.2 + 0.7 * t)
            zu0, zl0 = self.w.z_at(s.p0[0], s.p0[1])
            zu1, zl1 = self.w.z_at(s.p1[0], s.p1[1])
            
            vtx = [
                (s.p0[0], s.p0[1], zl0),
                (s.p1[0], s.p1[1], zl1),
                (s.p1[0], s.p1[1], zu1),
                (s.p0[0], s.p0[1], zu0)
            ]
            polys.append(vtx)
            colors.append(c)
            
        collection = Poly3DCollection(polys, facecolors=colors, edgecolors='none', alpha=0.8)
        ax.add_collection3d(collection)

    def _xs(self, ax, segs, eta):
        ys = eta * self.w.b
        xl = self.w.xle(eta)
        xc = xl + np.linspace(0, 1, 200) * self.w.c(eta)
        zu = [self.w.z_at(x, ys)[0] for x in xc]
        zl = [self.w.z_at(x, ys)[1] for x in xc]
        ax.fill_between(xc, zl, zu, alpha=0.1, color='steelblue')
        ax.plot(xc, zu, 'b-', lw=1.5)
        ax.plot(xc, zl, 'b-', lw=1.5)
        for s in segs:
            dy = s.p1[1] - s.p0[1]
            if abs(dy) < 1e-10:
                continue
            tp = (ys - s.p0[1]) / dy
            if 0 <= tp <= 1:
                xp = s.p0[0] + tp * (s.p1[0] - s.p0[0])
                zup, zlo = self.w.z_at(xp, ys)
                ax.plot([xp, xp], [zlo, zup], 'r-', lw=1.0, alpha=0.8)
        for xs in [self.w.xfs(eta), self.w.xrs(eta)]:
            zu_, zl_ = self.w.z_at(xs, ys)
            ax.plot([xs, xs], [zl_, zu_], 'b--', lw=0.7, alpha=0.3)
        ax.set_title(f'η={eta:.2f} (y={ys:.1f}m)', fontsize=10)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
