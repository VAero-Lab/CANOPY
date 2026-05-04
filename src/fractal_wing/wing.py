"""
Planform and cross-section geometry for a tapered, swept wing.
"""

import numpy as np
from .airfoil import NACAFourDigit


class Wing:
    """
    Tapered swept wing with spar positions.

    Parameters
    ----------
    b : float
        Semi-span [m].
    cr, ct : float
        Root and tip chord [m].
    sweep : float
        Leading-edge sweep angle [deg].
    af : NACAFourDigit or None
        Airfoil (default NACA 2412).
    sf, sr : float
        Front / rear spar chordwise fractions.
    dih : float
        Dihedral angle [deg].
    bm : str
        Boundary mode: ``'wingbox'`` or ``'full_wing'``.
    """

    def __init__(
        self,
        b: float = 15.0,
        cr: float = 5.0,
        ct: float = 2.0,
        sweep: float = 25.0,
        af: NACAFourDigit | None = None,
        sf: float = 0.15,
        sr: float = 0.65,
        dih: float = 0.0,
        bm: str = 'wingbox',
    ):
        self.b = b
        self.cr = cr
        self.ct = ct
        self.lam = np.radians(sweep)
        self.af = af or NACAFourDigit()
        self.sf = sf
        self.sr = sr
        self.dih = np.radians(dih)
        self.bm = bm

    # ── planform helpers ────────────────────────────────────────
    def c(self, e: float) -> float:
        """Chord at span fraction *e* ∈ [0, 1]."""
        return self.cr * (1 - e * (1 - self.ct / self.cr))

    def xle(self, e: float) -> float:
        """Leading-edge x at span fraction *e*."""
        return e * self.b * np.tan(self.lam)

    def xte(self, e: float) -> float:
        """Trailing-edge x at span fraction *e*."""
        return self.xle(e) + self.c(e)

    def xfs(self, e: float) -> float:
        """Front-spar x at span fraction *e*."""
        return self.xle(e) + self.sf * self.c(e)

    def xrs(self, e: float) -> float:
        """Rear-spar x at span fraction *e*."""
        return self.xle(e) + self.sr * self.c(e)

    def xdf(self, e: float) -> float:
        """Domain front x (LE or front spar depending on *bm*)."""
        return self.xle(e) if self.bm == 'full_wing' else self.xfs(e)

    def xdr(self, e: float) -> float:
        """Domain rear x (TE or rear spar depending on *bm*)."""
        return self.xte(e) if self.bm == 'full_wing' else self.xrs(e)

    def xdc(self, e: float) -> float:
        """Domain centre x at span fraction *e*."""
        return 0.5 * (self.xdf(e) + self.xdr(e))

    # ── 3-D surface ─────────────────────────────────────────────
    def z_at(self, x: float, y: float):
        """Return ``(z_upper, z_lower)`` at planform position *(x, y)*."""
        e = y / self.b
        if e < 0 or e > 1:
            return 0.0, 0.0
        ch = self.c(e)
        xl = self.xle(e)
        xi = float(np.clip((x - xl) / ch, 0.001, 0.999))
        dih_z = y * np.tan(self.dih)
        return self.af.zu(xi) * ch + dih_z, self.af.zl(xi) * ch + dih_z

    # ── boundary queries ────────────────────────────────────────
    def inside(self, x: float, y: float) -> bool:
        """True if *(x, y)* lies inside the structural domain."""
        if y < -1e-6 or y > self.b + 1e-6:
            return False
        e = np.clip(y / self.b, 0, 1)
        return self.xdf(e) - 1e-6 <= x <= self.xdr(e) + 1e-6

    def clip(self, p1: np.ndarray, p2: np.ndarray):
        """
        Clip segment *p1→p2* to the domain boundary.

        Returns the farthest point along the segment that is still inside,
        or ``None`` if the segment never enters the domain.
        """
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        last = 0.0
        for i in range(1, 81):
            t = i / 80
            if self.inside(p1[0] + t * dx, p1[1] + t * dy):
                last = t
            else:
                lo, hi = last, t
                for _ in range(25):
                    m = 0.5 * (lo + hi)
                    if self.inside(p1[0] + m * dx, p1[1] + m * dy):
                        lo = m
                    else:
                        hi = m
                last = lo
                break
        if last > 1e-6:
            return np.array([p1[0] + last * dx, p1[1] + last * dy])
        return None

    def cla(self, sc: float = 1.0) -> float:
        """Centerline angle from root to *sc* span-fraction."""
        return np.arctan2(sc * self.b, self.xdc(sc) - self.xdc(0))
