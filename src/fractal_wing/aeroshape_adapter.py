"""
AeroShape Integration Adapter

This module bridges the 3D NURBS MultiSegmentWing from AeroShape with the
2D generative structural boundaries required by fractal-wing.
"""

import numpy as np

class AeroWingAdapter:
    """
    Adapter that wraps an AeroShape MultiSegmentWing to provide the
    analytical 2D boundaries (leading edge, trailing edge, spars)
    needed by the fractal-wing TreeGenerator.
    
    Parameters
    ----------
    aerowing : aeroshape.geometry.wings.MultiSegmentWing
        The AeroShape wing model.
    sf : float
        Front spar chordwise fraction (e.g. 0.15).
    sr : float
        Rear spar chordwise fraction (e.g. 0.65).
    bm : str
        Boundary mode: ``'wingbox'`` or ``'full_wing'``.
    """

    def __init__(self, aerowing, sf=0.15, sr=0.65, bm='wingbox'):
        self.w = aerowing
        self.sf = sf
        self.sr = sr
        self.bm = bm
        
        # Extract pre-computed sections from AeroShape
        self.frames = self.w.get_section_frames()
        if not self.frames:
            raise ValueError("AeroShape wing must have defined segments/frames.")
            
        self.y_root = self.frames[0]['y']
        self.y_tip = self.frames[-1]['y']
        self.b = self.y_tip - self.y_root

    def _interp(self, y: float, key: str) -> float:
        """Linearly interpolate a frame property along the span."""
        if y <= self.y_root:
            return self.frames[0][key]
        if y >= self.y_tip:
            return self.frames[-1][key]
            
        for i in range(len(self.frames) - 1):
            f0 = self.frames[i]
            f1 = self.frames[i + 1]
            if f0['y'] <= y <= f1['y']:
                dy = f1['y'] - f0['y']
                if dy < 1e-8:
                    return f0[key]
                t = (y - f0['y']) / dy
                return f0[key] * (1 - t) + f1[key] * t
        return self.frames[-1][key]
        
    def _interp_airfoil(self, y: float):
        """Interpolate the AirfoilProfile object along the span."""
        if y <= self.y_root:
            return self.frames[0]['airfoil']
        if y >= self.y_tip:
            return self.frames[-1]['airfoil']
            
        for i in range(len(self.frames) - 1):
            f0 = self.frames[i]
            f1 = self.frames[i + 1]
            if f0['y'] <= y <= f1['y']:
                # For simplicity in z_at extraction, just return the closest airfoil
                # or the rootward one for fast Z bounds checking.
                t = (y - f0['y']) / (f1['y'] - f0['y'])
                return f0['airfoil'] if t < 0.5 else f1['airfoil']
        return self.frames[-1]['airfoil']

    # ── planform helpers ────────────────────────────────────────
    def c(self, e: float) -> float:
        return self._interp(e * self.b, 'chord')

    def xle(self, e: float) -> float:
        return self._interp(e * self.b, 'x_offset')

    def xte(self, e: float) -> float:
        return self.xle(e) + self.c(e)

    def xfs(self, e: float) -> float:
        return self.xle(e) + self.sf * self.c(e)

    def xrs(self, e: float) -> float:
        return self.xle(e) + self.sr * self.c(e)

    def xdf(self, e: float) -> float:
        return self.xle(e) if self.bm == 'full_wing' else self.xfs(e)

    def xdr(self, e: float) -> float:
        return self.xte(e) if self.bm == 'full_wing' else self.xrs(e)

    def xdc(self, e: float) -> float:
        return 0.5 * (self.xdf(e) + self.xdr(e))

    # ── boundary queries ────────────────────────────────────────
    def inside(self, x: float, y: float) -> bool:
        if y < self.y_root - 1e-6 or y > self.y_tip + 1e-6:
            return False
        e = np.clip(y / self.b, 0, 1) if self.b > 0 else 0
        return self.xdf(e) - 1e-6 <= x <= self.xdr(e) + 1e-6

    def clip(self, p1: np.ndarray, p2: np.ndarray):
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

    # ── 3-D surface ─────────────────────────────────────────────
    def z_at(self, x: float, y: float):
        """
        Approximate Z bounds for visualization in Matplotlib.
        For exact CAD STEP export, cad_export.py uses boolean intersection
        against the exact B-rep OML.
        """
        e = y / self.b if self.b > 0 else 0
        if e < 0 or e > 1:
            return 0.0, 0.0
            
        ch = self.c(e)
        xl = self.xle(e)
        z_off = self._interp(y, 'z_offset')
        
        # Local chord fraction
        xi = np.clip((x - xl) / ch, 0.001, 0.999) if ch > 1e-6 else 0.5
        
        af = self._interp_airfoil(y)
        # AeroShape AirfoilProfile contains x, z coordinates.
        if hasattr(af, 'x') and hasattr(af, 'z'):
            af_x = np.array(af.x)
            af_z = np.array(af.z)
            
            # Find leading edge index
            le_idx = np.argmin(af_x)
            
            # Upper surface is from LE to TE (x increases from 0 to 1)
            x_up = af_x[le_idx:]
            z_up_curve = af_z[le_idx:]
            
            # Lower surface is from TE to LE (x decreases from 1 to 0)
            # Reverse it so x is monotonically increasing for np.interp
            x_low = af_x[:le_idx+1][::-1]
            z_low_curve = af_z[:le_idx+1][::-1]
            
            z_max = np.interp(xi, x_up, z_up_curve)
            z_min = np.interp(xi, x_low, z_low_curve)
            
            return z_max * ch + z_off, z_min * ch + z_off
            
        # Fallback fallback
        return 0.05 * ch + z_off, -0.05 * ch + z_off
