import numpy as np
from aeroshape.geometry.airfoils import AirfoilProfile

def ensure_open_te(af, cut_fraction: float = 0.01):
    x = np.array(af.x)
    z = np.array(af.z)
    
    x_max = np.max(x)
    te_points = np.where(x > 0.999 * x_max)[0]
    z_te = z[te_points]
    
    if np.max(z_te) - np.min(z_te) < 1e-3 * x_max:
        keep_idx = np.where(x <= x_max * (1.0 - cut_fraction))[0]
        
        new_x = x[keep_idx]
        new_z = z[keep_idx]
        
        new_max_x = np.max(new_x)
        new_x = new_x / new_max_x
        new_z = new_z / new_max_x
        
        return AirfoilProfile.from_points(new_x, new_z, name=f"{af.name}_open")
        
    return af

af = AirfoilProfile.from_naca4("2412", num_points=100)
print(f"Original: {af.z[-1] - af.z[0]}")
af_open = ensure_open_te(af)
print(f"Open: {af_open.z[-1] - af_open.z[0]}")

