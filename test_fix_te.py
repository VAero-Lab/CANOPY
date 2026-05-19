import numpy as np
from aeroshape.geometry.airfoils import AirfoilProfile

def ensure_open_te_fixed(af, cut_fraction: float = 0.01):
    x = np.array(af.x)
    z = np.array(af.z)
    
    le_idx = np.argmin(x)
    x_max = np.max(x)
    cut_x = x_max * (1.0 - cut_fraction)
    
    # Lower curve (TE to LE, so x decreases)
    x_lower = x[:le_idx+1]
    z_lower = z[:le_idx+1]
    
    # Upper curve (LE to TE, so x increases)
    x_upper = x[le_idx:]
    z_upper = z[le_idx:]
    
    # We need to interpolate at cut_x. 
    # For lower curve, x is decreasing, so we must reverse it for np.interp
    z_lower_cut = np.interp(cut_x, x_lower[::-1], z_lower[::-1])
    z_upper_cut = np.interp(cut_x, x_upper, z_upper)
    
    # Filter points
    keep_lower = x_lower <= cut_x
    keep_upper = x_upper <= cut_x
    
    new_x_lower = x_lower[keep_lower]
    new_z_lower = z_lower[keep_lower]
    # Because x_lower starts at TE (which we cut), the first element should be the cut point.
    # Wait, if x_lower starts at TE (x~1) and goes to LE (x=0), the values x > cut_x are at the BEGINNING of the array!
    # So we replace the removed points with the cut point at the beginning!
    new_x_lower = np.insert(new_x_lower, 0, cut_x)
    new_z_lower = np.insert(new_z_lower, 0, z_lower_cut)
    
    new_x_upper = x_upper[keep_upper]
    new_z_upper = z_upper[keep_upper]
    # x_upper goes from LE (x=0) to TE (x~1). The cut point is at the END.
    new_x_upper = np.append(new_x_upper, cut_x)
    new_z_upper = np.append(new_z_upper, z_upper_cut)
    
    # Recombine
    # lower goes from TE to LE, upper goes from LE to TE
    # The LE point is duplicated (it's the last of lower, first of upper). We can remove the duplicate.
    new_x = np.concatenate([new_x_lower[:-1], new_x_upper])
    new_z = np.concatenate([new_z_lower[:-1], new_z_upper])
    
    # Rescale
    new_x = new_x / cut_x
    new_z = new_z / cut_x
    
    return AirfoilProfile.from_points(new_x, new_z, name=f"{af.name}_open")

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_fixed = ensure_open_te_fixed(af)

print("Original TE bounds:", af.x[0], af.x[-1])
print("Fixed TE bounds:", af_fixed.x[0], af_fixed.x[-1])
print("Fixed TE Z gap:", af_fixed.z[-1] - af_fixed.z[0])
