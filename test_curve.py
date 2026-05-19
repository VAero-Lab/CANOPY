import fractal_wing as fw
from aeroshape.geometry.airfoils import AirfoilProfile

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_open = fw.ensure_open_te(af)

wire = af_open.to_occ_wire()
# Let's sample the wire near the trailing edge (parameter close to 0 and 1)
import build123d as b3d
edge = b3d.Edge(wire)
print(f"Edge is closed? {edge.is_closed}")
p_start = edge.position_at(0.0)
p_end = edge.position_at(1.0)
p_mid = edge.position_at(0.5)
print(f"Start: {p_start}")
print(f"End: {p_end}")
print(f"Mid: {p_mid}")

# Let's find the max X point
from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda t: -edge.position_at(t).X, bounds=(0, 1), method='bounded')
p_max = edge.position_at(res.x)
print(f"Max X point: {p_max}")
