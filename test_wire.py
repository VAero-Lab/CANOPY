import fractal_wing as fw
from aeroshape.geometry.airfoils import AirfoilProfile
import build123d as b3d

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_open = fw.ensure_open_te(af)

wire_occ = af_open.to_occ_wire()
wire = b3d.Wire(wire_occ)
print(f"Is wire closed? {wire.is_closed}")
print(f"Number of edges in wire: {len(wire.edges())}")

