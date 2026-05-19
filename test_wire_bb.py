import fractal_wing as fw
from aeroshape.geometry.airfoils import AirfoilProfile
import build123d as b3d

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_open = fw.ensure_open_te(af)

wire_occ = af_open.to_occ_wire()
wire = b3d.Wire(wire_occ)
bb = wire.bounding_box()
print(f"Wire BB: X in [{bb.min.X:.3f}, {bb.max.X:.3f}], Z in [{bb.min.Z:.3f}, {bb.max.Z:.3f}]")

# Let's inspect the geometry near X=1.0
vertices = wire.vertices()
print(f"Number of vertices: {len(vertices)}")
for v in vertices:
    print(f"Vertex at X={v.X:.3f}, Z={v.Z:.3f}")
