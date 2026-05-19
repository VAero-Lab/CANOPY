import fractal_wing as fw
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
from aeroshape.geometry.airfoils import AirfoilProfile
from build123d import Plane, Face

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_open = fw.ensure_open_te(af)

spec = SegmentSpec(
    span=15.0, root_chord=5.0, tip_chord=2.0, sweep_le_deg=25.0,
    root_airfoil=af_open, num_sections=20
)
aero_w = MultiSegmentWing().add_segment(spec)
wing_shape = aero_w.to_occ_shape()

# Let's create a web at the trailing edge!
# The trailing edge is at X=5.0 at Y=0.0
# Let's create a branch that goes exactly to X=5.0
p0 = (1.0, 0.0)
p1 = (5.0, 0.0) # exact trailing edge!

length = 4.0
mid_x = 3.0
mid_y = 0.0

dir_x = (1.0, 0.0, 0.0)
dir_z = (0.0, 1.0, 0.0)

workplane = Plane(origin=(mid_x, mid_y, 0.0), x_dir=dir_x, z_dir=dir_z)
web_face = Face.make_rect(length, 100.0, plane=workplane)

trimmed_web = web_face.intersect(wing_shape)

print(f"Trimmed web has {len(trimmed_web.edges())} edges.")
for e in trimmed_web.edges():
    bb = e.bounding_box()
    dx = bb.max.X - bb.min.X
    dz = bb.max.Z - bb.min.Z
    print(f"  Edge: dx={dx:.3f}, dz={dz:.3f}, min_X={bb.min.X:.3f}, max_X={bb.max.X:.3f}")

