from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
from aeroshape.geometry.airfoils import AirfoilProfile
from build123d import Plane, Face

af = AirfoilProfile.from_naca4("2412", num_points=100)

spec = SegmentSpec(
    span=15.0, root_chord=5.0, tip_chord=2.0, sweep_le_deg=25.0,
    root_airfoil=af, num_sections=20
)
aero_w = MultiSegmentWing().add_segment(spec)
wing_shape = aero_w.to_occ_shape()

# Trailing edge web exactly to X=5.0
p0 = (1.0, 0.0)
p1 = (5.0, 0.0)
length = 4.0
mid_x = 3.0
mid_y = 0.0
dir_x = (1.0, 0.0, 0.0)
dir_z = (0.0, 1.0, 0.0)

workplane = Plane(origin=(mid_x, mid_y, 0.0), x_dir=dir_x, z_dir=dir_z)
web_face = Face.make_rect(length, 100.0, plane=workplane)

trimmed_web = web_face.intersect(wing_shape)
print(f"Original web has {len(trimmed_web.edges())} edges.")

if len(trimmed_web.edges()) == 3:
    cut_dist = 0.010
    new_length = length - cut_dist
    new_mid_x = mid_x - (cut_dist / 2.0) * dir_x[0]
    new_mid_y = mid_y - (cut_dist / 2.0) * dir_x[1]
    
    new_workplane = Plane(origin=(new_mid_x, new_mid_y, 0.0), x_dir=dir_x, z_dir=dir_z)
    new_web_face = Face.make_rect(new_length, 100.0, plane=new_workplane)
    
    trimmed_web = new_web_face.intersect(wing_shape)
    print(f"Trimmed web now has {len(trimmed_web.edges())} edges.")
    
    for e in trimmed_web.edges():
        bb = e.bounding_box()
        dx = bb.max.X - bb.min.X
        dz = bb.max.Z - bb.min.Z
        print(f"  Edge: dx={dx:.3f}, dz={dz:.3f}, min_X={bb.min.X:.3f}, max_X={bb.max.X:.3f}")

