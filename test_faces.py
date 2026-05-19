import fractal_wing as fw
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
from aeroshape.geometry.airfoils import AirfoilProfile

af = AirfoilProfile.from_naca4("2412", num_points=100)
af_open = fw.ensure_open_te(af)

spec = SegmentSpec(
    span=15.0, root_chord=5.0, tip_chord=2.0, sweep_le_deg=25.0,
    root_airfoil=af_open, num_sections=20
)
aero_w = MultiSegmentWing().add_segment(spec)
wing_shape = aero_w.to_occ_shape()

print(f"Number of faces: {len(wing_shape.faces())}")
for i, f in enumerate(wing_shape.faces()):
    bb = f.bounding_box()
    print(f"Face {i}: X in [{bb.min.X:.3f}, {bb.max.X:.3f}], Y in [{bb.min.Y:.3f}, {bb.max.Y:.3f}], Z in [{bb.min.Z:.3f}, {bb.max.Z:.3f}]")
