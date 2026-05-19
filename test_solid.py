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

te_faces = [f for f in wing_shape.faces() if f.bounding_box().max.X > 4.9]
print(f"Number of TE faces: {len(te_faces)}")
for i, f in enumerate(te_faces):
    bb = f.bounding_box()
    print(f"  Face {i} DX: {bb.max.X - bb.min.X:.3f}, DY: {bb.max.Y - bb.min.Y:.3f}, DZ: {bb.max.Z - bb.min.Z:.3f}")
