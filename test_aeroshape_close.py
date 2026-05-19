from aeroshape.geometry.airfoils import AirfoilProfile
import fractal_wing as fw

af = AirfoilProfile.from_naca4("2412")
af_open = fw.ensure_open_te(af)

print(f"Original TE gap: {af.z[-1] - af.z[0]}")
print(f"Open TE gap: {af_open.z[-1] - af_open.z[0]}")

# Let's see if the wing solid has a sharp TE!
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
spec = SegmentSpec(
    span=15.0,
    root_chord=5.0,
    tip_chord=2.0,
    sweep_le_deg=25.0,
    root_airfoil=af_open,
    num_sections=20
)
aero_w = MultiSegmentWing().add_segment(spec)

# Check the Z difference at the trailing edge of the solid!
# At y=0, xte = 5.0. 
# Let's get the bounding box of the OML or evaluate the B-Rep at X=5.0
import build123d
wing_solid = aero_w.to_occ_shape()

# Let's find edges at X=5.0
edges_te = [e for e in wing_solid.edges() if abs(e.bounding_box().max.X - 5.0) < 0.1 and abs(e.bounding_box().min.Y) < 0.1]
print(f"Found {len(edges_te)} edges near Root TE.")
for e in edges_te:
    print(e.bounding_box())

