import fractal_wing as fw
from aeroshape.geometry.airfoils import AirfoilProfile
from aeroshape.geometry.wings import SegmentSpec, MultiSegmentWing
import numpy as np

af = AirfoilProfile.from_naca4("2412", num_points=100)
new_x, new_z = [], []
for x, z in zip(af.x, af.z):
    if x < 0.99:
        new_x.append(x)
        new_z.append(z)
        
new_x = np.array(new_x)
new_z = np.array(new_z)
max_x = np.max(new_x)
new_x = new_x / max_x
new_z = new_z / max_x

af_open = AirfoilProfile.from_points(new_x, new_z, name="Open_2412")

spec = SegmentSpec(
    span=15.0,
    root_chord=5.0,
    tip_chord=2.0,
    sweep_le_deg=25.0,
    root_airfoil=af_open,
    num_sections=20
)
aero_w = MultiSegmentWing().add_segment(spec)
fw.export_hollow_skin(aero_w, output_step="test_open_te.step")
print("SUCCESS: AeroShape handled open TE!")
