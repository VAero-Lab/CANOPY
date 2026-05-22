import frond as fw
from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec
from aeroshape.geometry.airfoils import AirfoilProfile

def get_base_wing(bm='wingbox'):
    """Generate a standard swept wing using AeroShape."""
    spec = SegmentSpec(
        span=15.0,
        root_chord=5.0,
        tip_chord=2.0,
        sweep_le_deg=25.0,
        root_airfoil=AirfoilProfile.from_naca4("2412"),
        num_sections=20
    )
    aero_w = MultiSegmentWing().add_segment(spec)
    return aero_w, fw.AeroWingAdapter(aero_w, bm=bm)
