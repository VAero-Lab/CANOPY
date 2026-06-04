"""
Example 07 — AeroShape NURBS Integration
=========================================
Demonstrates how to generate the 2D fractal structure and then extrude and
intersect it with the AeroShape 3D NURBS Outer Mold Line (OML). 

This outputs exact B-rep STEP files for the webs whose top/bottom curves
are mathematically identical to the wing skin surface, perfect for 
non-conformal multi-point constraint (glue) FEM meshing.
"""

import os
import canopy as cp
from aeroshape.nurbs.export import NurbsExporter
from canopy import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_cad')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*65)
    print('  Example 07 — AeroShape CAD/NURBS Export')
    print('='*65)

    # 1. Generate AeroShape wing
    print('1. Generating AeroShape 3D NURBS Wing...')
    aero_w, wing_adapter = get_base_wing(bm='wingbox')

    # 2. Configure Fractal Structure
    print('2. Generating Fractal web network...')
    stations = cp.make_mixed_stations(
        n_stations=8,
        diag_angle=40,
        chord_angle=80,
        diag_length=2.5,
        chord_length=1.5,
        diag_sub=cp.SubParams(mode='sympodial', max_depth=2, min_length=0.1),
        chord_sub=cp.SubParams(mode='dichotomous', max_depth=1, min_length=0.1)
    )

    spec = cp.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,  # 5mm trunk thickness
        stations=stations,
        allow_crossing=False
    )

    gen = cp.TreeGenerator(wing_adapter)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'   -> Generated {st["n"]} segments.')

    # 3. Export to STEP
    step_skin = os.path.join(OUT, 'wing_skin.step')
    step_webs_shell = os.path.join(OUT, 'fractal_webs_shell.step')
    step_webs_solid = os.path.join(OUT, 'fractal_webs_solid.step')

    print('\n3. Performing Boolean Cuts & Exporting CAD files...')
    
    # Export Wing Skin
    print(f'   -> Exporting Wing OML to: {step_skin}')
    NurbsExporter.to_step(aero_w.to_occ_shape(), step_skin)

    # Export Webs as Shells (2D surfaces perfectly matching OML)
    print(f'   -> Generating Shell Webs (Mid-surfaces)...')
    webs_shell, props = cp.build_brep_webs(segs, aero_w, as_solid=False, output_step=step_webs_shell)
    print(f'      Saved to: {step_webs_shell}')

    # Export Webs as Solids (3D thickened blocks matching OML)
    print(f'   -> Generating Solid Webs (Thickened)...')
    webs_solid, _ = cp.build_brep_webs(segs, aero_w, as_solid=True, output_step=step_webs_solid)
    print(f'      Saved to: {step_webs_solid}')

    print('='*65)
    print('Integration Complete. You can import these STEP files into')
    print('Gmsh, Salome, or your FEM Pre-processor!')
    print('='*65)


if __name__ == '__main__':
    main()
