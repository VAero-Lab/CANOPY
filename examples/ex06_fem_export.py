"""
Example 06 — FEM Structured Shell Mesh Export
==========================================
Demonstrates how to extract a fully structured quadrilateral shell mesh
from the generated fractal segments. The resulting Gmsh export can be used to build
a Finite Element Method (FEM) shell model.
"""

import os
import json
import fractal_wing as fw
import matplotlib.pyplot as plt
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_fem')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*65)
    print('  Example 06 — FEM Structured Mesh Export')
    print('='*65)

    # 1. Generate a small test structure
    aero_wing, wing = get_base_wing(bm='wingbox')
    
    # We use a simple structure to keep the output readable
    stations = fw.make_mixed_stations(
        n_stations=5,
        diag_angle=45,
        chord_angle=90,
        diag_length=2.0,
        chord_length=1.5,
        diag_sub=fw.SubParams(mode='sympodial', max_depth=1, min_length=0.1),
        chord_sub=fw.SubParams(mode='dichotomous', max_depth=1, min_length=0.1)
    )

    spec = fw.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,  # 5mm trunk thickness
        stations=stations,
        allow_crossing=False
    )

    gen = fw.TreeGenerator(wing)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'Generated {st["n"]} raw segments (total length {st["L"]}m).')

    # 2. Export B-rep Surfaces to STEP
    print('\nGenerating B-Rep Shells from NURBS Wing Skin...')
    os.makedirs(OUT, exist_ok=True)
    step_path = os.path.join(OUT, 'fractal_mesh_shell.step')
    inp_path = os.path.join(OUT, 'fractal_mesh.inp')
    
    assembly, props = fw.build_brep_webs(segs, aero_wing, as_solid=False, output_step=step_path)
    print(f'  -> Exported STEP B-Rep shells to: {step_path}')

    # 3. Extract the structured FEM mesh using Gmsh
    print('\nExtracting Structured FEM mesh (Gmsh)...')
    
    # Target element size of 2.5 cm (0.025m) for a smoother, higher-fidelity mesh.
    try:
        mesher = fw.GmshMesher(target_elem_size=0.025)
        mesh_stats = mesher.mesh(step_path, inp_path)
        
        print(f'  -> Target Element Size: {mesher.target_size}m')
        print(f'  -> Vertical elements per web (nz): {mesh_stats["nz"]}')
        print(f'  -> Found {mesh_stats["n_nodes"]} nodes.')
        print(f'  -> Found {mesh_stats["n_elems"]} structured Quad (S4R) elements.')
        print(f'\nCalculiX .inp mesh exported: {inp_path}')
    except ImportError as e:
        print(f"\n[Error] {e}")
        print("Please install gmsh to run the meshing portion of this example.")
    
    # Visualize the 1D graph just for reference
    viz_fw = fw.Viz(wing)
    fig = viz_fw.view3d(segs)
    fig.savefig(os.path.join(OUT, 'ex06_fem_graph.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('='*65)

if __name__ == '__main__':
    main()
