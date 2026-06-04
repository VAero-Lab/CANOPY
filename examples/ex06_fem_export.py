"""
Example 06 — FEM Structured Shell Mesh & CalculiX Simulation
=============================================================
Demonstrates the complete FEM pipeline:
  1. Generate fractal structure
  2. Export B-Rep STEP geometry (webs + hollow wing skin)
  3. Generate unified structured Quad (S4) mesh with Gmsh
  4. Build CalculiX simulation deck (material, contacts, BCs, loads)
  5. Run CalculiX solver (with automatic parallelism)
  6. Convert results for ParaView visualization
"""

import os
import canopy as cp
import matplotlib.pyplot as plt
from canopy import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_fem')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*65)
    print('  Example 06 — FEM Structured Mesh & CalculiX Simulation')
    print('='*65)

    # ── 1. Generate fractal structure ──
    aero_wing, wing = get_base_wing(bm='wingbox')
    
    stations = cp.make_mixed_stations(
        n_stations=5,
        diag_angle=45,
        chord_angle=90,
        diag_length=2.0,
        chord_length=1.5,
        diag_sub=cp.SubParams(mode='sympodial', max_depth=1, min_length=0.1),
        chord_sub=cp.SubParams(mode='dichotomous', max_depth=1, min_length=0.1)
    )

    spec = cp.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,  # 5mm trunk thickness
        stations=stations,
        allow_crossing=False
    )

    gen = cp.TreeGenerator(wing)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'Generated {st["n"]} raw segments (total length {st["L"]}m).')

    # ── 2. Export B-rep STEP geometry ──
    print('\nGenerating B-Rep Shells from NURBS Wing Skin...')
    webs_step_path = os.path.join(OUT, 'fractal_mesh_shell.step')
    skin_step_path = os.path.join(OUT, 'fractal_mesh_skin.step')
    inp_path = os.path.join(OUT, 'fractal_mesh.inp')
    
    assembly, props = cp.build_brep_webs(segs, aero_wing, as_solid=False, output_step=webs_step_path)
    print(f'  -> Exported Fractal Webs STEP to: {webs_step_path}')
    
    cp.export_hollow_skin(aero_wing, output_step=skin_step_path)
    print(f'  -> Exported Hollow Wing Skin STEP to: {skin_step_path}')

    # ── 3. Generate unified structured FEM mesh ──
    print('\nExtracting Unified Structured FEM mesh (Gmsh)...')
    
    mesher = cp.GmshMesher(target_elem_size=0.025, skin_elem_size=0.05)
    mesh_stats = mesher.mesh(
        webs_step_path, inp_path,
        skin_step=skin_step_path,
        web_properties=props,
    )
    
    print(f'  -> Target Element Size: {mesher.target_size}m')
    print(f'  -> Vertical elements per web (nz): {mesh_stats["nz"]}')
    print(f'  -> Found {mesh_stats["n_nodes"]} nodes.')
    print(f'  -> Found {mesh_stats["n_elems"]} structured Quad (S4) elements.')
    print(f'  -> Webs: {mesh_stats["n_webs"]}, Skin faces: {mesh_stats["n_skin_faces"]}')

    # ── 4. Build CalculiX simulation deck ──
    print('\nBuilding CalculiX Simulation Deck...')
    sim_path = cp.build_ccx_deck(
        mesh_inp=inp_path,
        web_properties=props,
        segments=segs,
        skin_thickness=0.003,   # 3mm skin
    )

    # ── 5. Run CalculiX solver & 6. Convert results ──
    print('\nRunning CalculiX solver...')
    result = cp.run_ccx(sim_path, convert_vtu=True)
    
    # ── Visualize the 1D graph for reference ──
    viz_fw = cp.Viz(wing)
    fig = viz_cp.view3d(segs)
    fig.savefig(os.path.join(OUT, 'ex06_fem_graph.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\n' + '='*65)

if __name__ == '__main__':
    main()
