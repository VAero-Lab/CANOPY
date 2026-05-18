"""
Example 08 — Organic FEM Pipeline
=================================
Demonstrates the full FEM pipeline for a highly complex organic
fractal structure with multiple tip loads.
"""

import os
import fractal_wing as fw
import matplotlib.pyplot as plt
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_fem')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*65)
    print('  Example 08 — Organic FEM Pipeline')
    print('='*65)

    # ── 1. Generate fractal structure ──
    # Using 'full_wing' from utils
    aero_wing, wing = get_base_wing(bm='full_wing')
    
    zones = [
        {
            'eta_start': 0.0, 'eta_end': 0.35,
            'diag_angle': 25, 'chord_angle': 65,
            'diag_length': 3.0, 'chord_length': 2.5,
            'diag_sub': fw.SubParams(
                mode=['sympodial', 'monopodial'], max_depth=3,
                angles=[30, 25, 20], length_ratios=[0.6, 0.55, 0.5],
                min_length=0.015,
            ),
            'chord_sub': fw.SubParams(mode='dichotomous', max_depth=2),
            'thick_frac': 0.85,
        },
        {
            'eta_start': 0.35, 'eta_end': 0.7,
            'diag_angle': 40, 'chord_angle': 75,
            'diag_length': 1.8, 'chord_length': 1.5,
            'diag_sub': fw.SubParams(mode='monopodial', max_depth=2, min_length=0.02),
            'chord_sub': fw.SubParams(mode='sympodial', max_depth=1),
            'thick_frac': 0.7,
        },
        {
            'eta_start': 0.7, 'eta_end': 1.0,
            'diag_angle': 50, 'chord_angle': None,
            'diag_length': 1.2, 'chord_length': 0.8,
            'diag_sub': fw.SubParams(mode='monochasium', max_depth=2, min_length=0.02),
            'chord_sub': fw.SubParams(mode='dichotomous', max_depth=1),
            'thick_frac': 0.5,
        },
    ]
    stations = fw.make_zoned_stations(
        n_stations=18, zones=zones,
        spacing='geometric', spacing_kwargs={'ratio': 1.6},
    )

    spec = fw.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,
        stations=stations,
        allow_crossing=False
    )

    gen = fw.TreeGenerator(wing)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'Generated {st["n"]} raw segments (total length {st["L"]}m).')

    # ── 2. Export B-rep STEP geometry ──
    print('\nGenerating B-Rep Shells from NURBS Wing Skin...')
    webs_step_path = os.path.join(OUT, 'organic_shell.step')
    skin_step_path = os.path.join(OUT, 'organic_skin.step')
    inp_path = os.path.join(OUT, 'organic_mesh.inp')
    
    assembly, props = fw.build_brep_webs(segs, aero_wing, as_solid=False, output_step=webs_step_path)
    print(f'  -> Exported Fractal Webs STEP to: {webs_step_path}')
    
    fw.export_hollow_skin(aero_wing, output_step=skin_step_path)
    print(f'  -> Exported Hollow Wing Skin STEP to: {skin_step_path}')

    # ── 3. Generate unified structured FEM mesh ──
    print('\nExtracting Unified Structured FEM mesh (Gmsh)...')
    
    mesher = fw.GmshMesher(target_elem_size=0.05)
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
    
    # Adding multiple point loads at the tip (Leading Edge and Trailing Edge)
    point_loads = [
        {'x_frac': 0.0, 'load': -500.0},  # Leading Edge
        {'x_frac': 1.0, 'load': -500.0},  # Trailing Edge
    ]
    
    sim_path = fw.build_ccx_deck(
        mesh_inp=inp_path,
        web_properties=props,
        segments=segs,
        skin_thickness=0.003,
        point_loads=point_loads,
    )

    # ── 5. Run CalculiX solver & 6. Convert results ──
    print('\nRunning CalculiX solver...')
    result = fw.run_ccx(sim_path, convert_vtu=True)
    
    # ── Visualize the 1D graph for reference ──
    viz_fw = fw.Viz(wing)
    fig = viz_fw.view3d(segs)
    fig.savefig(os.path.join(OUT, 'ex08_organic_graph.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print('\n' + '='*65)

if __name__ == '__main__':
    main()
