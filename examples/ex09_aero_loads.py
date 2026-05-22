"""
Example 09 — Aerodynamic Load Transfer & FEM Pipeline
====================================================
Demonstrates how to run the FLOWPanel.jl 3D panel method solver, map the
resulting panel forces onto structural skin nodes using IDW mapping, and run
a CalculiX static structural analysis with the mapped aerodynamic loads.
"""

import os
import frond as fw
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_fem_aero')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*70)
    print('  Example 09 — Aerodynamic Load Transfer & FEM Pipeline')
    print('='*70)

    # 1. Generate wing planform and fractal structure
    print('\n1. Generating AeroShape 3D NURBS Wing and Fractal Web Graph...')
    aero_wing, wing = get_base_wing(bm='full_wing')

    # Keep structural grid simple for quick meshing/simulation in this demo
    stations = fw.make_mixed_stations(
        n_stations=6,
        diag_angle=35,
        chord_angle=75,
        diag_length=2.2,
        chord_length=1.5,
        diag_sub=fw.SubParams(mode='sympodial', max_depth=2, min_length=0.1),
        chord_sub=fw.SubParams(mode='dichotomous', max_depth=1, min_length=0.1)
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
    print(f'   -> Generated {len(segs)} fractal segments.')

    # 2. Export B-rep STEP geometry
    print('\n2. Exporting B-Rep STEP geometry for skin and webs...')
    webs_step_path = os.path.join(OUT, 'aero_webs_shell.step')
    skin_step_path = os.path.join(OUT, 'aero_skin.step')
    inp_path = os.path.join(OUT, 'aero_wing_mesh.inp')

    assembly, props = fw.build_brep_webs(segs, aero_wing, as_solid=False, output_step=webs_step_path)
    fw.export_hollow_skin(aero_wing, output_step=skin_step_path)
    print(f'   -> Saved B-Rep files to {OUT}')

    # 3. Generate FEM mesh in Gmsh
    print('\n3. Generating FEM mesh with Gmsh...')
    # Use slightly coarser mesh sizes for demonstration speed
    mesher = fw.GmshMesher(target_elem_size=0.05, skin_elem_size=0.075, skin_clustering=0.2)
    mesh_stats = mesher.mesh(
        webs_step_path, inp_path,
        skin_step=skin_step_path,
        web_properties=props,
    )
    print(f'   -> Mesh generated: {mesh_stats["n_nodes"]} nodes, {mesh_stats["n_elems"]} elements.')

    # 4. Run aerodynamic analysis in FLOWPanel.jl
    print('\n4. Running 3D Panel Method Aerodynamic Analysis (FLOWPanel.jl)...')
    aoa = 4.0            # Angle of Attack (deg)
    velocity = 30.0      # Freestream velocity (m/s)
    rho = 1.225          # Density (kg/m^3)
    
    # Run solver; debug=True will output a VTK surface mesh with Cps and forces
    aero_data = fw.run_aerodynamic_analysis(
        wing=aero_wing,
        aoa=aoa,
        magVinf=velocity,
        rho=rho,
        num_points_profile=60,  # 60 chordwise panels
        num_points_spanwise=20, # 20 spanwise panels (demonstrates spanwise mesh control)
        debug=True,
        temp_dir=OUT
    )
    
    # 5. Parse mesh for mapping and apply IDW load transfer
    print('\n5. Mapping aerodynamic panel forces onto FEM skin nodes...')
    nodes, skin_nodes = fw.parse_mesh_for_mapping(inp_path)
    print(f'   -> Total structural nodes: {len(nodes)}')
    print(f'   -> Skin structural nodes:  {len(skin_nodes)}')
    
    mapped_forces = fw.map_aerodynamic_loads(
        aero_centroids=aero_data["centroids"],
        aero_forces=aero_data["forces"],
        nodes_dict=nodes,
        skin_node_ids=skin_nodes,
        num_neighbors=4,
        power=2.0
    )
    print(f'   -> Distributed panel loads to {len(mapped_forces)} skin nodes.')

    # 6. Build CalculiX input deck with mapped loads
    print('\n6. Building CalculiX static structural simulation deck...')
    sim_path = fw.build_ccx_deck(
        mesh_inp=inp_path,
        web_properties=props,
        segments=segs,
        skin_thickness=0.003, # 3mm carbon skin
        mapped_aero_forces=mapped_forces
    )

    # 7. Run CalculiX FEM solver
    print('\n7. Solving structural analysis (CalculiX)...')
    ccx_results = fw.run_ccx(sim_path, convert_vtu=True)
    
    print('\n' + '='*70)
    print('  Aerodynamic Load Transfer & FEM Solve Complete!')
    print(f'  CalculiX results saved to: {sim_path.replace(".inp", ".vtu")}')
    print(f'  FLOWPanel VTK saved to:    {os.path.join(OUT, "flowpanel_wing.vtk")}')
    print('='*70)

if __name__ == '__main__':
    main()
