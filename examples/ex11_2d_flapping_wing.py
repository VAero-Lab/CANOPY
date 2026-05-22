import os
import sys
import numpy as np

# Add src to path for running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import fractal_wing as fw
from fractal_wing.meshing import GmshMesher2D, append_1d_beams
from fractal_wing.aero_solver import run_vlm_analysis, map_aerodynamic_loads
from fractal_wing.fem_solver import parse_mesh_for_mapping, build_ccx_deck
from utils import get_base_wing

OUT = 'examples/output_2d_flapping'
os.makedirs(OUT, exist_ok=True)

def main():
    print("=================================================================")
    print("  Example 11 — 2D Flapping Wing (VLM + 1D Beams)")
    print("=================================================================")

    # 1. Geometry and Adapter
    _, wing = get_base_wing(bm='full_wing')
    
    # 2. Fractal Structure Generation (Multi-Trunk)
    sub_mt = fw.SubParams(max_depth=2, angles=[35, 30], mode='sympodial', min_length=0.02)
    specs = [
        fw.TrunkSpec(
            chord_frac=0.25, span_cov=1.0, thick=0.005,
            stations=fw.make_mixed_stations(
                n_stations=12, diag_angle=40, chord_angle=70,
                diag_length=2.5, chord_length=2.0,
                diag_sub=fw.SubParams(max_depth=3, mode='sympodial', min_length=0.02),
                chord_sub=fw.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=True,
        ),
        fw.TrunkSpec(
            chord_frac=0.75, span_cov=1.0, thick=0.005,
            stations=fw.make_mixed_stations(
                n_stations=12, diag_angle=-40, chord_angle=-70,
                diag_length=2.5, chord_length=2.0,
                diag_sub=fw.SubParams(max_depth=2, mode='sympodial', min_length=0.02),
                chord_sub=fw.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=False,
        ),
    ]

    print("Generating fractal structure...")
    gen = fw.TreeGenerator(wing)
    segs = gen.generate_multi(specs)
    print(f"  Generated {len(segs)} segments.")

    # 3. Meshing (2D Skin + 1D Beams)
    print("\nMeshing...")
    mesher = GmshMesher2D(target_elem_size=0.2)
    skin_inp = os.path.join(OUT, 'wing_skin.inp')
    
    # 3.1. Flat 2D skin mesh
    mesh_stats = mesher.mesh_skin(wing, output_inp=skin_inp, export_msh=True)
    print(f"  Skin mesh generated: {mesh_stats['n_nodes']} nodes, {mesh_stats['n_elems']} elements.")

    # 3.2. Discretize 1D beams and append to INP
    print("  Discretizing fractal segments into 1D B31 beams...")
    elem_groups = append_1d_beams(segs, target_size=0.2, inp_path=skin_inp)

    # 4. Aerodynamics (VortexLattice.jl VLM)
    print("\nRunning VLM Aerodynamics...")
    aoa = 5.0
    V = 10.0
    aero_data = run_vlm_analysis(wing, aoa=aoa, magVinf=V, num_x=20, num_y=40, save_vtk=True)
    
    # 5. Load Mapping
    print("\nMapping Aerodynamic Loads to Skin Mesh...")
    nodes_dict, skin_node_ids = parse_mesh_for_mapping(skin_inp, elset_name="SKIN_2D")
    
    mapped_forces = map_aerodynamic_loads(
        aero_centroids=aero_data['centroids'],
        aero_forces=aero_data['forces'],
        nodes_dict=nodes_dict,
        skin_node_ids=skin_node_ids,
        num_neighbors=4,
        power=2.0
    )

    # 6. FEM Deck Build (CalculiX)
    print("\nBuilding CalculiX Simulation Deck...")
    # For a flapping wing membrane, we use a very thin shell (e.g. 0.05mm)
    skin_thickness = 5e-5 
    
    sim_inp = build_ccx_deck(
        mesh_inp=skin_inp,
        web_properties={}, # Not used for beams
        segments=segs,     # Not heavily used for B31 orientation, handled differently
        skin_thickness=skin_thickness,
        mapped_aero_forces=mapped_forces,
        binary_output=False,
        skin_elset_name="SKIN_2D"
    )
    print(f"  Simulation deck ready: {sim_inp}")

    # 7. Run FEM and convert to VTK
    print("\nRunning CalculiX...")
    import subprocess
    cwd = os.getcwd()
    os.chdir(OUT)
    
    # Run ccx
    try:
        subprocess.run(["ccx", "-i", "wing_skin_sim"], check=True, capture_output=True, text=True)
        print("  FEM solve complete. Converting results to VTK...")
        subprocess.run(["ccx2paraview", "wing_skin_sim.frd", "vtu"], check=True, capture_output=True, text=True)
        print("  Converted to VTU successfully.")
        from fractal_wing.run_utils import split_vtu_file
        split_vtu_file("wing_skin_sim.vtu")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Could not run FEM or ccx2paraview. Command {e.cmd} returned non-zero exit status {e.returncode}.")
        if e.stdout:
            print(f"  STDOUT: {e.stdout}")
        if e.stderr:
            print(f"  STDERR: {e.stderr}")
    except FileNotFoundError as e:
        print(f"  Warning: Could not run FEM or ccx2paraview. {e}")
    finally:
        os.chdir(cwd)

    print(f"\n{'='*65}")
    print(f"  Done — saved to {OUT}/")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
