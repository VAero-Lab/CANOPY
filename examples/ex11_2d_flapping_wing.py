import os
import sys
import numpy as np

# Add src to path for running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import canopy as cp
from canopy.meshing import GmshMesher2D, append_1d_beams
from canopy.aero_solver import run_vlm_analysis, map_aerodynamic_loads
from canopy.fem_solver import parse_mesh_for_mapping, build_ccx_deck
from canopy import get_base_wing

OUT = 'examples/output_2d_flapping'
os.makedirs(OUT, exist_ok=True)

def main():
    print("=================================================================")
    print("  Example 11 — 2D Flapping Wing (VLM + 1D Beams)")
    print("=================================================================")

    # 1. Geometry and Adapter
    from aeroshape.geometry.wings import MultiSegmentWing, SegmentSpec
    from aeroshape.geometry.airfoils import AirfoilProfile

    # Define a biologically-scaled wing (e.g. 0.4m span, 15cm root chord)
    spec = SegmentSpec(
        span=0.05,
        root_chord=0.05,
        tip_chord=0.040,
        sweep_le_deg=10,
        root_airfoil=AirfoilProfile.from_naca4("0012"),
        num_sections=20
    )
    aero_wing = MultiSegmentWing().add_segment(spec)
    wing = cp.AeroWingAdapter(aero_wing, bm='full_wing')
    
    # 2. Fractal Structure Generation (Multi-Trunk)
    sub_mt = cp.SubParams(max_depth=2, angles=[35, 30], mode='sympodial', min_length=0.005)
    specs = [
        cp.TrunkSpec(
            chord_frac=0.25, span_cov=1.0, thick=0.0006,
            stations=cp.make_mixed_stations(
                n_stations=8, diag_angle=40, chord_angle=70,
                diag_length=0.08, chord_length=0.05,
                diag_sub=cp.SubParams(max_depth=3, mode='sympodial', min_length=0.005),
                chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=True,
        ),
        cp.TrunkSpec(
            chord_frac=0.75, span_cov=1.0, thick=0.0006,
            stations=cp.make_mixed_stations(
                n_stations=8, diag_angle=-40, chord_angle=-70,
                diag_length=0.08, chord_length=0.05,
                diag_sub=cp.SubParams(max_depth=2, mode='sympodial', min_length=0.005),
                chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=True,
        )
    ]
    gen = cp.TreeGenerator(wing)
    segs = gen.generate_multi(specs)
    print(f"  Generated {len(segs)} segments.")

    print("\nMeshing...")
    mesher = GmshMesher2D(target_elem_size=0.0025) # scaled down element size
    skin_inp = os.path.join(OUT, 'wing_skin.inp')
    
    # 3.1. Flat 2D skin mesh
    mesh_stats = mesher.mesh_skin(wing, output_inp=skin_inp, export_msh=True)
    print(f"  Skin mesh generated: {mesh_stats['n_nodes']} nodes, {mesh_stats['n_elems']} elements.")

    # 3.2. Discretize 1D beams and append to INP
    print("  Discretizing fractal segments into 1D B31 beams...")
    elem_groups = append_1d_beams(segs, target_size=0.0075, inp_path=skin_inp)

    # 4. Aerodynamics (VortexLattice.jl VLM)
    print("\nRunning VLM Aerodynamics...")
    aoa = 3.0
    V = 10.0
    density = 0.95
    aero_data = run_vlm_analysis(wing, aoa=aoa, magVinf=V, rho=density, num_x=40, num_y=80, save_vtk=True)
    
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
    skin_thickness = 1e-4  # 1mm (prevents numerical instability in linear solver)
    pla = {
    "name": "PLA",
    "E1": 3.5e9,    # Pa
    "E2": 3.5e9,
    "E3": 3.5e9,
    "nu12": 0.36,
    "nu13": 0.36,
    "nu23": 0.36,
    "G12": 1.29e9,  # E / (2*(1+nu))
    "G13": 1.29e9,
    "G23": 1.29e9,
    "density": 1240.0,  # kg/m^3
    }
    
    sim_inp = build_ccx_deck(
        mesh_inp=skin_inp,
        material=pla,
        web_properties={}, # Not used for beams
        segments=segs,     # Not heavily used for B31 orientation, handled differently
        skin_thickness=skin_thickness,
        mapped_aero_forces=mapped_forces,
        binary_output=False,
        skin_elset_name="SKIN_2D",
        beam_section="CIRC",
        beam_wall_thickness_ratio=0.2,
        nlgeom=False,
        isotropic=True,
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
        from canopy.run_utils import split_vtu_file
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
