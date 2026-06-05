#!/usr/bin/env python3
"""
Example 14: Solid Sandwich Baseline Wing
=========================================================
Generates a perfectly conformal 3D finite element model of a sandwich wing:
- A structured Quadrilateral Carbon Fiber Skin (S4 Shells)
- A conformal Pyramidal transition layer (C3D5 Solids)
- An unstructured EPO Foam Core (C3D10 Tetrahedral Solids)

Uses CalculiX to solve the static response under baseline loads.
"""

import os
import sys
import time
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import canopy as cp
from canopy import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), "solid_wing")
os.makedirs(OUT, exist_ok=True)

def main():
    print("=" * 75)
    print("  Example 14 — Solid Sandwich Baseline Wing")
    print("=" * 75)

    # 1. Generate AeroShape Wing
    aero_wing, wing = get_base_wing(bm="full_wing")

    # 2. Export Solid Volume to STEP
    print("\\n[1] Exporting Solid B-Rep to STEP...")
    step_path = os.path.join(OUT, "solid_wing.step")
    cp.cad_export.export_solid_wing(aero_wing, step_path)
    print(f"  -> {step_path}")

    # 3. Generate Conformal 3D Mesh
    print("\\n[2] Generating Conformal 3D Mesh (Structured Skin + Unstructured Core)...")
    inp_path = os.path.join(OUT, "solid_mesh.inp")
    mesher = cp.GmshMesher(target_elem_size=0.04)  # Slightly coarser for fast testing
    
    # This automatically forces structured quads on the boundary and tets inside
    stats = mesher.mesh_solid_core(step_path, inp_path, export_msh=True)
    print(f"  -> Generated {stats['n_nodes']} nodes. Mesh saved to {inp_path}")

    # 4. Run Aerodynamics and Map Loads
    print("\\n[3] Running Aerodynamic Analysis...")
    aoa = 4.0          # Angle of Attack (deg)
    velocity = 30.0    # Freestream velocity (m/s)
    rho = 1.225        # Air density (kg/m^3)

    aero_data = cp.run_aerodynamic_analysis(
        wing=aero_wing,
        aoa=aoa,
        magVinf=velocity,
        rho=rho,
        num_points_profile=40,
        num_points_spanwise=15,
        debug=False,
        temp_dir=OUT,
    )
    
    print("\\n[4] Mapping Aerodynamic Loads to Solid Mesh...")
    nodes, skin_nodes = cp.parse_mesh_for_mapping(inp_path, elset_name='WingSkin')
    mapped_forces = cp.map_aerodynamic_loads(
        aero_centroids=aero_data["centroids"],
        aero_forces=aero_data["forces"],
        nodes_dict=nodes,
        skin_node_ids=skin_nodes,
        num_neighbors=4,
        power=2.0,
    )
    print(f"  -> Successfully mapped loads to {len(mapped_forces)} skin nodes.")

    # 5. Build CalculiX Deck
    print("\\n[5] Building Sandwich Composite Simulation Deck...")

    sim_path = cp.build_ccx_deck(
        mesh_inp=inp_path,
        web_properties={},  # No webs
        skin_thickness=0.003, # 3mm Carbon skin
        mapped_aero_forces=mapped_forces,
        material=cp.CFRP_T300,        # Skin Material
        core_material=cp.FOAM_EPO,    # Core Material
        output_path=os.path.join(OUT, "solid_sim.inp"),
        binary_output=False
    )

    # 6. Run CalculiX Solver
    print("\\n[6] Running CalculiX Solver...")
    result = cp.run_ccx(sim_path, convert_vtu=True)

    print("\\n===========================================================================")
    if result["success"]:
        print(f"  ✅ Solver Success! ({result['elapsed_s']:.1f}s)")
        print("\\n  Visualization files saved to 3D_wing_Opt/")
        print("  Open solid_sim.vtu in ParaView to inspect the 3D core stresses.")
    else:
        print("  ❌ Solver Failed!")
        print(result["stderr"])
    print("===========================================================================")


if __name__ == "__main__":
    main()
