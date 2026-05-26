"""
Example 12 — Parametric Fractal Wing Optimization
==================================================
Demonstrates how to optimize the continuous geometric parameters of an internal
fractal web network to minimize structural compliance under steady aerodynamic
loads. Supports COBYLA, SLSQP, and global Differential Evolution optimization.
"""

import os
import numpy as np
import frond as fw
from utils import get_base_wing

# Output directory for optimization iteration files and trace logs
OUT = os.path.join(os.path.dirname(__file__), "3D_wing_Opt")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 75)
    print("  Example 12 — Parametric Fractal Wing Optimization")
    print("=" * 75)

    # ── 1. Planform & Baseline Zoned organic fractal topology ──
    print("\n1. Generating AeroShape 3D NURBS Wing Planform...")
    aero_wing, wing = get_base_wing(bm="full_wing")

    # Baseline zones from Example 08 (organic zoned fractal topology)
    baseline_zones = [
        {
            "eta_start": 0.0,
            "eta_end": 0.35,
            "diag_angle": 25.0,
            "chord_angle": 65.0,
            "diag_length": 3.0,
            "chord_length": 2.5,
            "diag_sub": fw.SubParams(
                mode=["sympodial", 'monopodial'],
                max_depth=3,
                angles=[30, 25, 20],
                length_ratios=[0.6, 0.55, 0.5],
                min_length=0.015,
            ),
            "chord_sub": fw.SubParams(mode="dichotomous", max_depth=2),
            "thick_frac": 0.85,
        },
        {
            "eta_start": 0.35,
            "eta_end": 0.7,
            "diag_angle": 40.0,
            "chord_angle": 75.0,
            "diag_length": 1.8,
            "chord_length": 1.5,
            "diag_sub": fw.SubParams(mode="monopodial", max_depth=2, min_length=0.02),
            "chord_sub": fw.SubParams(mode="sympodial", max_depth=1),
            "thick_frac": 0.7,
        },
        {
            "eta_start": 0.7,
            "eta_end": 1.0,
            "diag_angle": 50.0,
            "chord_angle": None,
            "diag_length": 1.2,
            "chord_length": 0.8,
            "diag_sub": fw.SubParams(mode="monochasium", max_depth=2, min_length=0.02),
            "chord_sub": fw.SubParams(mode="dichotomous", max_depth=1),
            "thick_frac": 0.5,
        },
    ]

    # ── 2. Run baseline flow solve once & Map loads (Uncoupled Mode) ──
    print("\n2. Pre-calculating steady aerodynamic loads on baseline wing...")
    aoa = 4.0          # Angle of Attack (deg)
    velocity = 30.0    # Freestream velocity (m/s)
    rho = 1.225        # Air density (kg/m^3)

    aero_data = fw.run_aerodynamic_analysis(
        wing=aero_wing,
        aoa=aoa,
        magVinf=velocity,
        rho=rho,
        num_points_profile=40,
        num_points_spanwise=15,
        debug=True,
        temp_dir=OUT,
    )

    # ── 3. Parse baseline structural mesh for mapping ──
    print("\n3. Generating baseline mesh to map aerodynamic panel loads...")
    # First, generate baseline tree segments
    baseline_stations = fw.make_zoned_stations(n_stations=10, zones=baseline_zones)
    baseline_spec = fw.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,
        stations=baseline_stations,
        allow_crossing=False,
    )
    baseline_gen = fw.TreeGenerator(wing)
    baseline_segs = baseline_gen.generate(baseline_spec)

    webs_step = os.path.join(OUT, "baseline_webs.step")
    skin_step = os.path.join(OUT, "baseline_skin.step")
    mesh_inp = os.path.join(OUT, "baseline_mesh.inp")

    # B-Rep Step export
    fw.build_brep_webs(baseline_segs, aero_wing, as_solid=False, output_step=webs_step)
    fw.export_hollow_skin(aero_wing, output_step=skin_step)

    # Structured quad meshing
    mesher = fw.GmshMesher(target_elem_size=0.15, skin_elem_size=0.25)
    mesh_stats = mesher.mesh(
        webs_step, mesh_inp,
        skin_step=skin_step,
    )

    # Map baseline loads
    nodes, skin_nodes = fw.parse_mesh_for_mapping(mesh_inp)
    mapped_forces = fw.map_aerodynamic_loads(
        aero_centroids=aero_data["centroids"],
        aero_forces=aero_data["forces"],
        nodes_dict=nodes,
        skin_node_ids=skin_nodes,
        num_neighbors=4,
        power=2.0,
    )
    print(f"   -> Distributed panel loads to {len(mapped_forces)} skin nodes.")

    # ── 4. Set up the AeroStructuralOptimizer ──
    print("\n4. Initializing AeroStructuralOptimizer...")
    optimizer = fw.AeroStructuralOptimizer(
        aero_wing=aero_wing,
        wing=wing,
        baseline_zones=baseline_zones,
        output_dir=OUT,
        mode="uncoupled",
        mapped_loads_dict=mapped_forces,
        save_every_n_iterations=5,      # Save files every 5 iterations (plus initial/final)
        target_volume_fraction=0.30,   # Max 30% envelope volume fraction
        skin_thickness=0.003,          # 3mm skin thickness
    )

    # ── 5. Run Verification Optimization (Limit to 2 iterations) ──
    print("\n5. Running verification iterations (method='SLSQP', max_iter=2)...")
    print("-" * 75)
    
    # Run the SLSQP solver for 2 test iterations to check execution convergence
    results = optimizer.optimize(method="slsqp", max_iter=2)
    
    print("-" * 75)
    print("Verification iterations completed successfully!")
    print(f"  Total solver calls: {results['iterations']}")
    print(f"  History trace written to: {optimizer.history_csv}")
    print(f"  Saved baseline/final structures to: {OUT}")
    
    print("\n" + "=" * 75)
    print("  To run a full optimization to completion, modify max_iter and pick:")
    print("  -> 'cobyla' (fastest local search, 1 solver call/iter)")
    print("  -> 'slsqp' (gradient-based local search, N+1 solver calls/iter)")
    print("  -> 'differential_evolution' (robust global search, global population)")
    print("=" * 75)


if __name__ == "__main__":
    main()
