"""
Example 13 — Cluster-Ready Fractal Wing Optimization
==================================================
Demonstrates how to optimize the continuous geometric parameters of an internal
fractal web network to minimize structural compliance under steady aerodynamic
loads. Supports COBYLA, SLSQP, and global Differential Evolution optimization.
"""

import os
import shutil
import json
import numpy as np
import canopy as cp
from canopy import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), "3D_wing_Opt")

def clean_output_dir(out_dir):
    """Remove previous optimization files but intentionally preserve .json load files."""
    if os.path.exists(out_dir):
        print(f"Cleaning previous outputs from: {out_dir}")
        for filename in os.listdir(out_dir):
            if filename.endswith(".json"):
                continue  # PRESERVE json files!
            file_path = os.path.join(out_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    os.makedirs(out_dir, exist_ok=True)



def main():
    print("=" * 75)
    print("  Example 13 — Cluster-Ready Fractal Wing Optimization")
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
            "diag_sub": cp.SubParams(
                mode=["sympodial", 'monopodial'],
                max_depth=3,
                angles=[30, 25, 20],
                length_ratios=[0.6, 0.55, 0.5],
                min_length=0.015,
            ),
            "chord_sub": cp.SubParams(mode="dichotomous", max_depth=2),
            "thick_frac": 0.75,
        },
        {
            "eta_start": 0.35,
            "eta_end": 0.7,
            "diag_angle": 40.0,
            "chord_angle": 75.0,
            "diag_length": 1.8,
            "chord_length": 1.5,
            "diag_sub": cp.SubParams(mode="monopodial", max_depth=2, min_length=0.02),
            "chord_sub": cp.SubParams(mode="sympodial", max_depth=1),
            "thick_frac": 0.9,
        },
        {
            "eta_start": 0.7,
            "eta_end": 1.0,
            "diag_angle": 50.0,
            "chord_angle": None,
            "diag_length": 1.2,
            "chord_length": 0.8,
            "diag_sub": cp.SubParams(mode="monochasium", max_depth=2, min_length=0.02),
            "chord_sub": cp.SubParams(mode="dichotomous", max_depth=1),
            "thick_frac": 0.9,
        },
    ]

    # ── 2. Load Pre-calculated Aerodynamic Loads ──
    print("\n2. Loading steady aerodynamic loads from baseline JSON...")
    loads_file = os.path.join(OUT, "baseline_aero_loads.json")
    
    if not os.path.exists(loads_file):
        raise FileNotFoundError(
            f"Could not find {loads_file}. Please run ex12_fractal_optimization.py "
            "first to generate the baseline aerodynamic loads."
        )
        
    with open(loads_file, "r") as f:
        serializable_loads = json.load(f)
        
    # Convert string keys back to integers and lists back to numpy arrays
    mapped_forces = {int(k): np.array(v) for k, v in serializable_loads.items()}
    print(f"   -> Loaded distributed panel loads for {len(mapped_forces)} skin nodes.")

    # ── 4. Set up the AeroStructuralOptimizer ──
    

    # ── 4. Define Optimization Bounds & Topology Builder ──
    print("\n4. Defining optimization mapping logic...")
    bounds = [
        (0.004, 0.006),  # x[0]: global main trunk thickness [m]
        (15.0, 45.0),    # x[1]: Zone 1 diag_angle [deg]
        (45.0, 85.0),    # x[2]: Zone 1 chord_angle [deg]
        (1.5, 4.0),      # x[3]: Zone 1 diag_length [m]
        (1.0, 3.5),      # x[4]: Zone 1 chord_length [m]
        (0.5, 1.0),      # x[5]: Zone 1 thick_frac
        (20.0, 60.0),    # x[6]: Zone 2 diag_angle [deg]
        (50.0, 85.0),    # x[7]: Zone 2 chord_angle [deg]
        (1.0, 3.0),      # x[8]: Zone 2 diag_length [m]
        (0.8, 2.5),      # x[9]: Zone 2 chord_length [m]
        (0.4, 0.9),      # x[10]: Zone 2 thick_frac
        (30.0, 75.0),    # x[11]: Zone 3 diag_angle [deg]
        (0.5, 2.0),      # x[12]: Zone 3 diag_length [m]
        (0.2, 0.7),      # x[13]: Zone 3 thick_frac
    ]
    
    x_init = np.array([
        0.005,
        25.0, 65.0, 3.0, 2.5, 0.85,
        40.0, 75.0, 1.8, 1.5, 0.70,
        50.0, 1.2, 0.50
    ])
    
    import copy
    def build_topology(x, wing_obj):
        zones = copy.deepcopy(baseline_zones)
        trunk_thick = float(x[0])

        zones[0]["diag_angle"] = float(x[1])
        zones[0]["chord_angle"] = float(x[2])
        zones[0]["diag_length"] = float(x[3])
        zones[0]["chord_length"] = float(x[4])
        zones[0]["thick_frac"] = float(x[5])

        zones[1]["diag_angle"] = float(x[6])
        zones[1]["chord_angle"] = float(x[7])
        zones[1]["diag_length"] = float(x[8])
        zones[1]["chord_length"] = float(x[9])
        zones[1]["thick_frac"] = float(x[10])

        zones[2]["diag_angle"] = float(x[11])
        zones[2]["diag_length"] = float(x[12])
        zones[2]["thick_frac"] = float(x[13])
        
        stations = cp.make_zoned_stations(n_stations=10, zones=zones)
        spec = cp.TrunkSpec(
            chord_frac=0.5,
            span_cov=1.0,
            thick=trunk_thick,
            stations=stations,
            allow_crossing=False,
        )
        gen = cp.TreeGenerator(wing_obj)
        return gen.generate(spec)

    print("\n5. Initializing AeroStructuralOptimizer...")
    optimizer = cp.AeroStructuralOptimizer(
        aero_wing=aero_wing,
        wing=wing,
        bounds=bounds,
        x_init=x_init,
        topology_builder=build_topology,
        output_dir=OUT,
        mode="uncoupled",
        mapped_loads_dict=mapped_forces if 'mapped_forces' in locals() else loaded_forces,
        save_every_n_iterations=5,      
        skin_thickness=0.003,          
    )

    # ── 4. Run Cluster Optimization (e.g. Differential Evolution) ──
    print("\n4. Running full cluster optimization (method='DE')...")
    print("-" * 75)
    
    # Run the DE solver for robust global optimization
    results = optimizer.optimize(method="cobyla", max_iter=300, popsize=3)
    print("-" * 75)
    print("Cluster optimization completed successfully!")
    print(f"  Total solver calls: {results['iterations']}")
    print(f"  History trace written to: {optimizer.history_csv}")
    print(f"  Saved baseline/final structures to: {OUT}")
    
    print("\n" + "=" * 75)
    print("  To run a full optimization to completion, modify max_iter and pick:")
    print("  -> 'cobyla' (fastest local search, 1 solver call/iter)")
    print("  -> 'slsqp' (gradient-based local search, N+1 solver calls/iter)")
    print("  -> 'differential_evolution' (robust global search, global population)")
    print("=" * 75)



    print("\n[Post-Processing] Generating convergence animation...")
    post_processor = cp.OptimizationPostProcessor(OUT)
    post_processor.animate_history(build_topology, aero_wing, wing)

if __name__ == "__main__":
    main()
