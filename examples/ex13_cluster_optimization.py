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
import frond as fw
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), "3D_wing_Opt")

def clean_output_dir(out_dir):
    """Remove previous optimization files from the output directory to start fresh."""
    if os.path.exists(out_dir):
        print(f"Cleaning previous outputs from: {out_dir}")
        for filename in os.listdir(out_dir):
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
    print("\n4. Initializing AeroStructuralOptimizer...")
    optimizer = fw.AeroStructuralOptimizer(
        aero_wing=aero_wing,
        wing=wing,
        baseline_zones=baseline_zones,
        output_dir=OUT,
        mode="uncoupled",
        mapped_loads_dict=mapped_forces,
        save_every_n_iterations=5,      # Save files every 5 iterations (plus initial/final)
        skin_thickness=0.003,          # 3mm skin thickness
    )

    # ── 4. Run Cluster Optimization (e.g. Differential Evolution) ──
    print("\n4. Running full cluster optimization (method='DE')...")
    print("-" * 75)
    
    # Run the DE solver for robust global optimization
    results = optimizer.optimize(method="differential_evolution", max_iter=20, pop_size=5)
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


if __name__ == "__main__":
    main()
