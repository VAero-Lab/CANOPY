"""
Aerodynamic Solver and Mapping Bridge

This module handles running FLOWPanel.jl via subprocess and mapping the computed
aerodynamic panel forces onto the structural FEM skin nodes using scipy.spatial.cKDTree
and Inverse Distance Weighting (IDW).
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import numpy as np
from typing import Dict, List, Set, Tuple
from scipy.spatial import cKDTree

def run_aerodynamic_analysis(
    wing,
    aoa: float,
    magVinf: float,
    rho: float = 1.225,
    num_points_profile: int = 80,
    num_points_spanwise: int | None = None,
    spanwise_clustering = None,
    chordwise_clustering = None,
    debug: bool = False,
    julia_project_dir: str | None = None,
    temp_dir: str | None = None,
) -> Dict[str, list]:
    """
    Run the FLOWPanel.jl aerodynamic solver on the given wing.

    Parameters
    ----------
    wing : aeroshape.geometry.wings.MultiSegmentWing
        The wing object to extract the OML grid from.
    aoa : float
        Angle of attack (degrees).
    magVinf : float
        Freestream velocity magnitude (m/s).
    rho : float
        Air density (kg/m^3).
    num_points_profile : int
        Number of chordwise paneling points.
    num_points_spanwise : int, optional
        Number of spanwise paneling points. If None, uses the wing segments' num_sections.
    spanwise_clustering : callable, optional
        Distribution law f(n) -> array from aeroshape.analysis.clustering.
    chordwise_clustering : callable, optional
        Distribution law f(n) -> array from aeroshape.analysis.clustering.
    debug : bool
        If True, FLOWPanel will save a VTK file of the pressure distribution.
    julia_project_dir : str, optional
        Path to the Julia project directory. Defaults to the workspace root.
    temp_dir : str, optional
        Path to save temporary JSON files. Defaults to the workspace root.

    Returns
    -------
    dict
        Dictionary containing 'centroids', 'forces', 'areas', and 'Cps'.
    """
    # Use workspace root as default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    if julia_project_dir is None:
        julia_project_dir = workspace_root

    if temp_dir is None:
        temp_dir = workspace_root

    # Handle spanwise station count dynamically
    original_num_sections = []
    if num_points_spanwise is not None:
        total_span = sum(seg.span for seg in wing.segments)
        for seg in wing.segments:
            original_num_sections.append(seg.num_sections)
            # Allocate points proportionally to segment spans, ensuring at least 2 points per segment
            fraction = seg.span / total_span if total_span > 0 else 1.0
            seg.num_sections = max(2, int(round(num_points_spanwise * fraction)))

    try:
        # 1. Extract structured coordinates from AeroShape MultiSegmentWing
        print(f"Extracting OML structured grid coordinates from wing (num_points_profile={num_points_profile})...")
        X, Y, Z = wing.to_vertex_grids(
            num_points_profile=num_points_profile,
            spanwise_clustering=spanwise_clustering,
            chordwise_clustering=chordwise_clustering
        )
    finally:
        # Restore original segment divisions to prevent side-effects on original geometry
        if num_points_spanwise is not None:
            for seg, orig in zip(wing.segments, original_num_sections):
                seg.num_sections = orig

    # Convert NumPy arrays to nested lists for JSON serialization
    input_data = {
        "X": X.tolist(),
        "Y": Y.tolist(),
        "Z": Z.tolist(),
        "AoA": aoa,
        "magVinf": magVinf,
        "rho": rho,
        "debug": debug
    }

    # 2. Write to temp JSON file in the workspace
    input_json_path = os.path.join(temp_dir, "aero_input.json")
    output_json_path = os.path.join(temp_dir, "aero_loads.json")

    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    # 3. Call Julia FLOWPanel script via subprocess
    script_path = os.path.join(current_dir, "solve_flowpanel.jl")
    
    # Clean env of dynamic library paths to avoid dynamic link & Pkg segfault issues!
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""

    # Command: julia --project=<julia_project_dir> solve_flowpanel.jl input_json output_json
    cmd = [
        "julia",
        f"--project={julia_project_dir}",
        script_path,
        input_json_path,
        output_json_path
    ]

    print(f"Launching Julia FLOWPanel.jl subprocess...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Julia solve failed with error:")
        print(e.stderr)
        print(e.stdout)
        raise RuntimeError("FLOWPanel aerodynamic solve failed. Check Julia log above.") from e
    finally:
        # Clean up input JSON
        if os.path.exists(input_json_path):
            os.remove(input_json_path)

    # 4. Load the output JSON containing panel centroids, forces, areas, and Cp
    if not os.path.exists(output_json_path):
        raise FileNotFoundError(f"Aerodynamic output file was not generated: {output_json_path}")

    with open(output_json_path, 'r') as f:
        output_data = json.load(f)

    # Clean up output JSON
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    return output_data

def map_aerodynamic_loads(
    aero_centroids: list | np.ndarray,
    aero_forces: list | np.ndarray,
    nodes_dict: Dict[int, Tuple[float, float, float]],
    skin_node_ids: Set[int] | List[int],
    num_neighbors: int = 4,
    power: float = 2.0
) -> Dict[int, np.ndarray]:
    """
    Map aerodynamic panel force vectors onto the structural skin nodes.
    Uses cKDTree search and Inverse Distance Weighting (IDW).

    Parameters
    ----------
    aero_centroids : list or np.ndarray
        List of panel centroids (size N_panels x 3).
    aero_forces : list or np.ndarray
        List of panel force vectors (size N_panels x 3).
    nodes_dict : dict
        Structural nodes dictionary {node_id: (x, y, z)}.
    skin_node_ids : set or list
        Set of node IDs representing the structural wing skin.
    num_neighbors : int
        Number of nearest neighbors to distribute each force to.
    power : float
        IDW power coefficient.

    Returns
    -------
    dict
        Dictionary mapping node_id -> force_vector [Fx, Fy, Fz] (only for nodes with non-zero force).
    """
    skin_ids = sorted(list(skin_node_ids))
    if not skin_ids:
        raise ValueError("Cannot map loads: skin_node_ids set is empty.")

    skin_coords = np.array([nodes_dict[nid] for nid in skin_ids])
    tree = cKDTree(skin_coords)

    # Pre-allocate mapped forces at skin nodes
    mapped_forces = {nid: np.zeros(3) for nid in skin_ids}

    aero_centroids = np.array(aero_centroids)
    aero_forces = np.array(aero_forces)

    for centroid, force in zip(aero_centroids, aero_forces):
        # Query nearest skin nodes
        distances, indices = tree.query(centroid, k=num_neighbors)

        # Handle k=1 boundary condition format returned by scipy
        if num_neighbors == 1:
            distances = np.array([distances])
            indices = np.array([indices])

        # Avoid division-by-zero if panel centroid is exactly coincident with skin node
        zero_dist_idx = np.where(distances < 1e-10)[0]
        if len(zero_dist_idx) > 0:
            target_idx = indices[zero_dist_idx[0]]
            target_node_id = skin_ids[target_idx]
            mapped_forces[target_node_id] += force
        else:
            # IDW weights
            weights = 1.0 / (distances ** power)
            sum_w = np.sum(weights)
            normalized_weights = weights / sum_w

            for idx, weight in zip(indices, normalized_weights):
                target_node_id = skin_ids[idx]
                mapped_forces[target_node_id] += force * weight

    # Verify force conservation
    total_aero_force = np.sum(aero_forces, axis=0)
    total_mapped_force = np.sum(list(mapped_forces.values()), axis=0)
    discrepancy = np.linalg.norm(total_mapped_force - total_aero_force)

    print("Aerodynamic load mapping conservation summary:")
    print(f"  Total Aero Force:   Fx={total_aero_force[0]:.4f}, Fy={total_aero_force[1]:.4f}, Fz={total_aero_force[2]:.4f} N")
    print(f"  Total Mapped Force: Fx={total_mapped_force[0]:.4f}, Fy={total_mapped_force[1]:.4f}, Fz={total_mapped_force[2]:.4f} N")
    print(f"  Discrepancy (L2 norm): {discrepancy:.6e} N")

    # Filter out nodes with negligible force to keep CalculiX input deck compact
    return {nid: force for nid, force in mapped_forces.items() if np.linalg.norm(force) > 1e-10}
