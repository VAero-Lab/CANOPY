"""
CalculiX Solver Runner Utilities.

Provides a portable, reusable function for executing CalculiX simulations
with optimal parallelism settings and optional post-processing.
"""

from __future__ import annotations

import os
import subprocess
import time


def run_ccx(
    sim_inp: str,
    n_threads: int = None,
    convert_vtu: bool = False,
    ccx_binary: str = "ccx",
) -> dict:
    """
    Run the CalculiX solver with optimal parallelism settings.

    Sets environment variables for parallel stiffness/results assembly
    (``CCX_NPROC_STIFFNESS``, ``CCX_NPROC_RESULTS``, ``OMP_NUM_THREADS``)
    before invoking ``ccx``.  These variables are standard CalculiX/OpenMP
    controls and are silently ignored on builds that do not support them,
    making this function fully portable across machines.

    Parameters
    ----------
    sim_inp : str
        Path to the CalculiX simulation ``.inp`` file.
    n_threads : int, optional
        Number of threads for parallel assembly and results calculation.
        Defaults to ``min(os.cpu_count(), 8)`` to prevent OOM errors and
        contact-search race conditions on high-core-count clusters using
        the default SPOOLES solver.
    convert_vtu : bool
        If True, convert the ``.frd`` results file to ``.vtu`` format
        via ``ccx2paraview`` after a successful solve.  Default is False.
    ccx_binary : str
        Name or path of the CalculiX executable.  Default is ``"ccx"``.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``success`` (bool): Whether ``ccx`` returned exit code 0.
        - ``elapsed_s`` (float): Wall-clock time for the solve [seconds].
        - ``frd_path`` (str or None): Path to the ``.frd`` results file.
        - ``vtu_path`` (str or None): Path to the ``.vtu`` file (if converted).
        - ``returncode`` (int): The raw process exit code.
        - ``stdout`` (str): Solver standard output (last 1000 chars).
        - ``stderr`` (str): Solver standard error (last 1000 chars).
    """
    if not os.path.isfile(sim_inp):
        raise FileNotFoundError(f"Simulation input file not found: {sim_inp}")

    # ── Thread count ──
    if n_threads is None:
        # Cap at 8 to prevent OOM / segfaults on high-core-count clusters
        # (e.g. 72 threads) when using standard SPOOLES ccx.
        n_threads = min(os.cpu_count() or 4, 8)

    # ── Build environment with parallelism hints ──
    env = os.environ.copy()
    nt = str(n_threads)
    env["CCX_NPROC_STIFFNESS"] = nt
    env["CCX_NPROC_RESULTS"] = nt
    env["OMP_NUM_THREADS"] = nt

    # ── Prepare paths ──
    cwd = os.path.dirname(os.path.abspath(sim_inp))
    job_basename = os.path.basename(sim_inp).replace(".inp", "")
    frd_path = os.path.join(cwd, job_basename + ".frd")

    # ── Run solver ──
    print(f"  -> Running CalculiX ({ccx_binary}) with {n_threads} threads ...")
    t0 = time.perf_counter()

    result = subprocess.run(
        [ccx_binary, job_basename],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed = time.perf_counter() - t0

    success = result.returncode == 0
    if success:
        print(f"  -> CalculiX solver completed successfully in {elapsed:.1f}s")
    else:
        print(f"  -> CalculiX solver FAILED (exit code {result.returncode}) "
              f"after {elapsed:.1f}s")
        tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        if tail:
            print(f"     {tail}")

    # ── Optional VTU conversion ──
    vtu_path = None
    if success and convert_vtu and os.path.isfile(frd_path):
        print("  -> Converting results for ParaView (ccx2paraview)...")
        try:
            from ccx2paraview import Converter
            import logging
            logging.basicConfig(level=logging.WARNING)
            c = Converter(frd_path, ["vtu"])
            c.run()
            vtu_path = frd_path.replace(".frd", ".vtu")
            print(f"  -> VTU file generated: {vtu_path}")
            split_vtu_file(vtu_path)
        except ImportError:
            print("  -> ccx2paraview not installed. Run: pip install ccx2paraview")
        except Exception as e:
            print(f"  -> ccx2paraview conversion error: {e}")

    return {
        "success": success,
        "elapsed_s": elapsed,
        "frd_path": frd_path if os.path.isfile(frd_path) else None,
        "vtu_path": vtu_path,
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:] if result.stdout else "",
        "stderr": result.stderr[-1000:] if result.stderr else "",
    }


def split_vtu_file(vtu_path: str):
    """
    Splits a CalculiX VTU file (which ccx2paraview has extruded to 3D bricks)
    into:
      1. An extruded skin-only VTU file ('_skin.vtu')
      2. A 2D skin-only surface VTU file ('_skin_2d.vtu')
      3. A 1D beam-only line VTU file ('_beams_1d.vtu')
    This allows applying the Tube filter to the 1D beams in ParaView.
    """
    if not vtu_path or not os.path.isfile(vtu_path):
        return

    try:
        import meshio
        import numpy as np
        from scipy.spatial import KDTree

        # Read the extruded VTU file
        mesh = meshio.read(vtu_path)

        # Paths for output files
        base_dir = os.path.dirname(vtu_path)
        base_name = os.path.splitext(os.path.basename(vtu_path))[0]

        # 1. Output the extruded skin-only mesh (hexahedrons)
        skin_cells_3d = []
        skin_cell_data_3d = {}
        for key in mesh.cell_data:
            skin_cell_data_3d[key] = []

        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "hexahedron":
                skin_cells_3d.append(cell_block)
                for key in mesh.cell_data:
                    skin_cell_data_3d[key].append(mesh.cell_data[key][i])

        if skin_cells_3d:
            skin_mesh_3d = meshio.Mesh(
                points=mesh.points,
                cells=skin_cells_3d,
                point_data=mesh.point_data,
                cell_data=skin_cell_data_3d,
            )
            skin_3d_path = os.path.join(base_dir, f"{base_name}_skin.vtu")
            meshio.write(skin_3d_path, skin_mesh_3d)
            print(f"  -> Generated 3D extruded skin mesh: {skin_3d_path}")

        # 2. Parse the original CalculiX INP file to extract 1D beams and 2D skin elements
        inp_path = os.path.join(base_dir, f"{base_name}.inp")
        if not os.path.isfile(inp_path):
            print(
                f"  -> Warning: CalculiX INP file not found: {inp_path}. Cannot generate 1D beams or 2D skin."
            )
            return

        nodes = {}
        beam_elements = []
        skin_elements = []

        current_mode = None
        element_type = None

        with open(inp_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("**"):
                    continue
                if line_str.startswith("*"):
                    header = line_str.upper()
                    if header == "*NODE" or header.startswith("*NODE,"):
                        current_mode = "node"
                    elif header == "*ELEMENT" or header.startswith("*ELEMENT,"):
                        current_mode = "element"
                        element_type = None
                        parts = header.split(",")
                        for part in parts:
                            if "TYPE=" in part:
                                element_type = part.split("=")[1].strip()
                    else:
                        current_mode = None
                    continue

                if current_mode == "node":
                    parts = line_str.split(",")
                    node_id = int(parts[0].strip())
                    coords = [float(p.strip()) for p in parts[1:4]]
                    nodes[node_id] = coords
                elif current_mode == "element":
                    parts = line_str.split(",")
                    conn = [int(p.strip()) for p in parts[1:] if p.strip()]
                    if element_type in ["B31", "B32"]:
                        beam_elements.append(conn)
                    elif element_type in ["S3", "S4"]:
                        skin_elements.append(conn)

        if not nodes:
            print(f"  -> Warning: No nodes parsed from {inp_path}")
            return

        # Match original nodes to their closest indices in the VTU mesh points
        tree = KDTree(mesh.points)
        node_ids = sorted(nodes.keys())
        node_coords = np.array([nodes[nid] for nid in node_ids])

        distances, indices = tree.query(node_coords)

        # Map original node IDs to their indices (0-based) in our node_ids list
        nid_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Map point data to the original nodes
        point_data = {}
        for key, val in mesh.point_data.items():
            point_data[key] = val[indices]

        # 3. Write 1D Beams (VTU_LINE)
        if beam_elements:
            lines_conn = []
            for elem in beam_elements:
                cell_conn = [nid_to_idx[nid] for nid in elem]
                if len(cell_conn) == 2:
                    lines_conn.append([cell_conn[0], cell_conn[1]])
                elif len(cell_conn) >= 3:
                    lines_conn.append([cell_conn[0], cell_conn[2]])
                    lines_conn.append([cell_conn[2], cell_conn[1]])

            beam_mesh_1d = meshio.Mesh(
                points=node_coords,
                cells=[("line", np.array(lines_conn))],
                point_data=point_data,
            )
            beam_1d_path = os.path.join(base_dir, f"{base_name}_beams_1d.vtu")
            meshio.write(beam_1d_path, beam_mesh_1d)
            print(
                f"  -> Generated 1D beam-only mesh (for Tube filter): {beam_1d_path}"
            )

        # 4. Write 2D Skin (VTU_QUAD or VTU_TRIANGLE)
        if skin_elements:
            quad_conn = []
            tri_conn = []
            for elem in skin_elements:
                cell_conn = [nid_to_idx[nid] for nid in elem]
                if len(cell_conn) == 3:
                    tri_conn.append(cell_conn)
                elif len(cell_conn) == 4:
                    quad_conn.append(cell_conn)

            skin_cells_2d = []
            if quad_conn:
                skin_cells_2d.append(("quad", np.array(quad_conn)))
            if tri_conn:
                skin_cells_2d.append(("triangle", np.array(tri_conn)))

            skin_mesh_2d = meshio.Mesh(
                points=node_coords,
                cells=skin_cells_2d,
                point_data=point_data,
            )
            skin_2d_path = os.path.join(base_dir, f"{base_name}_skin_2d.vtu")
            meshio.write(skin_2d_path, skin_mesh_2d)
            print(f"  -> Generated 2D skin-only surface mesh: {skin_2d_path}")

    except Exception as e:
        print(f"  -> Warning: Could not split/reconstruct VTU files: {e}")


