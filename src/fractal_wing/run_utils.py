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
        generated_files = {}

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
            generated_files["Skin"] = f"{base_name}_skin.vtu"

        # 2. Parse the original CalculiX INP file to extract 1D beams and 2D skin elements
        inp_path = os.path.join(base_dir, f"{base_name}.inp")
        if not os.path.isfile(inp_path):
            print(
                f"  -> Warning: CalculiX INP file not found: {inp_path}. Cannot generate 1D beams or 2D skin."
            )
            return

        nodes = {}
        beam_elements = []  # stores (elset_name, conn)
        skin_elements = []

        current_mode = None
        element_type = None
        current_elset = None
        beam_sections = {}

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
                        current_elset = None
                        parts = header.split(",")
                        for part in parts:
                            if "TYPE=" in part:
                                element_type = part.split("=")[1].strip()
                            elif "ELSET=" in part:
                                current_elset = part.split("=")[1].strip()
                    elif header.startswith("*BEAM SECTION"):
                        current_mode = "beam_section"
                        current_elset = None
                        parts = header.split(",")
                        for part in parts:
                            if "ELSET=" in part:
                                current_elset = part.split("=")[1].strip()
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
                        beam_elements.append((current_elset, conn))
                    elif element_type in ["S3", "S4"]:
                        skin_elements.append(conn)
                elif current_mode == "beam_section":
                    try:
                        radius = float(line_str.split(",")[0].strip())
                        if current_elset:
                            beam_sections[current_elset] = radius
                    except ValueError:
                        pass
                    current_mode = None

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

        # 3. Write 1D Beams (VTK PolyData with TubeRadius)
        if beam_elements:
            try:
                import vtk
                beam_nodes = set()
                for elset_name, conn in beam_elements:
                    for nid in conn:
                        beam_nodes.add(nid)
                
                beam_node_list = sorted(list(beam_nodes))
                beam_nid_to_idx = {nid: idx for idx, nid in enumerate(beam_node_list)}
                
                pts_vtk = vtk.vtkPoints()
                for nid in beam_node_list:
                    c = nodes[nid]
                    pts_vtk.InsertNextPoint(c[0], c[1], c[2])
                    
                lines_vtk = vtk.vtkCellArray()
                node_radii = {nid: 0.0 for nid in beam_node_list}
                
                for elset_name, conn in beam_elements:
                    r = beam_sections.get(elset_name, 0.0025)
                    for nid in conn:
                        node_radii[nid] = max(node_radii[nid], r)
                    
                    if len(conn) == 2:
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, beam_nid_to_idx[conn[0]])
                        line.GetPointIds().SetId(1, beam_nid_to_idx[conn[1]])
                        lines_vtk.InsertNextCell(line)
                    elif len(conn) >= 3:
                        line1 = vtk.vtkLine()
                        line1.GetPointIds().SetId(0, beam_nid_to_idx[conn[0]])
                        line1.GetPointIds().SetId(1, beam_nid_to_idx[conn[2]])
                        lines_vtk.InsertNextCell(line1)
                        line2 = vtk.vtkLine()
                        line2.GetPointIds().SetId(0, beam_nid_to_idx[conn[2]])
                        line2.GetPointIds().SetId(1, beam_nid_to_idx[conn[1]])
                        lines_vtk.InsertNextCell(line2)

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(pts_vtk)
                polydata.SetLines(lines_vtk)
                
                radius_array = vtk.vtkFloatArray()
                radius_array.SetName("TubeRadius")
                radius_array.SetNumberOfComponents(1)
                for nid in beam_node_list:
                    radius_array.InsertNextValue(node_radii[nid])
                polydata.GetPointData().AddArray(radius_array)
                
                for key, val_array in mesh.point_data.items():
                    arr_vtk = vtk.vtkFloatArray() if val_array.dtype.kind == 'f' else vtk.vtkIntArray()
                    arr_vtk.SetName(key)
                    if len(val_array.shape) > 1:
                        n_comp = val_array.shape[1]
                        arr_vtk.SetNumberOfComponents(n_comp)
                        for nid in beam_node_list:
                            mapped_idx = nid_to_idx[nid]
                            val = val_array[indices[mapped_idx]]
                            arr_vtk.InsertNextTuple(val.tolist())
                    else:
                        arr_vtk.SetNumberOfComponents(1)
                        for nid in beam_node_list:
                            mapped_idx = nid_to_idx[nid]
                            arr_vtk.InsertNextValue(float(val_array[indices[mapped_idx]]))
                    polydata.GetPointData().AddArray(arr_vtk)

                writer = vtk.vtkXMLPolyDataWriter()
                beam_vtp_path = os.path.join(base_dir, f"{base_name}_beams.vtp")
                writer.SetFileName(beam_vtp_path)
                writer.SetInputData(polydata)
                writer.Write()
                print(f"  -> Generated 1D beam PolyData mesh (for Tube filter): {beam_vtp_path}")
                generated_files["Beams"] = f"{base_name}_beams.vtp"
            except ImportError:
                print("  -> Warning: 'vtk' module not installed. Cannot export VTP beams mesh.")

        # 4. Generate combined VTM and cleanup
        if generated_files:
            import xml.etree.ElementTree as ET
            root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
            mbds = ET.SubElement(root, "vtkMultiBlockDataSet")
            for i, (name, fname) in enumerate(generated_files.items()):
                ET.SubElement(mbds, "DataSet", index=str(i), file=fname, name=name)
                
            tree = ET.ElementTree(root)
            vtm_path = os.path.join(base_dir, f"{base_name}_combined.vtm")
            ET.indent(tree, space="  ")
            tree.write(vtm_path, encoding="utf-8", xml_declaration=True)
            print(f"  -> Generated combined MultiBlock Dataset: {vtm_path}")
            
            # Remove redundant mixed VTU file
            if os.path.exists(vtu_path):
                os.remove(vtu_path)

    except Exception as e:
        print(f"  -> Warning: Could not split/reconstruct VTU files: {e}")


