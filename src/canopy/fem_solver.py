"""
CalculiX FEM Simulation Builder.

Reads a Gmsh-generated mesh .inp file and appends the complete CalculiX
simulation deck: material definitions, shell sections, orientations,
MPC equations, boundary conditions, loads, and analysis step.
"""

from __future__ import annotations

import io
import re
import numpy as np
from typing import Dict, List, Tuple, Set


# Pre-compiled regex for parsing ELSET names (used in multiple parsers)
_RE_ELSET = re.compile(r'ELSET\s*=\s*(\S+)', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────
# T300/Epoxy Unidirectional Carbon Fiber (Engineering Constants)
# ─────────────────────────────────────────────────────────────────
CFRP_T300 = {
    "name": "CFRP_T300",
    "E1": 135e9,   # Pa  (Longitudinal / fiber direction)
    "E2": 10e9,    # Pa  (Transverse)
    "E3": 10e9,    # Pa  (Through-thickness)
    "nu12": 0.28,
    "nu13": 0.28,
    "nu23": 0.40,
    "G12": 5.0e9,  # Pa  (In-plane shear)
    "G13": 5.0e9,  # Pa
    "G23": 3.5e9,  # Pa  (Transverse shear)
    "density": 1600.0,  # kg/m^3
}

FOAM_EPO = {
    "name": "FOAM_EPO",
    "E": 35e6,          # Pa
    "nu": 0.3,
    "density": 35.0,    # kg/m^3
}


def _parse_inp_nodes(inp_path: str) -> Dict[int, Tuple[float, float, float]]:
    """Parse *NODE block from a CalculiX .inp file and return {id: (x, y, z)}.

    Uses numpy bulk parsing for large meshes (typically 3-5x faster than
    line-by-line Python string operations on 800K+ line files).
    """
    # 1. Extract the raw *NODE text block
    node_lines = []
    in_nodes = False
    with open(inp_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith('*NODE'):
                in_nodes = True
                continue
            if in_nodes:
                if stripped.startswith('*'):
                    in_nodes = False
                    # We don't break because there might be multiple *NODE blocks
                    if stripped.upper().startswith('*NODE'):
                        in_nodes = True
                    continue
                if stripped:
                    node_lines.append(stripped.rstrip(','))

    if not node_lines:
        return {}

    # 2. Bulk-parse with numpy for speed
    block = '\n'.join(node_lines)
    data = np.genfromtxt(
        io.StringIO(block),
        delimiter=',',
        usecols=(
            0,
            1,
            2,
            3))

    # Handle single-node edge case (genfromtxt returns 1-D array)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # 3. Build dict from array
    return {int(row[0]): (row[1], row[2], row[3]) for row in data}


def _parse_inp_elsets(inp_path: str) -> Dict[str, List[int]]:
    """Parse *ELSET blocks from a CalculiX .inp file."""
    elsets = {}
    current_set = None
    with open(inp_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('**'):
                continue
            if stripped.upper().startswith('*ELSET'):
                match = _RE_ELSET.search(stripped)
                if match:
                    current_set = match.group(1)
                    elsets[current_set] = []
                continue
            if current_set is not None:
                if stripped.startswith('*'):
                    current_set = None
                    continue
                # Parse comma-separated element IDs
                for token in stripped.rstrip(',').split(','):
                    token = token.strip()
                    if token.isdigit():
                        elsets[current_set].append(int(token))
    return elsets


def _parse_inp_elements(
        inp_path: str) -> Dict[str, List[Tuple[int, List[int]]]]:
    """Parse *ELEMENT blocks. Returns {elset_name: [(elem_id, [n1,n2,n3,n4]), ...]}."""
    elements = {}
    current_set = None
    with open(inp_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('**'):
                continue
            if stripped.upper().startswith('*ELEMENT'):
                match = _RE_ELSET.search(stripped)
                if match:
                    current_set = match.group(1)
                    if current_set not in elements:
                        elements[current_set] = []
                continue
            if current_set is not None:
                if stripped.startswith('*'):
                    current_set = None
                    continue
                parts = stripped.rstrip(',').split(',')
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 3:
                    eid = int(parts[0])
                    conn = [int(p) for p in parts[1:]]
                    elements[current_set].append((eid, conn))
    return elements


def _get_element_nodes_for_elset(elset_name: str, elements: dict,
                                 elsets: dict) -> set:
    """Get all unique node IDs belonging to a named element set."""
    node_ids = set()

    # Check if elements are directly in a surface/element block with this
    # ELSET name
    for eset_name, elem_list in elements.items():
        if eset_name == elset_name:
            for eid, conn in elem_list:
                node_ids.update(conn)

    # Also check the ELSET definitions (which reference element IDs)
    if elset_name in elsets:
        target_eids = set(elsets[elset_name])
        for eset_name, elem_list in elements.items():
            for eid, conn in elem_list:
                if eid in target_eids:
                    node_ids.update(conn)

    return node_ids


def parse_mesh_for_mapping(
        mesh_inp: str, elset_name: str = 'WingSkin') -> Tuple[Dict[int, Tuple[float, float, float]], Set[int]]:
    """Parse mesh file to get all nodes and the set of skin node IDs."""
    nodes = _parse_inp_nodes(mesh_inp)
    elements = _parse_inp_elements(mesh_inp)
    elsets = _parse_inp_elsets(mesh_inp)
    skin_nodes = _get_element_nodes_for_elset(elset_name, elements, elsets)
    return nodes, skin_nodes


def build_ccx_deck(
    mesh_inp: str,
    web_properties: Dict[int, dict],
    segments: list = None,
    skin_thickness: float = 0.003,
    web_thickness_override: float = None,
    tip_load: float = -1000.0,
    point_loads: List[Dict[str, float]] = None,
    material: dict = None,
    core_material: dict = None,
    output_path: str = None,
    solver: str = None,
    binary_output: bool = False,
    mapped_aero_forces: Dict[int, np.ndarray] = None,
    skin_elset_name: str = 'WingSkin',
    beam_section: str = 'CIRC',
    beam_wall_thickness_ratio: float = 0.2,
    nlgeom: bool = False,
    isotropic: bool = None,
) -> str:
    """
    Build a complete CalculiX simulation deck from a Gmsh mesh .inp file.

    Parameters
    ----------
    mesh_inp : str
        Path to the Gmsh-generated .inp mesh file (already has *NODE, *ELEMENT, *ELSET).
    web_properties : dict
        Per-web property dict from build_brep_webs(). Keys are web indices (0, 1, ...),
        values are dicts with 'thickness', 'level', 'id'.
    segments : list of Seg, optional
        The fractal segments, used to compute per-web fiber orientation vectors.
    skin_thickness : float
        Shell thickness for the wing skin (default: 3mm).
    web_thickness_override : float, optional
        If set, overrides all web thicknesses with this value.
    tip_load : float
        Fallback concentrated force if point_loads is not provided.
    point_loads : list of dict, optional
        List of dictionaries defining point loads at the tip.
        Format: [{'x_frac': 0.5, 'load': -1000.0}], where x_frac is the chordwise position (0=LE, 1=TE).
    material : dict, optional
        Material properties dict. Defaults to CFRP_T300.
    output_path : str, optional
        Path for the output simulation .inp file. Defaults to replacing the
        mesh file's extension with '_sim.inp'.
    solver : str, optional
        Equation solver keyword (e.g. ``'PARDISO'``, ``'PASTIX'``).  If None,
        CalculiX uses its default solver (SPOOLES).  Only effective when the
        ``ccx`` binary has been compiled with the corresponding library.
    binary_output : bool
        If True, use ``*NODE OUTPUT`` / ``*ELEMENT OUTPUT`` for compact
        mixed binary/ASCII ``.frd`` files (~40%% smaller, faster I/O).
        If False (default), use ``*NODE FILE`` / ``*EL FILE`` for fully
        ASCII output compatible with ``ccx2paraview`` and other
        post-processing tools.  Set to True only when VTU conversion
        is not needed.

    Returns
    -------
    str
        Path to the generated simulation .inp file.
    """
    if material is None:
        material = CFRP_T300
    if core_material is None:
        core_material = FOAM_EPO

    if output_path is None:
        output_path = mesh_inp.replace('.inp', '_sim.inp')

    # ── Parse the mesh ──
    nodes = _parse_inp_nodes(mesh_inp)
    elements = _parse_inp_elements(mesh_inp)
    elsets = _parse_inp_elsets(mesh_inp)

    # ── Identify coordinate-based node sets ──
    ys = [c[1] for c in nodes.values()]
    y_min, y_max = min(ys), max(ys)
    tol_y = 1e-3  # 1mm tolerance for coordinate matching

    root_nodes = sorted(
        [nid for nid, (x, y, z) in nodes.items() if abs(y - y_min) < tol_y])
    tip_nodes = sorted(
        [nid for nid, (x, y, z) in nodes.items() if abs(y - y_max) < tol_y])

    # Process point loads at the tip
    active_loads = []
    if point_loads is not None:
        for p in point_loads:
            frac = p.get('x_frac', 0.5)
            val = p.get('load', tip_load)
            active_loads.append({'x_frac': frac, 'load': val})
    else:
        active_loads.append({'x_frac': 0.5, 'load': tip_load})

    applied_loads = []
    if tip_nodes:
        tip_xs = [nodes[n][0] for n in tip_nodes]
        min_x, max_x = min(tip_xs), max(tip_xs)
        dx = max_x - min_x if max_x > min_x else 1.0

        for p in active_loads:
            target_x = min_x + p['x_frac'] * dx
            node_id = min(tip_nodes, key=lambda n: abs(nodes[n][0] - target_x))
            applied_loads.append((node_id, p['load']))
    else:
        # Fallback if no tip nodes found
        fallback_node = max(nodes.keys())
        for p in active_loads:
            applied_loads.append((fallback_node, p['load']))

    # ── Detect Web and Beam Elsets and Boundary Nodes ──
    web_elset_names = sorted([k for k in set(list(elsets.keys()) + list(elements.keys())) if k.startswith('Web_')])
    beam_elset_names = sorted([k for k in set(list(elsets.keys()) + list(elements.keys())) if k.startswith('BEAMS_T_')])
    has_core = 'Core' in elsets or 'Core' in elements

    # Flatten elements dict for quick lookup by ID
    flat_elements = {}
    for elem_list in elements.values():
        for eid, enodes in elem_list:
            flat_elements[eid] = enodes

    # Topologically find all web edges
    web_edges = {}
    for wname in web_elset_names:
        for eid in elsets.get(wname, []):
            if eid in flat_elements:
                enodes = flat_elements[eid]
                n_n = len(enodes)
                for i in range(n_n):
                    edge = tuple(sorted((enodes[i], enodes[(i + 1) % n_n])))
                    web_edges[edge] = web_edges.get(edge, 0) + 1

    # Nodes on edges shared by only 1 web element are on the outer boundary
    boundary_nodes = set()
    for edge, count in web_edges.items():
        if count == 1:
            boundary_nodes.add(edge[0])
            boundary_nodes.add(edge[1])

    # For 2D beams, EVERY beam node must be tied to the skin
    for bname in beam_elset_names:
        eids = set(elsets.get(bname, []))
        if bname in elements:
            eids.update([eid for eid, _ in elements[bname]])
        for eid in eids:
            if eid in flat_elements:
                boundary_nodes.update(flat_elements[eid])

    # Exclude root nodes to prevent SPC/MPC conflicts
    boundary_nodes.difference_update(root_nodes)

    # ── Match Web Boundary Nodes to Skin ──
    skin_nodes = _get_element_nodes_for_elset(skin_elset_name, elements, elsets)
    skin_nodes_list = list(skin_nodes)
    skin_coords = np.array([nodes[n] for n in skin_nodes_list if n in nodes])

    mpc_equations = []
    if len(skin_coords) > 0 and boundary_nodes:
        from scipy.spatial import cKDTree
        skin_tree = cKDTree(skin_coords)

        w_nodes_list = list(boundary_nodes)
        w_coords = np.array([nodes[n] for n in w_nodes_list if n in nodes])

        if len(w_coords) == len(w_nodes_list):
            # Query the 10 closest nodes to allow finding an unused one
            _, indices = skin_tree.query(w_coords, k=10)
            
            used_skin_nodes = set()
            for i in range(len(w_nodes_list)):
                web_n = w_nodes_list[i]
                tied = False
                # Iterate through the k=10 closest skin nodes
                for idx in indices[i]:
                    skin_n = skin_nodes_list[idx]
                    if skin_n not in used_skin_nodes:
                        used_skin_nodes.add(skin_n)
                        mpc_equations.append((web_n, skin_n))
                        tied = True
                        break
                
                # If we couldn't find an unused one in the top 10, fallback to finding the absolute closest unused
                if not tied:
                    dists = np.linalg.norm(skin_coords - w_coords[i], axis=1)
                    sorted_all = np.argsort(dists)
                    for idx in sorted_all:
                        skin_n = skin_nodes_list[idx]
                        if skin_n not in used_skin_nodes:
                            used_skin_nodes.add(skin_n)
                            mpc_equations.append((web_n, skin_n))
                            break

    # ── Compute per-web orientation vectors ──
    web_orientations = {}
    if segments is not None:
        for web_idx, props in web_properties.items():
            seg_id = props.get('id', web_idx)
            if seg_id < len(segments):
                seg = segments[seg_id]
                direction = seg.p1 - seg.p0
                length = np.linalg.norm(direction)
                if length > 1e-12:
                    direction = direction / length
                    web_orientations[web_idx] = (
                        float(
                            direction[0]), float(
                            direction[1]), 0.0)

    # ── Build the simulation deck ──
    lines = []

    # Read the original mesh content
    with open(mesh_inp, 'r') as f:
        mesh_content = f.read()
    if beam_section.upper() == 'PIPE':
        mesh_content = mesh_content.replace('TYPE=B32,', 'TYPE=B32R,').replace('type=B32,', 'type=B32R,')
    lines.append(mesh_content.rstrip())
    lines.append('')

    # ── Node Sets ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** NODE SETS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')

    # Root nodes
    lines.append('*NSET, NSET=NSET_ROOT')
    for k in range(0, len(root_nodes), 10):
        chunk = root_nodes[k:k + 10]
        lines.append(', '.join(str(n) for n in chunk) + ',')

    # Tip nodes
    lines.append('*NSET, NSET=NSET_TIP')
    for k in range(0, len(tip_nodes), 10):
        chunk = tip_nodes[k:k + 10]
        lines.append(', '.join(str(n) for n in chunk) + ',')

    # Tip load nodes
    lines.append('*NSET, NSET=NSET_TIP_LOAD')
    for n_id, _ in applied_loads:
        lines.append(f'{n_id},')

    # ── Material Definition ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** MATERIAL')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')
    lines.append(f'*MATERIAL, NAME={material["name"]}')
    is_iso = False
    if isotropic is True:
        is_iso = True
    elif isotropic is None:
        if "E" in material and "nu" in material:
            is_iso = True
        elif all(k in material for k in ["E1", "E2", "E3", "nu12"]):
            # Auto-detect if all E are equal
            if abs(material["E1"] - material["E2"]) < 1e-6 and abs(material["E2"] - material["E3"]) < 1e-6:
                is_iso = True

    if is_iso:
        E_val = material.get("E", material.get("E1"))
        nu_val = material.get("nu", material.get("nu12"))
        lines.append('*ELASTIC')
        lines.append(f'{E_val}, {nu_val}')
    else:
        lines.append('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
        lines.append(f'{material["E1"]}, {material["E2"]}, {material["E3"]}, '
                     f'{material["nu12"]}, {material["nu13"]}, {material["nu23"]}, '
                     f'{material["G12"]}, {material["G13"]},')
        lines.append(f'{material["G23"]}')
    lines.append('*DENSITY')
    lines.append(f'{material["density"]}')

    if has_core:
        lines.append('**')
        lines.append(f'*MATERIAL, NAME={core_material["name"]}')
        lines.append('*ELASTIC')
        lines.append(f'{core_material.get("E", 35e6)}, {core_material.get("nu", 0.3)}')
        lines.append('*DENSITY')
        lines.append(f'{core_material.get("density", 35.0)}')

    # ── Orientations ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** ORIENTATIONS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')

    lines.append('*ORIENTATION, NAME=OR_SKIN, SYSTEM=RECTANGULAR')
    lines.append('0., 1., 0.,  0., 0., 1.')

    for web_idx in range(len(web_elset_names)):
        if web_idx in web_orientations:
            dx, dy, dz = web_orientations[web_idx]
        else:
            dx, dy, dz = 0., 1., 0.
        lines.append(
            f'*ORIENTATION, NAME=OR_Web_{web_idx}, SYSTEM=RECTANGULAR')
        lines.append(f'{dx}, {dy}, {dz},  0., 0., 1.')

    # ── Shell Sections ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** SHELL SECTIONS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')

    lines.append(f'*SHELL SECTION, ELSET={skin_elset_name}, MATERIAL={material["name"]}, '
                 f'ORIENTATION=OR_SKIN')
    lines.append(f'{skin_thickness}')

    for web_idx in range(len(web_elset_names)):
        if web_thickness_override is not None:
            thick = web_thickness_override
        elif web_idx in web_properties:
            thick = web_properties[web_idx].get('thickness', 0.005)
        else:
            thick = 0.005
        lines.append(f'*SHELL SECTION, ELSET=Web_{web_idx}, MATERIAL={material["name"]}, '
                     f'ORIENTATION=OR_Web_{web_idx}')
        lines.append(f'{thick}')

    # ── Solid Sections ──
    if has_core:
        lines.append('**')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('** SOLID SECTIONS')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('**')
        lines.append(f'*SOLID SECTION, ELSET=Core, MATERIAL={core_material["name"]}')

    # ── Beam Sections ──
    if beam_elset_names:
        lines.append('**')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('** BEAM SECTIONS')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('**')
        for bname in beam_elset_names:
            try:
                # Name is like BEAMS_T_0_003000
                thick_str = bname.replace('BEAMS_T_', '')
                # Find where the decimal point should be based on length, or just replace last underscore
                # Actually, in meshing.py we did f"BEAMS_T_{thick:.6f}".replace('.', '_')
                # So if it's 0_003000, we can replace the first underscore with a dot
                parts = thick_str.split('_')
                if len(parts) >= 2:
                    thick = float(parts[0] + '.' + parts[1])
                else:
                    thick = 0.005
            except (ValueError, IndexError):
                thick = 0.005

            if beam_section.upper() == 'PIPE':
                r_outer = thick / 2.0
                t_wall = beam_wall_thickness_ratio * r_outer
                lines.append(f'*BEAM SECTION, ELSET={bname}, MATERIAL={material["name"]}, SECTION={beam_section.upper()}')
                lines.append(f'{r_outer:.8e}, {t_wall:.8e}')
            else:
                lines.append(f'*BEAM SECTION, ELSET={bname}, MATERIAL={material["name"]}, SECTION=CIRC')
                lines.append(f'{thick / 2.0:.8e}')
            lines.append('0., 0., -1.')  # Direction cosines for beam orientation (z-axis downwards)

    # ── MPC Equations (Web to Skin Contact) ──
    if mpc_equations:
        lines.append('**')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('** KINEMATIC COUPLING (MPC EQUATIONS)')
        lines.append('** ═══════════════════════════════════════════')
        lines.append('** Tying web boundary nodes to nearest skin nodes')
        for web_n, skin_n in mpc_equations:
            for dof in range(1, 4):  # ONLY tie translational DOFs (1, 2, 3)!
                lines.append('*EQUATION')
                lines.append('2')
                lines.append(f'{web_n}, {dof}, 1.0, {skin_n}, {dof}, -1.0')

    # ── Boundary Conditions ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** BOUNDARY CONDITIONS AND LOADS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')

    if nlgeom:
        lines.append('*STEP, NLGEOM')
    else:
        lines.append('*STEP')

    if solver:
        lines.append(f'*STATIC, SOLVER={solver.upper()}')
    else:
        lines.append('*STATIC')
    lines.append('1., 1.')
    lines.append('**')

    lines.append('*BOUNDARY')
    lines.append('NSET_ROOT, 1, 6')
    lines.append('**')

    if mapped_aero_forces is not None:
        lines.append('**')
        lines.append('** MAPPED AERODYNAMIC LOADS')
        lines.append('**')
        for n_id, force in mapped_aero_forces.items():
            lines.append('*CLOAD')
            if abs(force[0]) > 1e-6:
                lines.append(f'{n_id}, 1, {force[0]:.8e}'.upper())
            if abs(force[1]) > 1e-6:
                lines.append(f'{n_id}, 2, {force[1]:.8e}'.upper())
            if abs(force[2]) > 1e-6:
                lines.append(f'{n_id}, 3, {force[2]:.8e}'.upper())
        lines.append('**')
    else:
        for n_id, load_val in applied_loads:
            lines.append('*CLOAD')
            lines.append(f'{n_id}, 3, {load_val}')
            lines.append('**')

    if binary_output:
        lines.append('*NODE OUTPUT')
        lines.append('U')
        lines.append('*ELEMENT OUTPUT')
        lines.append('S')
    else:
        lines.append('*NODE FILE')
        lines.append('U')
        lines.append('*EL FILE')
        lines.append('S')
    lines.append('**')
    lines.append('*END STEP')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'  -> CalculiX simulation deck written: {output_path}')
    print(f'     Root nodes (clamped): {len(root_nodes)}')
    print(f'     Tip nodes: {len(tip_nodes)}')
    if mapped_aero_forces is not None:
        print(f'     Mapped aero loads applied: {len(mapped_aero_forces)} nodes')
    else:
        for n_id, load_val in applied_loads:
            print(f'     Tip load node: {n_id} ({load_val} N in Z)')
    print(f'     Web-to-Skin MPCs: {len(mpc_equations)}')
    print(f'     Web orientations defined: {len(web_orientations)}')

    return output_path
