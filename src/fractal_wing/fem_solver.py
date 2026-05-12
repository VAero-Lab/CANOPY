"""
CalculiX FEM Simulation Builder.

Reads a Gmsh-generated mesh .inp file and appends the complete CalculiX
simulation deck: material definitions, shell sections, orientations,
TIE contacts, boundary conditions, loads, and analysis step.
"""

from __future__ import annotations

import io
import os
import re
import numpy as np
from typing import Dict, List, Tuple


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
                    break
                if stripped:
                    node_lines.append(stripped.rstrip(','))

    if not node_lines:
        return {}

    # 2. Bulk-parse with numpy for speed
    block = '\n'.join(node_lines)
    data = np.genfromtxt(io.StringIO(block), delimiter=',', usecols=(0, 1, 2, 3))

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


def _parse_inp_elements(inp_path: str) -> Dict[str, List[Tuple[int, List[int]]]]:
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
                if len(parts) >= 5:
                    eid = int(parts[0])
                    conn = [int(p) for p in parts[1:5]]
                    elements[current_set].append((eid, conn))
    return elements


def _get_element_nodes_for_elset(elset_name: str, elements: dict, 
                                  elsets: dict) -> set:
    """Get all unique node IDs belonging to a named element set."""
    node_ids = set()
    
    # Check if elements are directly in a surface/element block with this ELSET name
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


def build_ccx_deck(
    mesh_inp: str,
    web_properties: Dict[int, dict],
    segments: list = None,
    skin_thickness: float = 0.003,
    web_thickness_override: float = None,
    tip_load: float = -1000.0,
    point_loads: List[Dict[str, float]] = None,
    material: dict = None,
    output_path: str = None,
    solver: str = None,
    binary_output: bool = False,
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
    
    root_nodes = sorted([nid for nid, (x, y, z) in nodes.items() if abs(y - y_min) < tol_y])
    tip_nodes = sorted([nid for nid, (x, y, z) in nodes.items() if abs(y - y_max) < tol_y])
    
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
    
    # ── Detect web-to-web TIE pairs ──
    # We use the fractal segment hierarchy to find connected webs
    web_elset_names = sorted([k for k in elsets if k.startswith('Web_')])
    web_node_sets = {}
    for wname in web_elset_names:
        web_node_sets[wname] = _get_element_nodes_for_elset(wname, elements, elsets)
    
    tie_pairs = []
    if segments is not None:
        seg_to_web = {}
        for web_idx, props in web_properties.items():
            seg_id = props.get('id', web_idx)
            seg_to_web[seg_id] = f'Web_{web_idx}'
            
        for i, sa in enumerate(segments):
            for j, sb in enumerate(segments):
                if i != j:
                    # Parent-child connection: parent's end == child's start
                    if np.linalg.norm(sa.p1 - sb.p0) < 1e-6:
                        name_a = seg_to_web.get(i)
                        name_b = seg_to_web.get(j)
                        if name_a and name_b and name_a in web_elset_names and name_b in web_elset_names:
                            # parent is master (name_a), child is slave (name_b)
                            tie_pairs.append((name_a, name_b))

    # ── Detect skin-adjacent nodes for each web ──
    skin_nodes = _get_element_nodes_for_elset('WingSkin', elements, elsets)
    skin_coords = np.array([nodes[n] for n in skin_nodes if n in nodes])
    
    web_skin_nsets = {}
    if len(skin_coords) > 0:
        from scipy.spatial import cKDTree
        skin_tree = cKDTree(skin_coords)
        
        for wname in web_elset_names:
            w_nodes = list(web_node_sets[wname])
            if not w_nodes: continue
            
            # Remove root nodes to avoid MPC/SPC conflict on slave nodes
            w_nodes = [n for n in w_nodes if n not in root_nodes]
            if not w_nodes: continue
            
            w_coords = np.array([nodes[n] for n in w_nodes])
            dists, _ = skin_tree.query(w_coords, k=1)
            
            # Nodes within 5cm of skin are "skin edge" nodes
            edge_nodes = [w_nodes[idx] for idx, d in enumerate(dists) if d < 0.05]
            if edge_nodes:
                web_skin_nsets[wname] = edge_nodes

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
                    web_orientations[web_idx] = (float(direction[0]), float(direction[1]), 0.0)
    
    # ── Build the simulation deck ──
    lines = []
    
    # Read the original mesh content
    with open(mesh_inp, 'r') as f:
        mesh_content = f.read()
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
        chunk = root_nodes[k:k+10]
        lines.append(', '.join(str(n) for n in chunk) + ',')
    
    # Tip nodes
    lines.append('*NSET, NSET=NSET_TIP')
    for k in range(0, len(tip_nodes), 10):
        chunk = tip_nodes[k:k+10]
        lines.append(', '.join(str(n) for n in chunk) + ',')
    
    # Tip load nodes
    lines.append('*NSET, NSET=NSET_TIP_LOAD')
    for n_id, _ in applied_loads:
        lines.append(f'{n_id},')
    
    # Web skin edge nodes
    for wname, nset in web_skin_nsets.items():
        lines.append(f'*NSET, NSET=NSET_{wname}_SkinEdge')
        for k in range(0, len(nset), 10):
            chunk = nset[k:k+10]
            lines.append(', '.join(str(n) for n in chunk) + ',')
    
    # ── Material Definition ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** MATERIAL')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')
    lines.append(f'*MATERIAL, NAME={material["name"]}')
    lines.append('*ELASTIC, TYPE=ENGINEERING CONSTANTS')
    lines.append(f'{material["E1"]}, {material["E2"]}, {material["E3"]}, '
                 f'{material["nu12"]}, {material["nu13"]}, {material["nu23"]}, '
                 f'{material["G12"]}, {material["G13"]},')
    lines.append(f'{material["G23"]}')
    lines.append('*DENSITY')
    lines.append(f'{material["density"]}')
    
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
        lines.append(f'*ORIENTATION, NAME=OR_Web_{web_idx}, SYSTEM=RECTANGULAR')
        lines.append(f'{dx}, {dy}, {dz},  0., 0., 1.')
    
    # ── Shell Sections ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** SHELL SECTIONS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')
    
    lines.append(f'*SHELL SECTION, ELSET=WingSkin, MATERIAL={material["name"]}, '
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
    
    # ── Surface Definitions for TIE ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** SURFACES AND TIE CONTACTS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')
    
    lines.append('*SURFACE, NAME=SURF_SKIN, TYPE=ELEMENT')
    lines.append('WingSkin, SPOS')
    
    for wname in web_elset_names:
        lines.append(f'*SURFACE, NAME=SURF_{wname}, TYPE=ELEMENT')
        lines.append(f'{wname}, SPOS')
        
    for wname in web_skin_nsets.keys():
        lines.append(f'*SURFACE, NAME=SURF_{wname}_SkinEdge, TYPE=NODE')
        lines.append(f'NSET_{wname}_SkinEdge')
    
    # Skin-to-Webs TIE
    lines.append('**')
    lines.append('** Skin ↔ Webs glue contact')
    for wname in web_skin_nsets.keys():
        lines.append(f'*TIE, NAME=SkinBond_{wname}, POSITION TOLERANCE=0.05, ADJUST=YES')
        lines.append(f'SURF_{wname}_SkinEdge, SURF_SKIN')
    
    # Web-to-Web TIEs
    lines.append('**')
    lines.append('** Web ↔ Web glue contact')
    for pair_idx, (name_a, name_b) in enumerate(tie_pairs):
        # We need to make sure we don't include clamped nodes in the slave surface.
        # However, Web-to-Web TIE is surface-to-surface. 
        # A simpler way is to just define it normally, since only Web_0 is clamped 
        # and Web_0 is the master (name_a) in all its relationships. 
        # So name_b (slave) does not contain clamped nodes.
        lines.append(f'*TIE, NAME=WebBond_{pair_idx}, POSITION TOLERANCE=0.02, ADJUST=YES')
        lines.append(f'SURF_{name_a}, SURF_{name_b}')
    
    # ── Boundary Conditions ──
    lines.append('**')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('** BOUNDARY CONDITIONS AND LOADS')
    lines.append('** ═══════════════════════════════════════════')
    lines.append('**')
    
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
    for n_id, load_val in applied_loads:
        print(f'     Tip load node: {n_id} ({load_val} N in Z)')
    print(f'     Web-to-Web TIE pairs: {len(tie_pairs)}')
    print(f'     Webs tied to skin: {len(web_skin_nsets)}')
    print(f'     Web orientations defined: {len(web_orientations)}')
    
    return output_path
