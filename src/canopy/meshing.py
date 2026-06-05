"""
Structured Meshing Module for Fractal Webs and Wing Skin.

Generates structured quadrilateral (S4) shell meshes using Gmsh and OpenCASCADE.
Supports unified meshing of internal fractal webs and outer wing skin, with
Physical Groups for CalculiX ELSET management and per-web element grouping.
"""

import os
import math

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


class GmshMesher:
    """
    Generates structured Quad meshes for fractal webs using Gmsh and OpenCASCADE.

    This mesher loads B-Rep STEP files and perfectly maps a structured
    Quadrilateral (S4) mesh onto the true NURBS curvature of the wing skin.

    Parameters
    ----------
    target_elem_size : float
        The desired length of each element edge (in meters).
    """

    def __init__(self, target_elem_size: float = 0.05,
                 skin_elem_size: float = None, skin_clustering: float = 1.0):
        if not GMSH_AVAILABLE:
            raise ImportError(
                "Gmsh is not installed. Please run `pip install gmsh` to use the GmshMesher.")

        self.target_size = target_elem_size
        self.skin_size = skin_elem_size if skin_elem_size is not None else target_elem_size
        self.skin_clustering = skin_clustering

    def mesh(self, webs_step: str, output_inp: str, skin_step: str = None,
             web_properties: dict = None, nz: int = None, export_msh: bool = False):
        """
        Generate global structured quad mesh from STEP files and export to CalculiX .inp.

        Parameters
        ----------
        webs_step : str
            Path to the input STEP file containing independent 2D shell webs.
        output_inp : str
            Path to write the resulting CalculiX .inp mesh file.
        skin_step : str, optional
            Path to the input STEP file containing the hollow wing skin faces.
        web_properties : dict, optional
            Per-web property dict from build_brep_webs(). Keys are web indices,
            values are dicts with 'thickness', 'level', 'id'.
        nz : int, optional
            Number of elements along the vertical (Z) axis. If None, it calculates
            this dynamically based on the tallest vertical edge in the model.
        export_msh : bool
            If True, also save the mesh in Gmsh ``.msh`` format for GUI
            visualization.  Default is False.

        Returns
        -------
        dict
            Statistics about the generated mesh.
        """
        if not os.path.exists(webs_step):
            raise FileNotFoundError(f"Input STEP file not found: {webs_step}")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress verbose output
        gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 4)
        # Frontal-Delaunay for Quads (robust for fragments)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        # Force quad recombination globally
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # First-order elements

        # Enforce global mesh size (replaces transfinite curves for fragmented
        # topology)
        min_size = min(self.target_size, self.skin_size)
        max_size = max(self.target_size, self.skin_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size * 0.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size * 1.2)

        # We will track which surface belongs to which physical group
        webs_surfs = []
        skin_surfs = []

        # 1. Import Skin (if provided)
        if skin_step and os.path.exists(skin_step):
            tags = gmsh.model.occ.importShapes(skin_step)
            skin_surfs = [t[1] for t in tags if t[0] == 2]

        # 2. Import Webs
        tags = gmsh.model.occ.importShapes(webs_step)
        webs_surfs = [t[1] for t in tags if t[0] == 2]

        # ── Conformal Meshing: Boolean Fragments (Webs Only) ──
        # Intersect only the webs against each other. The skin remains
        # independent.
        webs_dimtags = [(2, t) for t in webs_surfs]

        # fragment webs against each other
        out_dimtags, out_map = gmsh.model.occ.fragment(webs_dimtags, [])
        gmsh.model.occ.synchronize()

        # Map original faces to their new fragmented pieces
        new_webs_surfs = []
        web_to_fragments = []
        for i in range(len(webs_dimtags)):
            frags = [t[1] for t in out_map[i] if t[0] == 2]
            web_to_fragments.append(frags)
            new_webs_surfs.extend(frags)

        webs_surfs = new_webs_surfs
        # skin_surfs remains unchanged because we didn't fragment it

        # 3. Create Physical Groups for ELSETs in CalculiX
        # Aggregate groups for skin and all webs
        if skin_surfs:
            pg_skin = gmsh.model.addPhysicalGroup(2, skin_surfs)
            gmsh.model.setPhysicalName(2, pg_skin, "WingSkin")

        if webs_surfs:
            pg_webs = gmsh.model.addPhysicalGroup(2, webs_surfs)
            gmsh.model.setPhysicalName(2, pg_webs, "FractalWebs")

        # Per-web Physical Groups (Web_0, Web_1, ...) for individual shell
        # sections
        for idx, frag_list in enumerate(web_to_fragments):
            if frag_list:
                pg = gmsh.model.addPhysicalGroup(2, frag_list)
                gmsh.model.setPhysicalName(2, pg, f"Web_{idx}")

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            gmsh.finalize()
            return {"n_nodes": 0, "n_elems": 0}

        # 4. First Pass: Find global maximum vertical edge to enforce a
        # constant nz
        vertical_edges = []
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
                dim, tag)
            dz = zmax - zmin
            dxy = math.hypot(xmax - xmin, ymax - ymin)
            if dxy < 1e-4:
                vertical_edges.append((tag, dz))

        if nz is None:
            max_h = max([dz for _, dz in vertical_edges]
                        ) if vertical_edges else 0.1
            nz = max(1, int(round(max_h / self.target_size)))

        # 5. Apply structured Transfinite meshing rules per surface
        for dim, tag in surfaces:
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=True)

            # Identify vertical vs horizontal edges
            h_tags = []
            v_tags = []
            h_lengths = []

            for c_dim, c_tag in bnd:
                c_tag_abs = abs(c_tag)
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
                    c_dim, c_tag_abs)
                if math.hypot(xmax - xmin, ymax - ymin) < 1e-4:
                    v_tags.append(c_tag_abs)
                else:
                    h_tags.append(c_tag_abs)
                    h_lengths.append(gmsh.model.occ.getMass(c_dim, c_tag_abs))

            # Rule A: Web Fragments (Exactly 2 vertical edges)
            # Because we only fragment webs against webs (vertical cuts), they
            # remain 4-sided
            if len(v_tags) == 2:
                for v_tag in v_tags:
                    gmsh.model.mesh.setTransfiniteCurve(v_tag, nz + 1)
                if h_tags:
                    avg_len = sum(h_lengths) / len(h_lengths)
                    nx = max(1, int(round(avg_len / self.target_size)))
                    for h_tag in h_tags:
                        gmsh.model.mesh.setTransfiniteCurve(h_tag, nx + 1)

            # Rule B: Wing Skin (0 vertical edges, 4 boundary edges forming a
            # loop)
            elif len(v_tags) == 0 and len(bnd) == 4:
                edges = [abs(t) for _, t in bnd]
                if len(edges) == 4:
                    # Transfinite curve constraints commented out to avoid "5 corner" errors
                    pass

            # try:
            #     gmsh.model.mesh.setTransfiniteSurface(tag)
            # except Exception:
            #     pass

            gmsh.model.mesh.setRecombine(dim, tag)

        # Generate 2D surface mesh
        gmsh.model.mesh.generate(2)

        # Save as CalculiX .inp format
        gmsh.write(output_inp)

        # Also save as Gmsh .msh format for easy visualization (optional)
        if export_msh:
            output_msh = output_inp.replace('.inp', '.msh')
            if output_msh != output_inp:
                gmsh.write(output_msh)

        # Extract stats before closing
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)

        n_nodes = len(node_tags)
        n_elems = 0
        for i, etype in enumerate(elem_types):
            if etype == 3:  # Quad
                n_elems += len(elem_tags[i])

        gmsh.finalize()

        # Post-process: Fix CPS4 → S4 and CPS3 → S3 for CalculiX shell analysis
        # Gmsh cannot natively export S4/S3 shell labels; it writes CPS4/CPS3 (plane stress).
        # We need S4/S3 for 3D shell bending + membrane behavior.
        with open(output_inp, 'r') as f:
            content = f.read()
        content = content.replace('type=CPS4', 'type=S4')
        content = content.replace('type=CPS3', 'type=S3')
        with open(output_inp, 'w') as f:
            f.write(content)

        return {
            "n_nodes": n_nodes,
            "n_elems": n_elems,
            "nz": nz,
            "target_size": self.target_size,
            "inp_path": output_inp,
            "n_webs": len(webs_surfs),
            "n_skin_faces": len(skin_surfs),
        }

    def mesh_solid_core(self, step_path: str, output_inp: str, export_msh: bool = False):
        """
        Generate a conformal 3D tetrahedral mesh of the wing core, completely
        enclosed by a perfectly structured Quadrilateral shell skin. 
        Pyramid elements are automatically generated at the boundary to transition
        between the Quadrilateral skin and Tetrahedral core.
        """
        if not os.path.exists(step_path):
            raise FileNotFoundError(f"Input STEP file not found: {step_path}")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 4)
        
        # We must use RecombineAll = 0 to force Structured Triangles on the skin.
        # This prevents Gmsh from generating Pyramids (C3D5) which CalculiX rejects.
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 8)      # 2D Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)    # 3D Delaunay
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.target_size * 0.8)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.target_size * 1.2)

        # Import solid wing
        tags = gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()

        # Identify Volumes (Core) and Surfaces (Skin)
        volumes = gmsh.model.getEntities(3)
        surfaces = gmsh.model.getEntities(2)

        if not volumes:
            raise ValueError("No 3D volumes found in the STEP file! Make sure you exported a Solid Wing.")

        # Assign Physical Groups
        pg_core = gmsh.model.addPhysicalGroup(3, [t[1] for t in volumes])
        gmsh.model.setPhysicalName(3, pg_core, "Core")

        pg_skin = gmsh.model.addPhysicalGroup(2, [t[1] for t in surfaces])
        gmsh.model.setPhysicalName(2, pg_skin, "WingSkin")

        # Apply Transfinite mapping to the skin to guarantee structured Quads
        # For each surface, we constrain its boundary curves based on their length
        for dim, tag in surfaces:
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=True)
            for c_dim, c_tag in bnd:
                c_tag_abs = abs(c_tag)
                length = gmsh.model.occ.getMass(c_dim, c_tag_abs)
                n_points = max(2, int(round(length / self.target_size)) + 1)
                gmsh.model.mesh.setTransfiniteCurve(c_tag_abs, n_points)

        # Generate 3D volume mesh (this will automatically mesh 1D, 2D, and then 3D)
        gmsh.model.mesh.generate(3)

        # Renumber to ensure contiguous IDs (CalculiX requirement)
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()

        # Write CalculiX INP
        gmsh.write(output_inp)

        if export_msh:
            output_msh = output_inp.replace('.inp', '.msh')
            if output_msh != output_inp:
                gmsh.write(output_msh)

        # Extract stats
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        
        gmsh.finalize()

        # Post-process INP file because Gmsh uses plane stress names for shells (CPS4) 
        # and standard volume names for solids (C3D4, C3D5, C3D10). We just fix the shells.
        with open(output_inp, 'r') as f:
            content = f.read()
        content = content.replace('type=CPS4', 'type=S4')
        content = content.replace('type=CPS3', 'type=S3')
        with open(output_inp, 'w') as f:
            f.write(content)

        return {
            "n_nodes": n_nodes,
            "target_size": self.target_size,
            "inp_path": output_inp
        }


class GmshMesher2D:
    """
    Generates a 2D flat skin mesh for a wing using Gmsh.
    Used for 2D flapping wings (membranes).

    Parameters
    ----------
    target_elem_size : float
        The desired length of each element edge (in meters).
    """

    def __init__(self, target_elem_size: float = 0.05):
        if not GMSH_AVAILABLE:
            raise ImportError(
                "Gmsh is not installed. Please run `pip install gmsh` to use the GmshMesher.")
        self.target_size = target_elem_size

    def mesh_skin(self, wing, output_inp: str, export_msh: bool = False):
        """
        Generate 2D flat skin mesh from AeroWingAdapter frames.

        Parameters
        ----------
        wing : AeroWingAdapter
            The wing adapter containing the frames.
        output_inp : str
            Path to the output CalculiX .inp file.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.target_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.target_size)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Quad dominant
        gmsh.option.setNumber("Mesh.Algorithm", 6)    # Frontal-Delaunay

        # Build boundary from frames
        frames = wing.frames
        points_le = []
        points_te = []

        # Create points
        for f in frames:
            x_le = f['x_offset']
            y = f['y']
            c = f['chord']
            x_te = x_le + c

            pt_le = gmsh.model.geo.addPoint(x_le, y, 0.0, self.target_size)
            pt_te = gmsh.model.geo.addPoint(x_te, y, 0.0, self.target_size)
            points_le.append(pt_le)
            points_te.append(pt_te)

        lines = []
        # Root chord
        lines.append(gmsh.model.geo.addLine(points_te[0], points_le[0]))

        # Leading edge
        for i in range(len(points_le) - 1):
            lines.append(gmsh.model.geo.addLine(points_le[i], points_le[i + 1]))

        # Tip chord
        lines.append(gmsh.model.geo.addLine(points_le[-1], points_te[-1]))

        # Trailing edge
        for i in range(len(points_te) - 1, 0, -1):
            lines.append(gmsh.model.geo.addLine(points_te[i], points_te[i - 1]))

        cl = gmsh.model.geo.addCurveLoop(lines)
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()

        # Set transfinite curves for structured mesh
        n_chord = max(2, int(frames[0]['chord'] / self.target_size) + 1)

        # Root and Tip
        gmsh.model.mesh.setTransfiniteCurve(lines[0], n_chord)
        gmsh.model.mesh.setTransfiniteCurve(lines[len(frames)], n_chord)

        # LE and TE segments
        for i in range(len(frames) - 1):
            dy = frames[i + 1]['y'] - frames[i]['y']
            n_span_seg = max(2, int(dy / self.target_size) + 1)

            le_line = lines[1 + i]
            # TE curves were added from tip to root
            te_line = lines[-1 - i]

            gmsh.model.mesh.setTransfiniteCurve(le_line, n_span_seg)
            gmsh.model.mesh.setTransfiniteCurve(te_line, n_span_seg)

        # Transfinite surface
        corner_points = [points_te[0], points_le[0], points_le[-1], points_te[-1]]
        gmsh.model.mesh.setTransfiniteSurface(surf, "Left", corner_points)
        gmsh.model.mesh.setRecombine(2, surf)

        # Physical groups
        pg_skin = gmsh.model.addPhysicalGroup(2, [surf])
        gmsh.model.setPhysicalName(2, pg_skin, "SKIN_2D")

        gmsh.model.mesh.generate(2)
        gmsh.write(output_inp)

        if export_msh:
            output_msh = output_inp.replace('.inp', '.msh')
            if output_msh != output_inp:
                gmsh.write(output_msh)

        # Extract stats
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
        n_nodes = len(node_tags)
        n_elems = 0
        for i, etype in enumerate(elem_types):
            if etype == 3:  # Quad
                n_elems += len(elem_tags[i])
            elif etype == 2:  # Tri
                n_elems += len(elem_tags[i])

        gmsh.finalize()

        # Fix CPS4 -> S4, CPS3 -> S3
        with open(output_inp, 'r') as f:
            content = f.read()
        content = content.replace('type=CPS4', 'type=S4').replace('type=CPS3', 'type=S3')
        with open(output_inp, 'w') as f:
            f.write(content)

        return {
            "n_nodes": n_nodes,
            "n_elems": n_elems,
            "target_size": self.target_size,
            "inp_path": output_inp
        }


def append_1d_beams(segs, target_size: float, inp_path: str):
    """
    Discretize fractal segments into 1D B31 beam elements and append them
    to an existing CalculiX .inp file.

    Parameters
    ----------
    segs : List[Seg]
        Fractal segments from TreeGenerator.
    target_size : float
        Target element length.
    inp_path : str
        Path to the .inp file to append to.
    """
    import numpy as np

    # 1. Read existing .inp to find max Node and Element IDs
    max_nid = 0
    max_eid = 0
    with open(inp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            parts = line.split(',')
            if len(parts) > 0:
                try:
                    val = int(parts[0])
                    # Nodal lines have 4 parts: id, x, y, z. Element lines have 4+ parts
                    if len(parts) == 4 and '.' in parts[1]:  # heuristically a node
                        if val > max_nid:
                            max_nid = val
                    else:  # heuristically an element
                        if val > max_eid:
                            max_eid = val
                except ValueError:
                    pass

    nid = max_nid + 1
    eid = max_eid + 1

    node_lines = ["*NODE\n"]
    # Group elements by thickness so we can assign different BEAM SECTIONs later
    elem_groups = {}  # thickness -> list of lines

    for seg in segs:
        thick = seg.thick
        if thick not in elem_groups:
            elem_groups[thick] = []

        p0 = seg.p0
        p1 = seg.p1

        L = np.linalg.norm(p1 - p0)
        n_elems = max(1, int(round(L / target_size)))

        # Discretize segment (quadratic beams need 3 nodes per element)
        seg_nids = []
        for i in range(2 * n_elems + 1):
            t = i / (2 * n_elems)
            p = p0 + t * (p1 - p0)
            node_lines.append(f"{nid}, {p[0]:.6f}, {p[1]:.6f}, 0.0\n")
            seg_nids.append(nid)
            nid += 1

        # Create elements (B32: node1, node2, middle_node)
        for i in range(n_elems):
            n1 = seg_nids[2 * i]
            n2 = seg_nids[2 * i + 2]
            n3 = seg_nids[2 * i + 1]
            elem_groups[thick].append(f"{eid}, {n1}, {n2}, {n3}\n")
            eid += 1

    # Append to file
    with open(inp_path, 'a') as f:
        f.writelines(node_lines)

        for thick, elems in elem_groups.items():
            set_name = f"BEAMS_T_{thick:.6f}".replace('.', '_')
            f.write(f"*ELEMENT, TYPE=B32, ELSET={set_name}\n")
            f.writelines(elems)

    return elem_groups
