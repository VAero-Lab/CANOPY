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

    def __init__(self, target_elem_size: float = 0.05, skin_elem_size: float = None):
        if not GMSH_AVAILABLE:
            raise ImportError("Gmsh is not installed. Please run `pip install gmsh` to use the GmshMesher.")

        self.target_size = target_elem_size
        self.skin_size = skin_elem_size if skin_elem_size is not None else target_elem_size

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
        gmsh.option.setNumber("Mesh.Algorithm", 8)    # Frontal-Delaunay for Quads (robust for fragments)
        gmsh.option.setNumber("Mesh.RecombineAll", 1) # Force quad recombination globally
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # First-order elements

        # Enforce global mesh size (replaces transfinite curves for fragmented topology)
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
        # Intersect only the webs against each other. The skin remains independent.
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

        # Per-web Physical Groups (Web_0, Web_1, ...) for individual shell sections
        for idx, frag_list in enumerate(web_to_fragments):
            if frag_list:
                pg = gmsh.model.addPhysicalGroup(2, frag_list)
                gmsh.model.setPhysicalName(2, pg, f"Web_{idx}")

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            gmsh.finalize()
            return {"n_nodes": 0, "n_elems": 0}

        # 4. First Pass: Find global maximum vertical edge to enforce a constant nz
        vertical_edges = []
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
            dz = zmax - zmin
            dxy = math.hypot(xmax - xmin, ymax - ymin)
            if dxy < 1e-4:
                vertical_edges.append((tag, dz))

        if nz is None:
            max_h = max([dz for _, dz in vertical_edges]) if vertical_edges else 0.1
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
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(c_dim, c_tag_abs)
                if math.hypot(xmax - xmin, ymax - ymin) < 1e-4:
                    v_tags.append(c_tag_abs)
                else:
                    h_tags.append(c_tag_abs)
                    h_lengths.append(gmsh.model.occ.getMass(c_dim, c_tag_abs))

            # Rule A: Web Fragments (Exactly 2 vertical edges)
            # Because we only fragment webs against webs (vertical cuts), they remain 4-sided
            if len(v_tags) == 2:
                for v_tag in v_tags:
                    gmsh.model.mesh.setTransfiniteCurve(v_tag, nz + 1)
                if h_tags:
                    avg_len = sum(h_lengths) / len(h_lengths)
                    nx = max(1, int(round(avg_len / self.target_size)))
                    for h_tag in h_tags:
                        gmsh.model.mesh.setTransfiniteCurve(h_tag, nx + 1)

            # Rule B: Wing Skin (0 vertical edges, 4 boundary edges forming a loop)
            elif len(v_tags) == 0 and len(bnd) == 4:
                # bnd is ordered along the loop. 0 and 2 are opposite, 1 and 3 are opposite.
                edges = [abs(t) for _, t in bnd]
                l0 = gmsh.model.occ.getMass(1, edges[0])
                l1 = gmsh.model.occ.getMass(1, edges[1])
                l2 = gmsh.model.occ.getMass(1, edges[2])
                l3 = gmsh.model.occ.getMass(1, edges[3])

                nx02 = max(1, int(round(((l0 + l2) / 2.0) / self.skin_size)))
                nx13 = max(1, int(round(((l1 + l3) / 2.0) / self.skin_size)))

                gmsh.model.mesh.setTransfiniteCurve(edges[0], nx02 + 1)
                gmsh.model.mesh.setTransfiniteCurve(edges[2], nx02 + 1)
                gmsh.model.mesh.setTransfiniteCurve(edges[1], nx13 + 1)
                gmsh.model.mesh.setTransfiniteCurve(edges[3], nx13 + 1)

            try:
                gmsh.model.mesh.setTransfiniteSurface(tag)
            except Exception:
                pass

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
            if etype == 3: # Quad
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
