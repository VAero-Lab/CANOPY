"""
Structured Meshing Module for Fractal Webs.

Generates structured quadrilateral (Quad4) shell meshes for fractal webs
by sampling the exact NURBS boundaries from the AeroWingAdapter.

To guarantee perfect conformality (node sharing) at intersections between webs,
the vertical discretization (number of elements along Z) must be constant 
across all webs. This is calculated automatically based on the target element size 
and the maximum thickness of the wing.
"""

import numpy as np
from scipy.spatial import cKDTree

from .structures import Seg
from .aeroshape_adapter import AeroWingAdapter

import os
import math
import numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


class GmshMesher:
    """
    Generates structured Quad meshes for fractal webs using Gmsh and OpenCASCADE.
    
    This mesher loads a B-Rep STEP file (typically exported via cad_export.build_brep_webs)
    and perfectly maps a structured Quadrilateral (S4R) mesh onto the true NURBS curvature
    of the wing skin boundaries.
    
    Parameters
    ----------
    target_elem_size : float
        The desired length of each element edge (in meters).
    """

    def __init__(self, target_elem_size: float = 0.05):
        if not GMSH_AVAILABLE:
            raise ImportError("Gmsh is not installed. Please run `pip install gmsh` to use the GmshMesher.")
            
        self.target_size = target_elem_size

    def mesh(self, step_filepath: str, output_inp: str, nz: int = None):
        """
        Generate global structured quad mesh from STEP file and export to CalculiX .inp.
        
        Parameters
        ----------
        step_filepath : str
            Path to the input STEP file containing independent 2D shell webs.
        output_inp : str
            Path to write the resulting CalculiX .inp mesh file.
        nz : int, optional
            Number of elements along the vertical (Z) axis. If None, it calculates
            this dynamically based on the tallest vertical edge in the model.
            
        Returns
        -------
        dict
            Statistics about the generated mesh.
        """
        if not os.path.exists(step_filepath):
            raise FileNotFoundError(f"Input STEP file not found: {step_filepath}")
            
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # Suppress verbose output
        
        # We must use OCC to import the STEP so we can query bounding boxes and properties
        gmsh.model.occ.importShapes(step_filepath)
        gmsh.model.occ.synchronize()
        
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            gmsh.finalize()
            return {"nodes": 0, "elements": 0}
            
        # 1. First Pass: Find global maximum vertical edge to enforce a constant nz
        # A constant nz across all independent webs guarantees that nodes *could* align 
        # at intersections for non-conformal tied contacts.
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
            
        # 2. Second Pass: Apply structured Transfinite meshing rules per surface!
        # This is critical: Opposite edges of a 4-sided transfinite surface MUST have 
        # the exact same number of nodes. Since the top and bottom wing skin boundaries 
        # might have slightly different arc lengths, we must calculate the node count
        # per surface and apply it to BOTH of its horizontal boundaries identically.
        for dim, tag in surfaces:
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False)
            
            h_tags = []
            v_tags = []
            h_lengths = []
            
            for c_dim, c_tag in bnd:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(c_dim, c_tag)
                if math.hypot(xmax - xmin, ymax - ymin) < 1e-4:
                    v_tags.append(c_tag)
                else:
                    h_tags.append(c_tag)
                    h_lengths.append(gmsh.model.occ.getMass(c_dim, c_tag))
                    
            # Set vertical edges
            for v_tag in v_tags:
                gmsh.model.mesh.setTransfiniteCurve(v_tag, nz + 1)
                
            # Set horizontal edges to be identical
            if h_tags:
                avg_len = sum(h_lengths) / len(h_lengths)
                nx = max(1, int(round(avg_len / self.target_size)))
                for h_tag in h_tags:
                    gmsh.model.mesh.setTransfiniteCurve(h_tag, nx + 1)
                    
            # Apply structured surface rules (Quad Recombination)
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(dim, tag)
            
        # Generate 2D surface mesh
        gmsh.model.mesh.generate(2)
        
        # Save as CalculiX .inp format
        gmsh.write(output_inp)
        
        # Also save as Gmsh .msh format for easy visualization
        output_msh = output_inp.replace('.inp', '.msh')
        if output_msh != output_inp:
            gmsh.write(output_msh)
        
        # Extract stats before closing
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
        
        n_nodes = len(node_tags)
        # Quads are type 3 in Gmsh
        n_elems = 0
        for i, etype in enumerate(elem_types):
            if etype == 3: # Quad
                n_elems += len(elem_tags[i])
                
        gmsh.finalize()
        
        return {
            "n_nodes": n_nodes,
            "n_elems": n_elems,
            "nz": nz,
            "target_size": self.target_size,
            "inp_path": output_inp
        }

