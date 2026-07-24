"""
CAD Export Module

Generates exact B-rep (Boundary Representation) structures for the fractal webs
by performing OpenCASCADE boolean intersections against the AeroShape NURBS skin.
Supports exporting perfect non-conformal matching surfaces to STEP.
"""

from typing import List
from build123d import Face, Plane, Compound

from .structures import Seg


def build_brep_webs(segs: List[Seg], aerowing,
                    as_solid: bool = False, output_step: str = None):
    """
    Convert fractal segments into B-rep Faces/Solids perfectly matching
    the AeroShape wing curvature.

    Parameters
    ----------
    segs : list of Seg
        The generated fractal segments.
    aerowing : aeroshape.geometry.wings.MultiSegmentWing
        The AeroShape wing model providing the OML Solid.
    as_solid : bool
        If True, thickens the webs to 3D Solids. If False, exports 2D Faces (shells).
    output_step : str, optional
        Path to save the generated STEP file.

    Returns
    -------
    tuple
        (build123d.Compound of webs, dict mapping web index to thickness)
    """

    # 1. Obtain the Wing OML Solid from AeroShape
    wing_shape = aerowing.to_occ_shape()

    if not hasattr(wing_shape, "wrapped"):
        from build123d import Shape
        wing_shape = Shape(wing_shape)

    webs = []
    properties = {}

    # Large bounding box Z to guarantee intersection
    Z_INF = 100.0

    for i, seg in enumerate(segs):
        p0, p1 = seg.p0, seg.p1
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = (dx**2 + dy**2)**0.5

        if length < 1e-6:
            continue

        # Create a vertical plane exactly covering the segment length
        # Origin is the midpoint of the segment on Z=0
        mid_x = (p0[0] + p1[0]) / 2.0
        mid_y = (p0[1] + p1[1]) / 2.0

        # X-direction of the plane is along the segment
        dir_x = (dx / length, dy / length, 0.0)
        # Z-direction is normal to the segment (for offsetting if solid)
        dir_z = (-dy / length, dx / length, 0.0)

        workplane = Plane(origin=(mid_x, mid_y, 0.0), x_dir=dir_x, z_dir=dir_z)

        # Make a giant rectangle on this workplane that exceeds wing thickness
        # Rectangle width=length, height=2*Z_INF
        web_face = Face.make_rect(length, 2 * Z_INF, plane=workplane)

        if as_solid:
            # Thicken the face by thickness/2 in both directions
            from build123d import extrude
            thick = seg.thick
            # Extrude forward
            solid1 = extrude(web_face, amount=thick / 2)
            # Extrude backward
            solid2 = extrude(web_face, amount=-thick / 2)

            web_shape = solid1.fuse(solid2)
        else:
            web_shape = web_face

        # Boolean Intersect with Wing OML!
        # This automatically trims the top and bottom to match NURBS curvature
        # perfectly.
        try:
            trimmed_web = web_shape.intersect(wing_shape)
        except Exception:
            trimmed_web = None

        if trimmed_web is None:
            continue

        # build123d intersect can return a single Shape, or a ShapeList (list) if disjoint.
        pieces = trimmed_web if isinstance(trimmed_web, (list, tuple)) else [trimmed_web]
        
        for piece in pieces:
            if piece is None:
                continue
                
            # FIX: Avoid degenerate 3-edged webs at the trailing edge
            if not as_solid and hasattr(piece, 'is_valid') and piece.is_valid and len(piece.edges()) == 3:
                # The web converged to a sharp point at the trailing edge.
                # We pull the outboard tip inwards by 2% of the branch length.
                cut_dist = length * 0.02
                new_length = length - cut_dist
                new_mid_x = mid_x - (cut_dist / 2.0) * dir_x[0]
                new_mid_y = mid_y - (cut_dist / 2.0) * dir_x[1]

                new_workplane = Plane(
                    origin=(new_mid_x, new_mid_y, 0.0), x_dir=dir_x, z_dir=dir_z)
                new_web_face = Face.make_rect(
                    new_length, 2 * Z_INF, plane=new_workplane)
                try:
                    re_trimmed = new_web_face.intersect(wing_shape)
                    re_pieces = re_trimmed if isinstance(re_trimmed, (list, tuple)) else [re_trimmed]
                    
                    for rp in re_pieces:
                        if rp is not None and hasattr(rp, 'is_valid') and rp.is_valid:
                            webs.append(rp)
                            properties[len(webs) - 1] = {"thickness": seg.thick, "level": seg.level, "id": i}
                except Exception:
                    pass
            else:
                if hasattr(piece, 'is_valid') and piece.is_valid:
                    webs.append(piece)
                    properties[len(webs) - 1] = {"thickness": seg.thick, "level": seg.level, "id": i}

    # Combine webs into single CAD compound
    assembly = Compound(webs)

    if output_step:
        from aeroshape.nurbs.export import NurbsExporter
        NurbsExporter.to_step(assembly, output_step)

    return assembly, properties


def export_hollow_skin(aerowing, output_step: str):
    """
    Extracts the hollow aerodynamic skin (Upper and Lower surfaces)
    from the AeroShape wing solid and exports it to a STEP file.
    It ignores the solid root/tip caps.
    """
    from build123d import Compound
    from aeroshape.nurbs.export import NurbsExporter

    wing_solid = aerowing.to_occ_shape()

    # We only want hollow curved skin.
    # Skin surfaces are typically bounded by 4 edges (Root, Tip, LE, TE).
    # Caps (Root/Tip) are usually bounded by 1 or 2 edges depending on the
    # airfoil parameterization.
    skin_faces = [f for f in wing_solid.faces() if len(f.edges()) == 4]

    assembly = Compound(skin_faces)
    NurbsExporter.to_step(assembly, output_step)
    return assembly


def export_solid_wing(aerowing, output_step: str):
    """
    Extracts the full 3D solid wing volume from the AeroShape wing
    and exports it to a STEP file for solid volumetric meshing.
    """
    from aeroshape.nurbs.export import NurbsExporter

    wing_solid = aerowing.to_occ_shape()
    
    if not hasattr(wing_solid, "wrapped"):
        from build123d import Shape
        wing_solid = Shape(wing_solid)

    NurbsExporter.to_step(wing_solid, output_step)
    return wing_solid
