import gmsh
import math

gmsh.initialize()
gmsh.model.occ.importShapes('../output_fem/fractal_mesh_shell.step')
gmsh.model.occ.synchronize()

target_size = 0.025
surfaces = gmsh.model.getEntities(2)

vertical_edges = []
for dim, tag in gmsh.model.getEntities(1):
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
    if math.hypot(xmax - xmin, ymax - ymin) < 1e-4:
        vertical_edges.append(zmax - zmin)

nz = max(1, int(round(max(vertical_edges) / target_size))) if vertical_edges else 10

# Process surface by surface
for dim, tag in surfaces:
    bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    
    horizontal_tags = []
    vertical_tags = []
    lengths = []
    
    for c_dim, c_tag in bnd:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(c_dim, c_tag)
        if math.hypot(xmax - xmin, ymax - ymin) < 1e-4:
            vertical_tags.append(c_tag)
        else:
            horizontal_tags.append(c_tag)
            lengths.append(gmsh.model.occ.getMass(c_dim, c_tag))
            
    # Set vertical edges
    for v_tag in vertical_tags:
        gmsh.model.mesh.setTransfiniteCurve(v_tag, nz + 1)
        
    # Set horizontal edges to be IDENTICAL
    if horizontal_tags:
        avg_len = sum(lengths) / len(lengths)
        nx = max(1, int(round(avg_len / target_size)))
        for h_tag in horizontal_tags:
            gmsh.model.mesh.setTransfiniteCurve(h_tag, nx + 1)
            
    gmsh.model.mesh.setTransfiniteSurface(tag)
    gmsh.model.mesh.setRecombine(dim, tag)

gmsh.model.mesh.generate(2)
gmsh.write('../output_fem/test_gmsh_fix.msh')
gmsh.finalize()
