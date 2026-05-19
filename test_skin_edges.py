import gmsh

gmsh.initialize()
gmsh.merge("test_open_te.step")
gmsh.model.occ.synchronize()

print(f"Number of surfaces: {len(gmsh.model.getEntities(2))}")
for dim, tag in gmsh.model.getEntities(2):
    bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    print(f"Surface {tag} has {len(bnd)} boundary edges.")
    for b_dim, b_tag in bnd:
        mass = gmsh.model.occ.getMass(b_dim, b_tag)
        print(f"  Edge {b_tag}: mass={mass:.3f}")
gmsh.finalize()
