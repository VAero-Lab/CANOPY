import gmsh
gmsh.initialize()
gmsh.model.occ.importShapes('../output_fem/fractal_mesh_shell.step')
gmsh.model.occ.synchronize()

surfaces = gmsh.model.getEntities(2)
for dim, tag in surfaces:
    bnd = gmsh.model.getBoundary([(dim, tag)])
    print(f"Surface {tag} has {len(bnd)} bounding edges.")

gmsh.finalize()
