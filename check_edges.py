import gmsh

gmsh.initialize()
gmsh.merge("examples/output_fem/organic_shell.step")
gmsh.model.occ.synchronize()

print("Checking webs...")
for dim, tag in gmsh.model.getEntities(2):
    edges = gmsh.model.getBoundary([(dim, tag)], oriented=False)
    if len(edges) != 4:
        print(f"Web {tag} has {len(edges)} edges!")
        
        # Calculate bounding box to see where it is
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        print(f"  Bounds: X=[{xmin:.3f}, {xmax:.3f}], Y=[{ymin:.3f}, {ymax:.3f}], Z=[{zmin:.3f}, {zmax:.3f}]")
        
        for b_dim, b_tag in edges:
            b_xmin, b_ymin, b_zmin, b_xmax, b_ymax, b_zmax = gmsh.model.occ.getBoundingBox(b_dim, b_tag)
            dx = b_xmax - b_xmin
            dy = b_ymax - b_ymin
            dz = b_zmax - b_zmin
            print(f"  Edge {b_tag}: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

gmsh.finalize()
