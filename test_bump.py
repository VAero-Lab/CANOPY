import gmsh

gmsh.initialize()
gmsh.model.add("test_bump")

p1 = gmsh.model.geo.addPoint(0, 0, 0)
p2 = gmsh.model.geo.addPoint(1, 0, 0)
l1 = gmsh.model.geo.addLine(p1, p2)
gmsh.model.geo.synchronize()

gmsh.model.mesh.setTransfiniteCurve(l1, 20, "Beta", 1.5)
gmsh.model.mesh.generate(1)

nodes, coords, _ = gmsh.model.mesh.getNodes()
x_coords = sorted(coords[0::3])
for x in x_coords:
    print(f"{x:.4f}")
gmsh.finalize()
