import re

with open("examples/output_fem/organic_mesh_sim.inp", "r") as f:
    lines = f.readlines()

in_eq = False
self_mpcs = 0
for i, line in enumerate(lines):
    if line.startswith("*EQUATION"):
        in_eq = True
        continue
    if line.startswith("*") and in_eq:
        in_eq = False
    
    if in_eq:
        # *EQUATION
        # 2
        # node1, dof, 1.0, node2, dof, -1.0
        if len(line.split(",")) > 3:
            parts = line.split(",")
            n1 = parts[0].strip()
            n2 = parts[3].strip()
            if n1 == n2:
                self_mpcs += 1

print(f"Found {self_mpcs} self-MPCs (node connected to itself)!")
