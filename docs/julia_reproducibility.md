# Julia Environmental Isolation & Aerodynamic Solvers Reproducibility Guide

This guide documents the environment settings, package versions, and workarounds implemented to interface the **FLOWPanel.jl** (3D) and **VortexLattice.jl** (2D) Julia packages with the Python framework, ensuring reproducibility across different user setups.

---

## 1. System Requirements & Package Versions

The following versions are checked and pinned in the `Manifest.toml` of the project:

- **Julia Version**: `v1.10.x` LTS (recommended `v1.10.11`)
- **VortexLattice.jl**: `v0.2.3`
- **FLOWPanel.jl**: `v1.2.1`
- **JSON.jl**: `v1.6.0`
- **StaticArrays.jl**: `v1.9.18`
- **WriteVTK.jl**: `v1.21.2`

---

## 2. Dynamic Library Isolation (`LD_LIBRARY_PATH` Issue)

### The Problem

When running Julia from a Python subprocess (especially within a Conda or Pyenv environment), the process inherits the environment variables of the active Python kernel.

Python environments often prepend virtual-env-specific paths to `LD_LIBRARY_PATH` (e.g. `/home/user/.pyenv/versions/.../lib`). When Julia attempts to precompile or load packages that depend on low-level binary wrappers (such as `LibCURL_jll`, `LibSSH2_jll`, `nghttp2_jll`), it links against Python's shared libraries instead of Julia's internal ones. This results in missing symbol mismatches, precompilation failures, and segmentation faults (`SIGSEGV: 11`).

### The Solution (Non-Intrusive)

In the Python aerodynamic solver wrapper (`src/canopy/aero_solver.py`), we explicitly isolate the subprocess environment by copying the parent environment and clearing `LD_LIBRARY_PATH`:

```python
import os
import subprocess

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = ""  # Prevent linking against Python/Conda dependencies

# Run Julia execution
subprocess.run(["julia", "--project", "script.jl"], env=env, check=True)
```

---

## 3. VortexLattice.jl (v0.2.3) Control Point Memory Bug

### The Problem

When executing VortexLattice's VLM grid panel solver, the analysis would randomly crash with:

- `SingularException(0)` (LAPACK factorization failure)
- `SIGSEGV: 11` (segmentation fault inside the linear system solve)

This behavior was non-deterministic because it was driven by **uninitialized heap memory allocation**.

Inside `VortexLattice.jl/src/system.jl`, the `System` constructor initializes control point ratio offsets using the `undef` keyword:

```julia
# VortexLattice.jl/src/system.jl (Original Code)
ratios = [Array{TF}(undef, 2, nc[i], ns[i]) for i = 1:nsurf]
for i = 1:nsurf
    ratios[i] = ratios[i] .+ [0.5; 0.75]
end
```

Because `Array{TF}(undef, ...)` allocates memory without zeroing it out, the arrays contain arbitrary garbage data left over from previous processes. Adding `.+ [0.5; 0.75]` to garbage memory propagates `NaN`, `Inf`, or extreme floating-point garbage to the control points.

When these control points are used to evaluate the influence coefficients of the grid panels, the resulting aerodynamic influence coefficient (AIC) matrix gets populated with `NaN`s, causing the linear solver to collapse.

### The Bypass (Implemented in our Wrapper)

Rather than forcing users to modify their global Julia package directories, we bypass the bug in `src/canopy/solve_vortexlattice.jl` by manually resetting the ratios array to the correct defaults immediately after instantiation:

```julia
# solve_vortexlattice.jl (Bypass logic)
system = System(surfaces)

# Override uninitialized ratios to bypass VortexLattice.jl's heap memory bug
for i in 1:length(system.ratios)
    for c in 1:size(system.ratios[i], 2)
        for s in 1:size(system.ratios[i], 3)
            system.ratios[i][1, c, s] = 0.5
            system.ratios[i][2, c, s] = 0.75
        end
    end
end

# Solve steady analysis safely
ref = Reference(S, c, b, r, V)
steady_analysis!(system, ref, fs; derivatives=false)
```

---

## 4. Replication & Setup Commands

Anyone cloning and using this package should follow these steps to run the pipeline:

### Step 1: Install Julia 1.10

Download and install Julia 1.10 LTS from [julialang.org](https://julialang.org/downloads/). If you use `juliaup`:

```bash
juliaup add 1.10
juliaup default 1.10
```

### Step 2: Instantiate the Julia Environment

Ensure your terminal environment has `LD_LIBRARY_PATH` cleared when instantiating the packages:

```bash
LD_LIBRARY_PATH="" julia --project="." -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### Step 3: Run the Examples

Run Example 11 to verify the steady aerodynamic and structural analysis pipeline:

```bash
python3 examples/ex11_2d_flapping_wing.py
```

This script runs the VLM code, maps the forces, calls CalculiX, and exports `vlm_aero.vtm` and `wing_skin_sim.vtu` to `examples/output_2d_flapping/`.
