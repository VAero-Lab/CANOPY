# Fractal Wing Generator (`fractal-wing`)

A professional, modularized Python package for the generation of fractal tree-like structures for aerodynamic wing design. The package provides high-level versatility for structural optimization by implementing non-uniform station distribution laws, per-depth branching mode switching, per-branch thickness control, multi-trunk support, and automated self-intersection avoidance.

![Organic Zoned Design](docs/images/organic_zoned.png)

## Features

- **Adaptive Branching Modes**: Support for per-recursion-depth mode switching (e.g., transition from *sympodial* to *monopodial* deeper in the tree).
- **Non-Uniform Station Distribution**: Provides geometric, cosine, power-law, bimodal, and custom spacing strategies.
- **Granular Branch Control**: Per-branch thickness control and tunable recursion termination thresholds.
- **Multi-Trunk Support**: Generate complex designs with multiple independent trunks while maintaining global crossing constraints.
- **AeroShape NURBS Integration**: Directly query $C^2$-continuous aerodynamic surfaces to restrict fractal growth perfectly within real CAD boundaries.
- **Gmsh Structural Meshing**: Generates high-fidelity structured Quad (S4) shell meshes mapped precisely to the NURBS wing skin using OpenCASCADE B-Rep boundaries. Natively exports to CalculiX (`.inp`).
- **CalculiX & ParaView Integration**: Fully automated FEM pipeline that builds simulation decks, defines fractal-hierarchy-aware `*TIE` contacts, applies boundary conditions, solves the structural response, and converts results to `.vtu` format for ParaView visualization.

## Installation

The package can be installed in editable mode using `pip`:

```bash
git clone <repository_url>
cd fractal_structures_wing
pip install -e .
pip install gmsh  # Required for FEM meshing
```

## Quick Start

```python
import fractal_wing as fw
from aeroshape.geometry.wings import MultiSegmentWing

# 1. Provide an AeroShape MultiSegmentWing
aero_wing = MultiSegmentWing(...)

# 2. Wrap it with the Adapter
wing = fw.AeroWingAdapter(aero_wing)

# 3. Configure spanwise-zoned branching behavior
zones = [
    {
        'eta_start': 0.0, 'eta_end': 0.4,
        'diag_sub': fw.SubParams(mode='sympodial', max_depth=3),
        'chord_sub': fw.SubParams(mode='dichotomous', max_depth=2),
    },
    {
        'eta_start': 0.4, 'eta_end': 1.0,
        'diag_sub': fw.SubParams(mode='monopodial', max_depth=2),
        'chord_sub': fw.SubParams(mode='sympodial', max_depth=1),
    }
]

# 4. Create station distribution and generate segments
stations = fw.make_zoned_stations(n_stations=14, zones=zones, spacing='geometric')
spec = fw.TrunkSpec(chord_frac=0.5, stations=stations, allow_crossing=False)
gen = fw.TreeGenerator(wing)
segments = gen.generate(spec)

# 5. Export CAD B-Rep Shells to STEP
step_path = "fractal_mesh_shell.step"
assembly, props = fw.build_brep_webs(segments, aero_wing, as_solid=False, output_step=step_path)

# 6. Extract Structured FEM Mesh directly from STEP using Gmsh
mesher = fw.GmshMesher(target_elem_size=0.025)
stats = mesher.mesh(step_path, "fractal_mesh.inp")

# 7. Build Simulation Deck & Run CalculiX
sim_path = fw.build_ccx_deck(
    mesh_inp="fractal_mesh.inp",
    web_properties=props,
    segments=segments,
    point_loads=[{'x_frac': 0.0, 'load': -500.0}, {'x_frac': 1.0, 'load': -500.0}]
)

# 8. Run solver with automatic parallelism & convert results to VTU
result = fw.run_ccx(sim_path, convert_vtu=True)
print(f"Solver finished in {result['elapsed_s']:.1f}s")
```

## Performance Optimization

The FEM pipeline has been optimized to handle complex organic fractal meshes efficiently:
- **Parallel Assembly**: `run_ccx` automatically detects your CPU cores and sets `CCX_NPROC_STIFFNESS` and `OMP_NUM_THREADS` to parallelize stiffness matrix assembly and results calculation.
- **Fast I/O**: `build_ccx_deck` defaults to a mixed binary/ASCII `.frd` format (`*NODE OUTPUT`), reducing result file sizes by ~40% and accelerating disk I/O. Set `binary_output=False` when using `ccx2paraview`.
- **Fast Parsing**: The Python-based `.inp` parser is vectorized via NumPy, allowing million-line mesh files to be processed in seconds.
- **Multi-threaded Meshing**: `GmshMesher` configures the OpenCASCADE kernel and Frontal-Delaunay algorithms to utilize all available CPU threads.

## Examples

The `examples/` directory contains demonstration scripts covering core capabilities:
- `ex01_basic_modes.py`: Demonstrates the 4 main branching modes independently.
- `ex02_mixed_mode.py`: Shows combination of structural modes.
- `ex03_non_uniform_spacing.py`: Compares different distribution laws (uniform vs. non-uniform).
- `ex04_organic_structures.py`: Synthesizes deep tree-like structures with graded parameters.
- `ex05_crossing_control.py`: Demonstrates the self-intersection pruning logic (`allow_crossing`).
- `ex06_fem_export.py`: Extracts exact mathematical topology via OpenCASCADE and generates a Gmsh structured mesh for CalculiX, executing the full simulation pipeline.
- `ex07_aeroshape_integration.py`: Demonstrates CAD boolean operations with AeroShape NURBS models.
- `ex08_fem_organic.py`: End-to-end FEM simulation for a dense, spanwise-zoned organic fractal structure with multiple distributed point loads.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
