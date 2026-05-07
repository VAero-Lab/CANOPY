# Fractal Wing Generator (`fractal-wing`)

A professional, modularized Python package for the generation of fractal tree-like structures for aerodynamic wing design. The package provides high-level versatility for structural optimization by implementing non-uniform station distribution laws, per-depth branching mode switching, per-branch thickness control, multi-trunk support, and automated self-intersection avoidance.

![Organic Zoned Design](docs/images/organic_zoned.png)

## Features

- **Adaptive Branching Modes**: Support for per-recursion-depth mode switching (e.g., transition from *sympodial* to *monopodial* deeper in the tree).
- **Non-Uniform Station Distribution**: Provides geometric, cosine, power-law, bimodal, and custom spacing strategies.
- **Granular Branch Control**: Per-branch thickness control and tunable recursion termination thresholds.
- **Multi-Trunk Support**: Generate complex designs with multiple independent trunks while maintaining global crossing constraints.
- **AeroShape NURBS Integration**: Directly query $C^2$-continuous aerodynamic surfaces to restrict fractal growth perfectly within real CAD boundaries.
- **Gmsh Structural Meshing**: Generates high-fidelity structured Quad (S4R) shell meshes mapped precisely to the NURBS wing skin using OpenCASCADE B-Rep boundaries. Natively exports to CalculiX (`.inp`) and Gmsh (`.msh`).

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
```

## Examples

The `examples/` directory contains demonstration scripts covering core capabilities:
- `ex01_basic_modes.py`: Demonstrates the 4 main branching modes independently.
- `ex02_mixed_mode.py`: Shows combination of structural modes.
- `ex03_non_uniform_spacing.py`: Compares different distribution laws (uniform vs. non-uniform).
- `ex04_organic_structures.py`: Synthesizes deep tree-like structures with graded parameters.
- `ex05_crossing_control.py`: Demonstrates the self-intersection pruning logic (`allow_crossing`).
- `ex06_fem_export.py`: Extracts exact mathematical topology via OpenCASCADE and generates a Gmsh structured mesh for CalculiX.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
