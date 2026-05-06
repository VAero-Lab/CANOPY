# Fractal Wing Generator (`fractal-wing`)

A professional, modularized Python package for the generation of fractal tree-like structures for aerodynamic wing design. The package provides high-level versatility for structural optimization by implementing non-uniform station distribution laws, per-depth branching mode switching, per-branch thickness control, multi-trunk support, and automated self-intersection avoidance.

![Organic Zoned Design](docs/images/organic_zoned.png)

## Features

- **Adaptive Branching Modes**: Support for per-recursion-depth mode switching (e.g., transition from *sympodial* to *monopodial* deeper in the tree).
- **Non-Uniform Station Distribution**: Provides geometric, cosine, power-law, bimodal, and custom spacing strategies.
- **Granular Branch Control**: Per-branch thickness control and tunable recursion termination thresholds.
- **Multi-Trunk Support**: Generate complex designs with multiple independent trunks while maintaining global crossing constraints.
- **Spanwise-zoned & Graded Parameters**: Built-in factory helpers to enable root-to-tip variation and zoned behavior.
- **Self-intersection Control**: Automated segment pruning to avoid structural intersections.
- **FEM Structural Integration**: Export clean node-element-property graph topology for immediate loading into Finite Element Method solvers as 2D shells.

## Installation

The package can be installed in editable mode using `pip`:

```bash
git clone <repository_url>
cd fractal_structures_wing
pip install -e .
```

## Quick Start

```python
import fractal_wing as fw

# 1. Define the structural domain
wing = fw.Wing(bm='full_wing')

# 2. Configure spanwise-zoned branching behavior
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

# 3. Create station distribution (e.g., geometric progression)
stations = fw.make_zoned_stations(
    n_stations=14, 
    zones=zones, 
    spacing='geometric'
)

# 4. Generate the fractal structure
spec = fw.TrunkSpec(chord_frac=0.5, stations=stations, allow_crossing=False)
gen = fw.TreeGenerator(wing)
segments = gen.generate(spec)

# 5. Export to FEM structural graph (Nodes, Shell Elements, Properties)
graph = fw.build_structural_graph(segments, tol=1e-6)
fw.export_to_json(graph, "fractal_mesh.json")

# 6. Visualize
viz = fw.Viz(wing)
viz.view3d(segments, "Zoned Fractal Wing")
```

## Examples

The `examples/` directory contains demonstration scripts covering core capabilities:
- `ex01_basic_modes.py`: Demonstrates the 4 main branching modes independently.
- `ex02_mixed_mode.py`: Shows combination of structural modes.
- `ex03_non_uniform_spacing.py`: Compares different distribution laws (uniform vs. non-uniform).
- `ex04_organic_structures.py`: Synthesizes deep tree-like structures with graded parameters.
- `ex05_crossing_control.py`: Demonstrates the self-intersection pruning logic (`allow_crossing`).
- `ex06_fem_export.py`: Extracts the gapless mathematical topology and exports it as a JSON structural graph.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
