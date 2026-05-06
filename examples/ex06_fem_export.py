"""
Example 06 — FEM Structural Graph Export
==========================================
Demonstrates how to extract a structural graph (Nodes, Elements, Properties)
from the generated fractal segments. The resulting graph can be used to build
a Finite Element Method (FEM) shell model, with no gaps and explicit thickness
for each shell element.
"""

import os
import json
import fractal_wing as fw
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output_fem')
os.makedirs(OUT, exist_ok=True)

def main():
    print('='*65)
    print('  Example 06 — FEM Structural Graph Export')
    print('='*65)

    # 1. Generate a small test structure
    wing = fw.Wing(bm='wingbox')
    
    # We use a simple structure to keep the output readable
    stations = fw.make_mixed_stations(
        n_stations=5,
        diag_angle=45,
        chord_angle=90,
        diag_length=2.0,
        chord_length=1.5,
        diag_sub=fw.SubParams(mode='sympodial', max_depth=1, min_length=0.1),
        chord_sub=fw.SubParams(mode='dichotomous', max_depth=1, min_length=0.1)
    )

    spec = fw.TrunkSpec(
        chord_frac=0.5,
        span_cov=1.0,
        thick=0.005,  # 5mm trunk thickness
        stations=stations,
        allow_crossing=False
    )

    gen = fw.TreeGenerator(wing)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'Generated {st["n"]} raw segments (total length {st["L"]}m).')

    # 2. Extract the structural graph
    print('\nExtracting FEM structural graph (merging nodes)...')
    graph = fw.build_structural_graph(segs, tol=1e-6)
    
    # Visualize the graph
    viz_fw = fw.Viz(wing)
    fig = viz_fw.view3d(segs)
    plt.show()

    n_nodes = len(graph['nodes'])
    n_elems = len(graph['elements'])
    print(f'  -> Found {n_nodes} unique nodes.')
    print(f'  -> Found {n_elems} shell mid-surface elements.')
    
    if n_elems != st['n']:
        print(f'  Warning: Element count ({n_elems}) != Segment count ({st["n"]})')

    # 3. Save to JSON
    json_path = os.path.join(OUT, 'fractal_fem_mesh.json')
    fw.export_to_json(graph, json_path)
    print(f'\nExported structural graph to: {json_path}')

    # Let's print a small sample of the JSON to prove it works
    print('\n--- JSON Data Sample ---')
    with open(json_path, 'r') as f:
        lines = f.readlines()
        print(''.join(lines[:20]) + '    ...\n  ],\n  "elements": [\n    ...')

    print('='*65)


if __name__ == '__main__':
    main()
