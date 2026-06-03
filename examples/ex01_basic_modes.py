"""
Example 01 — Basic Branching Modes
===================================
Demonstrates the four fundamental fractal branching modes side by side.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import canopy as cp
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT, exist_ok=True)


def main():
    aero_wing, wing = get_base_wing(bm='wingbox')
    viz = cp.Viz(wing)

    modes = ['monopodial', 'sympodial', 'dichotomous', 'monochasium']
    results = {}

    for mode in modes:
        sub = cp.SubParams(max_depth=3, angles=[30, 35, 40], mode=mode)
        stations = cp.make_diagonal_only_stations(
            n_stations=10, angle=35, length=2.0, sub=sub,
        )
        spec = cp.TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=True,
        )
        gen = cp.TreeGenerator(wing)
        segs = gen.generate(spec)
        st = gen.stats()
        results[f'{mode}\n({st["n"]} segs)'] = segs
        print(f'  {mode:15s}: {st["n"]:4d} segs, L={st["L"]:6.1f}m')

    fig = viz.multi(results)
    fig.savefig(f'{OUT}/ex01_basic_modes.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Saved: {OUT}/ex01_basic_modes.png')


if __name__ == '__main__':
    main()
