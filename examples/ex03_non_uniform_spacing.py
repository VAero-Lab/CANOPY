"""
Example 03 — Non-uniform Station Spacing
==========================================
Demonstrates how different distribution laws affect the fractal structure.

This is a key feature: during optimization the optimizer can produce
arbitrary station positions, but these standard distributions serve as
good initialisation strategies.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fractal_wing as fw
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT, exist_ok=True)


def main():
    _, wing = get_base_wing(bm='wingbox')
    viz = fw.Viz(wing)

    n_st = 12

    # ── Compare all distribution laws ───────────────────────────
    print('  Comparing distribution laws...')

    distributions = {
        'uniform': {},
        'geometric\n(ratio=2.0, dense root)': {'ratio': 2.0},
        'geometric\n(ratio=0.5, dense tip)': {'ratio': 0.5},
        'cosine\n(dense both ends)': {},
        'power\n(exponent=2.5)': {'exponent': 2.5},
        'bimodal\n(weight=2.5)': {'weight': 2.5},
    }

    results = {}
    for label, kwargs in distributions.items():
        method = label.split('\n')[0]
        stations = fw.make_mixed_stations(
            n_stations=n_st,
            spacing=method,
            spacing_kwargs=kwargs,
        )
        spec = fw.TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=True,
        )
        gen = fw.TreeGenerator(wing)
        segs = gen.generate(spec)
        st = gen.stats()
        results[f'{label}\n({st["n"]} segs)'] = segs
        print(f'    {method:12s}: {st["n"]:4d} segs')

    fig = viz.multi(results, ww=4.5)
    fig.savefig(f'{OUT}/ex03_spacing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Custom positions (optimizer-style) ──────────────────────
    print('\n  Custom (optimizer-defined) positions...')

    custom_positions = [0.03, 0.05, 0.08, 0.15, 0.25, 0.5, 0.7, 0.85, 0.92, 0.95]
    stations = fw.make_mixed_stations(
        n_stations=len(custom_positions),
        spacing='custom',
        spacing_kwargs={'positions': custom_positions},
    )
    spec = fw.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations, allow_crossing=True,
    )
    gen = fw.TreeGenerator(wing)
    segs = gen.generate(spec)
    st = gen.stats()
    print(f'    custom: {st["n"]} segs')

    uniform_stations = fw.make_mixed_stations(n_stations=len(custom_positions))
    spec_u = fw.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=uniform_stations, allow_crossing=True,
    )
    gen_u = fw.TreeGenerator(wing)
    segs_u = gen_u.generate(spec_u)
    st_u = gen_u.stats()

    fig = viz.cmp(
        segs, segs_u,
        f'Custom positions ({st["n"]} segs)',
        f'Uniform ({st_u["n"]} segs)',
    )
    fig.savefig(f'{OUT}/ex03_custom_vs_uniform.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Distribution visualisation (positions only) ─────────────
    print('\n  Distribution position plot...')
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = [
        ('uniform', {}),
        ('geometric', {'ratio': 2.0}),
        ('geometric', {'ratio': 0.5}),
        ('cosine', {}),
        ('power', {'exponent': 2.5}),
        ('bimodal', {'weight': 2.5}),
    ]
    for i, (method, kw) in enumerate(methods):
        pos = fw.distribute_stations(n_st, method, **kw)
        label = method
        if kw:
            label += f' ({list(kw.keys())[0]}={list(kw.values())[0]})'
        ax.scatter(pos, [i] * len(pos), s=60, zorder=5, label=label)
        ax.hlines(i, 0, 1, colors='gray', alpha=0.2, lw=0.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[0] + (f'\n{m[1]}' if m[1] else '') for m in methods], fontsize=9)
    ax.set_xlabel('Position along trunk')
    ax.set_title('Station Distribution Laws', fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    fig.savefig(f'{OUT}/ex03_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n  Saved to: {OUT}/')


if __name__ == '__main__':
    main()
