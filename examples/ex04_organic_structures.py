"""
Example 04 — Organic Tree-like Structures
==========================================
Demonstrates the most advanced capabilities:
  - Per-depth mode switching (e.g. sympodial → monopodial → dichotomous)
  - Spanwise-zoned mode variation
  - Graded parameters from root to tip
  - Per-branch thickness control
  - Non-uniform spacing combined with mode switching
"""

import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import canopy as cp
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT, exist_ok=True)


def main():
    _, wing = get_base_wing(bm='full_wing')
    viz = cp.Viz(wing)

    # ════════════════════════════════════════════════════════════
    # 1. Per-depth mode switching
    # ════════════════════════════════════════════════════════════
    print('  1. Per-depth mode switching...')

    # Sympodial at depth 0 → monopodial at depth 1 → dichotomous at depth 2
    sub_switching = cp.SubParams(
        max_depth=3,
        angles=[35, 30, 25],
        length_ratios=[0.55, 0.5, 0.45],
        thickness_ratios=[0.7, 0.65, 0.6],
        mode=['sympodial', 'monopodial', 'dichotomous'],
        min_length=0.02,
    )

    # Compare: uniform mode vs switching mode
    sub_uniform = cp.SubParams(
        max_depth=3,
        angles=[35, 30, 25],
        length_ratios=[0.55, 0.5, 0.45],
        thickness_ratios=[0.7, 0.65, 0.6],
        mode='sympodial',
        min_length=0.02,
    )

    results = {}
    for label, sub in [
        ('sympodial only', sub_uniform),
        ('symp→mono→dich', sub_switching),
    ]:
        stations = cp.make_diagonal_only_stations(
            n_stations=12, angle=35, length=2.0, sub=sub,
        )
        spec = cp.TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=stations, allow_crossing=True,
        )
        gen = cp.TreeGenerator(wing)
        segs = gen.generate(spec)
        st = gen.stats()
        results[f'{label}\n({st["n"]} segs)'] = segs
        print(f'    {label}: {st["n"]} segs')

    fig = viz.cmp(
        list(results.values())[0], list(results.values())[1],
        list(results.keys())[0], list(results.keys())[1],
    )
    fig.savefig(f'{OUT}/ex04_depth_mode_switching.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 2. Spanwise-zoned mode variation
    # ════════════════════════════════════════════════════════════
    print('\n  2. Spanwise-zoned mode variation...')

    zones = [
        {
            'eta_start': 0.0, 'eta_end': 0.35,
            'diag_angle': 30, 'chord_angle': 70,
            'diag_length': 2.5, 'chord_length': 2.0,
            'diag_sub': cp.SubParams(
                mode='sympodial', max_depth=3,
                angles=[30, 25, 20],
                length_ratios=[0.55, 0.5, 0.45],
                min_length=0.02,
            ),
            'chord_sub': cp.SubParams(
                mode='dichotomous', max_depth=1,
                angles=[25],
            ),
            'thick_frac': 0.8,
        },
        {
            'eta_start': 0.35, 'eta_end': 0.7,
            'diag_angle': 40, 'chord_angle': 75,
            'diag_length': 1.8, 'chord_length': 1.5,
            'diag_sub': cp.SubParams(
                mode=['monopodial', 'sympodial'], max_depth=2,
                angles=[35, 30],
                min_length=0.02,
            ),
            'chord_sub': cp.SubParams(
                mode='sympodial', max_depth=1,
                angles=[20],
            ),
            'thick_frac': 0.7,
        },
        {
            'eta_start': 0.7, 'eta_end': 1.0,
            'diag_angle': 50, 'chord_angle': None,
            'diag_length': 1.2, 'chord_length': 0.8,
            'diag_sub': cp.SubParams(
                mode='monochasium', max_depth=2,
                angles=[40, 35],
                min_length=0.02,
            ),
            'chord_sub': cp.SubParams(mode='dichotomous', max_depth=1),
            'thick_frac': 0.5,
        },
    ]

    stations_zoned = cp.make_zoned_stations(
        n_stations=16,
        zones=zones,
        spacing='geometric',
        spacing_kwargs={'ratio': 1.5},
    )

    spec_zoned = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations_zoned, allow_crossing=True,
    )
    gen = cp.TreeGenerator(wing)
    segs_zoned = gen.generate(spec_zoned)
    st_z = gen.stats()
    print(f'    Zoned: {st_z["n"]} segs')

    # Compare with uniform mixed
    stations_uniform = cp.make_mixed_stations(n_stations=16)
    spec_uniform = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations_uniform, allow_crossing=True,
    )
    gen2 = cp.TreeGenerator(wing)
    segs_uniform = gen2.generate(spec_uniform)
    st_u = gen2.stats()

    fig = viz.cmp(
        segs_zoned, segs_uniform,
        f'Zoned: symp→mono→monochasium\n({st_z["n"]} segs)',
        f'Uniform mixed\n({st_u["n"]} segs)',
    )
    fig.savefig(f'{OUT}/ex04_zoned_spanwise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 3. Graded parameters root → tip
    # ════════════════════════════════════════════════════════════
    print('\n  3. Graded parameters root → tip...')

    stations_graded = cp.make_graded_stations(
        n_stations=14,
        spacing='geometric',
        spacing_kwargs={'ratio': 1.4},
        angle_range=(25, 55),
        length_range=(2.5, 0.8),
        depth_range=(3, 1),
        mode_sequence=['sympodial', 'monopodial', 'dichotomous'],
        thick_frac_range=(0.85, 0.4),
    )

    spec_graded = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations_graded, allow_crossing=True,
    )
    gen = cp.TreeGenerator(wing)
    segs_graded = gen.generate(spec_graded)
    st_g = gen.stats()
    print(f'    Graded: {st_g["n"]} segs')

    fig = viz.cmp(
        segs_graded, segs_uniform,
        f'Graded root→tip\n({st_g["n"]} segs)',
        f'Uniform mixed\n({st_u["n"]} segs)',
    )
    fig.savefig(f'{OUT}/ex04_graded.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 4. Per-branch thickness control
    # ════════════════════════════════════════════════════════════
    print('\n  4. Per-branch thickness control...')

    # Heavy diagonal branches + light chordwise branches
    sub_heavy = cp.SubParams(
        max_depth=2, angles=[30], mode='sympodial', min_length=0.02,
    )
    sub_light = cp.SubParams(
        max_depth=1, angles=[25], mode='dichotomous',
    )

    stations_thick = []
    for p in cp.distribute_stations(12, 'uniform'):
        bl = 2.0 * (1 - 0.4 * p)
        stations_thick.append(cp.Station(
            position=p,
            branches=[
                cp.BranchDef(angle=35,  length=bl, thick_frac=0.9, sub=copy.deepcopy(sub_heavy)),
                cp.BranchDef(angle=-35, length=bl, thick_frac=0.9, sub=copy.deepcopy(sub_heavy)),
                cp.BranchDef(angle=75,  length=bl*0.8, thick_frac=0.3, sub=copy.deepcopy(sub_light)),
                cp.BranchDef(angle=-75, length=bl*0.8, thick_frac=0.3, sub=copy.deepcopy(sub_light)),
            ],
        ))

    spec_thick = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations_thick, allow_crossing=True,
    )
    gen = cp.TreeGenerator(wing)
    segs_thick = gen.generate(spec_thick)
    st_t = gen.stats()
    print(f'    Custom thickness: {st_t["n"]} segs')

    fig, ax = plt.subplots(figsize=(8, 7))
    viz._dom(ax)
    viz._br(ax, segs_thick)
    ax.set_title(f'Per-branch thickness\n(heavy diagonal, light chordwise, {st_t["n"]} segs)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{OUT}/ex04_thickness_control.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 5. Full organic design (combines everything)
    # ════════════════════════════════════════════════════════════
    print('\n  5. Full organic design...')

    organic_zones = [
        {
            'eta_start': 0.0, 'eta_end': 0.3,
            'diag_angle': 25, 'chord_angle': 65,
            'diag_length': 3.0, 'chord_length': 2.5,
            'diag_sub': cp.SubParams(
                mode=['sympodial', 'monopodial', 'dichotomous'],
                max_depth=3,
                angles=[30, 25, 20],
                length_ratios=[0.6, 0.55, 0.5],
                thickness_ratios=[0.75, 0.7, 0.65],
                min_length=0.01,
            ),
            'chord_sub': cp.SubParams(
                mode='dichotomous', max_depth=2,
                angles=[20, 15],
                length_ratios=[0.5, 0.45],
                min_length=0.02,
            ),
            'thick_frac': 0.85,
        },
        {
            'eta_start': 0.3, 'eta_end': 0.65,
            'diag_angle': 35, 'chord_angle': 75,
            'diag_length': 2.0, 'chord_length': 1.5,
            'diag_sub': cp.SubParams(
                mode=['monopodial', 'sympodial'],
                max_depth=2,
                angles=[35, 30],
                length_ratios=[0.5, 0.45],
                min_length=0.02,
            ),
            'chord_sub': cp.SubParams(
                mode='sympodial', max_depth=1,
                angles=[20],
                min_length=0.03,
            ),
            'thick_frac': 0.65,
        },
        {
            'eta_start': 0.65, 'eta_end': 1.0,
            'diag_angle': 45, 'chord_angle': 80,
            'diag_length': 1.0, 'chord_length': 0.8,
            'diag_sub': cp.SubParams(
                mode='monochasium', max_depth=2,
                angles=[40, 35],
                length_ratios=[0.45, 0.4],
                min_length=0.02,
            ),
            'chord_sub': cp.SubParams(
                mode='dichotomous', max_depth=1,
                angles=[25],
            ),
            'thick_frac': 0.45,
        },
    ]

    stations_organic = cp.make_zoned_stations(
        n_stations=18,
        zones=organic_zones,
        spacing='geometric',
        spacing_kwargs={'ratio': 1.6},
    )

    spec_organic = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=stations_organic, allow_crossing=True,
    )
    gen = cp.TreeGenerator(wing)
    segs_organic = gen.generate(spec_organic)
    st_o = gen.stats()
    print(f'    Organic: {st_o["n"]} segs, L={st_o["L"]}m')

    # 2D view
    fig, ax = plt.subplots(figsize=(10, 8))
    viz._dom(ax)
    viz._br(ax, segs_organic)
    ax.set_title(
        f'Organic Design — zoned modes + geometric spacing + depth switching\n'
        f'({st_o["n"]} segs, L={st_o["L"]}m)',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{OUT}/ex04_organic_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D view
    fig = viz.view3d(segs_organic, f'Organic design ({st_o["n"]} webs)')
    fig.savefig(f'{OUT}/ex04_organic_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Cross-sections
    fig = viz.xsec(segs_organic, [0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    fig.savefig(f'{OUT}/ex04_organic_xsec.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # Gallery panel
    # ════════════════════════════════════════════════════════════
    print('\n  Gallery panel...')
    fig = viz.multi({
        f'Zoned\n({st_z["n"]} segs)': segs_zoned,
        f'Graded\n({st_g["n"]} segs)': segs_graded,
        f'Organic\n({st_o["n"]} segs)': segs_organic,
    })
    fig.savefig(f'{OUT}/ex04_gallery.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n  Saved to: {OUT}/')


if __name__ == '__main__':
    main()
