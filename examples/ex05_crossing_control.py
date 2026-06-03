"""
Example 05 — Self-intersection Control (allow_crossing)
=========================================================
Demonstrates how ``allow_crossing=False`` prunes branches that would
cross existing segments, producing cleaner, non-overlapping structures.

Each configuration is shown side-by-side: crossing ON vs OFF.
"""

import os
import copy
import matplotlib
# matplotlib.use('Agg') # Commented out to allow interactive plotting (plt.show)
import matplotlib.pyplot as plt
import numpy as np
import canopy as cp
from utils import get_base_wing

OUT = os.path.join(os.path.dirname(__file__), 'output_crossing')
os.makedirs(OUT, exist_ok=True)


def generate_pair(wing, stations_fn, label, viz, **spec_kw):
    """Generate crossing-ON and crossing-OFF and return both."""
    results = {}
    for cross in [True, False]:
        tag = 'ON' if cross else 'OFF'
        stations = stations_fn()
        spec = cp.TrunkSpec(
            stations=stations,
            allow_crossing=cross,
            **spec_kw,
        )
        gen = cp.TreeGenerator(wing)
        segs = gen.generate(spec)
        st = gen.stats()
        results[tag] = (segs, st)
        print(f'    {label} crossing {tag}: {st["n"]} segs, L={st["L"]}m')
    return results


def main():
    print('='*65)
    print('  Example 05 — Self-intersection Control')
    print('='*65)

    # ════════════════════════════════════════════════════════════
    # 1. Basic sympodial — crossing ON vs OFF
    # ════════════════════════════════════════════════════════════
    print('\n  1. Sympodial — wingbox')
    _, wing_wb = get_base_wing(bm='wingbox')
    viz_wb = cp.Viz(wing_wb)

    def symp_stations():
        return cp.make_diagonal_only_stations(
            n_stations=14, angle=40, length=2.5,
            sub=cp.SubParams(max_depth=3, angles=[30, 35, 40], mode='sympodial'),
        )

    r = generate_pair(wing_wb, symp_stations, 'sympodial_wb',
                      viz_wb, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_wb.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Sympodial — wingbox', fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_01_sympodial_wb.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 2. Mixed mode — full wing (biggest effect)
    # ════════════════════════════════════════════════════════════
    print('\n  2. Mixed mode — full wing')
    _, wing_fw = get_base_wing(bm='full_wing')
    viz_fw = cp.Viz(wing_fw)

    def mixed_stations():
        return cp.make_mixed_stations(n_stations=16)

    r = generate_pair(wing_fw, mixed_stations, 'mixed_fw',
                      viz_fw, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_cp.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Mixed mode — full wing', fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_02_mixed_fullwing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 3. Dense monopodial — where crossing control matters most
    # ════════════════════════════════════════════════════════════
    print('\n  3. Dense monopodial — full wing')

    def mono_dense():
        return cp.make_diagonal_only_stations(
            n_stations=16, angle=25, length=3.0,
            sub=cp.SubParams(
                max_depth=3, angles=[20, 25, 30],
                length_ratios=[0.6, 0.55, 0.5],
                mode='monopodial', min_length=0.02,
            ),
        )

    r = generate_pair(wing_fw, mono_dense, 'monopodial_dense',
                      viz_fw, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_cp.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Dense monopodial — full wing', fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_03_monopodial_dense.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 4. Per-depth mode switching + crossing OFF
    # ════════════════════════════════════════════════════════════
    print('\n  4. Per-depth mode switching + crossing OFF')

    def depth_switch():
        return cp.make_diagonal_only_stations(
            n_stations=14, angle=35, length=2.0,
            sub=cp.SubParams(
                max_depth=3,
                angles=[35, 30, 25],
                length_ratios=[0.55, 0.5, 0.45],
                mode=['sympodial', 'monopodial', 'dichotomous'],
                min_length=0.02,
            ),
        )

    r = generate_pair(wing_fw, depth_switch, 'depth_switch',
                      viz_fw, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_cp.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Mode switching (symp→mono→dich) — crossing comparison',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_04_depth_switch.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 5. Geometric spacing + crossing OFF
    # ════════════════════════════════════════════════════════════
    print('\n  5. Geometric spacing + crossing OFF')

    def geom_mixed():
        return cp.make_mixed_stations(
            n_stations=14, spacing='geometric', spacing_kwargs={'ratio': 2.0},
        )

    r = generate_pair(wing_fw, geom_mixed, 'geom_mixed',
                      viz_fw, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_cp.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Geometric spacing (dense root) — crossing comparison',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_05_geom_spacing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 6. Zoned organic + crossing OFF
    # ════════════════════════════════════════════════════════════
    print('\n  6. Zoned organic + crossing OFF')

    def organic_zoned():
        zones = [
            {
                'eta_start': 0.0, 'eta_end': 0.35,
                'diag_angle': 25, 'chord_angle': 65,
                'diag_length': 3.0, 'chord_length': 2.5,
                'diag_sub': cp.SubParams(
                    mode=['sympodial', 'monopodial'], max_depth=3,
                    angles=[30, 25, 20], length_ratios=[0.6, 0.55, 0.5],
                    min_length=0.015,
                ),
                'chord_sub': cp.SubParams(mode='dichotomous', max_depth=2),
                'thick_frac': 0.85,
            },
            {
                'eta_start': 0.35, 'eta_end': 0.7,
                'diag_angle': 40, 'chord_angle': 75,
                'diag_length': 1.8, 'chord_length': 1.5,
                'diag_sub': cp.SubParams(mode='monopodial', max_depth=2, min_length=0.02),
                'chord_sub': cp.SubParams(mode='sympodial', max_depth=1),
                'thick_frac': 0.7,
            },
            {
                'eta_start': 0.7, 'eta_end': 1.0,
                'diag_angle': 50, 'chord_angle': None,
                'diag_length': 1.2, 'chord_length': 0.8,
                'diag_sub': cp.SubParams(mode='monochasium', max_depth=2, min_length=0.02),
                'chord_sub': cp.SubParams(mode='dichotomous', max_depth=1),
                'thick_frac': 0.5,
            },
        ]
        return cp.make_zoned_stations(
            n_stations=18, zones=zones,
            spacing='geometric', spacing_kwargs={'ratio': 1.6},
        )

    r = generate_pair(wing_fw, organic_zoned, 'organic_zoned',
                      viz_fw, chord_frac=0.5, span_cov=1.0, thick=0.004)

    fig = viz_cp.cmp(
        r['ON'][0], r['OFF'][0],
        f'Crossing ON ({r["ON"][1]["n"]} segs)',
        f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )

    fig.suptitle('Zoned organic design — crossing comparison',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_06_zoned_organic.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = viz_cp.view3d(
        r['OFF'][0],
        title=f'Crossing OFF ({r["OFF"][1]["n"]} segs)',
    )
    fig.suptitle('Zoned organic design — crossing OFF',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.show()
    plt.close(fig)



    # ════════════════════════════════════════════════════════════
    # 7. Multi-trunk + crossing OFF (global enforcement)
    # ════════════════════════════════════════════════════════════
    print('\n  7. Multi-trunk + crossing OFF (global enforcement)')

    # Use full_wing + wide-angle mixed stations so branches from
    # the two trunks overlap significantly, making the cross-trunk
    # crossing enforcement clearly visible.
    sub_mt = cp.SubParams(
        max_depth=2, angles=[35, 30], mode='sympodial', min_length=0.02,
    )

    for allow in [True, False]:
        tag = 'ON' if allow else 'OFF'
        specs = [
            cp.TrunkSpec(
                chord_frac=0.25, span_cov=1.0, thick=0.004,
                stations=cp.make_mixed_stations(
                    n_stations=12, diag_angle=40, chord_angle=70,
                    diag_length=2.5, chord_length=2.0,
                    diag_sub=copy.deepcopy(sub_mt),
                    chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
                ),
                allow_crossing=allow,
            ),
            cp.TrunkSpec(
                chord_frac=0.75, span_cov=1.0, thick=0.004,
                stations=cp.make_mixed_stations(
                    n_stations=12, diag_angle=-40, chord_angle=-70,
                    diag_length=2.5, chord_length=2.0,
                    diag_sub=copy.deepcopy(sub_mt),
                    chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
                ),
                allow_crossing=allow,
            ),
        ]
        gen = cp.TreeGenerator(wing_fw)
        segs = gen.generate_multi(specs)
        st = gen.stats()
        print(f'    multi-trunk crossing {tag}: {st["n"]} segs')

        if allow:
            segs_on, st_on = segs, st
        else:
            segs_off, st_off = segs, st

    fig = viz_cp.cmp(
        segs_on, segs_off,
        f'Crossing ON ({st_on["n"]} segs, 2 trunks)',
        f'Crossing OFF ({st_off["n"]} segs, 2 trunks)',
    )
    fig.suptitle('Multi-trunk — global crossing enforcement (cross-trunk)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_07_multi_trunk.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 8. Unprotected Trunk Dynamic Termination (protect_trunk=False)
    # ════════════════════════════════════════════════════════════
    print('\n  8. Unprotected Trunk Dynamic Termination (protect_trunk=False)')

    # Primary trunk is fully protected, secondary trunk stops when it hits primary's branches
    specs_dyn = [
        cp.TrunkSpec(
            chord_frac=0.3, span_cov=1.0, thick=0.004,
            stations=cp.make_mixed_stations(
                n_stations=14, diag_angle=40, chord_angle=70,
                diag_length=3.0, chord_length=2.0,
                diag_sub=cp.SubParams(max_depth=3, angles=[30, 25, 20], mode='sympodial', min_length=0.02),
                chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=True, # Primary trunk ignores crossings
        ),
        cp.TrunkSpec(
            chord_frac=0.8, span_cov=1.0, thick=0.004,
            stations=cp.make_mixed_stations(
                n_stations=14, diag_angle=-40, chord_angle=-70,
                diag_length=2.5, chord_length=2.0,
                diag_sub=cp.SubParams(max_depth=2, angles=[35, 30], mode='sympodial', min_length=0.02),
                chord_sub=cp.SubParams(mode='dichotomous', max_depth=1),
            ),
            allow_crossing=False,
            protect_trunk=False, # Secondary trunk dynamically stops if it hits a crossing
        ),
    ]

    gen_dyn = cp.TreeGenerator(wing_fw)
    segs_dyn = gen_dyn.generate_multi(specs_dyn)
    st_dyn = gen_dyn.stats()
    print(f'    dynamic termination: {st_dyn["n"]} segs')

    fig = viz_cp.view3d(
        segs_dyn,
        title=f'Dynamic Trunk Termination ({st_dyn["n"]} segs)\nPrimary protected (chord_frac=0.3), Secondary unprotected (chord_frac=0.8)',
    )
    fig.savefig(f'{OUT}/ex05_08_dynamic_termination.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # ════════════════════════════════════════════════════════════
    # 9. Gallery: all crossing-OFF results
    # ════════════════════════════════════════════════════════════
    print('\n  9. Gallery — all crossing-OFF designs')

    # Re-generate the best crossing-OFF configs for a gallery
    configs = [
        ('Sympodial\nwingbox', wing_wb, viz_wb, symp_stations),
        ('Mixed\nfull wing', wing_fw, viz_fw, mixed_stations),
        ('Monopodial\ndense', wing_fw, viz_fw, mono_dense),
        ('Mode switch\nsymp→mono→dich', wing_fw, viz_fw, depth_switch),
        ('Organic\nzoned', wing_fw, viz_fw, organic_zoned),
    ]

    gallery = {}
    for label, wing, viz, st_fn in configs:
        spec = cp.TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=0.004,
            stations=st_fn(), allow_crossing=False,
        )
        gen = cp.TreeGenerator(wing)
        segs = gen.generate(spec)
        st = gen.stats()
        gallery[f'{label}\n({st["n"]} segs)'] = segs

    # Use the full_wing viz for the gallery (wider domain looks better)
    fig = viz_cp.multi(gallery, ww=4.5)
    fig.suptitle('Gallery — all designs with crossing OFF',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'{OUT}/ex05_09_gallery_nocross.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{"="*65}')
    print(f'  Done — saved to {OUT}/')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
