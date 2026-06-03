"""
Example 02 — Mixed Mode (diagonal + chordwise branches)
========================================================
Demonstrates the mixed mode that combines diagonal and chordwise branches
with a comparison to the conventional wingbox baseline.
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
    _, wing_wb = get_base_wing(bm='wingbox')
    _, wing_fw = get_base_wing(bm='full_wing')
    viz_wb = cp.Viz(wing_wb)
    viz_fw = cp.Viz(wing_fw)

    conv = cp.ConvWB(wing_wb)
    br_conv = conv.generate()

    # ── Mixed mode on wingbox ───────────────────────────────────
    print('  Mixed mode on wingbox...')
    spec_wb = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=cp.make_mixed_stations(14),
        allow_crossing=True,
    )
    gen = cp.TreeGenerator(wing_wb)
    br_wb = gen.generate(spec_wb)
    st = gen.stats()
    print(f'    {st["n"]} segs, L={st["L"]}m')

    fig = viz_wb.cmp(
        br_wb, br_conv,
        f'Mixed mode ({st["n"]} webs)',
        f'Conventional ({len(br_conv)})',
    )
    fig.savefig(f'{OUT}/ex02_mixed_vs_conv.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Mixed mode on full wing ─────────────────────────────────
    print('  Mixed mode on full wing...')
    spec_fw = cp.TrunkSpec(
        chord_frac=0.5, span_cov=1.0, thick=0.004,
        stations=cp.make_mixed_stations(14),
        allow_crossing=True,
    )
    gen2 = cp.TreeGenerator(wing_fw)
    br_fw = gen2.generate(spec_fw)
    st2 = gen2.stats()
    print(f'    {st2["n"]} segs, L={st2["L"]}m')

    fig = viz_cp.cmp(
        br_fw, br_conv,
        f'Full-wing mixed ({st2["n"]} webs)',
        f'Conventional ({len(br_conv)})',
    )
    fig.savefig(f'{OUT}/ex02_mixed_fullwing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── 3D view ─────────────────────────────────────────────────
    print('  3D views...')
    fig = viz_wb.view3d(br_wb, f'Wingbox mixed mode ({st["n"]} webs)')
    fig.savefig(f'{OUT}/ex02_3d_wb.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = viz_cp.view3d(br_fw, f'Full-wing mixed mode ({st2["n"]} webs)')
    fig.savefig(f'{OUT}/ex02_3d_cp.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n  Saved to: {OUT}/')


if __name__ == '__main__':
    main()
