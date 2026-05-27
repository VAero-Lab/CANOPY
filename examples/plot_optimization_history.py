#!/usr/bin/env python3
"""
Post-processing Script: Animated 2D Visualization of Fractal Web Optimization
==============================================================================
Reads the parameter trace from opt_history.csv, reconstructs the fractal
geometry at each iteration, and saves a beautiful animated GIF showing:
  - Left panel: 2D planform with web topology colored by thickness.
  - Right panel: Convergence trace (compliance, Von Mises stress, volume ratio).

Usage
-----
    python3 examples/plot_optimization_history.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import frond as fw
from examples.utils import get_base_wing


def main():
    # ── paths ──
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    opt_dir = os.path.join(examples_dir, "3D_wing_Opt")
    csv_path = os.path.join(opt_dir, "opt_history.csv")
    output_gif = os.path.join(opt_dir, "optimization_history.gif")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} evaluations from {csv_path}")

    # ── wing adapters ──
    aero_wing, wing = get_base_wing(bm="full_wing")

    # ── baseline zone definitions (same as ex08 / ex12) ──
    baseline_zones = [
        {
            "eta_start": 0.0, "eta_end": 0.35,
            "diag_angle": 25.0, "chord_angle": 65.0,
            "diag_length": 3.0, "chord_length": 2.5,
            "diag_sub": fw.SubParams(
                mode=["sympodial", "monopodial"], max_depth=3,
                angles=[30, 25, 20], length_ratios=[0.6, 0.55, 0.5],
                min_length=0.015),
            "chord_sub": fw.SubParams(mode="dichotomous", max_depth=2),
            "thick_frac": 0.85,
        },
        {
            "eta_start": 0.35, "eta_end": 0.7,
            "diag_angle": 40.0, "chord_angle": 75.0,
            "diag_length": 1.8, "chord_length": 1.5,
            "diag_sub": fw.SubParams(mode="monopodial", max_depth=2, min_length=0.02),
            "chord_sub": fw.SubParams(mode="sympodial", max_depth=1),
            "thick_frac": 0.7,
        },
        {
            "eta_start": 0.7, "eta_end": 1.0,
            "diag_angle": 50.0, "chord_angle": None,
            "diag_length": 1.2, "chord_length": 0.8,
            "diag_sub": fw.SubParams(mode="monochasium", max_depth=2, min_length=0.02),
            "chord_sub": fw.SubParams(mode="dichotomous", max_depth=1),
            "thick_frac": 0.5,
        },
    ]
    mapper = fw.FractalParameterMapper(baseline_zones)
    gen = fw.TreeGenerator(wing)

    # ── compute planform outline from the adapter (matches Viz._dom) ──
    eta = np.linspace(0, 1, 200)
    y_span = eta * wing.b
    x_le = np.array([wing.xle(e) for e in eta])
    x_te = np.array([wing.xte(e) for e in eta])
    x_fs = np.array([wing.xfs(e) for e in eta])
    x_rs = np.array([wing.xrs(e) for e in eta])
    x_df = np.array([wing.xdf(e) for e in eta])
    x_dr = np.array([wing.xdr(e) for e in eta])

    # ── read convergence data from CSV ──
    eval_ids     = np.arange(1, len(df) + 1)
    compliances  = df["Compliance"].values
    vol_fracs    = df["VolFraction"].values    # web_vol / max_vol (ratio, ~0.03)
    max_stresses = df["MaxStress"].values      # Von Mises in Pa

    # ── reconstruct geometry at each CSV row ──
    iterations_data = []
    global_min_thick = 1e9
    global_max_thick = 0.0

    for idx, row in df.iterrows():
        iter_num = int(row["Iteration"])
        compliance = float(row["Compliance"])
        x_vec = np.array([float(row[f"x_{i}"]) for i in range(14)])

        zones, trunk_thick = mapper.vector_to_params(x_vec)
        stations = fw.make_zoned_stations(n_stations=10, zones=zones)
        spec = fw.TrunkSpec(
            chord_frac=0.5, span_cov=1.0, thick=trunk_thick,
            stations=stations, allow_crossing=False,
        )
        try:
            segs = gen.generate(spec)
        except Exception:
            segs = []

        lines = []
        thicks = []
        if segs:
            for seg in segs:
                lines.append(((seg.p0[0], seg.p0[1]), (seg.p1[0], seg.p1[1])))
                thicks.append(seg.thick * 1000.0)  # mm
            global_min_thick = min(global_min_thick, min(thicks))
            global_max_thick = max(global_max_thick, max(thicks))

        iterations_data.append({
            "eval": idx + 1,
            "iter": iter_num,
            "compliance": compliance,
            "lines": lines,
            "thicknesses": thicks,
            "n_segs": len(segs),
        })

    if global_max_thick <= global_min_thick:
        global_max_thick = global_min_thick + 1.0

    print(f"Reconstructed {len(iterations_data)} frames. "
          f"Thickness range: [{global_min_thick:.2f}, {global_max_thick:.2f}] mm")

    # ── figure setup (dark premium aesthetic, 2 subplots) ──
    fig, (ax_topo, ax_conv) = plt.subplots(
        1, 2, figsize=(16, 7.5), dpi=120,
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.patch.set_facecolor("#121214")

    # ────────────────────────────────────────────────────────────
    # LEFT PANEL — Topology
    # ────────────────────────────────────────────────────────────
    ax = ax_topo
    ax.set_facecolor("#18181c")
    ax.set_aspect("equal")
    ax.grid(color="#2b2b30", linestyle="--", linewidth=0.4)
    ax.tick_params(colors="#8a8a93", labelsize=9)
    ax.set_xlabel("x [m]", color="#a0a0a5", fontsize=11, labelpad=8)
    ax.set_ylabel("y [m]", color="#a0a0a5", fontsize=11, labelpad=8)
    for spine in ax.spines.values():
        spine.set_color("#2b2b30")

    margin = 0.5
    ax.set_xlim(x_le.min() - margin, x_te.max() + margin)
    ax.set_ylim(y_span.min() - margin, y_span.max() + margin)

    # Static planform
    ax.plot(x_le, y_span, color="#e4e4e7", lw=1.5, zorder=2)
    ax.plot(x_te, y_span, color="#e4e4e7", lw=1.5, zorder=2)
    ax.plot([x_le[0], x_te[0]], [y_span[0], y_span[0]], color="#e4e4e7", lw=1.5, zorder=2)
    ax.plot([x_le[-1], x_te[-1]], [y_span[-1], y_span[-1]], color="#e4e4e7", lw=1.5, zorder=2)
    ax.plot(x_fs, y_span, color="#5588cc", lw=0.6, ls="--", alpha=0.25, zorder=1)
    ax.plot(x_rs, y_span, color="#5588cc", lw=0.6, ls="--", alpha=0.25, zorder=1)
    ax.fill_betweenx(y_span, x_df, x_dr, alpha=0.04, color="green", zorder=0)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=global_min_thick, vmax=global_max_thick)
    lc = LineCollection([], cmap=cmap, norm=norm, zorder=3)
    ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#8a8a93", labelcolor="#8a8a93")
    cbar.ax.set_ylabel("Web Thickness [mm]", color="#a0a0a5", fontsize=10, labelpad=8)
    cbar.outline.set_edgecolor("#2b2b30")

    card_text = ax.text(
        0.03, 0.97, "", transform=ax.transAxes, color="#ffffff",
        fontsize=11, fontweight="semibold", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e1e24",
                  edgecolor="#ffb86c", alpha=0.88),
        zorder=10,
    )
    segs_text = ax.text(
        0.97, 0.03, "", transform=ax.transAxes, color="#8a8a93",
        fontsize=9, ha="right", va="bottom", zorder=10,
    )
    ax.set_title("Fractal Web Topology", color="#ffffff", fontsize=13,
                 fontweight="bold", pad=12)

    # ────────────────────────────────────────────────────────────
    # RIGHT PANEL — Convergence trace
    # ────────────────────────────────────────────────────────────
    ax_c = ax_conv
    ax_c.set_facecolor("#18181c")
    ax_c.grid(color="#2b2b30", linestyle="--", linewidth=0.4)
    ax_c.tick_params(colors="#8a8a93", labelsize=9)
    ax_c.set_xlabel("Evaluation #", color="#a0a0a5", fontsize=11, labelpad=8)
    ax_c.set_ylabel("Compliance [N·m]", color="#ff9e64", fontsize=11, labelpad=8)
    for spine in ax_c.spines.values():
        spine.set_color("#2b2b30")
    ax_c.set_title("Convergence History", color="#ffffff", fontsize=13,
                   fontweight="bold", pad=12)

    # X-axis range
    ax_c.set_xlim(0.5, len(eval_ids) + 0.5)

    # Primary Y-axis: compliance (log scale)
    valid_comps = compliances[compliances > 0]
    if len(valid_comps) > 0:
        comp_min = valid_comps.min() * 0.5
        comp_max = compliances.max() * 2.0
    else:
        comp_min, comp_max = 1e6, 1e11
    ax_c.set_yscale("log")
    ax_c.set_ylim(comp_min, comp_max)
    ax_c.tick_params(axis="y", colors="#ff9e64")

    comp_line, = ax_c.plot([], [], "o-", color="#ff9e64", lw=1.5, ms=4,
                           label="Compliance", zorder=5)

    # Secondary Y-axis: stress and volume ratio
    ax_r = ax_c.twinx()
    ax_r.set_ylabel("Max Von Mises [MPa] / Vol Ratio", color="#7aa2f7",
                     fontsize=10, labelpad=8)
    ax_r.tick_params(axis="y", colors="#7aa2f7", labelsize=9)
    ax_r.spines["right"].set_color("#2b2b30")
    ax_r.spines["left"].set_color("#2b2b30")
    ax_r.spines["top"].set_color("#2b2b30")
    ax_r.spines["bottom"].set_color("#2b2b30")

    # Determine secondary axis range
    stress_mpa = max_stresses * 1e-6
    all_secondary = np.concatenate([stress_mpa[stress_mpa > 0], vol_fracs[vol_fracs > 0]])
    if len(all_secondary) > 0:
        sec_min = max(all_secondary.min() * 0.5, 1e-4)
        sec_max = all_secondary.max() * 3.0
    else:
        sec_min, sec_max = 1e-3, 1e3
    ax_r.set_yscale("log")
    ax_r.set_ylim(sec_min, sec_max)

    stress_line, = ax_r.plot([], [], "s-", color="#7aa2f7", lw=1.2, ms=3,
                             label="Max σ_VM [MPa]", zorder=4, alpha=0.8)
    volfrac_line, = ax_r.plot([], [], "^-", color="#9ece6a", lw=1.2, ms=3,
                              label="Vol Ratio", zorder=4, alpha=0.8)

    # Combined legend
    lines_for_legend = [comp_line, stress_line, volfrac_line]
    labels_for_legend = ["Compliance", "Max σ_VM [MPa]", "Vol Ratio"]
    ax_c.legend(lines_for_legend, labels_for_legend, loc="upper right",
                fontsize=8, facecolor="#1e1e24", edgecolor="#2b2b30",
                labelcolor="#c0c0c5")

    # Vertical marker line showing current frame position
    vline = ax_c.axvline(x=1, color="#ff5555", lw=1.0, ls=":", alpha=0.6, zorder=6)

    fig.tight_layout()

    # ── animation ──
    def update(frame_idx):
        data = iterations_data[frame_idx]
        lines_ = data["lines"]
        thicks_ = data["thicknesses"]
        comp = data["compliance"]

        # --- Left panel: topology ---
        lc.set_segments(lines_ if lines_ else [])
        if thicks_:
            lc.set_array(np.array(thicks_))
            lc.set_linewidths(np.clip(np.array(thicks_) * 0.7, 0.4, 6.0))
        else:
            lc.set_array(np.array([]))
            lc.set_linewidths([])

        if comp >= 1e9:
            comp_str = "PENALTY"
            card_text.get_bbox_patch().set_edgecolor("#ff5555")
        else:
            comp_str = f"{comp:.4e} N·m"
            is_last = (frame_idx == len(iterations_data) - 1)
            card_text.get_bbox_patch().set_edgecolor("#39ff14" if is_last else "#ffb86c")

        card_text.set_text(
            f"Eval {data['eval']:03d}  (Iter {data['iter']:02d})\n"
            f"Compliance: {comp_str}"
        )
        segs_text.set_text(f"{data['n_segs']} webs")

        # --- Right panel: progressive traces up to current frame ---
        n = frame_idx + 1
        xs = eval_ids[:n]

        # Compliance (filter out penalty values for clean plot)
        comp_vals = compliances[:n].copy()
        comp_vals[comp_vals >= 1e9] = np.nan
        comp_line.set_data(xs, comp_vals)

        # Von Mises stress in MPa
        sv = stress_mpa[:n].copy()
        sv[sv <= 0] = np.nan
        stress_line.set_data(xs, sv)

        # Volume ratio
        vf = vol_fracs[:n].copy()
        vf[vf <= 0] = np.nan
        volfrac_line.set_data(xs, vf)

        # Vertical marker
        vline.set_xdata([data["eval"]])

        return lc, card_text, segs_text, comp_line, stress_line, volfrac_line, vline

    ani = animation.FuncAnimation(
        fig, update, frames=len(iterations_data),
        blit=True, interval=400,
    )

    print(f"Saving animation ({len(iterations_data)} frames) → {output_gif}")
    ani.save(output_gif, writer="pillow", fps=3)
    print("Done!")


if __name__ == "__main__":
    main()
