import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

class OptimizationPostProcessor:
    """
    Reads parameter traces and constructs animations of structural topology and convergence history.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history_csv = os.path.join(self.output_dir, "opt_history.csv")
        self.convergence_csv = os.path.join(self.output_dir, "opt_convergence.csv")
        self.output_gif = os.path.join(self.output_dir, "optimization_history.gif")

    def animate_history(self, topology_builder, aero_wing, wing):
        """
        Generates an animation of the optimization trace.
        
        Parameters
        ----------
        topology_builder : callable
            The function `builder(x, wing)` used in optimization to generate segments.
        aero_wing : AeroWing
            The AeroShape adapter wing object.
        wing : Wing
            The base AeroShape wing object.
        """
        if not os.path.exists(self.history_csv):
            print(f"Error: {self.history_csv} not found.")
            return

        df = pd.read_csv(self.history_csv)
        print(f"Loaded {len(df)} evaluations from {self.history_csv}")

        # ── read convergence data from CSVs ──
        eval_ids     = np.arange(1, len(df) + 1)
        compliances  = df["Compliance"].values
        if "ConstraintResidual" in df.columns:
            vol_fracs = df["ConstraintResidual"].values + 1.0
        else:
            vol_fracs = df["VolFraction"].values
        max_irfs = df["MaxIRF"].values

        has_conv = False
        if os.path.exists(self.convergence_csv):
            df_conv = pd.read_csv(self.convergence_csv)
            if "Evaluation" in df_conv.columns:
                conv_evals = df_conv["Evaluation"].values
                conv_comps = df_conv["Compliance"].values
                has_conv = True

        # ── compute planform outline from the adapter (matches Viz._dom) ──
        eta = np.linspace(0, 1, 200)
        y_span = eta * wing.b
        x_le = np.array([wing.xle(e) for e in eta])
        x_te = np.array([wing.xte(e) for e in eta])
        x_fs = np.array([wing.xfs(e) for e in eta])
        x_rs = np.array([wing.xrs(e) for e in eta])
        x_df = np.array([wing.xdf(e) for e in eta])
        x_dr = np.array([wing.xdr(e) for e in eta])

        # ── reconstruct geometry at each CSV row ──
        iterations_data = []
        global_min_thick = 1e9
        global_max_thick = 0.0

        for idx, row in df.iterrows():
            iter_num = int(row["Evaluation"])
            compliance = float(row["Compliance"])
            
            # dynamically extract all x_* columns
            x_cols = [col for col in row.keys() if col.startswith("x_")]
            x_cols.sort(key=lambda c: int(c.split('_')[1]))
            x_vec = np.array([float(row[col]) for col in x_cols])

            try:
                segs = topology_builder(x_vec, wing)
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

        # Scatter for raw evaluations (lighter, transparent)
        eval_scatter, = ax_c.plot([], [], "o", color="#ff9e64", ms=3, alpha=0.3,
                                  label="Raw Evaluations", zorder=4)
                                  
        # Solid line for generational convergence
        comp_line, = ax_c.plot([], [], "-", color="#ff9e64", lw=2.5,
                               label="Algorithmic Convergence", zorder=5)
        
        if has_conv:
            conv_scatter, = ax_c.plot([], [], "s", color="#ff5555", ms=5,
                                      label="Generation Best", zorder=6)

        # Secondary Y-axis: IRF and volume ratio
        ax_r = ax_c.twinx()
        ax_r.set_ylabel("Tsai-Wu Inverse Reserve Factor (IRF) / Constraint Ratio", color="#7aa2f7",
                         fontsize=10, labelpad=8)
        ax_r.tick_params(axis="y", colors="#7aa2f7", labelsize=9)
        ax_r.spines["right"].set_color("#2b2b30")
        ax_r.spines["left"].set_color("#2b2b30")
        ax_r.spines["top"].set_color("#2b2b30")
        ax_r.spines["bottom"].set_color("#2b2b30")

        # Determine secondary axis range
        irf_values = max_irfs
        all_secondary = np.concatenate([irf_values[irf_values > 0], vol_fracs[vol_fracs > 0]])
        if len(all_secondary) > 0:
            sec_min = max(all_secondary.min() * 0.5, 1e-4)
            sec_max = all_secondary.max() * 3.0
        else:
            sec_min, sec_max = 1e-3, 1e3
        ax_r.set_yscale("log")
        ax_r.set_ylim(sec_min, sec_max)

        irf_line, = ax_r.plot([], [], "s-", color="#7aa2f7", lw=1.2, ms=3,
                                 label="Tsai-Wu IRF", zorder=4, alpha=0.8)
        volfrac_line, = ax_r.plot([], [], "^-", color="#9ece6a", lw=1.2, ms=3,
                                  label="Constraint Ratio", zorder=4, alpha=0.8)

        # Combined legend
        lines_for_legend = [comp_line, eval_scatter, irf_line, volfrac_line]
        labels_for_legend = ["Convergence", "Evaluations", "Tsai-Wu IRF", "Constraint Ratio"]
        if has_conv:
            lines_for_legend.insert(1, conv_scatter)
            labels_for_legend.insert(1, "Gen Best")
            
        ax_c.legend(lines_for_legend, labels_for_legend, loc="upper right",
                    fontsize=8, facecolor="#1e1e24", edgecolor="#2b2b30",
                    labelcolor="#c0c0c5")

        # Vertical marker line showing current frame position
        vline = ax_c.axvline(x=1, color="#ff5555", lw=1.0, ls=":", alpha=0.6, zorder=6)
        
        # Horizontal failure line for IRF = 1.0
        irf_fail_line = ax_r.axhline(y=1.0, color="#ff5555", lw=1.5, ls="--", alpha=0.6, zorder=1)

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
                f"Eval {data['eval']:03d}  (Iter {data['iter']:02d})\\n"
                f"Compliance: {comp_str}"
            )
            segs_text.set_text(f"{data['n_segs']} webs")

            # --- Right panel: progressive traces up to current frame ---
            n = frame_idx + 1
            xs = eval_ids[:n]

            # Raw Evaluations (Scatter)
            comp_vals = compliances[:n].copy()
            comp_vals[comp_vals >= 1e9] = np.nan
            eval_scatter.set_data(xs, comp_vals)
            
            # Generational Convergence (Line)
            if has_conv:
                valid_conv_mask = conv_evals <= data["eval"]
                cur_conv_evals = conv_evals[valid_conv_mask]
                cur_conv_comps = conv_comps[valid_conv_mask]
                
                if len(cur_conv_evals) == 0:
                    comp_line.set_data(xs, comp_vals)
                    conv_scatter.set_data([], [])
                else:
                    comp_line.set_data(cur_conv_evals, cur_conv_comps)
                    conv_scatter.set_data(cur_conv_evals, cur_conv_comps)
            else:
                comp_line.set_data(xs, comp_vals)

            # Tsai-Wu IRF
            sv = irf_values[:n].copy()
            sv[sv <= 0] = np.nan
            irf_line.set_data(xs, sv)

            # Volume ratio
            vf = vol_fracs[:n].copy()
            vf[vf <= 0] = np.nan
            volfrac_line.set_data(xs, vf)

            # Vertical marker
            vline.set_xdata([data["eval"]])

            return [lc, card_text, segs_text, comp_line, eval_scatter, irf_line, volfrac_line, vline, irf_fail_line] + ([conv_scatter] if has_conv else [])

        ani = animation.FuncAnimation(
            fig, update, frames=len(iterations_data),
            blit=True, interval=400,
        )

        print(f"Saving animation ({len(iterations_data)} frames) → {self.output_gif}")
        ani.save(self.output_gif, writer="pillow", fps=3)
        print("Done!")
        plt.close(fig)

    def export_vtu(self, mode: str = "all", specific_iter: int = None):
        """
        Converts saved .frd files in the output directory to .vtu format for ParaView.
        
        Parameters
        ----------
        mode : str
            "all" : Convert all .frd files found in the directory.
            "first_last" : Convert the first iteration, the last iteration, and the best final design.
            "specific" : Convert only the iteration specified by `specific_iter`.
        specific_iter : int, optional
            The iteration number to convert if mode="specific".
        """
        import glob
        from ccx2paraview import Converter
        import logging
        from .run_utils import split_vtu_file

        logging.basicConfig(level=logging.WARNING)
        print(f"\\n[Post-Processing] Converting .frd to .vtu (mode: {mode})...")

        frd_files = []
        if mode == "all":
            frd_files = sorted(glob.glob(os.path.join(self.output_dir, "*.frd")))
        elif mode == "specific":
            if specific_iter is None:
                print("Error: specific_iter must be provided when mode='specific'")
                return
            target = os.path.join(self.output_dir, f"opt_iter_{specific_iter:03d}_sim.frd")
            if os.path.exists(target):
                frd_files.append(target)
            else:
                print(f"File not found: {target}")
        elif mode == "first_last":
            # Best final
            best_final = os.path.join(self.output_dir, "opt_best_final_wing.frd")
            if os.path.exists(best_final):
                frd_files.append(best_final)
            
            # Find all iter frd files to get first and last
            iter_files = sorted(glob.glob(os.path.join(self.output_dir, "opt_iter_*_sim.frd")))
            if iter_files:
                frd_files.append(iter_files[0])  # First
                if len(iter_files) > 1:
                    frd_files.append(iter_files[-1]) # Last
        else:
            print(f"Unknown mode: {mode}")
            return

        # Remove duplicates while preserving order
        frd_files = list(dict.fromkeys(frd_files))

        if not frd_files:
            print("No .frd files found to convert.")
            return

        for frd_path in frd_files:
            vtu_path = frd_path.replace(".frd", ".vtu")
            if os.path.exists(vtu_path):
                print(f"  -> Skipping existing: {os.path.basename(vtu_path)}")
                continue
                
            print(f"  -> Converting: {os.path.basename(frd_path)}")
            try:
                c = Converter(frd_path, ["vtu"])
                c.run()
                if os.path.exists(vtu_path):
                    split_vtu_file(vtu_path)
            except Exception as e:
                print(f"  -> ccx2paraview conversion error on {frd_path}: {e}")
        
        print("Done converting VTU files.")
        
    def export_cad(self, topology_builder, aero_w, filename="optimized_3d_structure"):
        """
        Parses opt_convergence.csv to extract the global best configuration,
        rebuilds the topology, and exports it as Solid STEP and STL models.
        """
        if not os.path.exists(self.convergence_csv):
            print(f"Error: {self.convergence_csv} not found. Cannot export CAD.")
            return
            
        import pandas as pd
        from .cad_export import export_solid_topology
        
        df = pd.read_csv(self.convergence_csv)
        if df.empty:
            print("Error: opt_convergence.csv is empty.")
            return
            
        # Grab the absolute last row (best configuration in convergence file)
        best_row = df.iloc[-1]
        best_comp = float(best_row["Compliance"])
        print(f"  [CAD Export] Extracting best configuration from convergence file (Compliance: {best_comp:.4e})")
        
        # Extract variables
        x_cols = [c for c in df.columns if c.startswith("x_")]
        x_cols.sort(key=lambda c: int(c.split('_')[1]))
        x_opt = [float(best_row[col]) for col in x_cols]
        
        # Build wing obj for generator
        from .aeroshape_adapter import AeroWingAdapter
        wing_obj = AeroWingAdapter(aero_w, bm='full_wing')
        
        # Generate segments
        try:
            segs = topology_builder(x_opt, wing_obj)
        except Exception as e:
            print(f"  [CAD Export] Failed to build topology for CAD export: {e}")
            return
            
        # Export
        export_solid_topology(segs, aero_w, self.output_dir, filename=filename)
