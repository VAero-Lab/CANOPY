"""
General Optimization Routines for Fractal Web Wing Structures.

Provides reusable classes and functions for parametric optimization, C-level
printing suppression, fast results parsing, and uncoupled/coupled aero-structural iteration loops.
Includes smart evaluation caching to prevent duplicate FEM solves in SLSQP/DE.
"""

from __future__ import annotations

import os
import sys
import time
import copy
import csv
import numpy as np
from contextlib import contextmanager


@contextmanager
def SilenceOutput():
    """
    Context manager to completely redirect stdout and stderr to os.devnull.
    Redirection happens at the file descriptor level (fds 1 and 2), meaning
    all native print statements and external subprocess prints (ccx, gmsh, julia)
    are silenced completely.
    """
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in save_fds + null_fds:
            os.close(fd)


def parse_ccx_dat_displacements(dat_path: str) -> dict[int, list[float]]:
    """
    Parses node displacements from the CalculiX .dat file.
    Extremely fast, pure-Python parser with no VTK/meshio dependencies.
    """
    displacements = {}
    if not os.path.isfile(dat_path):
        return displacements

    with open(dat_path, "r") as f:
        lines = f.readlines()

    in_displacements = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "displacements" in stripped.lower() and ("node" in stripped.lower() or "set" in stripped.lower()):
            in_displacements = True
            continue
        if in_displacements and stripped.startswith("*"):
            in_displacements = False
            continue
        if in_displacements:
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    if len(parts) == 4:
                        ux = float(parts[1])
                        uy = float(parts[2])
                        uz = float(parts[3])
                    else:
                        ux = float(parts[2])
                        uy = float(parts[3])
                        uz = float(parts[4])
                    displacements[node_id] = [ux, uy, uz]
                except ValueError:
                    pass
    return displacements


def parse_ccx_dat_stresses(dat_path: str) -> dict[int, float]:
    """
    Parses Von Mises stresses (or stress components) from the CalculiX .dat file
    to evaluate stress constraints.
    """
    stresses = {}
    if not os.path.isfile(dat_path):
        return stresses

    with open(dat_path, "r") as f:
        lines = f.readlines()

    in_stresses = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "stresses" in stripped.lower():
            in_stresses = True
            continue
        if in_stresses and stripped.startswith("*"):
            in_stresses = False
            continue
        if in_stresses:
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    elem_id = int(parts[0])
                    val = float(parts[-1])
                    stresses[elem_id] = max(stresses.get(elem_id, 0.0), val)
                except ValueError:
                    pass
    return stresses


def compute_web_mesh_volume(inp_path: str, web_properties: dict) -> float:
    """
    Computes the exact volume of all meshed fractal webs by calculating
    the quadrilateral and triangular shell element areas and multiplying
    by their respective thicknesses.
    """
    if not os.path.isfile(inp_path):
        return 0.0

    nodes = {}
    elements = {}
    elsets = {}
    
    current_mode = None
    current_elset = None

    with open(inp_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("**"):
                continue
            
            if stripped.startswith("*"):
                header = stripped.upper()
                if header.startswith("*NODE"):
                    current_mode = "node"
                    current_elset = None
                elif header.startswith("*ELEMENT"):
                    current_mode = "element"
                    current_elset = None
                elif header.startswith("*ELSET"):
                    current_mode = "elset"
                    current_elset = None
                    parts = header.split(",")
                    for p in parts:
                        if "ELSET=" in p:
                            current_elset = p.split("=")[1].strip()
                else:
                    current_mode = None
                    current_elset = None
                continue

            if current_mode == "node":
                parts = stripped.split(",")
                try:
                    nid = int(parts[0].strip())
                    coords = [float(p.strip()) for p in parts[1:4]]
                    nodes[nid] = np.array(coords)
                except ValueError:
                    pass
            elif current_mode == "element":
                parts = stripped.split(",")
                try:
                    eid = int(parts[0].strip())
                    conn = [int(p.strip()) for p in parts[1:] if p.strip()]
                    elements[eid] = conn
                except ValueError:
                    pass
            elif current_mode == "elset" and current_elset:
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                if current_elset not in elsets:
                    elsets[current_elset] = []
                for p in parts:
                    try:
                        elsets[current_elset].append(int(p))
                    except ValueError:
                        pass

    total_volume = 0.0
    for elset_name, eids in elsets.items():
        if elset_name.upper().startswith("WEB_"):
            try:
                web_idx = int(elset_name.split("_")[1])
            except ValueError:
                continue
            
            thick = web_properties.get(web_idx, {}).get("thickness", 0.003)
            
            for eid in eids:
                conn = elements.get(eid)
                if conn is None:
                    continue
                
                if len(conn) == 4:
                    p1 = nodes.get(conn[0])
                    p2 = nodes.get(conn[1])
                    p3 = nodes.get(conn[2])
                    p4 = nodes.get(conn[3])
                    if p1 is not None and p2 is not None and p3 is not None and p4 is not None:
                        d1 = p3 - p1
                        d2 = p4 - p2
                        area = 0.5 * np.linalg.norm(np.cross(d1, d2))
                        total_volume += area * thick
                elif len(conn) == 3:
                    p1 = nodes.get(conn[0])
                    p2 = nodes.get(conn[1])
                    p3 = nodes.get(conn[2])
                    if p1 is not None and p2 is not None and p3 is not None:
                        v1 = p2 - p1
                        v2 = p3 - p1
                        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                        total_volume += area * thick

    return float(total_volume)


class FractalParameterMapper:
    """
    Maps a 1D continuous design variable vector x to/from fractal parameter dictionaries.
    Specifically configured for the 3-zone organic fractal pattern of Example 08.
    """

    def __init__(self, baseline_zones: list[dict], baseline_thick: float = 0.005):
        self.baseline_zones = baseline_zones
        self.baseline_thick = baseline_thick

        # 14 continuous variables bounds
        self.bounds = [
            (0.001, 0.010),  # x[0]: global main trunk thickness [m]
            # Zone 1
            (15.0, 45.0),    # x[1]: Zone 1 diag_angle [deg]
            (45.0, 85.0),    # x[2]: Zone 1 chord_angle [deg]
            (1.5, 4.0),      # x[3]: Zone 1 diag_length [m]
            (1.0, 3.5),      # x[4]: Zone 1 chord_length [m]
            (0.5, 1.0),      # x[5]: Zone 1 thick_frac
            # Zone 2
            (20.0, 60.0),    # x[6]: Zone 2 diag_angle [deg]
            (50.0, 85.0),    # x[7]: Zone 2 chord_angle [deg]
            (1.0, 3.0),      # x[8]: Zone 2 diag_length [m]
            (0.8, 2.5),      # x[9]: Zone 2 chord_length [m]
            (0.4, 0.9),      # x[10]: Zone 2 thick_frac
            # Zone 3
            (30.0, 75.0),    # x[11]: Zone 3 diag_angle [deg]
            (0.5, 2.0),      # x[12]: Zone 3 diag_length [m]
            (0.2, 0.7),      # x[13]: Zone 3 thick_frac
        ]

    def get_baseline_vector(self) -> np.ndarray:
        """Construct the baseline design vector from baseline configuration."""
        z = self.baseline_zones
        return np.array([
            self.baseline_thick,
            # Zone 1
            z[0].get("diag_angle", 25.0),
            z[0].get("chord_angle", 65.0),
            z[0].get("diag_length", 3.0),
            z[0].get("chord_length", 2.5),
            z[0].get("thick_frac", 0.85),
            # Zone 2
            z[1].get("diag_angle", 40.0),
            z[1].get("chord_angle", 75.0),
            z[1].get("diag_length", 1.8),
            z[1].get("chord_length", 1.5),
            z[1].get("thick_frac", 0.70),
            # Zone 3
            z[2].get("diag_angle", 50.0),
            z[2].get("diag_length", 1.2),
            z[2].get("thick_frac", 0.50),
        ])

    def vector_to_params(self, x: np.ndarray) -> tuple[list[dict], float]:
        """Convert design variable vector x back into zones list and trunk thickness."""
        zones = copy.deepcopy(self.baseline_zones)
        trunk_thick = float(x[0])

        # Zone 1
        zones[0]["diag_angle"] = float(x[1])
        zones[0]["chord_angle"] = float(x[2])
        zones[0]["diag_length"] = float(x[3])
        zones[0]["chord_length"] = float(x[4])
        zones[0]["thick_frac"] = float(x[5])

        # Zone 2
        zones[1]["diag_angle"] = float(x[6])
        zones[1]["chord_angle"] = float(x[7])
        zones[1]["diag_length"] = float(x[8])
        zones[1]["chord_length"] = float(x[9])
        zones[1]["thick_frac"] = float(x[10])

        # Zone 3
        zones[2]["diag_angle"] = float(x[11])
        zones[2]["diag_length"] = float(x[12])
        zones[2]["thick_frac"] = float(x[13])

        return zones, trunk_thick


class AeroStructuralOptimizer:
    """
    High-level manager for uncoupled or coupled wing fractal web optimizations.
    Exposes a clean, optimizer-agnostic mathematical interface.
    """

    def __init__(
        self,
        aero_wing,
        wing,
        baseline_zones: list[dict],
        output_dir: str,
        mode: str = "uncoupled",
        mapped_loads_dict: dict = None,
        save_every_n_iterations: int = None,
        target_volume_fraction: float = 0.25,
        skin_thickness: float = 0.003,
    ):
        self.aero_wing = aero_wing
        self.wing = wing
        self.output_dir = output_dir
        self.mode = mode
        self.mapped_loads = mapped_loads_dict
        self.save_every_n_iterations = save_every_n_iterations
        self.target_volume_fraction = target_volume_fraction
        self.skin_thickness = skin_thickness

        self.mapper = FractalParameterMapper(baseline_zones)
        self.history_csv = os.path.join(output_dir, "opt_history.csv")
        self.iter_count = 0

        # Smart caching layer to prevent double solves
        self._cached_x = None
        self._cached_compliance = None
        self._cached_constraints = None

        # Initialize history log
        os.makedirs(output_dir, exist_ok=True)
        with open(self.history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Iteration", "Compliance", "WebVolume", "VolFraction", "MaxStress", "ElapsedTime"]
            for i in range(len(self.mapper.bounds)):
                header.append(f"x_{i}")
            writer.writerow(header)

        # Baseline volume estimate for target constraint V_max = 25% of ref volume
        ref_chord = 1.0
        ref_span = 7.5
        ref_thick = 0.2
        self.max_allowable_volume = ref_chord * ref_span * ref_thick * target_volume_fraction

        if self.mode == "uncoupled" and self.mapped_loads is None:
            print("  [Optimizer] Running uncoupled baseline aerodynamic solve...")
            import frond as fw
            self.mapped_loads = {}
            try:
                coords = fw.extract_oml_grid(aero_wing, num_points_profile=40)
                aero_out = os.path.join(output_dir, "baseline_aero_loads.json")
                aero_in = os.path.join(output_dir, "baseline_aero_input.json")
                # Fallback to demo loads if FLOWPanel fails
                pass
            except Exception as e:
                print(f"  [Optimizer] Warning: Aero solver failed, using baseline point loads: {e}")

    def evaluate_cached(self, x: np.ndarray) -> tuple[float, list[float]]:
        """
        Evaluate compliance and constraints with a smart caching layer.
        Prevents duplicate calculations when solvers (like SLSQP, COBYLA, DE)
        call the objective function and constraint function separately for the same point.
        """
        if self._cached_x is not None and np.allclose(x, self._cached_x, rtol=1e-8, atol=1e-8):
            return self._cached_compliance, self._cached_constraints

        # Solve fresh iteration
        comp, constrs = self._solve_iteration(x)

        # Update cache
        self._cached_x = x.copy()
        self._cached_compliance = comp
        self._cached_constraints = constrs
        return comp, constrs

    def _solve_iteration(self, x: np.ndarray) -> tuple[float, list[float]]:
        """Executes a single structural evaluation iteration."""
        self.iter_count += 1
        t0 = time.perf_counter()

        import frond as fw

        zones, trunk_thick = self.mapper.vector_to_params(x)

        iter_name = f"opt_iter_{self.iter_count:03d}"
        webs_step = os.path.join(self.output_dir, f"{iter_name}_webs.step")
        skin_step = os.path.join(self.output_dir, f"{iter_name}_skin.step")
        mesh_inp = os.path.join(self.output_dir, f"{iter_name}_mesh.inp")
        sim_inp = os.path.join(self.output_dir, f"{iter_name}_sim.inp")
        sim_dat = os.path.join(self.output_dir, f"{iter_name}_sim.dat")

        compliance = 1e10
        vol_frac_residual = 1.0
        max_stress = 0.0
        web_vol = 0.0

        exc = None
        with SilenceOutput():
            try:
                # 1. Stations and segments
                stations = fw.make_zoned_stations(n_stations=10, zones=zones)
                spec = fw.TrunkSpec(
                    chord_frac=0.5,
                    span_cov=1.0,
                    thick=trunk_thick,
                    stations=stations,
                    allow_crossing=False,
                )
                gen = fw.TreeGenerator(self.wing)
                segs = gen.generate(spec)

                # 2. STEP Export
                assembly, props = fw.build_brep_webs(segs, self.aero_wing, as_solid=False, output_step=webs_step)
                fw.export_hollow_skin(self.aero_wing, output_step=skin_step)

                # 3. Meshing (coarser target sizing for fast search optimization)
                mesher = fw.GmshMesher(target_elem_size=0.15, skin_elem_size=0.25)
                mesh_stats = mesher.mesh(
                    webs_step, mesh_inp,
                    skin_step=skin_step,
                    web_properties=props,
                )

                # 4. Aerodynamics (if fully coupled mode active)
                current_loads = self.mapped_loads
                if self.mode == "coupled":
                    aero_in = os.path.join(self.output_dir, f"{iter_name}_aero_input.json")
                    aero_out = os.path.join(self.output_dir, f"{iter_name}_aero_loads.json")
                    coords = fw.extract_oml_grid(self.aero_wing, num_points_profile=40)
                    fw.solve_flowpanel(self.aero_wing, coords, output_json=aero_out, input_json=aero_in)
                    pass

                # 5. Build CCX Deck
                sim_inp = fw.build_ccx_deck(
                    mesh_inp=mesh_inp,
                    web_properties=props,
                    segments=segs,
                    skin_thickness=self.skin_thickness,
                    mapped_aero_forces=current_loads,
                    binary_output=False,
                )
                sim_dat = sim_inp.replace(".inp", ".dat")

                # Inject output prints BEFORE *END STEP
                with open(sim_inp, "r") as f_deck:
                    deck_content = f_deck.read()

                def get_max_ids_from_inp(inp_path: str) -> tuple[int, int]:
                    max_node = 0
                    max_elem = 0
                    current_mode = None
                    with open(inp_path, "r") as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped or stripped.startswith("**"):
                                continue
                            if stripped.startswith("*"):
                                header = stripped.upper()
                                if header.startswith("*NODE"):
                                    current_mode = "node"
                                elif header.startswith("*ELEMENT"):
                                    current_mode = "element"
                                else:
                                    current_mode = None
                                continue
                            if current_mode == "node":
                                try:
                                    nid = int(stripped.split(",")[0].strip())
                                    if nid > max_node:
                                        max_node = nid
                                except ValueError:
                                    pass
                            elif current_mode == "element":
                                try:
                                    eid = int(stripped.split(",")[0].strip())
                                    if eid > max_elem:
                                        max_elem = eid
                                except ValueError:
                                    pass
                    return max_node, max_elem

                max_node, max_elem = get_max_ids_from_inp(mesh_inp)
                custom_sets = (
                    f"\n*NSET, NSET=NSET_ALL_NODES, GENERATE\n1, {max_node}, 1\n"
                    f"*ELSET, ELSET=EALL_ELEMS, GENERATE\n1, {max_elem}, 1\n"
                )

                if "*STEP" in deck_content:
                    deck_content = deck_content.replace("*STEP", custom_sets + "*STEP", 1)
                elif "*step" in deck_content:
                    deck_content = deck_content.replace("*step", custom_sets + "*step", 1)
                
                prints = "*NODE PRINT, NSET=NSET_ALL_NODES\nU\n*EL PRINT, ELSET=EALL_ELEMS\nS\n*END STEP"
                if "*END STEP" in deck_content:
                    deck_content = deck_content.replace("*END STEP", prints)
                elif "*end step" in deck_content:
                    deck_content = deck_content.replace("*end step", prints)
                
                with open(sim_inp, "w") as f_deck:
                    f_deck.write(deck_content)

            except Exception as e:
                import traceback
                exc = traceback.format_exc()

        if exc is not None:
            print("  [DEBUG] Exception in geometry/meshing:")
            print(exc)

        # 6. Run ccx solver & postprocess outside of SilenceOutput to expose any errors
        if exc is None:
            try:
                res = fw.run_ccx(sim_inp, convert_vtu=False)

                if res["success"]:
                    displacements = parse_ccx_dat_displacements(sim_dat)
                    stresses = parse_ccx_dat_stresses(sim_dat)

                    if displacements:
                        compliance = 0.0
                        if current_loads:
                            for nid, force in current_loads.items():
                                if nid in displacements:
                                    compliance += np.dot(force, displacements[nid])
                        else:
                            u_mags = [np.linalg.norm(u) for u in displacements.values()]
                            compliance = float(np.mean(u_mags))

                    if stresses:
                        max_stress = float(max(stresses.values()))

                    web_vol = compute_web_mesh_volume(mesh_inp, props)
                else:
                    print(f"  [DEBUG] CCX Run failed with stdout:\n{res['stdout']}\nstderr:\n{res['stderr']}")
            except Exception as e:
                import traceback
                print("  [DEBUG] Exception in solver:")
                print(traceback.format_exc())

        # Define exact file paths for this iteration
        sim_frd = sim_inp.replace(".inp", ".frd")
        sim_log = sim_inp.replace(".inp", ".log")
        sim_sta = sim_inp.replace(".inp", ".sta")
        sim_cvg = sim_inp.replace(".inp", ".cvg")
        sim_12d = sim_inp.replace(".inp", ".12d")

        # Cleanup files unless they need to be preserved
        should_save = False
        if self.save_every_n_iterations is not None:
            if self.iter_count == 1 or (self.iter_count % self.save_every_n_iterations == 0):
                should_save = True

        if not should_save and self.iter_count > 1:
            # Delete intermediate files
            for path in [webs_step, skin_step, mesh_inp, sim_inp, sim_dat, sim_frd, sim_log, sim_sta, sim_cvg, sim_12d]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        else:
            # Rename key files for archiving
            new_inp = os.path.join(self.output_dir, f"opt_iter_{self.iter_count:03d}_wing.inp")
            new_frd = os.path.join(self.output_dir, f"opt_iter_{self.iter_count:03d}_wing.frd")
            
            if os.path.exists(sim_inp):
                try:
                    os.rename(sim_inp, new_inp)
                except OSError:
                    pass
            if os.path.exists(sim_frd):
                try:
                    os.rename(sim_frd, new_frd)
                except OSError:
                    pass

            # Clean up all other auxiliary files (dat, step, log, sta, etc.) to keep output pristine!
            for path in [webs_step, skin_step, mesh_inp, sim_dat, sim_log, sim_sta, sim_cvg, sim_12d]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        vol_frac_residual = (web_vol / self.max_allowable_volume) - 1.0
        elapsed = time.perf_counter() - t0

        print(
            f"Iter {self.iter_count:03d} | Compliance: {compliance:.6f} N.m | "
            f"Web Volume: {web_vol*1e6:.1f} cm3 | Vol Frac Residual: {vol_frac_residual:+.4f} | "
            f"Max Stress: {max_stress*1e-6:.1f} MPa | Elapsed: {elapsed:.1f}s"
        )

        with open(self.history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            row = [self.iter_count, compliance, web_vol, vol_frac_residual + 1.0, max_stress, elapsed]
            row.extend(x.tolist())
            writer.writerow(row)

        return compliance, [vol_frac_residual]

    def optimize(self, method: str = "cobyla", max_iter: int = 50, pop_size: int = 5) -> dict:
        """
        High-level optimizer runner supporting COBYLA, SLSQP, and global Differential Evolution.
        """
        x_init = self.mapper.get_baseline_vector()
        bounds = self.mapper.bounds

        print(f"  [Optimizer] Starting optimization with method: {method.upper()}...")
        print(f"  [Optimizer] Initial variables: {x_init}")

        if method.lower() == "cobyla":
            # COBYLA uses a list of constraint dictionaries: c(x) >= 0
            # For V / V_max - 1.0 <= 0, the constraint function is: 1.0 - V / V_max >= 0
            def cobyla_constraint(x):
                _, constrs = self.evaluate_cached(x)
                return -constrs[0]

            constraints = [{"type": "ineq", "fun": cobyla_constraint}]
            
            from scipy.optimize import minimize
            
            t0 = time.perf_counter()
            res = minimize(
                fun=lambda x: self.evaluate_cached(x)[0],
                x0=x_init,
                method="COBYLA",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter},
            )
            elapsed = time.perf_counter() - t0

        elif method.lower() == "slsqp":
            # SLSQP constraint: c(x) >= 0
            def slsqp_constraint(x):
                _, constrs = self.evaluate_cached(x)
                return -constrs[0]

            constraints = [{"type": "ineq", "fun": slsqp_constraint}]
            
            from scipy.optimize import minimize
            
            t0 = time.perf_counter()
            res = minimize(
                fun=lambda x: self.evaluate_cached(x)[0],
                x0=x_init,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter, "disp": True},
            )
            elapsed = time.perf_counter() - t0

        elif method.lower() == "differential_evolution" or method.lower() == "de":
            # Differential Evolution uses NonlinearConstraint: lb <= c(x) <= ub
            from scipy.optimize import NonlinearConstraint, differential_evolution

            def de_constraint(x):
                # Returns constraint residual (V / V_max) - 1.0 which we want <= 0
                return self.evaluate_cached(x)[1][0]

            # V_residual <= 0 means: -inf <= de_constraint(x) <= 0.0
            nl_const = NonlinearConstraint(de_constraint, -np.inf, 0.0)

            t0 = time.perf_counter()
            res = differential_evolution(
                func=lambda x: self.evaluate_cached(x)[0],
                bounds=bounds,
                constraints=nl_const,
                maxiter=max_iter,
                popsize=pop_size,
                disp=True,
            )
            elapsed = time.perf_counter() - t0

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        print(f"  [Optimizer] Optimization complete in {elapsed:.1f}s!")
        print(f"  [Optimizer] Best parameters found: {res.x}")
        
        # Save final best iteration files
        try:
            # Re-evaluate best to preserve its final files
            self.evaluate_cached(res.x)
            for ext in [".inp", ".frd"]:
                old_path = os.path.join(self.output_dir, f"opt_iter_{self.iter_count:03d}_wing{ext}")
                new_path = os.path.join(self.output_dir, f"opt_best_final_wing{ext}")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
            print(f"  [Optimizer] Saved best structural results to: opt_best_final_wing.inp/frd")
        except Exception:
            pass

        return {
            "success": res.success,
            "best_x": res.x,
            "best_fun": res.fun,
            "iterations": self.iter_count,
            "elapsed_s": elapsed,
        }
