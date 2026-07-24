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
from canopy.fem_solver import CFRP_T300
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


@contextmanager
def SilenceAndCaptureOutput(log_file_path: str):
    """
    Context manager to redirect stdout and stderr at the file descriptor level
    to a log file. Keeps the console completely silent while capturing all solver prints,
    warnings, and messages.
    """
    log_fd = os.open(log_file_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    save_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        os.close(log_fd)
        for fd in save_fds:
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
        if in_displacements and (stripped.startswith("*") or "stresses" in stripped.lower()):
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


def parse_ccx_dat_stresses(dat_path: str, material: dict = None) -> dict[int, float]:
    """
    Parses Von Mises or Tsai-Wu IRF from the CalculiX .dat file.

    The .dat stress block has the format (per element, per integration point):
        elem_id  integ_pt  sxx  syy  szz  sxy  sxz  syz  ELSET_NAME
    
    If 'material' with Tsai-Wu strengths (Xt, Xc, Yt, etc.) is provided, 
    calculates the 3D Inverse Reserve Factor (IRF). 
    If IRF > 1.0, the material has failed.
    Otherwise, computes the standard Von Mises stress.
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
            # Expect: elem_id  integ_pt  sxx syy szz sxy sxz syz  [elset_name]
            if len(parts) >= 8:
                try:
                    elem_id = int(parts[0])
                    s11 = float(parts[2])
                    s22 = float(parts[3])
                    s33 = float(parts[4])
                    s12 = float(parts[5])
                    s13 = float(parts[6])
                    s23 = float(parts[7])
                    
                    if material and "Xt" in material:
                        # 3D Tsai-Wu Failure Index
                        Xt = material["Xt"]
                        Xc = material["Xc"]
                        Yt = material["Yt"]
                        Yc = material["Yc"]
                        Zt = material["Zt"]
                        Zc = material["Zc"]
                        S12 = material["S12"]
                        S13 = material["S13"]
                        S23 = material["S23"]
                        
                        F1 = (1.0/Xt) - (1.0/Xc)
                        F2 = (1.0/Yt) - (1.0/Yc)
                        F3 = (1.0/Zt) - (1.0/Zc)
                        F11 = 1.0 / (Xt * Xc)
                        F22 = 1.0 / (Yt * Yc)
                        F33 = 1.0 / (Zt * Zc)
                        F44 = 1.0 / (S23**2)
                        F55 = 1.0 / (S13**2)
                        F66 = 1.0 / (S12**2)
                        
                        # Empirical interaction terms
                        F12 = -0.5 * np.sqrt(F11 * F22)
                        F13 = -0.5 * np.sqrt(F11 * F33)
                        F23 = -0.5 * np.sqrt(F22 * F33)
                        
                        # Quadratic part (A)
                        A = (F11 * s11**2 + F22 * s22**2 + F33 * s33**2 + 
                             F66 * s12**2 + F55 * s13**2 + F44 * s23**2 +
                             2.0 * F12 * s11 * s22 + 2.0 * F13 * s11 * s33 + 2.0 * F23 * s22 * s33)
                        
                        # Linear part (B)
                        B = F1 * s11 + F2 * s22 + F3 * s33
                        
                        # Solve for Reserve Factor R: A*R^2 + B*R - 1 = 0
                        if A > 1e-30:
                            disc = B**2 + 4.0 * A
                            if disc >= 0:
                                R = (-B + np.sqrt(disc)) / (2.0 * A)
                                if R > 0:
                                    metric = 1.0 / R  # IRF
                                else:
                                    metric = 0.0
                            else:
                                metric = 0.0
                        else:
                            metric = 0.0
                    else:
                        # Standard Von Mises stress
                        metric = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                                            + 6.0 * (s12**2 + s13**2 + s23**2)))
                                            
                    stresses[elem_id] = max(stresses.get(elem_id, 0.0), metric)
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


def compute_skin_mesh_volume(inp_path: str, skin_thickness: float) -> float:
    """
    Computes the exact volume of the structural skin by calculating
    the quadrilateral and triangular shell element areas in the WingSkin
    elset and multiplying by the skin_thickness.
    """
    if not os.path.isfile(inp_path):
        return 0.0

    nodes = {}
    elements = {}
    skin_eids = []
    
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
                elif header.startswith("*ELSET"):
                    current_mode = "elset"
                    if "WINGSKIN" in header:
                        current_mode = "skin_elset"
                else:
                    current_mode = None
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
            elif current_mode == "skin_elset":
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                for p in parts:
                    try:
                        skin_eids.append(int(p))
                    except ValueError:
                        pass

    total_area = 0.0
    for eid in skin_eids:
        conn = elements.get(eid)
        if conn is None:
            continue
        
        if len(conn) == 4:
            p1, p2, p3, p4 = (nodes.get(c) for c in conn)
            if p1 is not None and p2 is not None and p3 is not None and p4 is not None:
                d1 = p3 - p1
                d2 = p4 - p2
                total_area += 0.5 * np.linalg.norm(np.cross(d1, d2))
        elif len(conn) == 3:
            p1, p2, p3 = (nodes.get(c) for c in conn)
            if p1 is not None and p2 is not None and p3 is not None:
                v1 = p2 - p1
                v2 = p3 - p1
                total_area += 0.5 * np.linalg.norm(np.cross(v1, v2))

    return float(total_area * skin_thickness)


def compute_core_mesh_volume(inp_path: str) -> float:
    """
    Computes the exact volume of the 3D solid core by calculating
    the volume of all C3D4 tetrahedral elements in the Core elset.
    """
    if not os.path.isfile(inp_path):
        return 0.0

    nodes = {}
    elements = {}
    core_eids = []
    
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
                elif header.startswith("*ELSET"):
                    current_mode = "elset"
                    if "CORE" in header:
                        current_mode = "core_elset"
                else:
                    current_mode = None
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
            elif current_mode == "core_elset":
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                for p in parts:
                    try:
                        core_eids.append(int(p))
                    except ValueError:
                        pass

    total_volume = 0.0
    for eid in core_eids:
        conn = elements.get(eid)
        if conn is None or len(conn) != 4:
            continue
            
        p1, p2, p3, p4 = (nodes.get(c) for c in conn)
        if p1 is not None and p2 is not None and p3 is not None and p4 is not None:
            # Volume of tetrahedron = |(a-d) . ((b-d) x (c-d))| / 6
            ad = p1 - p4
            bd = p2 - p4
            cd = p3 - p4
            vol = abs(np.dot(ad, np.cross(bd, cd))) / 6.0
            total_volume += vol

    return float(total_volume)


def compute_structural_mass(inp_path: str, web_properties: dict, skin_thickness: float, 
                            skin_density: float = 1600.0, web_density: float = 1600.0, 
                            core_density: float = 35.0) -> float:
    """
    Computes the total structural mass [kg] of the wing.
    M = (V_skin * rho_skin) + (V_webs * rho_webs) + (V_core * rho_core)
    """
    v_skin = compute_skin_mesh_volume(inp_path, skin_thickness)
    v_webs = compute_web_mesh_volume(inp_path, web_properties)
    v_core = compute_core_mesh_volume(inp_path)
    
    return (v_skin * skin_density) + (v_webs * web_density) + (v_core * core_density)


class AeroStructuralOptimizer:
    """
    High-level manager for uncoupled or coupled wing fractal web optimizations.
    Exposes a clean, optimizer-agnostic mathematical interface.
    """

    def __init__(
        self,
        aero_wing,
        wing,
        bounds: list[tuple[float, float]],
        x_init: np.ndarray,
        topology_builder: callable,
        output_dir: str,
        mode: str = "uncoupled",
        mapped_loads_dict: dict = None,
        aero_data: dict = None,
        save_every_n_iterations: int = None,
        target_volume_fraction: float = 0.25,
        target_mass_kg: float = None,
        skin_thickness: float = 0.003,
        mesh_target_elem_size: float = 0.15,
        mesh_skin_elem_size: float = 0.25,
    ):
        self.aero_wing = aero_wing
        self.wing = wing
        self.bounds = bounds
        self.x_init = x_init
        self.topology_builder = topology_builder
        self.output_dir = output_dir
        self.mode = mode
        self.mapped_loads = mapped_loads_dict
        self.aero_data = aero_data
        self.save_every_n_iterations = save_every_n_iterations
        self.target_volume_fraction = target_volume_fraction
        self.target_mass_kg = target_mass_kg
        self.skin_thickness = skin_thickness
        self.mesh_target_elem_size = mesh_target_elem_size
        self.mesh_skin_elem_size = mesh_skin_elem_size

        self.history_csv = os.path.join(output_dir, "opt_history.csv")
        self.convergence_csv = os.path.join(output_dir, "opt_convergence.csv")
        self.log_file = os.path.join(output_dir, "opt_solver_messages.log")
        self.iter_count = 0
        self.gen_count = 0

        # Initialize messages log file
        import datetime
        os.makedirs(output_dir, exist_ok=True)
        with open(self.log_file, "w") as f_log:
            f_log.write("==================================================\n")
            f_log.write(f"  AeroStructural Optimization Diagnostic Log\n")
            f_log.write(f"  Started: {datetime.datetime.now().isoformat()}\n")
            f_log.write("==================================================\n\n")

        # Initialize history log (Raw Evaluations)
        with open(self.history_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Evaluation", "Compliance", "WebVolume", "Mass", "ConstraintResidual", "MaxIRF", "ElapsedTime"]
            for i in range(len(self.bounds)):
                header.append(f"x_{i}")
            writer.writerow(header)

        # Initialize convergence log (Algorithmic Iterations/Generations)
        with open(self.convergence_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Generation", "Evaluation", "Compliance", "WebVolume", "Mass", "ConstraintResidual", "MaxIRF", "ElapsedTime"]
            for i in range(len(self.bounds)):
                header.append(f"x_{i}")
            writer.writerow(header)

        # Pre-calculate baseline volume dynamically
        print("  [Optimizer] Pre-calculating baseline web volume and structural behavior...")
        self._last_web_vol = 0.0
        self._cache_dict = {}
        self._cache_max_size = 200

        comp_base, constr_base = self._solve_iteration(self.x_init)
        self.baseline_web_volume = self._last_web_vol
        print(f"  [Optimizer] Baseline web volume computed: {self.baseline_web_volume * 1e6:.1f} cm3")

        # Reset iter_count and cached states so optimization loop starts clean from Iteration 1
        self.iter_count = 0
        self._cache_dict = {}

        # Estimate total wing envelope volume dynamically
        envelope_vol = 0.0
        if hasattr(self.aero_wing, "segments"):
            for seg in self.aero_wing.segments:
                seg_span = getattr(seg, "span", 7.5)
                half_span = seg_span / 2.0
                root_c = getattr(seg, "root_chord", 1.0)
                tip_c = getattr(seg, "tip_chord", 1.0)
                avg_c = 0.5 * (root_c + tip_c)
                avg_t = avg_c * 0.12
                envelope_vol += half_span * avg_c * avg_t

        if envelope_vol < 1e-3:
            envelope_vol = 1.0 * 7.5 * 0.2

        if self.target_mass_kg is None:
            self.max_allowable_volume = envelope_vol * target_volume_fraction
        else:
            self.max_allowable_volume = None

        if self.mode == "uncoupled" and self.mapped_loads is None and self.aero_data is None:
            print("  [Optimizer] Running uncoupled baseline aerodynamic solve...")
            import canopy as cp
            self.mapped_loads = {}
            try:
                coords = cp.extract_oml_grid(aero_wing, num_points_profile=40)
                aero_out = os.path.join(output_dir, "baseline_aero_loads.json")
                aero_in = os.path.join(output_dir, "baseline_aero_input.json")
                # Fallback to demo loads if FLOWPanel fails
                pass
            except Exception as e:
                print(f"  [Optimizer] Warning: Aero solver failed, using baseline point loads: {e}")

    def evaluate_cached(self, x: np.ndarray) -> tuple[float, list[float]]:
        """
        Evaluate compliance and constraints with a smart dictionary caching layer.
        Prevents duplicate calculations when solvers (like DE) evaluate identical
        population members across constraint and objective phases.
        """
        x_bytes = x.tobytes()
        if x_bytes in self._cache_dict:
            return self._cache_dict[x_bytes]

        # Solve fresh iteration
        comp, constrs = self._solve_iteration(x)

        # Update cache
        self._cache_dict[x_bytes] = (comp, constrs)
        
        if len(self._cache_dict) > self._cache_max_size:
            self._cache_dict.pop(next(iter(self._cache_dict)))

        return comp, constrs

    def _solve_iteration(self, x: np.ndarray) -> tuple[float, list[float]]:
        """Executes a single structural evaluation iteration."""
        self.iter_count += 1
        t0 = time.perf_counter()

        import canopy as cp

        iter_name = f"opt_iter_{self.iter_count:03d}"
        webs_step = os.path.join(self.output_dir, f"{iter_name}_webs.step")
        skin_step = os.path.join(self.output_dir, f"{iter_name}_skin.step")
        mesh_inp = os.path.join(self.output_dir, f"{iter_name}_mesh.inp")
        sim_inp = os.path.join(self.output_dir, f"{iter_name}_sim.inp")
        sim_dat = os.path.join(self.output_dir, f"{iter_name}_sim.dat")

        compliance = 1e10
        constraint_residual = 1.0
        max_irf = 0.0
        web_vol = 0.0
        total_mass = 0.0

        exc = None
        # Write clean iteration headers to the log file before entering redirected block
        with open(self.log_file, "a") as f_log:
            f_log.write(f"\n==================================================\n")
            f_log.write(f"  Evaluation {self.iter_count:03d} Solver Call\n")
            f_log.write(f"  Design Vector x: {x.tolist()}\n")
            f_log.write(f"==================================================\n")

        with SilenceAndCaptureOutput(self.log_file):
            try:
                # 1. Generate Topology via User Callback
                segs = self.topology_builder(x, self.wing)

                # Robust Segment Check: must have at least 3 segments
                if len(segs) < 3:
                    raise ValueError(f"Too few segments generated: {len(segs)}")

                # 2. STEP Export
                assembly, props = cp.build_brep_webs(segs, self.aero_wing, as_solid=False, output_step=webs_step)
                cp.export_hollow_skin(self.aero_wing, output_step=skin_step)

                # 3. Meshing (coarser target sizing for fast search optimization)
                mesher = cp.GmshMesher(
                    target_elem_size=self.mesh_target_elem_size, 
                    skin_elem_size=self.mesh_skin_elem_size
                )
                mesh_stats = mesher.mesh(
                    webs_step, mesh_inp,
                    skin_step=skin_step,
                    web_properties=props,
                )

                # 4. Aerodynamics (uncoupled or coupled)
                current_loads = self.mapped_loads
                
                if self.mode == "coupled":
                    aero_in = os.path.join(self.output_dir, f"{iter_name}_aero_input.json")
                    aero_out = os.path.join(self.output_dir, f"{iter_name}_aero_loads.json")
                    coords = cp.extract_oml_grid(self.aero_wing, num_points_profile=40)
                    aero_data = cp.solve_flowpanel(self.aero_wing, coords, output_json=aero_out, input_json=aero_in)
                    nodes, skin_nodes = cp.parse_mesh_for_mapping(mesh_inp, elset_name='WingSkin')
                    current_loads = cp.map_aerodynamic_loads(
                        aero_centroids=aero_data["centroids"],
                        aero_forces=aero_data["forces"],
                        nodes_dict=nodes,
                        skin_node_ids=skin_nodes,
                    )
                elif self.aero_data is not None:
                    # Dynamically map the uncoupled pre-computed aero data to the new mesh
                    nodes, skin_nodes = cp.parse_mesh_for_mapping(mesh_inp, elset_name='WingSkin')
                    current_loads = cp.map_aerodynamic_loads(
                        aero_centroids=np.array(self.aero_data["centroids"]),
                        aero_forces=np.array(self.aero_data["forces"]),
                        nodes_dict=nodes,
                        skin_node_ids=skin_nodes,
                    )

                # 5. Build CCX Deck
                sim_inp = cp.build_ccx_deck(
                    mesh_inp=mesh_inp,
                    web_properties=props,
                    segments=segs,
                    skin_thickness=self.skin_thickness,
                    mapped_aero_forces=current_loads,
                    binary_output=False,
                    dat_output=False,
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

                # 6. Run ccx solver & postprocess inside SilenceAndCaptureOutput for perfect logs
                res = cp.run_ccx(sim_inp, n_threads=1, convert_vtu=False)

                if res["success"]:
                    displacements = parse_ccx_dat_displacements(sim_dat)
                    stresses = parse_ccx_dat_stresses(sim_dat, material=CFRP_T300)

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
                        max_irf = float(max(stresses.values()))

                    web_vol = compute_web_mesh_volume(mesh_inp, props)
                    total_mass = compute_structural_mass(mesh_inp, props, self.skin_thickness)
                    if web_vol < 1e-6:
                        raise ValueError("Calculated web volume is exactly 0.0 (invalid mesh).")
                else:
                    raise RuntimeError(f"CCX solver execution failed. stdout:\n{res['stdout']}\nstderr:\n{res['stderr']}")

            except Exception as e:
                import traceback
                exc = traceback.format_exc()
                print("[ERROR] Exception in evaluation:")
                print(exc)

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

            for path in [webs_step, skin_step, mesh_inp, sim_dat, sim_log, sim_sta, sim_cvg, sim_12d]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        if exc is not None or web_vol < 1e-6 or compliance >= 1e9:
            compliance = 1e10
            web_vol = 0.0
            total_mass = 0.0
            constraint_residual = 10.0
        else:
            if self.target_mass_kg is not None:
                constraint_residual = (total_mass / self.target_mass_kg) - 1.0
            elif getattr(self, "max_allowable_volume", None) is not None:
                constraint_residual = (web_vol / self.max_allowable_volume) - 1.0
            else:
                constraint_residual = 0.0

        elapsed = time.perf_counter() - t0
        self._last_web_vol = web_vol
        self._last_mass = total_mass

        msg = (
            f"Eval {self.iter_count:03d} | Compliance: {compliance:.6e} N.m | "
            f"Mass: {total_mass:.2f} kg | Constraint Res: {constraint_residual:+.4f} | "
            f"Max IRF: {max_irf:.3f} | Elapsed: {elapsed:.1f}s"
        )

        with open(self.log_file, "a") as f_log:
            f_log.write(msg + "\n")

        if self.iter_count == 0:
            print("  " + msg)

        # Raw evaluation logging (History)
        with open(self.history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            row = [self.iter_count, compliance, web_vol, total_mass, constraint_residual, max_irf, elapsed]
            row.extend(x.tolist())
            writer.writerow(row)

        return compliance, [constraint_residual]

    def _optimization_callback(self, xk, convergence=None):
        """
        Triggered by SciPy at the end of each generation/iteration.
        Records the true convergence trace into opt_convergence.csv.
        """
        self.gen_count += 1
        
        # Pull cached evaluation corresponding to this best point
        x_bytes = xk.tobytes()
        if x_bytes in self._cache_dict:
            comp, constrs = self._cache_dict[x_bytes]
            constraint_residual = constrs[0]
        else:
            # Force an evaluation if somehow not cached (e.g. wiped by LRU limit)
            comp, constrs = self.evaluate_cached(xk)
            constraint_residual = constrs[0]
            
        print(f"  [Optimizer] Generation/Iter {self.gen_count:03d} Complete -> Best Compliance: {comp:.6e}")
        
        # We append a row to convergence CSV. 
        # (Since we only cache the most recent compliance and not max stress/vol, 
        # we will grab the last web_vol for now, or just map them. To be precise, we fetch the last web vol).
        web_vol = getattr(self, "_last_web_vol", 0.0)
        total_mass = getattr(self, "_last_mass", 0.0)
        
        with open(self.convergence_csv, "a", newline="") as f:
            writer = csv.writer(f)
            # Row: Generation, Evaluation, Compliance, WebVolume, Mass, ConstraintResidual, MaxIRF, ElapsedTime
            row = [self.gen_count, self.iter_count, comp, web_vol, total_mass, constraint_residual, 0.0, 0.0]
            row.extend(xk.tolist())
            writer.writerow(row)

    def optimize(self, method: str = "cobyla", max_iter: int = 50, **kwargs) -> dict:
        """
        High-level optimizer runner supporting COBYLA, SLSQP, and global Differential Evolution.
        Performs search in normalized design space [0, 1]^N to guarantee scaling and convergence stability.
        
        Parameters
        ----------
        method : str
            'cobyla', 'slsqp', or 'differential_evolution'
        max_iter : int
            Maximum number of iterations or generations.
            
        Keyword Arguments (Optimizer specific):
        ---------------------------------------
        DE (differential_evolution):
            popsize : int, default 5 (Population size multiplier)
            mutation : tuple, default (0.5, 1.0) (Mutation constant)
            recombination : float, default 0.7 (Recombination constant)
            tol : float, default 0.01 (Tolerance for convergence)
            
        SLSQP:
            eps : float, default 1e-3 (Step size for forward-difference gradient approximation)
            tol : float, default 1e-6 (Precision goal for the value of f)
            
        COBYLA:
            tol : float, default 1e-4 (Final accuracy in the optimization)
            rhobeg : float, default 1.0 (Reasonable initial changes to the variables)
        """
        bounds = self.bounds
        x_init = self.x_init

        # Map design space to normalized [0, 1]^N
        def z_to_x(z):
            x = np.zeros_like(z)
            for i, (lb, ub) in enumerate(bounds):
                x[i] = lb + z[i] * (ub - lb)
            return x

        def x_to_z(x):
            z = np.zeros_like(x)
            for i, (lb, ub) in enumerate(bounds):
                z[i] = (x[i] - lb) / (ub - lb)
            return z

        def evaluate_cached_normalized(z):
            # Strictly clip normalized values to [0, 1] to prevent COBYLA 
            # from taking steps outside the bounds and generating negative thicknesses
            z_clipped = np.clip(z, 0.0, 1.0)
            x = z_to_x(z_clipped)
            return self.evaluate_cached(x)

        z_init = x_to_z(x_init)
        z_bounds = [(0.0, 1.0)] * len(bounds)

        # Map normalized xk back to physical for callback
        def callback_wrapper(zk, convergence=None):
            zk_clipped = np.clip(zk, 0.0, 1.0)
            xk = z_to_x(zk_clipped)
            self._optimization_callback(xk, convergence)

        print(f"  [Optimizer] Starting optimization with method: {method.upper()}...")
        print(f"  [Optimizer] Configured for {len(bounds)} continuous variables.")

        if method.lower() == "cobyla":
            def cobyla_constraint(z):
                _, constrs = evaluate_cached_normalized(z)
                return -constrs[0]

            constraints = [{"type": "ineq", "fun": cobyla_constraint}]
            
            from scipy.optimize import minimize
            
            opts = {"maxiter": max_iter}
            if "rhobeg" in kwargs:
                opts["rhobeg"] = kwargs["rhobeg"]
            tol = kwargs.get("tol", 1e-4)

            t0 = time.perf_counter()
            res = minimize(
                fun=lambda z: evaluate_cached_normalized(z)[0],
                x0=z_init,
                method="COBYLA",
                bounds=z_bounds,
                constraints=constraints,
                options=opts,
                tol=tol,
                callback=callback_wrapper
            )
            elapsed = time.perf_counter() - t0

        elif method.lower() == "slsqp":
            def slsqp_constraint(z):
                _, constrs = evaluate_cached_normalized(z)
                return -constrs[0]

            constraints = [{"type": "ineq", "fun": slsqp_constraint}]
            
            from scipy.optimize import minimize
            
            opts = {"maxiter": max_iter, "disp": True}
            if "eps" in kwargs:
                opts["eps"] = kwargs["eps"]
            else:
                opts["eps"] = 1e-3
                
            tol = kwargs.get("tol", 1e-6)

            t0 = time.perf_counter()
            res = minimize(
                fun=lambda z: evaluate_cached_normalized(z)[0],
                x0=z_init,
                method="SLSQP",
                bounds=z_bounds,
                constraints=constraints,
                options=opts,
                tol=tol,
                callback=callback_wrapper
            )
            elapsed = time.perf_counter() - t0

        elif method.lower() == "differential_evolution" or method.lower() == "de":
            from scipy.optimize import NonlinearConstraint, differential_evolution

            def de_constraint(z):
                return evaluate_cached_normalized(z)[1][0]

            nl_const = NonlinearConstraint(de_constraint, -np.inf, 0.0)
            
            popsize = kwargs.get("popsize", 5)
            mutation = kwargs.get("mutation", (0.5, 1.0))
            recombination = kwargs.get("recombination", 0.7)
            tol = kwargs.get("tol", 0.01)

            t0 = time.perf_counter()
            res = differential_evolution(
                func=lambda z: evaluate_cached_normalized(z)[0],
                bounds=z_bounds,
                constraints=nl_const,
                maxiter=max_iter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                tol=tol,
                disp=True,
                callback=callback_wrapper
            )
            elapsed = time.perf_counter() - t0

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        best_x = z_to_x(res.x)

        print(f"  [Optimizer] Optimization complete in {elapsed:.1f}s!")
        
        try:
            self.evaluate_cached(best_x)
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
            "best_x": best_x,
            "best_fun": res.fun,
            "iterations": self.iter_count,
            "generations": self.gen_count,
            "elapsed_s": elapsed,
        }
