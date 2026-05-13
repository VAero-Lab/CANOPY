"""
CalculiX Solver Runner Utilities.

Provides a portable, reusable function for executing CalculiX simulations
with optimal parallelism settings and optional post-processing.
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional


def run_ccx(
    sim_inp: str,
    n_threads: int = None,
    convert_vtu: bool = False,
    ccx_binary: str = "ccx",
) -> dict:
    """
    Run the CalculiX solver with optimal parallelism settings.

    Sets environment variables for parallel stiffness/results assembly
    (``CCX_NPROC_STIFFNESS``, ``CCX_NPROC_RESULTS``, ``OMP_NUM_THREADS``)
    before invoking ``ccx``.  These variables are standard CalculiX/OpenMP
    controls and are silently ignored on builds that do not support them,
    making this function fully portable across machines.

    Parameters
    ----------
    sim_inp : str
        Path to the CalculiX simulation ``.inp`` file.
    n_threads : int, optional
        Number of threads for parallel assembly and results calculation.
        Defaults to ``min(os.cpu_count(), 8)`` to prevent OOM errors and
        contact-search race conditions on high-core-count clusters using
        the default SPOOLES solver.
    convert_vtu : bool
        If True, convert the ``.frd`` results file to ``.vtu`` format
        via ``ccx2paraview`` after a successful solve.  Default is False.
    ccx_binary : str
        Name or path of the CalculiX executable.  Default is ``"ccx"``.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``success`` (bool): Whether ``ccx`` returned exit code 0.
        - ``elapsed_s`` (float): Wall-clock time for the solve [seconds].
        - ``frd_path`` (str or None): Path to the ``.frd`` results file.
        - ``vtu_path`` (str or None): Path to the ``.vtu`` file (if converted).
        - ``returncode`` (int): The raw process exit code.
        - ``stdout`` (str): Solver standard output (last 1000 chars).
        - ``stderr`` (str): Solver standard error (last 1000 chars).
    """
    if not os.path.isfile(sim_inp):
        raise FileNotFoundError(f"Simulation input file not found: {sim_inp}")

    # ── Thread count ──
    if n_threads is None:
        # Cap at 8 to prevent OOM / segfaults on high-core-count clusters 
        # (e.g. 72 threads) when using standard SPOOLES ccx.
        n_threads = min(os.cpu_count() or 4, 8)

    # ── Build environment with parallelism hints ──
    env = os.environ.copy()
    nt = str(n_threads)
    env["CCX_NPROC_STIFFNESS"] = nt
    env["CCX_NPROC_RESULTS"] = nt
    env["OMP_NUM_THREADS"] = nt

    # ── Prepare paths ──
    cwd = os.path.dirname(os.path.abspath(sim_inp))
    job_basename = os.path.basename(sim_inp).replace(".inp", "")
    frd_path = os.path.join(cwd, job_basename + ".frd")

    # ── Run solver ──
    print(f"  -> Running CalculiX ({ccx_binary}) with {n_threads} threads ...")
    t0 = time.perf_counter()

    result = subprocess.run(
        [ccx_binary, job_basename],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )

    elapsed = time.perf_counter() - t0

    success = result.returncode == 0
    if success:
        print(f"  -> CalculiX solver completed successfully in {elapsed:.1f}s")
    else:
        print(f"  -> CalculiX solver FAILED (exit code {result.returncode}) "
              f"after {elapsed:.1f}s")
        tail = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        if tail:
            print(f"     {tail}")

    # ── Optional VTU conversion ──
    vtu_path = None
    if success and convert_vtu and os.path.isfile(frd_path):
        print("  -> Converting results for ParaView (ccx2paraview)...")
        try:
            from ccx2paraview import Converter
            import logging
            logging.basicConfig(level=logging.WARNING)
            c = Converter(frd_path, ["vtu"])
            c.run()
            vtu_path = frd_path.replace(".frd", ".vtu")
            print(f"  -> VTU file generated: {vtu_path}")
        except ImportError:
            print("  -> ccx2paraview not installed. Run: pip install ccx2paraview")
        except Exception as e:
            print(f"  -> ccx2paraview conversion error: {e}")

    return {
        "success": success,
        "elapsed_s": elapsed,
        "frd_path": frd_path if os.path.isfile(frd_path) else None,
        "vtu_path": vtu_path,
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:] if result.stdout else "",
        "stderr": result.stderr[-1000:] if result.stderr else "",
    }
