################################################################################
# solve_flowpanel.jl
#
# Standalone Julia solver bridge for FLOWPanel.jl.
# Reads structured grid coordinates from JSON, runs the 3D panel method solver,
# and writes centroids, forces, areas, and Cp values to an output JSON file.
#
# Usage:
#   julia --project=. solve_flowpanel.jl input.json output.json
#
# Author: Antigravity
################################################################################

import FLOWPanel as pnl
import JSON
using LinearAlgebra: norm

function run_solver(input_path::String, output_path::String)
    println("--- Julia FLOWPanel Solver Bridge ---")
    println("Loading input from: ", input_path)
    
    # 1. Load input JSON
    data = JSON.parsefile(input_path)
    X_matrix = data["X"]
    Y_matrix = data["Y"]
    Z_matrix = data["Z"]
    AOA = Float64(data["AoA"])
    magVinf = Float64(data["magVinf"])
    rho = Float64(data["rho"])
    debug_mode = Bool(data["debug"])

    n_span = length(X_matrix)
    n_chord = length(X_matrix[1])
    println("Grid size: ", n_span, " spanwise x ", n_chord, " chordwise points.")

    # Convert coordinates to Float64 matrices
    X = [Float64(X_matrix[i][j]) for i in 1:n_span, j in 1:n_chord]
    Y = [Float64(Y_matrix[i][j]) for i in 1:n_span, j in 1:n_chord]
    Z = [Float64(Z_matrix[i][j]) for i in 1:n_span, j in 1:n_chord]

    # 2. Build structured grid
    sec_NDIVS = n_chord - 1
    b_NDIVS = n_span - 1
    NDIVS = [sec_NDIVS, b_NDIVS, 0]
    P_min = [0.0, 0.0, 0.0]
    P_max = [1.0, 1.0, 0.0]
    loop_dim = 1
    
    grid = pnl.gt.Grid(P_min, P_max, NDIVS; loop_dim=loop_dim)

    # Apply 3D coordinate transform
    function my_space_transform(inds)
        i_chord = inds[1]
        i_span = inds[2]
        return [X[i_span, i_chord], Y[i_span, i_chord], Z[i_span, i_chord]]
    end
    pnl.gt.transform2!(grid, my_space_transform)

    # 3. Triangulate quadrilateral grid
    dimsplit = 1
    triang_grid = pnl.gt.GridTriangleSurface(grid, dimsplit)

    # Define shedding wake panels along the trailing edge loop
    ndivs = pnl.gt.get_ndivscells(triang_grid)
    lin_cells = LinearIndices((ndivs[1], ndivs[2]))
    U = [ lin_cells[ndivs[1]-1, i] for i in 1:ndivs[2] ]
    L = [ lin_cells[2, i] for i in 1:ndivs[2] ]

    nedges = length(U)
    shedding = zeros(Int, 6, nedges)
    for (ei, (u, l)) in enumerate(zip(U, L))
        shedding[1, ei] = u
        shedding[2, ei] = 3
        shedding[3, ei] = 2

        shedding[4, ei] = l
        shedding[5, ei] = 3
        shedding[6, ei] = 2
    end
    bodytype = pnl.RigidWakeBody{pnl.VortexRing}
    
    # 4. Determine normal vector orientation automatically
    # Check with default positive CPoffset
    body = bodytype(triang_grid, shedding; CPoffset=1e-14)
    normals = pnl.calc_normals(body)
    ctrl_pts = pnl.calc_controlpoints(body, normals)
    
    # Find topmost control point to verify its normal points upward (+Z)
    max_z_idx = 1
    max_z = -Inf
    for ci in 1:body.ncells
        if ctrl_pts[3, ci] > max_z
            max_z = ctrl_pts[3, ci]
            max_z_idx = ci
        end
    end
    
    if normals[3, max_z_idx] < 0.0
        println("Detected inward-pointing normals. Re-indexing body with CPoffset=-1e-14.")
        body = bodytype(triang_grid, shedding; CPoffset=-1e-14)
        normals = pnl.calc_normals(body)
        ctrl_pts = pnl.calc_controlpoints(body, normals)
    end

    # 5. Set up solver parameters
    Vinf = magVinf * [cos(AOA*pi/180), 0, sin(AOA*pi/180)]
    Uinfs = repeat(Vinf, 1, body.ncells)
    Das = repeat(Vinf/magVinf, 1, body.nsheddings)
    Dbs = repeat(Vinf/magVinf, 1, body.nsheddings)

    # Solve circulation
    println("Solving circulation...")
    pnl.solve(body, Uinfs, Das, Dbs)

    # 6. Post-process aerodynamic forces and Cp
    println("Post-processing aerodynamic fields...")
    Us = pnl.calcfield_U(body, body)
    pnl.calcfield_Ugradmu(body)
    pnl.addfields(body, "Ugradmu", "U")
    Cps = pnl.calcfield_Cp(body, magVinf)
    Fs = pnl.calcfield_F(body, magVinf, rho)
    areas = pnl.calc_areas(body)

    # 7. Serialize results to output JSON
    centroids_out = [ [ctrl_pts[1, i], ctrl_pts[2, i], ctrl_pts[3, i]] for i in 1:body.ncells ]
    forces_out = [ [Fs[1, i], Fs[2, i], Fs[3, i]] for i in 1:body.ncells ]
    
    results = Dict(
        "centroids" => centroids_out,
        "forces" => forces_out,
        "areas" => collect(areas),
        "Cps" => collect(Cps)
    )

    open(output_path, "w") do f
        JSON.print(f, results)
    end
    println("Saved panel results to: ", output_path)

    # Optionally save VTK for visualization
    if debug_mode
        vtk_path = pnl.save(body, "flowpanel_wing"; path=dirname(output_path))
        println("Saved VTK for debugging: ", vtk_path)
    end
    
    println("FLOWPanel solve finished successfully.")
end

# Check CLI arguments
if length(ARGS) < 2
    println("Error: Insufficient arguments.")
    println("Usage: julia --project=. solve_flowpanel.jl input.json output.json")
    exit(1)
else
    run_solver(ARGS[1], ARGS[2])
end
