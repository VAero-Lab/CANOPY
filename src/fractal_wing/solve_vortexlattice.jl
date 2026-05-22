import Pkg
Pkg.activate(".")

using VortexLattice
using JSON
using WriteVTK
using StaticArrays

function main()
    input_file = "aero_input.json"
    output_file = "aero_loads.json"

    # Read inputs
    dict = JSON.parsefile(input_file)
    
    # Grid: list of (x, y, z) matrices per surface
    # Wait, the Python side will provide X, Y, Z arrays for the mesh.
    # Mesh has shape (nx, ny, 3)
    X_py = dict["X"]
    Y_py = dict["Y"]
    Z_py = dict["Z"]
    
    aoa = dict["aoa"]
    V = dict["V"]
    rho = dict["rho"]
    save_vtk = get(dict, "save_vtk", false)
    
    nx = length(X_py)
    ny = length(X_py[1])
    
    # Reconstruct the grid as a 3D Array{Float64, 3} of size (3, nx, ny)
    grid = Array{Float64, 3}(undef, 3, nx, ny)
    for i in 1:nx
        for j in 1:ny
            grid[1, i, j] = Float64(X_py[i][j])
            grid[2, i, j] = Float64(Y_py[i][j])
            grid[3, i, j] = Float64(Z_py[i][j])
        end
    end
    
    # Define freestream
    fs = Freestream(V, aoa * pi/180.0, 0.0, zeros(3))
    # Create the System pre-allocated structure from the grid
    system = System([grid])
    
    # Populate the ratios explicitly to bypass the uninitialized memory bug
    for i in 1:length(system.ratios)
        system.ratios[i] .= 0.0
        for c in 1:size(system.ratios[i], 2)
            for s in 1:size(system.ratios[i], 3)
                system.ratios[i][1, c, s] = 0.5
                system.ratios[i][2, c, s] = 0.75
            end
        end
    end
    
    # Solve steady analysis using the pre-allocated system
    ref = Reference(1.0, 1.0, 1.0, SVector(0.0, 0.0, 0.0), V)
    steady_analysis!(system, ref, fs; symmetric=fill(false, 1), derivatives=false)
    
    # Retrieve panels from system surfaces
    panels = system.surfaces[1]
    
    # Calculate forces manually from system properties
    props = system.properties[1]
    q = 0.5 * rho * V^2
    
    forces_out = []
    centroids_out = []
    
    n_cells = (nx-1) * (ny-1)
    circ_cell = Vector{Float64}(undef, n_cells)
    force_x = Vector{Float64}(undef, n_cells)
    force_y = Vector{Float64}(undef, n_cells)
    force_z = Vector{Float64}(undef, n_cells)
    force_mag = Vector{Float64}(undef, n_cells)
    
    cell_idx = 1
    for j in 1:(ny-1)
        for i in 1:(nx-1)
            # Centroid
            c = (grid[:, i, j] + grid[:, i+1, j] + grid[:, i, j+1] + grid[:, i+1, j+1]) / 4.0
            
            # Panel force coefficients
            p_prop = props[i, j]
            cf_panel = p_prop.cfb + p_prop.cfl + p_prop.cfr
            
            # Dimensional force: F = cf * q * S
            f = cf_panel * q * ref.S
            
            push!(centroids_out, [c[1], c[2], c[3]])
            push!(forces_out, [f[1], f[2], f[3]])
            
            circ_cell[cell_idx] = p_prop.gamma
            force_x[cell_idx] = f[1]
            force_y[cell_idx] = f[2]
            force_z[cell_idx] = f[3]
            force_mag[cell_idx] = sqrt(f[1]^2 + f[2]^2 + f[3]^2)
            
            cell_idx += 1
        end
    end
    
    # Output to JSON
    out_dict = Dict(
        "centroids" => centroids_out,
        "forces" => forces_out
    )
    
    open(output_file, "w") do f
        write(f, JSON.json(out_dict))
    end
    
    # Save VTK if requested
    if save_vtk
        vtk_prefix = get(dict, "vtk_prefix", "vlm_aero")
        # VortexLattice provides write_vtk(filename, system)
        write_vtk(vtk_prefix, system)
        
        # Save custom 2D panel surface VTU file
        n_pts = nx * ny
        pts = Matrix{Float64}(undef, 3, n_pts)
        li = LinearIndices((nx, ny))
        for j in 1:ny
            for i in 1:nx
                pts[:, li[i, j]] = grid[:, i, j]
            end
        end
        
        cells = MeshCell[]
        for j in 1:(ny-1)
            for i in 1:(nx-1)
                v1 = li[i, j]
                v2 = li[i+1, j]
                v3 = li[i+1, j+1]
                v4 = li[i, j+1]
                push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, [v1, v2, v3, v4]))
            end
        end
        
        panels_filename = string(vtk_prefix, "_panels")
        vtk_grid(panels_filename, pts, cells) do vtk
            vtk["circulation"] = circ_cell
            vtk["force_x"] = force_x
            vtk["force_y"] = force_y
            vtk["force_z"] = force_z
            vtk["force_magnitude"] = force_mag
        end
    end
    
    println("solve successful")
end

main()
