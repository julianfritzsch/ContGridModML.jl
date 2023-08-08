export get_mesh

"""
    get_grid(filein::String, dx::Real[, fileout::String=""])::Tuple{Grid,Real}

Generate a grid using Gmsh from a json file containing the border coordinates.
The file can be saved to a file if fileout is specified.
"""
function get_mesh(
    filein::String,
    dx::Real;
    mesh_size_max::Real = 0.1,
    algo::Int = 7,
    fileout::String="")::Tuple{Grid,Real}

    border, scale_factor = import_border(filein)
    border = border[1:end-1, :]

    # Initialize gmsh
    Gmsh.initialize()

    # Add the points
    for i in eachindex(border[:, 1])
        gmsh.model.geo.add_point(border[i, :]..., 0, dx, i)
    end

    # Add the lines
    for i in 1:size(border, 1)-1
        gmsh.model.geo.add_line(i, i + 1, i)
    end
    gmsh.model.geo.add_line(size(border, 1), 1, size(border, 1))

    # Create the closed curve loop and the surface
    gmsh.model.geo.add_curve_loop(Vector(1:size(border, 1)), 1)
    gmsh.model.geo.add_plane_surface([1])

    # Synchronize the model
    gmsh.model.geo.synchronize()
    
    # Define algo and coarseness
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.model.mesh.set_algorithm(2, 1, algo) # dim = 2, assuming there's only one surface

    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh, and read back in as a Ferrite Grid
    if fileout == ""
        grid = mktempdir() do dir
            path = joinpath(dir, "temp.msh")
            gmsh.write(path)
            togrid(path)
        end
    else
        gmsh.write(fileout)
        open(fileout, "a") do f
            write(f, "\$BeginScaleFactor\n")
            write(f, string(scale_factor) * "\n")
            write(f, "\$EndScaleFactor\n")
        end
        grid = togrid(fileout)
    end

    Gmsh.finalize()

    return grid, scale_factor

end

"""
    get_grid(file::String)::Tuple{Grid, Real}

Load a grid from a gmsh file. The file needs to contain a field ScaleFactor.
"""
function get_mesh(
    file::String
)::Tuple{Grid,Real}
    grid = togrid(file)
    global scale_factor = nothing
    open(file) do f
        while !eof(f)
            s = readline(f)
            if s == "\$BeginScaleFactor"
                scale_factor = tryparse(Float64, readline(f))
                break
            end
        end
    end
    if isnothing(scale_factor)
        throw("scale_factor could not be read")
    end
    return grid, scale_factor
end
