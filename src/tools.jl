export save_sol, load_sol

"""
$(TYPEDSIGNATURES)

Save a solution to a HDF5 file.
"""
function save_sol(fn::String, sol::ContSol)::Nothing
    tmp = sol_to_dict(sol)
    fid = h5open(fn, "w")
    dict_to_hdf5(tmp, fid)
    close(fid)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Turn a static solution into a dictionary.
"""
function static_to_dict(sol::StaticSol)::Dict{String, <:Any}
    re = Dict{String, Any}()
    cm_dict = cont_to_dict(sol.model)
    re["model"] = cm_dict
    re["type"] = "StaticSol"
    for key in fieldnames(StaticSol)
        if key == :model
            continue
        end
        re[string(key)] = getfield(sol, key)
    end
    return re
end

"""
$(TYPEDSIGNATURES)

Turn a dynamic solution into a dictionary.
"""
function dynamic_to_dict(sol::DynamicSol)::Dict{String, <:Any}
    re = Dict{String, Any}()
    cm_dict = cont_to_dict(sol.model)
    re["model"] = cm_dict
    re["type"] = "DynamicSol"
    for key in fieldnames(DynamicSol)
        if key == :model
            continue
        end
        re[string(key)] = getfield(sol, key)
    end
    return re
end

"""
$(TYPEDSIGNATURES)

Turn a solution into a dict.
"""
function sol_to_dict(sol::ContSol)::Dict{String, <:Any}
    if typeof(sol) === StaticSol
        return static_to_dict(sol)
    elseif typeof(sol) === DynamicSol
        return dynamic_to_dict(sol)
    end
end

"""
$(TYPEDSIGNATURES)

Load a solution from a HDF5 file.
"""
function load_sol(fn::String)::ContSol
    fid = h5open(fn)
    data = hdf5_to_dict(fid)
    close(fid)
    return dict_to_sol(data)
end

"""
$(TYPEDSIGNATURES)

Load a solution from a dictionary.
"""
function dict_to_sol(data::Dict{String, <:Any})::ContSol
    if data["type"] == "StaticSol"
        return dict_to_static(data)
    elseif data["type"] == "DynamicSol"
        return dict_to_dynamic(data)
    end
end

"""
$(TYPEDSIGNATURES)

Load a static solution from a dictionary.
"""
function dict_to_static(data::Dict{String, <:Any})::StaticSol
    data["model"] = cont_from_dict(data["model"])
    return StaticSol((data[string(key)] for key in fieldnames(StaticSol))...)
end

"""
$(TYPEDSIGNATURES)

Load a dynamic solution from a dictionary.
"""
function dict_to_dynamic(data::Dict{String, <:Any})::DynamicSol
    data["model"] = cont_from_dict(data["model"])
    return DynamicSol((data[string(key)] for key in fieldnames(DynamicSol))...)
end






"""
    update_model!(model::ContModel, u_name::Symbol, u::Vector{<:Real})::Nothing

Update the continuos model from a vector of nodal values. It is sufficient to pass the symbol name without the _nodal suffix.
"""
function update_model!(model::ContModel, u_name::Symbol, u::Vector{<:Real})::Nothing
    if size(u) != (getnnodes(model.grid),)
        throw(DimensionMismatch("The size of the vector does not match the number of nodes."))
    end
    setproperty!(model, Symbol(string(u_name) * "_nodal"), u)
    setproperty!(model, u_name, (x; extrapolate=true, warn=:semi) -> interpolate(x, model.grid, model.dh₁, u, :u, extrapolate=extrapolate, warn=warn))
    return nothing
end

"""
    update_model(model::ContModel,u_name::Symbol, dm::DiscModel, tf::Real[, κ::Real=1.0, u_min::Real=0.1, σ::Real=0.01, bfactor::Real=1.0, bmin::Real=1000.0])::Nothing

Update the continuous model from a discrete model. The paramter are distributed using a diffusion procees (similar to the get_params function)
"""
function update_model!(
    model::ContModel,
    u_name::Symbol,
    dm::DiscModel,
    tf::Real;
    κ::Real=1.0,
    u_min::Real=0.1,
    σ::Real=0.01,
    bfactor::Real=1.0,
    bmin::Real=1000.0
)::Nothing

    ## This whole function and the get_params should be refactored ##
    area = integrate(model.dh₁, model.cellvalues, (x) -> 1)
    if u_name == :d
        function d₀(x, _)
            re = 0
            for i in 1:dm.Ngen
                if dm.p_gen == 0
                    continue
                end
                dif = x .- dm.coord[dm.id_gen[i], :]
                re += dm.d_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
            end
            for i in 1:dm.Nbus
                dif = x .- dm.coord[i, :]
                re += dm.d_load[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
            end
            return max(re, u_min)
        end
        d = diffusion(model.dh₁, model.cellvalues, model.grid, d₀, tf, κ)
        d = normalize_values!(d, sum(dm.d_load) + sum(dm.d_gen[dm.p_gen.>0]), area, model.grid, model.dh₁, model.cellvalues)
        update_model!(model, u_name, d)
    elseif u_name == :m
        function m₀(x, _)
            re = 0
            for i in 1:dm.Ngen
                if dm.p_gen[i] == 0
                    continue
                end
                dif = x .- dm.coord[dm.id_gen[i], :]
                re += dm.m_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
            end
            return max(re, u_min)
        end
        m = diffusion(model.dh₁, model.cellvalues, model.grid, m₀, tf, κ)
        m = normalize_values!(m, sum(dm.m_gen[dm.p_gen.>0]), area, model.grid, model.dh₁, model.cellvalues)
        update_model!(model, u_name, m)
    elseif u_name == :p
        function p₀(x, _)
            re = 0
            for i in 1:dm.Ngen
                dif = x .- dm.coord[dm.id_gen[i], :]
                re += dm.p_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
            end
            for i in 1:dm.Nbus
                dif = x .- dm.coord[i, :]
                re -= dm.p_load[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
            end
            return re
        end
        p = diffusion(model.dh₁, model.cellvalues, model.grid, p₀, tf, κ)
        p = normalize_values!(p, 0.0, area, model.grid, model.dh₁, model.cellvalues, mode= :off)
        update_model!(model, u_name, p)
    elseif u_name == :bx
        function bx₀(x, _)
            return bb(dm, x, σ, bfactor)[1]
        end
        bx = diffusion(model.dh₁, model.cellvalues, model.grid, bx₀, tf, κ)
        bx .= max.(bx, bmin)
        update_model!(model, u_name, bx)
    elseif u_name == :by
        function by₀(x, _)
            return bb(dm, x, σ, bfactor)[2]
        end
        by = diffusion(model.dh₁, model.cellvalues, model.grid, by₀, tf, κ)
        by .= max.(by, bmin)
        update_model!(model, u_name, by)
    end

    return nothing
end

"""
    albers_projection(coord::Array{<:Real,2}; # as latitude, longitude  lon0::Real=13.37616 / 180 * pi, lat0::Real=46.94653 / 180 * pi, lat1::Real=10 / 180 * pi, lat2::Real=50 / 180 * pi, R::Real=6371.0)::Array{<:Real,2}

Apply the Albers projection to a vector of coordinates. The coordinates need to be given as latitude, longitude.
See https://en.wikipedia.org/wiki/Albers_projection
"""
function albers_projection(
    coord::Array{<:Real,2};
    lon0::Real=13.37616 / 180 * pi,
    lat0::Real=46.94653 / 180 * pi,
    lat1::Real=10 / 180 * pi,
    lat2::Real=50 / 180 * pi,
    R::Real=6371.0
)::Array{<:Real,2}
    n = 1 / 2 * (sin(lat1) + sin(lat2))
    theta = n * (coord[:, 2] .- lon0)
    c = cos(lat1)^2 + 2 * n * sin(lat1)
    rho = R / n * sqrt.(c .- 2 * n * sin.(coord[:, 1]))
    rho0 = R / n * sqrt.(c - 2 * n * sin(lat0))
    x = rho .* sin.(theta)
    y = rho0 .- rho .* cos.(theta)
    return [x y]
end

"""
    import_border(filename::String)::Tuple{Array{2,<:Real},Real}

Import border from a json file, apply the Albers projection and rescale it such that the longest dimension is 1.
The coordinates need to be given as latitude, longitude.
"""
function import_border(
    filename::String
)::Tuple{Array{<:Real,2},<:Real}
    data = JSON.parsefile(filename)
    N = size(data["border"], 1)

    b = zeros(N, 2)
    for i = 1:N
        b[i, 1] = data["border"][i][1] / 180 * pi
        b[i, 2] = data["border"][i][2] / 180 * pi
    end

    if b[1, :] != b[end, :]
        b = vcat(b, reshape(b[1, :], 1, 2))
    end

    b = albers_projection(b)
    scale_factor = max(
        maximum(b[:, 1]) - minimum(b[:, 1]),
        maximum(b[:, 2]) - minimum(b[:, 2])
    )
    return b / scale_factor, scale_factor
end

"""
    to_jld2(fn::String, model::Union{ContModel,DiscModel})::nothing

Save a continuous or discrete model to a hdf5 file.
"""
function save_model(
    fn::String,
    model::GridModel
)::Nothing
    tmp = model_to_dict(model)
    fid = h5open(fn, "w")
    dict_to_hdf5(tmp, fid)
    close(fid)
end

"""
    from_jld2(fn::String)::Union{ContModel, DiscModel}

Load a continuous or discrete model from a hdf5 file.
"""
function load_model(fn::String)::GridModel
    tmp = h5open(fn)
    data = hdf5_to_dict(tmp)
    close(tmp)
    if data["model"] == "ContModel"
        return cont_from_dict(data)
    elseif data["model"] == "DiscModel"
        return DiscModel(
            (data[string(key)] for key in fieldnames(DiscModel))...
        )
    else
        throw(ArgumentError("The provided file does not have a model entry matching ContModel or DiscModel"))
    end
end

function hdf5_to_dict(fid::HDF5.H5DataStore)::Dict{String,<:Any}
    re = Dict{String,Any}()
    for k in keys(fid)
        if typeof(fid[k]) === HDF5.Dataset
            re[k] = read(fid[k])
        else
            re[k] = hdf5_to_dict(fid[k])
        end
    end
    return re
end

function dict_to_hdf5(data::Dict, fid::HDF5.H5DataStore)::Nothing
    for (k, i) in data
        if typeof(i) <: Dict
            g = create_group(fid, string(k))
            dict_to_hdf5(i, g)
        else
            fid[string(k)] = i
        end
    end
    return nothing
end

function cont_from_dict(data::Dict{String,<:Any})::ContModel
    cells = [Cell{data["grid"]["celltype"]...}(Tuple(x)) for x in eachrow(data["grid"]["cells"])]
    nodes = [Node(Ferrite.Vec(x...)) for x in eachrow(data["grid"]["nodes"])]
    grid = Grid(cells, nodes)
    dh₁ = DofHandler(grid)
    dh₂ = DofHandler(grid)
    add!(dh₁, :u, eval(Meta.parse(data["dh"]["ui"])))
    add!(dh₂, :θ, eval(Meta.parse(data["dh"]["ti"])))
    add!(dh₂, :ω, eval(Meta.parse(data["dh"]["oi"])))
    close!(dh₁)
    close!(dh₂)
    points = [Ferrite.Vec(x...) for x in eachrow(data["cellvalues"]["points"])]
    qr = eval(Meta.parse(data["cellvalues"]["type"]))(data["cellvalues"]["weights"], points)
    cv = CellScalarValues(qr, eval(Meta.parse(data["cellvalues"]["ip"])))
    db = Dirichlet(:u, Set(data["ch"]["slack"]), (x) -> 0)
    ch = ConstraintHandler(dh₁)
    add!(ch, db)
    close!(ch)
    return ContModel(
        grid,
        dh₁,
        dh₂,
        cv,
        data["values"]["area"],
        data["values"]["m"],
        data["values"]["d"],
        data["values"]["p"],
        data["values"]["bx"],
        data["values"]["by"],
        data["values"]["t"],
        data["values"]["fault"],
        ch
    )
end

function cont_to_dict(cm::ContModel)::Dict{String,<:Any}
    re = Dict{String,Any}()
    cells = reduce(vcat, [[x.nodes...]' for x in cm.grid.cells])
    nodes = reduce(vcat, [[x.x...]' for x in cm.grid.nodes])
    type = [typeof(cm.grid.cells[1]).parameters...]
    re["grid"] = Dict{String,Any}()
    re["grid"]["cells"] = cells
    re["grid"]["nodes"] = nodes
    re["grid"]["celltype"] = type

    ui = string(cm.dh₁.field_interpolations[1])
    ti = string(cm.dh₂.field_interpolations[1])
    oi = string(cm.dh₂.field_interpolations[2])
    re["dh"] = Dict{String,Any}()
    re["dh"]["ui"] = ui
    re["dh"]["ti"] = ti
    re["dh"]["oi"] = oi

    type = string(typeof(cm.cellvalues.qr))
    ip = string(cm.cellvalues.func_interp)
    weights = cm.cellvalues.qr.weights
    points = reduce(vcat, [[x.data...]' for x in cm.cellvalues.qr.points])
    re["cellvalues"] = Dict{String,Any}()
    re["cellvalues"]["type"] = type
    re["cellvalues"]["ip"] = ip
    re["cellvalues"]["points"] = points
    re["cellvalues"]["weights"] = weights

    slack = cm.ch.dbcs[1].local_face_dofs[1]
    re["ch"] = Dict{String,Any}()
    re["ch"]["slack"] = slack

    re["values"] = Dict{String,Any}()
    re["values"]["area"] = cm.area
    re["values"]["m"] = cm.m_nodal
    re["values"]["d"] = cm.d_nodal
    re["values"]["p"] = cm.p_nodal
    re["values"]["bx"] = cm.bx_nodal
    re["values"]["by"] = cm.by_nodal
    re["values"]["fault"] = cm.fault_nodal
    re["values"]["t"] = cm.θ₀_nodal

    re["model"] = "ContModel"

    return re
end

function disc_to_dict(dm::DiscModel)::Dict{String,<:Any}
    re = Dict(string(key) => getfield(dm, key) for key in fieldnames(DiscModel))
    re["model"] = "DiscModel"
    return re
end

function model_to_dict(model::GridModel)::Dict{String,<:Any}
    if typeof(model) === ContModel
        return cont_to_dict(model)
    else
        return disc_to_dict(model)
    end
end


