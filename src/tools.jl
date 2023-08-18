export save_sol, load_sol, load_discrete_model

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
    if typeof(sol) <: StaticSol
        return static_to_dict(sol)
    elseif typeof(sol) <: DynamicSol
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
    albers_projection(coord::Array{<:Real,2}; # as latitude, longitude  lon0::Real=13.37616 / 180 * pi, lat0::Real=46.94653 / 180 * pi, lat1::Real=10 / 180 * pi, lat2::Real=50 / 180 * pi, R::Real=6371.0)::Array{<:Real,2}

Apply the Albers projection to a vector of coordinates. The coordinates need to be given as latitude, longitude.
See https://en.wikipedia.org/wiki/Albers_projection
"""
function albers_projection(coord::Matrix{<:Real};
    lon0::Real = 13.37616 / 180 * pi,
    lat0::Real = 46.94653 / 180 * pi,
    lat1::Real = 10 / 180 * pi,
    lat2::Real = 50 / 180 * pi,
    R::Real = 6371.0)::Matrix{<:Real}
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
function import_border(filename::String)::Tuple{Matrix{<:Real}, <:Real}
    data = JSON3.read(filename)[:border]
    N = size(data["border"], 1)

    b = zeros(N, 2)
    for i in 1:N
        b[i, 1] = data[i][1] / 180 * pi
        b[i, 2] = data[i][2] / 180 * pi
    end

    if b[1, :] != b[end, :]
        b = vcat(b, reshape(b[1, :], 1, 2))
    end

    b = albers_projection(b)
    scale_factor = max(maximum(b[:, 1]) - minimum(b[:, 1]),
        maximum(b[:, 2]) - minimum(b[:, 2]))
    return b / scale_factor, scale_factor
end

"""
    to_jld2(fn::String, model::Union{ContModel,DiscModel})::nothing

Save a continuous or discrete model to a hdf5 file.
"""
function save_model(fn::String,
    model::GridModel)::Nothing
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
        return DiscModel((data[string(key)] for key in fieldnames(DiscModel))...)
    else
        throw(ArgumentError("The provided file does not have a model entry matching ContModel or DiscModel"))
    end
end

function hdf5_to_dict(fid::HDF5.H5DataStore)::Dict{String, <:Any}
    re = Dict{String, Any}()
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

function cont_from_dict(data::Dict{String, <:Any})::ContModel
    cells = [Cell{data["grid"]["celltype"]...}(Tuple(x))
             for x in eachrow(data["grid"]["cells"])]
    nodes = [Node(Ferrite.Vec(x...)) for x in eachrow(data["grid"]["nodes"])]
    grid = Grid(cells, nodes)
    dh₁ = DofHandler(grid)
    dh₂ = DofHandler(grid)
    add!(dh₁, :u, eval(Meta.parse(data["dh"]["ui"])))
    add!(dh₂, :θ, eval(Meta.parse(data["dh"]["ti"])))
    add!(dh₂, :ω, eval(Meta.parse(data["dh"]["oi"])))
    close!(dh₁)
    close!(dh₂)
    renumber!(dh₂, DofOrder.FieldWise())
    points = [Ferrite.Vec(x...) for x in eachrow(data["cellvalues"]["points"])]
    qr = eval(Meta.parse(data["cellvalues"]["type"]))(data["cellvalues"]["weights"], points)
    cv = CellScalarValues(qr, eval(Meta.parse(data["cellvalues"]["ip"])))
    q_proj = dict_to_sparse(data["matrices"]["q_proj"])
    disc_proj = dict_to_sparse(data["matrices"]["disc_proj"])
    return ContModel(grid,
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
        data["ch"]["slack"],
        disc_proj,
        q_proj)
end

function cont_to_dict(cm::ContModel)::Dict{String, <:Any}
    re = Dict{String, Any}()
    cells = reduce(vcat, [[x.nodes...]' for x in cm.grid.cells])
    nodes = reduce(vcat, [[x.x...]' for x in cm.grid.nodes])
    type = [typeof(cm.grid.cells[1]).parameters...]
    re["grid"] = Dict{String, Any}()
    re["grid"]["cells"] = cells
    re["grid"]["nodes"] = nodes
    re["grid"]["celltype"] = type

    ui = string(cm.dh₁.field_interpolations[1])
    ti = string(cm.dh₂.field_interpolations[1])
    oi = string(cm.dh₂.field_interpolations[2])
    re["dh"] = Dict{String, Any}()
    re["dh"]["ui"] = ui
    re["dh"]["ti"] = ti
    re["dh"]["oi"] = oi

    type = string(typeof(cm.cellvalues.qr))
    ip = string(cm.cellvalues.func_interp)
    weights = cm.cellvalues.qr.weights
    points = reduce(vcat, [[x.data...]' for x in cm.cellvalues.qr.points])
    re["cellvalues"] = Dict{String, Any}()
    re["cellvalues"]["type"] = type
    re["cellvalues"]["ip"] = ip
    re["cellvalues"]["points"] = points
    re["cellvalues"]["weights"] = weights

    slack = cm.id_slack
    re["ch"] = Dict{String, Any}()
    re["ch"]["slack"] = slack

    re["values"] = Dict{String, Any}()
    re["values"]["area"] = cm.area
    re["values"]["m"] = cm.m
    re["values"]["d"] = cm.d
    re["values"]["p"] = cm.p
    re["values"]["bx"] = cm.bx
    re["values"]["by"] = cm.by
    re["values"]["fault"] = cm.fault
    re["values"]["t"] = cm.θ₀

    re["matrices"] = Dict{String, Any}()
    re["matrices"]["q_proj"] = sparse_to_dict(cm.q_proj)
    re["matrices"]["disc_proj"] = sparse_to_dict(cm.disc_proj)

    re["model"] = "ContModel"

    return re
end

function sparse_to_dict(sm::SparseMatrixCSC)::Dict{String, <:Any}
    re = Dict(string(key) => getfield(sm, key) for key in fieldnames(SparseMatrixCSC))
    return re
end

function dict_to_sparse(dict::Dict{String, Any})::SparseMatrixCSC
    return SparseMatrixCSC([dict[string(key)] for key in fieldnames(SparseMatrixCSC)]...)
end

function disc_to_dict(dm::DiscModel)::Dict{String, <:Any}
    re = Dict(string(key) => getfield(dm, key) for key in fieldnames(DiscModel))
    re["model"] = "DiscModel"
    return re
end

function model_to_dict(model::GridModel)::Dict{String, <:Any}
    if typeof(model) <: ContModel
        return cont_to_dict(model)
    else
        return disc_to_dict(model)
    end
end

"""
    load_discrete_model(dataname::String, scaling_factor::Float64)::DiscModel

Load a discrete model from a file and rescale the coordinates to match the continuous model.
"""
function load_discrete_model(dataname::String,
    scaling_factor::Float64)::DiscModel
    if (contains(dataname, ".h5"))
        load_discrete_model_from_hdf5(dataname, scaling_factor)
    elseif (contains(dataname, ".json"))
        load_discrete_model_from_powermodels(dataname, scaling_factor)
    else
        error("Couldn't read $dataname")
    end
end

function load_discrete_model_from_hdf5(dataname::String,
    scaling_factor::Float64)::DiscModel
    data = h5read(dataname, "/")
    coord = albers_projection(data["bus_coord"] ./ (180 / pi))
    coord ./= scaling_factor
    dm = DiscModel(vec(data["gen_inertia"]),
        vec(data["gen_prim_ctrl"]),
        Int64.(vec(data["gen"][:, 1])),
        findall(vec(data["bus"][:, 2]) .== 3)[1],
        coord,
        vec(data["load_freq_coef"]),
        Int64.(data["branch"][:, 1:2]),
        1.0 ./ data["branch"][:, 4],
        vec(data["bus"][:, 3]) / 100.0,
        vec(data["bus"][:, 9]) / 180.0 * pi,
        vec(data["gen"][:, 2]) / 100.0,
        vec(data["gen"][:, 9]) / 100.0,
        size(data["bus"], 1),
        size(data["gen"], 1),
        size(data["branch"], 1))
    return dm
end

function load_discrete_model_from_json(dataname::String,
    project::Bool,
    scale_factor::Real)
    data = JSON3.read(dataname, Dict)
    return load_discrete_model_from_powermodels(data,
        project,
        scale_factor)
end

function load_discrete_model_from_powermodels(data::Dict{String, Any}, project::Bool,
    scale_factor::Real)::DiscModel
    bus_label = sort!([parse(Int, i) for i in keys(data["bus"])])
    bus_id = Dict{String, Int}(string(j) => i for (i, j) in enumerate(bus_label))
    n_bus = size(bus_label, 1)
    coord, va = zeros(n_bus, 2), zeros(n_bus)
    id_slack = 0

    for (i, bus) in data["bus"]
        ix = bus_id[i]
        if project
            coord[ix, :] = albers_projection(bus["coord"][[2, 1], :]' / 180 * π) /
                           scale_factor # This is ugly, we might want to change the albers projection argmuent order
        else
            coord[ix, :] = bus["coord"]
        end
        if bus["bus_type"] == 3
            id_slack = ix
        end
        va[ix] = bus["va"]
    end

    gen_label = sort!([parse(Int, i) for i in keys(data["gen"])])
    gen_id = Dict{String, Int}(string(j) => i for (i, j) in enumerate(gen_label))
    n_gen = size(gen_label, 1)
    inertia, gen_prim_ctrl, pmax, pg, id_gen = zeros(n_gen),
    zeros(n_gen),
    zeros(n_gen),
    zeros(n_gen),
    zeros(Int, n_gen)
    for (i, gen) in data["gen"]
        ix = gen_id[i]
        inertia[ix] = gen["inertia"]
        gen_prim_ctrl[ix] = gen["gen_prim_ctrl"]
        id_gen[ix] = bus_id[string(gen["gen_bus"])]
        pg[ix] = gen["pg"]
        pmax[ix] = gen["pmax"]
    end

    load_freq_coef, pl = zeros(n_bus), zeros(n_bus)
    for load in values(data["load"])
        ix = bus_id[string(load["load_bus"])]
        pl[ix] += load["pd"]
        load_freq_coef[ix] += load["load_freq_coef"]
    end

    n_branch = length(data["branch"])
    id_branch, susceptance = zeros(Int, n_branch, 2), zeros(n_branch)
    for (i, branch) in enumerate(values(data["branch"]))
        susceptance[i] = branch["br_x"] / (branch["br_x"]^2 + branch["br_r"]^2)
        id_branch[i, :] = [bus_id[string(branch["f_bus"])], bus_id[string(branch["t_bus"])]]
    end

    dm = DiscModel(inertia, gen_prim_ctrl, id_gen, id_slack, coord,
        load_freq_coef, id_branch, susceptance, pl, va, pg,
        pmax, n_bus, n_gen, n_branch)

    return dm
end

function remove_nan(grid::Dict{String, Any})
    for v in values(grid["gen"])
        isnan(v["pg"]) && (v["pg"] = 0)
        isnan(v["qg"]) && (v["qg"] = 0)
    end
    for v in values(grid["branch"])
        isnan(v["pf"]) && (v["pf"] = 0)
        isnan(v["pt"]) && (v["pt"] = 0)
        isnan(v["qf"]) && (v["qf"] = 0)
        isnan(v["qt"]) && (v["qt"] = 0)
    end
    return grid
end

function distribute_country_load(grid::Dict{String, Any}, country::Dict{String, <:Real})
    Sb = grid["baseMVA"]
    for key in keys(grid["load"])
        ctry = grid["bus"][string(grid["load"][key]["load_bus"])]["country"]
        coeff = grid["bus"][string(grid["load"][key]["load_bus"])]["load_prop"]
        grid["load"][key]["pd"] = country[ctry] * coeff / Sb
    end
    return grid
end

function opf_from_country(grid::Dict{String, Any}, country::Dict{String, <:Real})
    grid = distribute_country_load(grid, country)
    pm = instantiate_model(grid, DCMPPowerModel, build_opf)
    set_silent(pm.model)
    result = optimize_model!(pm, optimizer = Gurobi.Optimizer)
    if !(result["termination_status"] in [OPTIMAL LOCALLY_SOLVED ALMOST_OPTIMAL ALMOST_LOCALLY_SOLVED])
        throw(ErrorException("The OPF did not converge."))
    end
    update_data!(grid, result["solution"])
    return remove_nan(grid)
end

function stable_sol!(cm::ContModel)
    if cm.id_slack == 0
        println("Slack bus not set, setting it to 1")
        cm.id_slack = 1
    end
    Af, Ak, Islack = assemble_matrices_static(cm)
    q_proj = projectors_static_b(cm)
    b = zeros(2 * ndofs(cm.dh₁))
    b[1:2:end] = cm.bx
    b[2:2:end] = cm.by
    K = Ak * spdiagm(q_proj * b) * Ak' + Islack
    cm.θ₀[:] = K \ (Af * cm.q_proj * cm.p)
    return nothing
end
