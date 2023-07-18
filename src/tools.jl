export save_sol, load_sol

function save_sol(fn::String, sol::ContSol)::Nothing
    tmp = sol_to_dict(sol)
    fid = h5open(fn, "w")
    ContGridMod.dict_to_hdf5(tmp, fid)
    close(fid)
    return nothing
end

function static_to_dict(sol::StaticSol)::Dict{String, <:Any}
    re = Dict{String, Any}()
    cm_dict = ContGridMod.cont_to_dict(sol.model)
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

function sol_to_dict(sol::ContSol)::Dict{String, <:Any}
    if typeof(sol) === StaticSol
        return static_to_dict(sol)
    end
end

function load_sol(fn::String)::ContSol
    fid = h5open(fn)
    data = ContGridMod.hdf5_to_dict(fid)
    close(fid)
    return dict_to_sol(data)
end

function dict_to_sol(data::Dict{String, <:Any})::ContSol
    if data["type"] == "StaticSol"
        return dict_to_static(data)
    end
end

function dict_to_static(data::Dict{String, <:Any})::StaticSol
    data["model"] = ContGridMod.cont_from_dict(data["model"])
    return StaticSol((data[string(key)] for key in fieldnames(StaticSol))...)
end
