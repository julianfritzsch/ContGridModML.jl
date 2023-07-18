module ContGridModML

using ContGridMod
using Ferrite
using SparseArrays
using Flux
using Random
using HDF5

const MODULE_FOLDER = pkgdir(@__MODULE__)

abstract type ContSol end

struct StaticSol <: ContSol
    b::Vector{<:Real}
    losses::Array{<:Real,2}
    train_pred::Array{<:Real,2}
    test_pred::Array{<:Real,2}
    t_train::Array{<:Real,2}
    t_test::Array{<:Real,2}
    train_losses::Vector{<:Real}
    test_losses::Vector{<:Real}
    model::ContGridMod.ContModel
end

include("static.jl")
include("tools.jl")

end # module ContGridModML
