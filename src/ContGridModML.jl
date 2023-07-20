module ContGridModML

using ContGridMod
using Ferrite
using SparseArrays
using Flux
using Random
using HDF5
using BlockArrays
using StatsBase
using OrdinaryDiffEq
using JuMP
using Gurobi
using Base.Threads

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

struct DynamicSol <: ContSol
    test_ix::Vector{<:Integer}
    train_ix::Vector{<:Integer}
    comp_ix::Vector{<:Integer}
    m::Vector{<:Real}
    d::Vector{<:Real}
    coeffs::Vector{<:Real}
    eve::Array{<:Real,2}
    losses::Array{<:Real,2}
    train_losses::Vector{<:Real}
    test_losses::Vector{<:Real}
    model::ContGridMod.ContModel
end

include("static.jl")
include("tools.jl")
include("dynamic.jl")

end # module ContGridModML
