module ContGridModML

using LinearAlgebra
using Ferrite
using FerriteGmsh
using FerriteViz
using CairoMakie
using Gmsh
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
using DocStringExtensions
using PowerModels
using JSON3
using Dates
using DataFrames
using CSV

const MODULE_FOLDER = pkgdir(@__MODULE__)

abstract type GridModel end

mutable struct DiscModel <: GridModel
    m_gen::Vector{<:Real}
    d_gen::Vector{<:Real}
    id_gen::Vector{<:Int}
    id_slack::Int
    coord::Matrix{<:Real}
    d_load::Vector{<:Real}
    id_line::Matrix{Int}
    b::Vector{<:Real}
    p_load::Vector{<:Real}
    th::Vector{<:Real}
    p_gen::Vector{<:Real}
    max_gen::Vector{<:Real}
    Nbus::Int
    Ngen::Int
    Nline::Int
end

mutable struct ContModel <: GridModel
    grid::Grid
    dh₁::DofHandler
    dh₂::DofHandler
    cellvalues::CellScalarValues
    area::Real
    m_nodal::Vector{<:Real}
    d_nodal::Vector{<:Real}
    p_nodal::Vector{<:Real}
    bx_nodal::Vector{<:Real}
    by_nodal::Vector{<:Real}
    θ₀_nodal::Vector{<:Real}
    fault_nodal::Vector{<:Real}
    m::Function
    d::Function
    p::Function
    bx::Function
    by::Function
    θ₀::Function
    fault::Function
    ch::ConstraintHandler
    function ContModel(grid::Grid,
        dh₁::DofHandler,
        dh₂::DofHandler,
        cellvalues::CellScalarValues,
        area::Real,
        m_nodal::Vector{<:Real},
        d_nodal::Vector{<:Real},
        p_nodal::Vector{<:Real},
        bx_nodal::Vector{<:Real},
        by_nodal::Vector{<:Real},
        θ₀_nodal::Vector{<:Real},
        fault_nodal::Vector{<:Real},
        ch::ConstraintHandler)
        new(grid,
            dh₁,
            dh₂,
            cellvalues,
            area,
            m_nodal,
            d_nodal,
            p_nodal,
            bx_nodal,
            by_nodal,
            θ₀_nodal,
            fault_nodal,
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                m_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                d_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                p_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                bx_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                by_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                θ₀_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            (x; extrapolate = true, warn = :semi) -> interpolate(x,
                grid,
                dh₁,
                fault_nodal,
                :u,
                extrapolate = extrapolate,
                warn = warn),
            ch)
    end
    function ContModel(; grid::Grid,
        dh₁::DofHandler,
        dh₂::DofHandler,
        cellvalues::CellScalarValues,
        area::Real,
        m_nodal::Vector{<:Real},
        d_nodal::Vector{<:Real},
        p_nodal::Vector{<:Real},
        bx_nodal::Vector{<:Real},
        by_nodal::Vector{<:Real},
        θ₀_nodal::Vector{<:Real},
        fault_nodal::Vector{<:Real},
        ch::ConstraintHandler)
        ContModel(grid,
            dh₁::DofHandler,
            dh₂,
            cellvalues,
            area,
            m_nodal,
            d_nodal,
            p_nodal,
            bx_nodal,
            by_nodal,
            θ₀_nodal,
            fault_nodal,
            ch)
    end
end

abstract type ContSol end

"""
Contains the results of the static training.

$(TYPEDFIELDS)
"""
struct StaticSol <: ContSol
    """
    Vector of the nodal susceptances. The ordering is alternating ``b_x`` and ``b_y``.
    The values are ordered in the same order as in the DoF handler `cm.dh₁`.
    """
    b::Vector{<:Real}
    """
    The training losses. The row corresponds to the epoch and the column to the data set.
    """
    losses::Array{<:Real, 2}
    """
    Prediction of the values for each training data set with the trained values.
    """
    train_pred::Array{<:Real, 2}
    """
    Prediction of the values for each test data set with the trained values.
    """
    test_pred::Array{<:Real, 2}
    """
    Ground truth data for the training data sets.
    """
    t_train::Array{<:Real, 2}
    """
    Ground truth data for the test data sets.
    """
    t_test::Array{<:Real, 2}
    """
    Loss values for all training data sets.
    """
    train_losses::Vector{<:Real}
    """
    Loss values for all test data sets.
    """
    test_losses::Vector{<:Real}
    """
    The continuous model with the updated susceptances.
    """
    model::ContModel
end

"""
Contains the results of the dynamic training.

$(TYPEDFIELDS)
"""
struct DynamicSol <: ContSol
    """
    Indices of the generators used to create the training ground truth data.
    """
    train_ix::Vector{<:Integer}
    """
    Indices of the generators used to create the training ground truth data.
    """
    test_ix::Vector{<:Integer}
    """
    Indices of the nodes on which the frequency is compared to calculate the loss function.
    """
    comp_ix::Vector{<:Integer}
    """
    Vector of the nodal inertia. The values are ordered in the same order as in the DoF
    handler `cm.dh₁`.
    """
    m::Vector{<:Real}
    """
    Vector of the nodal damping. The values are ordered in the same order as in the DoF
    handler `cm.dh₁`.
    """
    d::Vector{<:Real}
    """
    Coefficients of the modal expansion of the dynamical parameters. The realtion between
    the nodal values, the coefficients, and the eigenvectors is given by
    ```julia
    n = size(coeffs, 1) ÷ 2
    m = eve[:, 1:n] * coeffs[1:n]
    d = eve[:, 1:n] * coeffs[n+1:end]
    ```
    """
    coeffs::Vector{<:Real}
    """
    Eigenvectors of the unweighted Laplacian of the grid of the continuous model.
    """
    eve::Array{<:Real, 2}
    """
    The training losses. The row corresponds to the epoch and the column to the data set.
    """
    losses::Array{<:Real, 2}
    """
    Loss values for all training data sets.
    """
    train_losses::Vector{<:Real}
    """
    Loss values for all test data sets.
    """
    test_losses::Vector{<:Real}
    """
    The continuous model with updated inertia and damping.
    """
    model::ContModel
end

include("init.jl")
include("mesh.jl")
include("plot.jl")
include("static.jl")
include("tools.jl")
include("dynamic.jl")

end # module ContGridModML
