module ContGridModML

using LinearAlgebra, Ferrite, FerriteGmsh, FerriteViz, CairoMakie,
    Ferrite, Gmsh, SparseArrays, Flux, Random, HDF5, BlockArrays,
    StatsBase, OrdinaryDiffEq, JuMP, Gurobi, Ipopt, Base.Threads,
    DocStringExtensions, JSON3, PowerModels, Dates, DataFrames, CSV

const MODULE_FOLDER = pkgdir(@__MODULE__)
function __init__()
    try
        Gurobi.Env()
        global opti = Gurobi.Optimizer
    catch
        println("Gurobi does not work on your system. Falling back to IPOPT.")
        global opti = Ipopt.Optimizer
    end
end

abstract type GridModel end

mutable struct DiscModel{T <: Real} <: GridModel
    m_gen::Vector{T}
    d_gen::Vector{T}
    id_gen::Vector{<:Int}
    id_slack::Int
    coord::Matrix{T}
    d_load::Vector{T}
    id_line::Matrix{Int}
    b::Vector{T}
    p_load::Vector{T}
    th::Vector{T}
    p_gen::Vector{T}
    max_gen::Vector{T}
    Nbus::Int
    Ngen::Int
    Nline::Int
end

mutable struct ContModel{T <: Real} <: GridModel
    grid::Grid
    dh::DofHandler
    cellvalues::CellScalarValues
    area::Real
    m::Vector{T}
    d::Vector{T}
    p::Vector{T}
    bx::Vector{T}
    by::Vector{T}
    θ₀::Vector{T}
    fault::Vector{T}
    id_slack::Int
    disc_proj::SparseMatrixCSC{T, Int}
    q_proj::SparseMatrixCSC{T, Int}
end

abstract type ContSol end

"""
Contains the results of the static training.

$(TYPEDFIELDS)
"""
struct StaticSol{T <: Real} <: ContSol
    """
    Vector of the nodal susceptances. The ordering is alternating ``b_x`` and ``b_y``.
    The values are ordered in the same order as in the DoF handler `cm.dh₁`.
    """
    b::Vector{T}
    """
    The training losses. The row corresponds to the epoch and the column to the data set.
    """
    losses::Matrix{T}
    """
    Prediction of the values for each training data set with the trained values.
    """
    train_pred::Matrix{T}
    """
    Prediction of the values for each test data set with the trained values.
    """
    test_pred::Matrix{T}
    """
    Ground truth data for the training data sets.
    """
    t_train::Matrix{T}
    """ q_proj_b
    Ground truth data for the test data sets.
    """
    t_test::Matrix{T}
    """
    Loss values for all training data sets.
    """
    train_losses::Vector{T}
    """
    Loss values for all test data sets.
    """
    test_losses::Vector{T}
    """
    The continuous model with the updated susceptances.
    """
    model::ContModel
    """
    The discrete models used for training.
    """
    train_models::Vector{<:DiscModel}
    """
    The discrete models used for testing.
    """
    test_models::Vector{<:DiscModel}
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
include("disc_dynamics.jl")

end # module ContGridModML
