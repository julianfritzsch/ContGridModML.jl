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
using DocStringExtensions

const MODULE_FOLDER = pkgdir(@__MODULE__)

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
    model::ContGridMod.ContModel
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
    model::ContGridMod.ContModel
end

include("static.jl")
include("tools.jl")
include("dynamic.jl")

end # module ContGridModML
