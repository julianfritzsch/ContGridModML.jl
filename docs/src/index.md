# Machine Learning Package for ContGridMod

Implementation of physics-informed machine learning routines for the continous power grid model package [`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl).

## Installation

ContGridModML can easily be installed using the Julia package manager.
```julia-repl
julia> using Pkg
julia> Pkg.add(url="https://github.com/laurentpagnier/ContGridMod.jl#FiniteElements")
julia> Pkg.add(url="https://github.com/julianfritzsch/ContGridModML.jl")
```
!!! note
    This package needs the `FiniteElements` branch of [`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl).
    This might conflict with preinstalled versions of [`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl).
    If there are any problems using this package, remove any versions of [`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl) checked out for development and try again.

## Quickstart Guide

To reproduce the results you only need to run two functions.
The static paramaters, *i.e.*, the susceptances in ``x`` and ``y`` direction (``b_x(\mathbf{r})`` and ``b_y(\mathbf{r})``), can be learned by running
```julia-repl
julia> sol = learn_susceptances()
```

Similarly, to learn the dynamical parameters, *i.e.*, the inertia ``m(\mathbf{r})`` and the damping ``d(\mathbf{r})``, run

```julia-repl
julia> sol = learn_dynamical_parameters()
```
