# Notes on the General Implementation

In the following there are some general notes on the implementation of the continuous model using [`Ferrite`](https://ferrite-fem.github.io/Ferrite.jl/) and [`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl).

## Ferrite

In general [`Ferrite`](https://ferrite-fem.github.io/Ferrite.jl/) takes care of most things when it comes to keeping track of the degrees of freedom (nodal values) as well as calculating all the needed values of the base functions and its gradients.
Additionally, it makes assembling the mass and stiffness matrix, and the force vector easy and allows finding the cells in which arbitrary points are located.
The central element is the [`DofHandler`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/dofhandler/#Ferrite.DofHandler).
It keeps track of the different fields (*e.g.* ``\theta`` and ``\omega``), contains the grid, and the order of the degrees of freedom.
!!! danger
    In general the degrees of freedom do NOT have the same order as the nodes of the grid.
    Ferrite loops over the cells in the grid and adds the degrees of freedom in the order it encounters them.
    For example if the first cell is made of nodes 40, 45, and 50, they will correspond to the degrees of freedom 1, 2, and 3.
    This is very important to keep in mind when trying to create matrices or to map result vectors onto the grid.

Another important structure is the [`CellValues`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/fevalues/#Ferrite.CellValues).
It contains the values of the base functions in the quadrature points as well as their gradients.
The gradient values are automatically transformed to their proper values from their reference values.
It also contains the product of the determinant of the Jacobian with the quadrature weight.

The final important structure is the [`ContraintHandler`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/boundary_conditions/#Ferrite.ConstraintHandler).
It allows us to enfoce Dirichlet boundary conditions whithout having to care about the degree of freedom ordering.
It is used to enforce the slack bus and to create the initial condition vector for the dynamic simulations.

To learn more about how [`Ferrite`](https://ferrite-fem.github.io/Ferrite.jl/) can be used to assemble the different matrices, check out the [Documentation](https://ferrite-fem.github.io/Ferrite.jl/).

## ContGridMod

[`ContGridMod`](https://github.com/laurentpagnier/ContGridMod.jl) offers data structures for continious and discrete models as well as some utilities.
The discrete model is used for the ground truth model.
Its dyanmics are governed by

```math
\begin{align*}
m_i \ddot{\theta}_i + d_i \dot{\theta}_i = P_i - \sum_j b_{ij}\sin(\theta_i - \theta_j), &\quad\text{for } i \in \text{generator},\\
d_i \dot{\theta}_i = P_i - \sum_j b_{ij}\sin(\theta_i - \theta_j),&\quad\text{for } i \in \text{load }.\\
\end{align*}
```

### Structures

#### Discrete Model

The `DiscModel` struct provides the following fields:

- `m_gen` inertia of each generator
- `d_gen` damping of each generator
- `id_gen` IDs of each generator
- `id_slack` ID of the slack bus
- `coord` the coordinates of each bus. They are ordered as (longitude, latitude). The coordinates have been transformed using an Albers Projection and afterwards scaled
    by the same factor as the continuous grid such that the longest dimension has length 1.
- `d_load` damping of each load buses
- `id_line` matrix containing the lines. They are sorted as (start, end)
- `b` line suscepetances
- `p_load` consumption of each node
- `th` steady state solution
- `p_gen` power generation of each generator. If the `p_gen` = 0. The generator is inactive.
- `max_gen` maximum generation of each generator
- `Nbus` number of buses
- `Ngen` number of generators
- `Nline` number of lines

#### Continuous Model

The `ContModel` struct provides the following fields:

- `grid` GMSH grid of the continuous model
- `dh₁` DofHandler containing one field `:u`. Used for the stable solution and to obtain field values.
- `dh₂` DofHandler containing two fields `:θ` and `:ω`. Used for the dynamical simulations.
- `cellvalues` CellValues object for the two DofHandlers
- `area` area of the grid used for normalization
- `m_nodal` nodal values for the inertia
- `d_nodal` nodal values for the damping
- `p_nodal` nodal values for the power
- `bx_nodal` nodal values for the ``x`` component of the susceptance
- `by_nodal` nodal values for the ``y`` component of the susceptance
- `θ₀_nodal` nodal values for the steady state solution
- `fault_nodal` nodal values for the power fault
- `m` function returning inertia at point `x`
- `d` function returning damping at point `x`
- `p` function returning power at point `x`
- `bx` function returning ``x`` component of the susceptance at point `x`
- `by` function returning ``y`` component of the susceptance at point `x`
- `θ₀` function returning steady state solution at point `x`
- `fault` function returning the power fault at point `x`
- `ch` ConstraintHandler to enforce the slack bus

#### Creating Models

To create the models from scratch, we first need to load a grid.
The grid can be created from a polygon.
The corners of the polygon need to be placed in a JSON file with the following structure

```json
{
    "border": [[x0, y0],
                [x1, y1],
                ...
              ]
}
```

The file is then loaded, the Albers projection applied, rescaled, and turned into a GMSH grid as follows (`dx` is the mesh element size)

```julia-repl
julia> using ContGridMod
julia> grid, scale_factor = get_grid(file_name, dx)
```

Alternatively, one can also directly load an existing GMSH file with the same command.

Next the discrete model can be loaded from a HDF5 file.
The data in the HDF5 file needs to contain the grid in the Matpower Data format with the additional data fields defined in the original [PanTaGruEl model](https://zenodo.org/record/2642175).
If you want to know more you can investigate the included scenarios in the `data/ml` folder of this repository.

```julia-repl
julia> dm = load_discrete_model(file_name, scale_factor)
```

Finally, the continuous model can be created using the grid and the discrete model.
The parameters are distributed using a heat equation diffusion.

```julia-repl
julia> cm = get_params(grid, 0.05, dm, κ=0.02, bfactor=50000.0, σ=0.01, bmin=1)
```

Faults in the discrete model can be simulated by

```julia-repl
julia> sol = disc_dynamics(dm, t0, tf, dP, faultid=id)
```

For faults in the continuous model we first need to calculate the steady state solution, then create the fault, and finally run the simulation
```julia-repl
julia> stable_sol!(cm)
julia> add_local_disturbance!(cm, coords, dP, σ)
julia> sol = perform_dyn_sim(cm, t0, tf)
```

The models can easily be saved as HDF5 file and loaded
```julia-repl
julia> save_model(file_name, model)
julia> model = load_model(file_name)
```
