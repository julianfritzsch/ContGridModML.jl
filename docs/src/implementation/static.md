# Implementation of the Susceptance Learning

The line susceptance are learned by comparing the static solution obtained from the continuous model with the solutions obtained from the power flow equations for the discrete case.
Learning is done by [`learn_susceptances`](@ref).
The process consists of the following steps

1. [Load the discrete models](@ref)
1. [Create all the necessary matrices](@ref)
1. [Create all projectors](@ref)
1. [Create the force vectors](@ref)
1. [Assemble the training and test data sets](@ref)
1. [Do the learning](@ref)
1. [Calculate the predictions](@ref)
1. [Calculate the loss values for the predictions](@ref)

### Load the discrete models

First the grid is loaded from a GMSH file.
After this we load all the discrete models used for the training and the test.
This is all done as described in [Creating Models](@ref).
Each single discrete model corresponds to one dispatch.

See: [`ContGridModML.discrete_models`](@ref)

### Create all the necessary matrices

#### Motivation

The matrices created are motivated as follows.
We begin with the stiffness matrix.
Recall that it is given by (dropping the transformation of the gradients as this taken care of by `Ferrite`)

```math
\mathbf{K}_{ij} = \sum_E\sum_k\omega_k \nabla\phi_i(\mathbf{q_k})
\begin{bmatrix}b_x(\mathbf{q}_k) & 0 \\ 0 & b_y(\mathbf{q}_k)\end{bmatrix}
\nabla\phi_j(\mathbf{q}_k)\det(J),
```

 Note that the sum over ``k`` can be written as

```math
 \begin{bmatrix}\nabla\phi_i(\mathbf{q_1})\\\nabla\phi_i(\mathbf{q_2})\\\vdots\end{bmatrix}^\top
\begin{bmatrix}b_x(\mathbf{q}_1) & 0 & 0 & 0 & \cdots\\ 0 & b_y(\mathbf{q}_1) & 0 & 0 & \cdots\\0 & 0 & b_x(\mathbf{q}_2) & 0 & \cdots \\ 0 & 0 & 0 & b_y(\mathbf{q}_2) & \cdots\\\vdots&\vdots&\vdots&\vdots&\ddots\end{bmatrix}
\begin{bmatrix}\nabla\phi_j(\mathbf{q_1})\\\nabla\phi_j(\mathbf{q_2})\\\vdots\end{bmatrix}
```

To eliminate the sum over ``E``, we can pull the determinant and quadrature weight into the gradient and label all quadrature points continuously.
Introducing the matrix ``\mathbf{A}`` which is made of ``1\times2`` blocks with

```math
\mathbf{A}_{ij} = (\nabla \phi_i(\mathbf{q}_{j}))^\top \sqrt{\omega_j\det(J_{j})},
```

where ``\omega_j`` and ``J_j`` are the weight and determinant that belong to quadrature point ``j``.
It is obvious that we can write the stiffness matrix as

```math
\mathbf{K} = \mathbf{A} \mathrm{diag}(b_x(\mathbf{q}_1), b_y(\mathbf{q}_1), \dots) \mathbf{A}^\top.
```

Therefore, the first matrix we need to create is ``\mathbf{A}``.
As we need the values of ``b`` in the quadrature points, we will collect the coordinates of the quadrature points in the process of creating ``\mathbf{A}``.
The next matrix we need is related to enforcing the value of the slack bus.
This means that at one node the solution to the static problem is fixed at 0.
This is achieved by setting the row and column of the stiffness matrix corresponding to that node to zero and adding a 1 on the diagonal for that node.
Additionally the corresponding entry of the force vector will be set to zero.
The next matrix therefore has only one entry: 1 on the diagonal corresponding to the slack node.
Finally, we need to create one more matrix.
Similar to the above, we want to write the force vector as

```math
\mathbf{f} = \mathbf{A}_f \begin{bmatrix} P(\mathbf{q}_1)\\ P(\mathbf{q}_2) \\ \vdots \end{bmatrix}.
```

The derivation is very similar to the above and we find

```math
\mathbf{A}_{f,ij} = \phi_i(\mathbf{q}_j) \omega_j \det(J_j).
```

#### Implementation

First the continuous model is loaded as described in [Creating Models](@ref).
We begin by creating allocating the required matrices, getting the number of base functions, and initializing the quadrature point iterators.

```julia
q_coords = zeros(getnquadpoints(model.cellvalues) * size(model.grid.cells, 1), 2)
Af = zeros(ndofs(model.dh₁),
    getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
Ak = zeros(ndofs(model.dh₁),
    2 * getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))

n_basefuncs = getnbasefunctions(model.cellvalues)
ix_af = 1
ix_ak = 1
```

Afterwards we loop over all cells using the [`CellIterator`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/dofhandler/#Ferrite.CellIterator).
For each cell the cell values need to be initialized so the structure contains the correct gradient values.
We also get the degrees of freedom that are associated with this cell.

```julia
for (ix, cell) in enumerate(CellIterator(model.dh₁))
    Ferrite.reinit!(model.cellvalues, cell)
    dofs = celldofs(cell)
```

Next, we loop over all the quadrature points in the cell.
First we get the coordinates of that point and the value of the product of the quadrature weight and the Jacobian determinant at this quadrature point.
The coordinates will be saved in the associated matrix.

```julia
    for q_point in 1:getnquadpoints(model.cellvalues)
        x = spatial_coordinate(model.cellvalues, q_point, getcoordinates(cell))
        dΩ = getdetJdV(model.cellvalues, q_point)
        q_coords[ix_af, :] = x
```

Next we iterate over the base functions associated with that cell (only base functions associated with this cell have a non-zero contribution in the quadrature points).
We can easily obtain the value and its gradient at the quadrature point from Ferrite and fill the matrix as described in the motivation section.

```julia
        for i in 1:n_basefuncs
            φᵢ = shape_value(model.cellvalues, q_point, i)
            ∇φᵢ = shape_gradient(model.cellvalues, q_point, i)
            Af[dofs[i], ix_af] = φᵢ * dΩ
            Ak[dofs[i], ix_ak:ix_ak+1] = ∇φᵢ * sqrt(dΩ)
        end
```

Finally, we need to increase the quadrature point iterators.

```julia
        ix_af += 1
        ix_ak += 2
    end
end
```

The last thing left to do is to enforce the slack node.
The correct degree of freedom can be straightforwardly obtained from the [`ConstraintHandler`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/boundary_conditions/#Ferrite.ConstraintHandler).

```julia
Ak[model.ch.prescribed_dofs, :] .= 0.0
Af[model.ch.prescribed_dofs, :] .= 0.0
dim = zeros(ndofs(model.dh₁), ndofs(model.dh₁))
dim[model.ch.prescribed_dofs, model.ch.prescribed_dofs] .= 1
```

See: [`ContGridModML.assemble_matrices_static`](@ref)

### Create all projectors

#### Motivation

While [`Ferrite`](https://ferrite-fem.github.io/Ferrite.jl/stable) has the option to obtain values at arbitrary point given a `DofHandler` and the nodal values doing this at every step of the training is time consuming.
Additionally, this is not compatible with the automatic differentiation of `Flux`.
However, we can easily turn the interpolation into an easy matrix multiplication.

Let's revisit the way that we can obtain values at arbitrary points using the finite element method.
In general a field  ``u(\mathbf{r})`` is expanded as follows

```math
u(\mathbf{r}) = \sum_i \phi_i(\mathbf{r}) \hat{u}_i,
```

where ``\phi_i`` are the base functions and ``\hat{u}_i`` are the nodal values.
As the position of the quadrature points and the discrete values in the grid does not change we can simply calculate ``\phi(\mathbf{r})`` for all these points and then multiply them by the nodal values.
As an example take the values of ``b_x`` and ``b_y`` in the first quadrature points they can be obtained by

```math
\begin{bmatrix}b_x(\mathbf{q}_1) \\ b_y(\mathbf{q}_2)\end{bmatrix} = 
\begin{bmatrix}\phi_1(\mathbf{q}_1) & 0 & \phi_2(\mathbf{q}_1) & 0 & \dots \\ 0 & \phi_1(\mathbf{q}_1) & 0 & \phi_2(\mathbf{q}_1) & \dots\end{bmatrix}
\begin{bmatrix}
\widehat{b_x}_{1} \\ \widehat{b_y}_{1} \\ \vdots
\end{bmatrix}
```

!!! info
    In general we wouldn't need to project values of ``P(\mathbf{r})`` but rather use the actual values.
    However, the distribution has been obtained through the heat equation diffusion and therefore we only have the values on the nodes.
    The `DofHandler` is the same as the one used for the steady state solution.

#### Implementation

We need to create three projection matrices: for the buses in the discrete model, for the sucsceptances onto the quadrature points, and for the power onto the quadrature points.
To demonstrate the creation of these matrices with `Ferrite`, we show the assembly of the projection matrix for the buses of the discrete model.

We start by allocating the memory for the matrix, getting the function interpolations used, and obtaining the coordinates of the grid nodes.

```julia
func_interpolations = Ferrite.get_func_interpolations(model.dh₁, :u)
grid_coords = [node.x for node in model.grid.nodes]
n_base_funcs = getnbasefunctions(model.cellvalues)
θ_proj = zeros(size(dm.th, 1), ndofs(model.dh₁))
```

Next, we loop over all the coordinates in the discrete model.
Using the [`PointEvalHandler`](https://ferrite-fem.github.io/Ferrite.jl/stable/reference/export/#Ferrite.PointEvalHandler) allows us to find the cell in which the coordinate is located.
Additionally, it yields the position of the coordinate within the cell in reference coordinates.
In case the coordinate is not inside the grid (for example, some buses are located on islands), we choose the closest node of the grid.

```julia
for (i, point) in enumerate(eachrow(dm.coord))
    ph = PointEvalHandler(model.grid, [Ferrite.Vec(point...)], warn = :false)
    if ph.cells[1] === nothing
        min_ix = argmin([norm(coord .- Ferrite.Vec(dm.coord[i, :]...))
                         for coord in grid_coords])
        ph = PointEvalHandler(model.grid, [grid_coords[min_ix]])
    end
```

Next, we obtain the values of the base functions at this point, get the degrees of freedom, and assemble the matrix.

```julia
    pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(model.dh₁, ph.cells[1]))
    Ferrite.celldofs!(cell_dofs, model.dh₁, ph.cells[1])
    for j in 1:n_base_funcs
        θ_proj[i, cell_dofs[j]] = pv.N[j]
    end
end
```

The assembly of the quadrature point projectors works analogously.
We do not need to check whether the points are inside the grid.
The pattern of the projector for the susceptances is obtained as follows

```julia
q_proj_b[2 * i - 1, 2 * cell_dofs[j] - 1] = pv.N[j]
q_proj_b[2 * i, 2 * cell_dofs[j]] = pv.N[j]
```

See: [`ContGridModML.projectors_static`](@ref)

### Create the force vectors

The creation of the force vectors is quite simple.
We update the model using the discrete models so that it contains the correct power data.
Then we use the projection matrix to obtain the values in the quadrature points and finally use the `Af` matrix to contain the force vector.

```julia
for i in 1:size(training, 1)
    update_model!(model, :p, training[i], tf, κ = κ, σ = σ)
    f_train[:, i] = Af * q_proj * model.p_nodal
end
```

See: [`ContGridModML.assemble_f_static`](@ref)

### Assemble the training and test data sets

We simply create a matrix that contains all the ground truth data, *i.e.*, all the steady state angles of the discrete models.
This allows us to train on all the data sets simultaneously.

```julia
for i in 1:size(training, 1)
    t_train[:, i] = training[i].th
end
```

See: [`ContGridModML.assemble_disc_theta`](@ref)

### Do the learning

Doing the actual learning is straightforward.
We use `Flux` to calculate the gradient and update the parameters.
At each step we assemble the stiffness matrix.
Then we obtain the nodal values of the steady state solution by solving the linear system ``\mathbf{K}\mathbf{\theta}_0 = \mathbf{f}`` and project them onto discrete bus locations.
The loss function is given by the Huber loss.
An epoch looks like

```julia
for batch in 1:n_batches
    local loss
    gs = Flux.gradient(param) do
        btemp = max.(b, bmin)
        K = A * sparse(1:n_q, 1:n_q, q_proj * btemp) * A' + dim
        θ = proj * (K \ f_train[:,
            shuffled_ix[((batch - 1) * batch_size + 1):(batch * batch_size)]])
        loss = Flux.huber_loss(θ,
            t_train[:,
                shuffled_ix[((batch - 1) * batch_size + 1):(batch * batch_size)]],
            delta = δ)
    end
    losses[e, batch] = loss
    if (mod(e, 50) == 0 && batch == 3)
        println(string(e) * ", " * string(mean(losses[e, :])))
    end
    Flux.update!(opt, param, gs)
end
```

### Calculate the predictions

The predictions are calculated like shown in the previous section.

See: [`ContGridModML.prediction`](@ref)

### Calculate the loss values for the predictions

The Huber loss for all predictions is calculated.

```julia
train_losses = vcat(Flux.huber_loss(train_pred,
    t_train,
    delta = δ,
    agg = x -> mean(x, dims = 1))...)
```

See: [`ContGridModML.get_losses`](@ref)
