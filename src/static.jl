export learn_susceptances

"""
$(TYPEDSIGNATURES)

Load all the discrete models for the training and test data sets.
"""
function discrete_models(train_folder::String,
    test_folder::String,
    n_train::Integer,
    n_test::Integer,
    scale_factor::Real)::Tuple{Vector{DiscModel}, Vector{DiscModel}}
    training = load_discrete_models(train_folder, scale_factor)
    test = load_discrete_models(test_folder, scale_factor)
    return training, test
end

function load_discrete_models(foldername::String,
    scale_factor::Real)::Vector{DiscModel}
    dms = DiscModel[]
    for fn in joinpath.(foldername, readdir(foldername))
        push!(dms, load_discrete_model(fn, scale_factor))
    end
    return dms
end


"""
$(TYPEDSIGNATURES)

Check if all the slack buses in the training and test data sets are the same.
"""
function check_slack(dataset::Vector{DiscModel})::Bool
    unique(dataset .|> d -> d.id_slack) |> length == 1
end

"""
$(TYPEDSIGNATURES)

Assemble the force vectors for the static solutions.
"""
function assemble_f_static(model::ContModel,
    dataset::Vector{DiscModel},
    Af::SparseMatrixCSC,
    q_proj::SparseMatrixCSC;
    tf::Real = 0.05,
    κ::Real = 1.0,
    σ::Real = 0.01)::Matrix{<:Real}
    f = zeros(ndofs(model.dh₁), length(dataset))

    for (i, dm) in enumerate(dataset)
        update_model!(model, :p, dm, tf, κ = κ, σ = σ)
        f[:, i] = Af * q_proj * model.p_nodal
    end
    
    return f
end

"""
$(TYPEDSIGNATURES)

Assemble all the ground truth data into one matrix for the training and one for the test
sets.
"""
function assemble_disc_theta(dataset::Vector{DiscModel})::Matrix{<:Real}
    return reduce(hcat, dataset .|> d.th)
end

"""
$(TYPEDSIGNATURES)

Create all the necessary finite element matrices from a given model.

The returned matrices are
- `Af` the matrix needed to create the force vector as `f = Af * p_quad`
- `Ak` the matrix needed to create the stiffness matrix as `K = A * b_quad * A'`. The
    susceptances need to be ordered as
    ``b_x(\\mathbf{q_1}), b_y(\\mathbf{q_1}), b_x(\\mathbf{q_2}),\\dots``
- `dim` matrix with single entry for the slack bus to ensure well-posedness of the system
    of linear equations
- `q_coords` Coordinates of the quadrature points in the same order as stored in
    the DoF-handler
"""
function assemble_matrices_static(model::ContModel)::Tuple{
    SparseMatrixCSC,
    SparseMatrixCSC,
    SparseMatrixCSC,
    Array{<:Real, 2},
}
    q_coords = zeros(getnquadpoints(model.cellvalues) * size(model.grid.cells, 1), 2)
    Af = zeros(ndofs(model.dh₁),
        getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
    Ak = zeros(ndofs(model.dh₁),
        2 * getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))

    n_basefuncs = getnbasefunctions(model.cellvalues)
    ix_af = 1
    ix_ak = 1
    for cell in CellIterator(model.dh₁)
        Ferrite.reinit!(model.cellvalues, cell)
        dofs = celldofs(cell)
        for q_point in 1:getnquadpoints(model.cellvalues)
            x = spatial_coordinate(model.cellvalues, q_point, getcoordinates(cell))
            dΩ = getdetJdV(model.cellvalues, q_point)
            q_coords[ix_af, :] = x
            for i in 1:n_basefuncs
                φᵢ = shape_value(model.cellvalues, q_point, i)
                ∇φᵢ = shape_gradient(model.cellvalues, q_point, i)
                Af[dofs[i], ix_af] = φᵢ * dΩ
                Ak[dofs[i], ix_ak:(ix_ak + 1)] = ∇φᵢ * sqrt(dΩ)
            end
            ix_af += 1
            ix_ak += 2
        end
    end
    # Enforce slack bus
    Ak[model.ch.prescribed_dofs, :] .= 0.0
    Af[model.ch.prescribed_dofs, :] .= 0.0
    # Dim is the matrix containing only the one on the diagonal for the slack bus to make sure that the linear equations will have a unique solution
    dim = zeros(ndofs(model.dh₁), ndofs(model.dh₁))
    dim[model.ch.prescribed_dofs, model.ch.prescribed_dofs] .= 1
    return sparse(Af), sparse(Ak), sparse(dim), q_coords
end

"""
$(TYPEDSIGNATURES)

Projectors of nodal values onto the discrete values and the quadrature points.

The returned matrices are
- `θ_proj` project the nodal values onto the discrete nodes for comparison
- `q_proj` project the nodal values onto the quadrature points
- `q_proj_b` project the susceptances onto the quadrature points. The susceptances need to 
    be ordered as ``b_x(\\mathbf{r_1}), b_y(\\mathbf{r_1}), b_x(\\mathbf{r_2}),\\dots``
"""
function projectors_static(model::ContModel,
    dm::DiscModel,
    q_coords::Array{<:Real, 2})::Tuple{SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC}
    func_interpolations = Ferrite.get_func_interpolations(model.dh₁, :u)
    grid_coords = [node.x for node in model.grid.nodes]
    n_base_funcs = getnbasefunctions(model.cellvalues)
    θ_proj = zeros(size(dm.th, 1), ndofs(model.dh₁))
    q_proj = zeros(size(q_coords, 1), ndofs(model.dh₁))
    q_proj_b = zeros(2 * size(q_coords, 1), 2 * ndofs(model.dh₁))

    for (i, point) in enumerate(eachrow(dm.coord))
        ph = PointEvalHandler(model.grid, [Ferrite.Vec(point...)], warn = :false)
        # If no cell is found (the point is outside the grid), use the closest grid point instead
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- Ferrite.Vec(point...))
                             for coord in grid_coords])
            ph = PointEvalHandler(model.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(model.dh₁, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, model.dh₁, ph.cells[1])
        for j in 1:n_base_funcs
            θ_proj[i, cell_dofs[j]] = pv.N[j]
        end
    end

    for (i, point) in enumerate(eachrow(q_coords))
        ph = PointEvalHandler(model.grid, [Ferrite.Vec(point...)])
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(model.dh₁, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, model.dh₁, ph.cells[1])
        for j in 1:n_base_funcs
            q_proj[i, cell_dofs[j]] = pv.N[j]
            q_proj_b[2 * i - 1, 2 * cell_dofs[j] - 1] = pv.N[j]
            q_proj_b[2 * i, 2 * cell_dofs[j]] = pv.N[j]
        end
    end
    return sparse(θ_proj), sparse(q_proj), sparse(q_proj_b)
end

"""
$(TYPEDSIGNATURES)

Do the actual learning of the parameters.
"""
function learn_susceptances(A::AbstractSparseMatrix,
    dim::AbstractSparseMatrix,
    q_proj::AbstractSparseMatrix,
    proj::AbstractSparseMatrix,
    f_train::Matrix{T},
    t_train::Matrix{T},
    b::Vector{T},
    n_epochs::Integer,
    n_batches::Int;
    opt = ADAM(0.1),
    bmin::Real = 0.1,
    rng::AbstractRNG = Xoshiro(123),
    δ::Real = 1.0
)::Tuple{Vector{T}, Matrix{T}} where T<:Real
    
    param = Flux.params(b)
    n_train = size(f_train, 2)
    @assert mod(n_train, n_batches) == 0 "The number of batches must be a divisor of the number of training cases."
    batch_size = Int(n_train / n_batches)
    
    shuffled_ix = randperm(rng, n_train)
    losses = zeros(n_epochs, n_batches)
    n_q = size(q_proj, 1)
    for e in 1:n_epochs
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
            if (mod(e, 50) == 0 && batch == 1)
                println(string(e) * ", " * string(mean(losses[e, :])))
            end
            Flux.update!(opt, param, gs)
        end
    end
    return max.(b, bmin), losses
end

"""
$(TYPEDSIGNATURES)

Learn the line susceptances.

The parameters are learned by calculating the stable solution for multiple dispatches and
comparing them to the results from the discrete model. The comparison points are obtained
by using the liear approximation provided by the finite element method.

# Arguments
- `train_fn::String = MODULE_FOLDER * "/data/ml/training_`": The names of the files
    containing the training scenarios. The files must be labeled `train_fn1.h5`,
    `train_fn2.h5`, etc.
- `test_fn::String = MODULE_FOLDER * "/data/ml/test_"`: The names of the files
    containing the test scenarios. The files must be labeled `test.h5`, `test.h5`, etc.
- `grid_fn::String = MODULE_FOLDER * "/data/panta.msh"`: Name of the file containing
  the mesh
- `n_train::Int = 48`: Number of training data sets
- `n_test::Int = 12`: Number of test data sets
- `n_epochs::Int = 10000`: Number of epochs
- `n_batches::Int = 3`: Number of batches per epoch
- `tf::Real = 0.05`: Duration of the heat equation diffusion for the power distribution 
- `κ::Real = 0.02`: Diffusion constant of the heat equation diffusion for the power
    distribution
- `σ::Real = 0.01`: Standard deviation for the initial Gaussian distribution of the
    parameters
- `rng::AbstractRNG = Xoshiro()`: Random number generator used to draw all random numbers
- `bmin::Real = 0.1`: Minimimum value of the suscpetances
- `δ = 0.5`: Parameter of the Huber loss function
"""
function run_learn_susceptances(;
    train_folder::String = MODULE_FOLDER * "/data/ml/train",
    test_folder::String = MODULE_FOLDER * "/data/ml/test",
    mesh_fn::String = MODULE_FOLDER * "/data/panta.msh",
    n_epochs::Int = 10000,
    n_batches::Int = 3,
    tf::Real = 0.05,
    κ::Real = 0.02,
    σ::Real = 0.01,
    rng::AbstractRNG = Xoshiro(123),
    opt = ADAM(0.1),
    bmin::Real = 0.1,
    δ = 0.5
)::StaticSol
    
    mesh, scale_factor = get_mesh(mesh_fn)
    train = load_discrete_models(train_folder, scale_factor)
    test = load_discrete_models(test_folder, scale_factor)
    @assert check_slack([train; test]) "The slack bus must be the same for all scenarios"
    model = init_model(mesh, tf, train[1], κ = κ, σ = σ)
    Af, Ak, dim, q_coords = assemble_matrices_static(model)
    
    θ_proj, q_proj, q_proj_b = projectors_static(model, train[1], q_coords)
    
    f_train = assemble_f_static(model, train, Af, q_proj, tf = tf, σ = σ, κ  = κ)
    f_test = assemble_f_static(model, test, Af, q_proj, tf = tf, σ = σ, κ  = κ)
        
    t_train, t_test = assemble_disc_theta(train, test)
    
    binit = 20 * rand(rng, 2 * ndofs(model.dh₁)) .+ 90
    
    b, losses = learn_susceptances(Ak, dim, q_proj_b, θ_proj, f_train,
        t_train, binit, n_epochs, n_batches, bmin = bmin, rng = rng)
        
    K = Ak * spdiagm(q_proj_b * b) * Ak' + dim
    
    train_pred, test_pred = prediction(K, f_train, f_test, θ_proj)
    #train_losses, test_losses = get_losses(train_pred, test_pred, t_train, t_test, δ = δ)
    train_losses, test_losses = zeros(2), zeros(2)
    
    update_model!(model, :bx, b[1:2:end])
    update_model!(model, :by, b[2:2:end])
    
    return StaticSol(b, losses, train_pred, test_pred, t_train, t_test,
        train_losses, test_losses, model)
end

"""
$(TYPEDSIGNATURES)

Obtain the prediction of the stable solution for the training and test data sets.
"""
function prediction(K::AbstractSparseMatrix,
    f_train::Matrix{<:Real},
    f_test::Matrix{<:Real},
    proj::AbstractSparseMatrix)::Tuple{Matrix{<:Real}, Matrix{<:Real}}
    #return proj * (K \ f_train), proj * (K \ f_test)
    return (K \ f_train), (K \ f_test)
end

"""
$(TYPEDSIGNATURES)

Obtain the loss values for the training and test data sets.
"""
function get_losses(train_pred::Matrix{<:Real},
    test_pred::Matrix{<:Real},
    t_train::Matrix{<:Real},
    t_test::Matrix{<:Real};
    δ::Real = 1.0)::Tuple{Vector{<:Real}, Vector{<:Real}}
    train_losses = vcat(Flux.huber_loss(train_pred,
        t_train,
        delta = δ,
        agg = x -> mean(x, dims = 1))...)
    test_losses = vcat(Flux.huber_loss(test_pred,
        t_test,
        delta = δ,
        agg = x -> mean(x, dims = 1))...)
    return train_losses, test_losses
end
