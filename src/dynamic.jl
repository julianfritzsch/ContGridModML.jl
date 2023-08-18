export learn_dynamical_parameters

"""
$(TYPEDSIGNATURES)

Create all the necessary finite element matrices from a given model.

The matrices returned are
- `M_const` the constant (non-parameter dependent) part of the mass matrix
- `K_const` the constant (non-parameter dependent) part of the stiffness matrix
- `A` the matrix needed to create the non-constant part of both the mass and stiffness
    matrices. For example the mass matrix can be calculated as 
    `M = M_const + A * m_quad * A'`
- `Af` the matrix needed to create the force vector as `f = Af * (p_quad + fault_quad)`
"""
function assemble_matrices_dynamic(model::ContModel)::Tuple{
    SparseMatrixCSC,
    SparseMatrixCSC,
    SparseMatrixCSC,
    SparseMatrixCSC,
}
    ndof = ndofs(model.dh₁)
    K_const = spzeros(2 * ndof, 2 * ndof)
    M_const = spzeros(2 * ndof, 2 * ndof)
    A = spzeros(ndofs(model.dh₂),
        getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
    Af = spzeros(ndofs(model.dh₂),
        getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
    n_basefuncs_θ = getnbasefunctions(model.cellvalues)
    n_basefuncs_ω = getnbasefunctions(model.cellvalues)
    bx = model.q_proj * model.bx
    by = model.q_proj * model.by

    q_ix = 1
    for cell in CellIterator(model.dh₁)
        Ferrite.reinit!(model.cellvalues, cell)

        dofs = celldofs(cell)
        for q_point in 1:getnquadpoints(model.cellvalues)
            b = spdiagm([bx[q_ix], by[q_ix]])
            dΩ = getdetJdV(model.cellvalues, q_point)
            for i in 1:n_basefuncs_θ
                φᵢ = shape_value(model.cellvalues, q_point, i)
                ∇φᵢ = shape_gradient(model.cellvalues, q_point, i)
                A[dofs[i] + ndof, q_ix] = φᵢ * sqrt(dΩ)
                Af[dofs[i] + ndof, q_ix] = φᵢ * dΩ
                for j in 1:n_basefuncs_ω
                    φⱼ = shape_value(model.cellvalues, q_point, j)
                    ∇φⱼ = shape_gradient(model.cellvalues, q_point, j)
                    K_const[dofs[i], dofs[j] + ndof] += φᵢ ⋅ φⱼ * dΩ
                    K_const[dofs[i] + ndof, dofs[j]] -= ∇φᵢ ⋅ (b * ∇φⱼ) * dΩ
                    M_const[dofs[i], dofs[j]] += φᵢ ⋅ φⱼ * dΩ
                end
            end
            q_ix += 1
        end
    end
    return M_const, K_const, A, Af
end

"""
$(TYPEDSIGNATURES)

Create projectors for nodal values onto quadrature points and onto comparison locations.

Returned projectors
- `q_proj` Project the nodal values onto the quadrature points
- `ω_proj` Project the nodal values onto the given comparison points. This is used to
    calculate the loss function.
"""
function projectors_dynamic(cm::ContModel,
    dm::DiscModel,
    ω_idxs::Vector{<:Integer})::SparseMatrixCSC
    func_interpolations = Ferrite.get_func_interpolations(cm.dh₁, :u)
    grid_coords = [node.x for node in cm.grid.nodes]
    n_base_funcs = getnbasefunctions(cm.cellvalues)

    ndof = ndofs(cm.dh₁)
    ω_proj = spzeros(size(ω_idxs, 1), 2 * ndof)
    for (i, point) in enumerate(eachrow(dm.coord[ω_idxs, :]))
        ph = PointEvalHandler(cm.grid, [Ferrite.Vec(point...)], warn = :false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- Ferrite.Vec(point...))
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
        cell_dofs = Vector{Int64}(undef, ndofs_per_cell(cm.dh₂, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh₂, ph.cells[1])
        for j in 1:n_base_funcs
            ω_proj[i, cell_dofs[j] + ndof] = pv.N[j]
        end
    end
    return ω_proj
end

"""
$(TYPEDSIGNATURES)

Assemble all the force vectors for the dynamical simulations.
"""
function assemble_f_dynamic(cm::ContModel,
    dm::DiscModel,
    fault_ix::Vector{<:Integer},
    dP::Union{Real, Vector{<:Real}},
    Af::SparseMatrixCSC)::Matrix{<:Real}
    @assert size(fault_ix) == size(dP)||isa(dP, Real) "The size of `fault_ix` and `dP` must match"
    if isa(dP, Real)
        dP = dP .* ones(size(fault_ix, 1))
    end
    f_old = cm.fault
    f = zeros(ndofs(cm.dh₂), size(fault_ix, 1))
    for (i, (ix, p)) in enumerate(zip(fault_ix, dP))
        set_local_disturbance!(cm, dm.coord[ix, :], p)
        f[:, i] = Af * cm.q_proj * (cm.p .+ cm.fault)
    end
    cm.fault = f_old
    return f
end

"""
$(TYPEDSIGNATURES)

Generate a list of indices used for the comparison of the ground truth with the results
    from the continuous model.

An equally spaced grid is overlaid over the area. If the points are within the area, the
closest bus in the discrete model is found and added to the list of indices. The generators
of the test and training set are not eligible for comparison and are remove from the list of
possible indices. The number of points can be controlled by `n`, which gives the number of
points in the largest dimension.
"""
function generate_comp_idxs(cm::ContModel,
    dm::DiscModel,
    tri::Vector{<:Integer},
    tei::Vector{<:Integer},
    n::Int)::Vector{<:Integer}
    grid_vals = Vector{Real}[]
    x_min, x_max = extrema(dm.coord[:, 1])
    y_min, y_max = extrema(dm.coord[:, 2])
    dx = max((x_max - x_min) / n, (y_max - y_min) / n)
    for i in x_min:dx:x_max
        for j in y_min:dx:y_max
            ph = PointEvalHandler(cm.dh₁.grid,
                [Ferrite.Vec(i, j)],
                warn = false)
            if ph.cells[1] !== nothing
                push!(grid_vals, [i; j])
            end
        end
    end
    grid_vals = reduce(vcat, grid_vals')
    idxs = Set()
    pidxs = Int.(sort(collect(setdiff(Set(1:3809), union(Set(tri), Set(tei))))))
    for point in eachrow(grid_vals)
        push!(idxs, argmin(map(norm, eachslice(dm.coord[pidxs, :]' .- point, dims = 2))))
    end
    return pidxs[Int.(sort(collect(idxs)))]
end

"""
$(TYPEDSIGNATURES)

Randomly choose generators for the training and test sets.
"""
function gen_idxs(dm::DiscModel,
    dP::Real,
    n_train::Integer,
    n_test::Int;
    seed::Union{Nothing, Integer} = nothing)::Tuple{Vector{<:Integer}, Vector{<:Integer}}
    rng = Xoshiro(seed)
    idg = dm.id_gen[dm.p_gen .>= abs(dP)]
    tri = sort(sample(rng, idg, n_train, replace = false))
    tei = sort(sample(rng,
        Int64.(collect(setdiff(Set(idg), Set(tri)))),
        n_test,
        replace = false))
    return tri, tei
end

"""
$(TYPEDSIGNATURES)

Run a dynamical simulation of the discrete model.
"""
function disc_dyn(dm::DiscModel,
    fault_node::Integer,
    fault_size::Real,
    dt::Real,
    tf::Real;
    solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::ODESolution
    return disc_dynamics(dm,
        0.0,
        tf,
        fault_size,
        faultid = fault_node,
        dt = dt,
        solve_kwargs = solve_kwargs)
end

"""
$(TYPEDSIGNATURES)

Run a dynamical simulation of the continuous model.
"""
function cont_dyn(M::SparseMatrixCSC,
    K::SparseMatrixCSC,
    f::Vector{<:Real},
    u₀::Vector{<:Real},
    tf::Real;
    solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::ODESolution
    function dif!(du, u, _, _)
        du[:] .= K * u .+ f
    end

    function jac!(J, _, _, _)
        J[:, :] .= K
    end
    rhs = ODEFunction(dif!, mass_matrix = M, jac_prototype = K, jac = jac!)
    problem = ODEProblem(rhs, u₀, (0.0, tf))
    sol_cont = solve(problem, Trapezoid(); solve_kwargs...)
    return sol_cont
end

"""
$(TYPEDSIGNATURES)

Create the initial conditions for the continuous simulations.
"""
function initial_conditions(cm::ContModel)::Vector{<:Real}
    stable_sol!(cm)
    ch = ConstraintHandler(cm.dh₂)
    db = Dirichlet(:θ, Set(1:getnnodes(cm.grid)), (x, t) -> cm.θ₀(x))
    add!(ch, db)
    close!(ch)
    update!(ch)
    u₀ = zeros(ndofs(cm.dh₂))
    apply!(u₀, ch)
    return u₀
end

"""
$(TYPEDSIGNATURES)

Solve the adjoint dynamics.

The continuous and discrete solutions are needed as well as the comparison indices
to calculate the contributions from the loss function.
"""
function lambda_dyn(cont_sol::ODESolution,
    disc_sol::ODESolution,
    M::SparseMatrixCSC,
    K::SparseMatrixCSC,
    ω_proj::SparseMatrixCSC,
    tf::Real,
    idxs::Vector{<:Integer};
    solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::ODESolution
    function dif_lambda!(du::Vector{<:Real}, u::Vector{<:Real}, _, t::Real)
        du[:] .= -K' * u .+
                 ω_proj' * (ω_proj * cont_sol(t) .- disc_sol(t, Val{1}, idxs = idxs))
    end

    function jac_lambda!(J::SparseMatrixCSC, _, _, _)
        J[:, :] .= -K'
    end
    rhs_lambda = ODEFunction(dif_lambda!,
        mass_matrix = M',
        jac_prototype = K',
        jac = jac_lambda!)
    problem_lambda = ODEProblem(rhs_lambda, zeros(size(M', 1)), (tf, 0))
    sol_lambda = solve(problem_lambda, Trapezoid(); solve_kwargs...)
    return sol_lambda
end

"""
$(TYPEDSIGNATURES)

Calculate the eigenvectors of the unweighted Laplacian of the grid used for the continuous
    simulations.
"""
function lap_eigenvectors(cm::ContModel)::Matrix{<:Real}
    N = ndofs(cm.dh₁)
    lap = zeros(N, N)
    for cell in CellIterator(cm.dh₁)
        i, j, k = celldofs(cell)
        lap[i, j] = lap[j, i] = -1
        lap[i, k] = lap[k, i] = -1
        lap[k, j] = lap[j, k] = -1
    end
    for i in 1:N
        lap[i, i] = -sum(lap[i, :])
    end
    _, eve = eigen(lap)
    return eve
end

"""
$(TYPEDSIGNATURES)

Create the initial values for the parameters.

The parameters are expanded in eigenvectors of the grid Laplacian. The first `n_modes`
eigenvectors are chosen. The first `n_coeffs` coefficients are chosen by projecting the
results of the heat equation diffusion onto the eigenvectors. The `n_modes - n_coeffs` are
set to zero.
"""
function init_expansion(cm::ContModel,
    eve::Matrix{<:Real},
    n_modes::Integer,
    n_coeffs::Integer)::Tuple{Vector{<:Real}, Matrix{<:Real}}
    @assert n_coeffs<=n_modes "The number of coefficients must be less or equal the number of modes"
    coeffs = zeros(2 * n_modes)
    for i in 1:n_coeffs
        coeffs[i] = cm.m_nodal' * eve[:, i]
        coeffs[i + n_modes] = cm.d_nodal' * eve[:, i]
    end
    return coeffs, eve[:, 1:n_modes]
end

"""
$(TYPEDSIGNATURES)

Do a full simulation step for one data set.

The continuous solution is calculated first and then used to obtain the adjoint solution. Afterwards,
the gradient and value of the loss function are calculated and returned.
"""
function simul(disc_sol::ODESolution,
    M_const::SparseMatrixCSC,
    K_const::SparseMatrixCSC,
    m::Vector{T},
    d::Vector{T},
    f::Vector{T},
    A::SparseMatrixCSC,
    q_proj::SparseMatrixCSC,
    ω_proj::SparseMatrixCSC,
    g_proj::Vector{SparseMatrixCSC},
    idxs::Vector{<:Integer},
    u₀::Vector{T},
    tf::Real;
    cont_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}(),
    lambda_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::Tuple{Vector{T}, T} where {T <: Real}
    M = M_const + A * spdiagm(q_proj * m) * A'
    K = K_const - A * spdiagm(q_proj * d) * A'
    cont_sol = cont_dyn(M, K, f, u₀, tf, solve_kwargs = cont_kwargs)
    sol_lambda = lambda_dyn(cont_sol, disc_sol, M, K, ω_proj, tf,
        idxs, solve_kwargs = lambda_kwargs)
    gr = grad(cont_sol, sol_lambda, g_proj)
    loss_val = loss(cont_sol, disc_sol, ω_proj, idxs)[1]
    return gr, loss_val
end

"""
$(TYPEDSIGNATURES)

Calculate the gradient from the solution of the adjoint dynamics.
"""
function grad(sol_cont::ODESolution,
    sol_lambda::ODESolution,
    g_proj::Vector{SparseMatrixCSC},
    dt::Real = 0.1)
    Nparam = size(g_proj, 1)
    function integrand!(du::Vector{<:Real}, t::Real)
        for i in 1:Nparam
            du[i] = sol_lambda(t)' * g_proj[i] * sol_cont(t, Val{1})
            du[i + Nparam] = sol_lambda(t)' * g_proj[i] * sol_cont(t)
        end
    end
    return trapz(0, 25, dt, integrand!, 2 * Nparam)
end

"""
$(TYPEDSIGNATURES)

Trapezoidal rule for the integration of a given function.
"""
function trapz(t₀::Real, tf::Real, dt::Real, int!::Function, N::Integer)::Vector{<:Real}
    du = zeros(N)
    int!(du, t₀)
    re = zeros(N)
    for t in (t₀ + dt):dt:tf
        prev = copy(du)
        int!(du, t)
        re .+= dt * (du .+ prev) / 2
    end
    return re
end

"""
$(TYPEDSIGNATURES)

Calculate the value of the loss function.
"""
function loss(sol_cont::ODESolution,
    sol_disc::ODESolution,
    ω_proj::SparseMatrixCSC,
    idxs::Vector{<:Integer})::Vector{<:Real}
    function integrand!(du::Vector{<:Real}, t::Real)
        tmp = (ω_proj * sol_cont(t) .- sol_disc(t, Val{1}, idxs = idxs))
        du[:] .= 0.5 * tmp' * tmp
    end
    return trapz(0, 25, 0.01, integrand!, 1)
end

"""
$(TYPEDSIGNATURES)

Calculate the projection matrix of the discrete solution onto the adjoint solution.
"""
function grad_proj(A::SparseMatrixCSC,
    q_proj::SparseMatrixCSC,
    evecs::Matrix{<:Real},
    n_coeffs::Integer)::Vector{SparseMatrixCSC}
    g_proj = Vector{SparseMatrixCSC}()
    for i in 1:n_coeffs
        push!(g_proj, A * spdiagm(q_proj * evecs[:, i]) * A')
    end
    return g_proj
end

"""
$(TYPEDSIGNATURES)

Update the parameters using a restricted gradient descent to ensure positiveness.
"""
function update(p::Vector{T},
    g::Vector{T},
    eve::Matrix{T},
    i::Integer,
    f::Function)::Vector{T} where {T <: Real}
    n = size(p, 1) ÷ 2
    opt = Model(Gurobi.Optimizer)
    set_silent(opt)
    @variable(opt, x[1:(2 * n)])
    @constraint(opt, mp, eve * (p[1:n] .+ x[1:n]).>=0.0)
    @constraint(opt, dp, eve * (p[(n + 1):end] .+ x[(n + 1):end]).>=0.0)
    @constraint(opt, ms, sum(x[1:(2 * n)] .^ 2)<=f(i))
    @objective(opt, Min, g'*x[1:(2 * n)])
    optimize!(opt)
    p .+= value.(x)
    return p
end

"""
$(TYPEDSIGNATURES)

Learn the inertia and damping distribution for the continuous model.

The frequency response of faults at multiple generators are compared on homogeneously spread
buses across the grid. The gradient is calculated using an adjoint sensitivity method and
the updates to the parameters are calculated using a constraint gradient descent method.

# Arguments
 - `dm_fn::String = MODULE_FOLDER * "/data/dm.h5"`: File name of the discrete model
 - `cm_fn::String = MODULE_FOLDER * "/data/cm.h5"`: File name of the continuous model
 - `dP::Real = -9.0`: Fault size to be simulated
 - `n_train::Integer = 12`: Amount of faults to consider for training
 - `n_test::Integer = 4`: Amount of faults to consider for testing
 - `dt::Real = 0.01`: Step size at which the solutions of the ODEs are saved
 - `tf::Real = 25.0`: Duration of the simulations
 - `disc_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}()`: Keyword arguments passed
    to the ODE solver for the discrete model
 - `cont_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}()`: Keyword arguments passed
    to the ODE solver for the continuous model
 - `lambda_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}(:saveat => 0.1,
    :abstol => 1e-3, :reltol => 1e-2)`: Keyword arguments passed to the ODE solver of the
    adjoint equations
 - `seed::Union{Nothing, Integer} = 1709`: Seed for the random number generator to be used
     to pick the training and test generators
 - `σ = 0.05`: Standard deviation of the Gaussian used to distribute the fault
 - `n_points = 40`: Number of points for comparison along the largest dimension
 - `n_coeffs = 1`: Number of coefficients that are non-zero at the beginning of the
    training. They correspond to the `n_coeffs` lowest modes of the Laplacian.
 - `n_modes = 20`: Number of modes the Laplacian that are used to expand the parameters.
 - `n_epochs = 8000`: Number of epochs used for the training.
 - `max_function::Function = (x) -> 30 * 2^(x / 500)`: Function that changes the magnitude
    of the change vector of the parameters wrt the epoch.
 - `train_ix::Union{Nothing, Vector{<:Integer}} = nothing`: Indices of the generators used
    for training if they are not supposed to be picked randomly.
 - `test_ix::Union{Nothing, Vector{<:Integer}} = nothing`: Indices of the generators used
    for testing if they are not supposed to be picked randomly.
!!! warning
    If the training and test generators are not supposed to be picked randomly, both
    `train_ix` and `test_ix` need to be passed.
"""
function learn_dynamical_parameters(;
    dm_fn::String = MODULE_FOLDER * "/data/dm.h5",
    cm_fn::String = MODULE_FOLDER * "/data/cm.h5",
    dP::Real = -9.0,
    n_train::Integer = 12,
    n_test::Integer = 4,
    dt::Real = 0.01,
    tf::Real = 25.0,
    disc_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}(),
    cont_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}(),
    lambda_solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}(:saveat => 0.1,
        :abstol => 1e-3,
        :reltol => 1e-2),
    seed::Union{Nothing, Integer} = 1709,
    σ = 0.05,
    n_points = 40,
    n_coeffs = 1,
    n_modes = 20,
    n_epochs = 8000,
    max_function::Function = (x) -> 30 * 2^(x / 500),
    train_ix::Union{Nothing, Vector{<:Integer}} = nothing,
    test_ix::Union{Nothing, Vector{<:Integer}} = nothing)::DynamicSol
    dm = load_model(dm_fn)
    cm = load_model(cm_fn)
    if train_ix !== nothing && test_ix !== nothing
        n_train = size(train_ix, 1)
        n_test = size(test_ix, 1)
    else
        train_ix, test_ix = gen_idxs(dm, dP, n_train, n_test, seed = seed)
    end
    comp_idxs = generate_comp_idxs(cm, dm, train_ix, test_ix, n_points)
    M, K, A, Af, q_coords = assemble_matrices_dynamic(cm)
    q_proj, ω_proj = projectors_dynamic(cm, dm, q_coords, comp_idxs)
    disc_sols_train = Vector{ODESolution}()
    disc_sols_test = Vector{ODESolution}()
    for i in train_ix
        push!(disc_sols_train,
            disc_dyn(dm, i, dP, dt, tf, solve_kwargs = disc_solve_kwargs))
    end
    for i in test_ix
        push!(disc_sols_test, disc_dyn(dm, i, dP, dt, tf, solve_kwargs = disc_solve_kwargs))
    end
    f_train = assemble_f_dynamic(cm, dm, train_ix, dP, Af, q_proj, σ)
    f_test = assemble_f_dynamic(cm, dm, test_ix, dP, Af, q_proj, σ)
    eve = lap_eigenvectors(cm)
    p, eve_p = init_expansion(cm, eve, n_modes, n_coeffs)
    losses = zeros(n_epochs, n_train)
    grads = zeros(2 * n_modes, n_train)
    g_proj = grad_proj(A, q_proj, eve_p, n_modes)
    u₀ = initial_conditions(cm)
    for i in 1:n_epochs
        println("Epoch $i")
        m = eve_p * p[1:n_modes]
        d = eve_p * p[(n_modes + 1):end]
        @threads for j in 1:n_train
            g, l = simul(disc_sols_train[j], M, K, m, d, f_train[:, j],
                A, q_proj, ω_proj, g_proj, comp_idxs, u₀, tf,
                cont_kwargs = cont_solve_kwargs,
                lambda_kwargs = lambda_solve_kwargs)
            grads[:, j] .= g
            losses[i, j] = l
            GC.gc()
        end
        p = update(p, sum(grads, dims = 2)[:, 1], eve_p, i, max_function)
    end
    m = eve_p * p[1:n_modes]
    d = eve_p * p[(n_modes + 1):end]
    test_losses = test_loss(disc_sols_test, M, K, m, d, f_test, A,
        q_proj, ω_proj, comp_idxs, u₀, tf, cont_kwargs = cont_solve_kwargs)
    update_model!(cm, :m, m)
    update_model!(cm, :d, d)

    return DynamicSol(train_ix, test_ix, comp_idxs, m, d, p, eve,
        losses, losses[end, :], test_losses, cm)
end

"""
$(TYPEDSIGNATURES)

Calculate the loss values for the test data sets.
"""
function test_loss(disc_sols::Vector{ODESolution},
    M_const::SparseMatrixCSC,
    K_const::SparseMatrixCSC,
    m::Vector{<:Real},
    d::Vector{<:Real},
    f_test::Array{<:Real, 2},
    A::SparseMatrixCSC,
    q_proj::SparseMatrixCSC,
    ω_proj::SparseMatrixCSC,
    idxs::Vector{<:Integer},
    u₀::Vector{<:Real},
    tf::Real;
    cont_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::Vector{<:Real}
    n = size(f_test, 2)
    M = M_const + A * spdiagm(q_proj * m) * A'
    K = K_const - A * spdiagm(q_proj * d) * A'
    losses = zeros(n)
    @threads for i in 1:n
        cont_sol = cont_dyn(M, K, f_test[:, i], u₀, tf, solve_kwargs = cont_kwargs)
        losses[i] = loss(cont_sol, disc_sols[i], ω_proj, idxs)[1]
    end
    return losses
end
