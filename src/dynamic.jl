function assemble_matrices_dynamic(
    model::ContGridMod.ContModel
)::Tuple{SparseMatrixCSC,SparseMatrixCSC,SparseMatrixCSC,SparseMatrixCSC,Array{<:Real,2}}
    K_const = create_sparsity_pattern(model.dh₂)
    M_const = create_sparsity_pattern(model.dh₂)
    A = zeros(ndofs(model.dh₂), getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
    Af = zeros(ndofs(model.dh₂), getnquadpoints(model.cellvalues) * size(model.grid.cells, 1))
    dofr = dof_range(model.dh₂, :ω)
    q_coords = zeros(getnquadpoints(model.cellvalues) * size(model.grid.cells, 1), 2)
    n_basefuncs_θ = getnbasefunctions(model.cellvalues)
    n_basefuncs_ω = getnbasefunctions(model.cellvalues)
    n_basefuncs = n_basefuncs_θ + n_basefuncs_ω
    θ▄, ω▄ = 1, 2

    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_θ, n_basefuncs_ω], [n_basefuncs_θ, n_basefuncs_ω])
    Mₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_θ, n_basefuncs_ω], [n_basefuncs_θ, n_basefuncs_ω])

    assembler_K = start_assemble(K_const)
    assembler_M = start_assemble(M_const)

    for (ix, cell) in enumerate(CellIterator(model.dh₂))
        fill!(Kₑ, 0)
        fill!(Mₑ, 0)

        Ferrite.reinit!(model.cellvalues, cell)

        dofs = celldofs(cell)
        for q_point in 1:getnquadpoints(model.cellvalues)
            x = spatial_coordinate(model.cellvalues, q_point, getcoordinates(cell))
            b = SparseMatrixCSC(diagm([model.bx(x), model.by(x)]))
            dΩ = getdetJdV(model.cellvalues, q_point)
            idx = (ix - 1) * getnquadpoints(model.cellvalues) + q_point
            q_coords[idx, :] = x
            for i in 1:n_basefuncs_θ
                φᵢ = shape_value(model.cellvalues, q_point, i)
                ∇φᵢ = shape_gradient(model.cellvalues, q_point, i)
                A[dofs[dofr[i]], idx] = φᵢ * sqrt(dΩ)
                Af[dofs[dofr[i]], idx] = φᵢ * dΩ
                for j in 1:n_basefuncs_ω
                    φⱼ = shape_value(model.cellvalues, q_point, j)
                    ∇φⱼ = shape_gradient(model.cellvalues, q_point, j)
                    Kₑ[BlockIndex((θ▄, ω▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                    Kₑ[BlockIndex((ω▄, θ▄), (i, j))] -= ∇φᵢ ⋅ (b * ∇φⱼ) * dΩ
                    Mₑ[BlockIndex((θ▄, θ▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Kₑ)
        assemble!(assembler_M, celldofs(cell), Mₑ)
    end
    dropzeros!(K_const)
    dropzeros!(M_const)
    return K_const, M_const, sparse(A), sparse(Af), q_coords
end

function projectors_dynamic(
    cm::ContGridMod.ContModel,
    dm::ContGridMod.DiscModel,
    q_coords::Array{<:Real,2},
    ω_idxs::Vector{<:Integer}
)::Tuple{SparseMatrixCSC,SparseMatrixCSC}
    q_proj = zeros(size(q_coords, 1), ndofs(cm.dh₂) ÷ 2)
    func_interpolations = Ferrite.get_func_interpolations(cm.dh₂, :ω)
    grid_coords = [node.x for node in cm.grid.nodes]
    ω_proj = zeros(size(ω_idxs, 1), ndofs(cm.dh₂))
    for (i, id) in enumerate(ω_idxs)
        ph = PointEvalHandler(cm.grid, [Ferrite.Vec(dm.coord[id, :]...)], warn=:false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- Ferrite.Vec(dm.coord[id, :]...)) for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
        cell_dofs = Vector{Integer}(undef, ndofs_per_cell(cm.dh₂, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh₂, ph.cells[1])
        n_base_funcs = getnbasefunctions(pv)
        dofr = dof_range(cm.dh₂, :ω)
        for j = 1:n_base_funcs
            ω_proj[i, cell_dofs[dofr[j]]] = shape_value(pv, 1, j)
        end
    end
    omega_dofs = Set{Integer}()
    dofr = dof_range(cm.dh₂, :ω)
    for i = 1:size(cm.grid.cells, 1)
        cell_dofs = Vector{Integer}(undef, ndofs_per_cell(cm.dh₂, i))
        Ferrite.celldofs!(cell_dofs, cm.dh₂, i)
        push!(omega_dofs, cell_dofs[dofr]...)
    end
    odofs_dict = Dict(j => i for (i, j) in enumerate(sort(collect(omega_dofs))))
    for (i, point) in enumerate(eachrow(q_coords))
        ph = PointEvalHandler(cm.grid, [Ferrite.Vec(point...)])
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], func_interpolations[1])
        cell_dofs = Vector{Integer}(undef, ndofs_per_cell(cm.dh₂, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh₂, ph.cells[1])
        n_base_funcs = getnbasefunctions(pv)
        for j = 1:n_base_funcs
            q_proj[i, odofs_dict[cell_dofs[dofr[j]]]] = shape_value(pv, 1, j)
        end
    end
    return sparse(q_proj), sparse(ω_proj)
end

function assemble_f_dynamic(
    cm::ContGridMod.ContModel,
    dm::ContGridMod.DiscModel,
    fault_ix::Vector{<:Integer},
    dP::Union{Real,Vector{<:Real}},
    Af::SparseMatrixCSC,
    q_proj::SparseMatrixCSC;
    σ::Real=0.01
)::Array{<:Real,2}
    @assert size(fault_ix) == size(dP) || isa(dP, Real) "The size of `fault_ix` and `dP` must match"
    if isa(dP, Real)
        dP = dP .* ones(size(fault_ix, 1))
    end
    f_old = cm.fault_nodal
    f = zeros(ndofs(cm.dh₂), size(fault_ix, 1))
    for (i, (ix, p)) in enumerate(zip(fault_ix, dP))
        add_local_disturbance!(cm, dm.coord[ix, :], p, σ)
        f[:, i] = Af * q_proj * (cm.p_nodal .+ cm.fault_nodal)
    end
    cm.fault_nodal = f_old
    return f
end

function generate_comp_idxs(
    cm::ContGridMod.ContModel,
    dm::ContGridMod.DiscModel,
    tri::Vector{<:Integer},
    tei::Vector{<:Integer},
    n::Int
)::Vector{<:Integer}
    grid_vals = []
    for i = 1:n
        for j = 1:n
            ph = PointEvalHandler(cm.dh₁.grid, [Ferrite.Vec(-0.6 + (i - 1) / (n - 1.0), -0.4 + (j - 1) / (n - 1.0))], warn=false)
            if ph.cells[1] !== nothing
                push!(grid_vals, [-0.6 + (i - 1) / (n - 1.0) -0.4 + (j - 1) / (n - 1.0)])
            end
        end
    end
    grid_vals = reduce(vcat, grid_vals)
    idxs = Set()
    pidxs = Int.(sort(collect(setdiff(Set(1:3809), union(Set(tri), Set(tei))))))
    for point in eachrow(grid_vals)
        push!(idxs, argmin(map(norm, eachslice(dm.coord[pidxs, :]' .- point, dims=2))))
    end
    return pidxs[Int.(sort(collect(idxs)))]
end

function gen_idxs(
    dm::ContGridMod.DiscModel,
    dP::Real,
    n_train::Integer,
    n_test::Int;
    seed::Union{Nothing,Integer}=nothing
)::Tuple{Vector{<:Integer},Vector{<:Integer}}
    rng = Xoshiro(seed)
    idg = dm.id_gen[dm.p_gen.>=abs(dP)]
    tri = sample(rng, idg, n_train, replace=false, ordered=true)
    tei = sample(rng, Int64.(collect(setdiff(Set(idg), Set(tri)))), n_test, replace=false, ordered=true)
    return tri, tei
end

function disc_dyn(
    dm::ContGridMod.DiscModel,
    fault_node::Integer,
    fault_size::Real,
    dt::Real,
    tf::Real;
    solve_kwargs::Dict{Symbol,<:Any}=Dict{Symbol,Any}()
)::ODESolution
    return disc_dynamics(dm, 0.0, tf, fault_size, faultid=fault_node, dt=dt, solve_kwargs=solve_kwargs)
end

function cont_dyn(
    M::SparseMatrixCSC,
    K::SparseMatrixCSC,
    f::Vector{<:Real},
    u₀::Vector{<:Real},
    tf::Real;
    solve_kwargs::Dict{Symbol,<:Any}=Dict{Symbol,Any}()
)::ODESolution
    function dif!(du, u, _, _)
        du[:] .= K * u .+ f
    end

    function jac!(J, _, _, _)
        J[:, :] .= K
    end
    rhs = ODEFunction(dif!, mass_matrix=M, jac_prototype=K, jac=jac!)
    problem = ODEProblem(rhs, u₀, (0.0, tf))
    sol_cont = solve(problem, Trapezoid(); solve_kwargs...)
    return sol_cont
end

function initial_conditions(
    cm::ContGridMod.ContModel
)::Vector{<:Real}
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

function lambda_dyn(
    cont_sol::ODESolution,
    disc_sol::ODESolution,
    M::SparseMatrixCSC,
    K::SparseMatrixCSC,
    ω_proj::SparseMatrixCSC,
    idxs::Vector{Integer};
    solve_kwargs::Dict{Symbol,<:Any}=Dict{Symbol,Any}()
)::ODESolution
    function dif_lambda!(du::Vector{<:Real}, u::Vector{<:Real}, _, t::Real)
        du[:] .= -K' * u .+ ω_proj' * (ω_proj * cont_sol(t) .- disc_sol(t, Val{1}, idxs=idxs))
    end

    function jac_lambda!(J::SparseMatrixCSC, _, _, _)
        J[:, :] .= -K'
    end
    rhs_lambda = ODEFunction(dif_lambda!, mass_matrix=M', jac_prototype=K', jac=jac_lambda!)
    problem_lambda = ODEProblem(rhs_lambda, zeros(size(M', 1)), (25, 0))
    sol_lambda = solve(problem_lambda, Trapezoid(); solve_kwargs...)
    return sol_lambda
end

function lap_eigenvectors(
    cm::ContGridMod.ContModel
)::Array{<:Real,2}
    N = ndofs(cm.dh₁)
    lap = zeros(N, N)
    for cell in CellIterator(cm.dh₁)
        i, j, k = celldofs(cell)
        lap[i, j] = lap[j, i] = -1
        lap[i, k] = lap[k, i] = -1
        lap[k, j] = lap[j, k] = -1
    end
    for i = 1:N
        lap[i, i] = -sum(lap[i, :])
    end
    _, eve = eigen(lap)
    return eve
end

function init_expansion(
    cm::ContGridMod.ContModel,
    eve::Array{<:Real,2},
    n_modes::Integer,
    n_coeffs::Integer)::Tuple{Vector{<:Real},Array{<:Real,2}}
    @assert n_coeffs <= n_modes "The number of coefficients must be less or equal the number of modes"
    coeffs = zeros(2 * n_modes)
    for i = 1:n_coeffs
        coeffs[i] = cm.m_nodal' * eve[:, i]
        coeffs[i+n_modes] = cm.d_nodal' * eve[:, i]
    end
    return coeffs, eve[:, 1:n_modes]
end

function simul(disc_sol::ODESolution, M_const::SparseMatrixCSC, K_const::SparseMatrixCSC, m::Vector{<:Real}, d::Vector{<:Real}, f::Vector{<:Real}, A::SparseMatrixCSC, q_proj::SparseMatrixCSC, ω_proj::SparseMatrixCSC, g_proj::Vector{SparseMatrixCSC}, idxs::Vector{<:Integer}, u₀::Vector{<:Real}, tf::Real; cont_kwargs::Dict{Symbol,<:Any}=Dict{Symbol,Any}(), lambda_kwargs::Dict{Symbol,<:Any}=Dict{Symbol,Any}())::Tuple{Vector{<:Real},Real}
    M = M_const + A * spdiagm(q_proj * m) * A'
    K = K_const - A * spdiagm(q_proj * d) * A'
    cont_sol = cont_dyn(M, K, f, u₀, tf, solve_kwargs=cont_kwargs)
    sol_lambda = lambda_dyn(cont_sol, disc_sol, M, K, ω_proj, idxs, solve_kwargs=lambda_kwargs)
    gr = grad(cont_sol, sol_lambda, g_proj)
    loss_val = loss(cont_sol, disc_sol, ω_proj, idxs)[1]
    return gr, loss_val
end

function grad(sol_cont::ODESolution, sol_lambda::ODESolution, g_proj::Vector{SparseMatrixCSC}, dt::Real=0.1)
    Nparam = size(g_proj, 1)
    function integrand!(du::Vector{<:Real}, t::Real)
        for i = 1:Nparam
            du[i] = sol_lambda(t)' * g_proj[i] * sol_cont(t, Val{1})
            du[i+Nparam] = sol_lambda(t)' * g_proj[i] * sol_cont(t)
        end
    end
    return trapz(0, 25, dt, integrand!, 2 * Nparam)
end

function trapz(t₀::Real, tf::Real, dt::Real, int!::Function, N::Integer)::Vector{<:Real}
    du = zeros(N)
    int!(du, t₀)
    re = zeros(N)
    for t in t₀+dt:dt:tf
        prev = copy(du)
        int!(du, t)
        re .+= dt * (du .+ prev) / 2
    end
    return re
end

function loss(sol_cont::ODESolution, sol_disc::ODESolution, ω_proj::SparseMatrixCSC, idxs::Vector{<:Integer})::Vector{<:Real}
    function integrand!(du::Vector{<:Real}, t::Real)
        tmp = (ω_proj * sol_cont(t) .- sol_disc(t, Val{1}, idxs=idxs))
        du[:] .= 0.5 * tmp' * tmp
    end
    return trapz(0, 25, 0.01, integrand!, 1)
end

function grad_proj(A::SparseMatrixCSC, q_proj::SparseMatrixCSC, evecs::Array{<:Real,2}, n_coeffs::Integer)::Vector{SparseMatrixCSC}
    g_proj = Vector{SparseMatrixCSC}()
    for i = 1:n_coeffs
        push!(g_proj, A * spdiagm(q_proj * evecs[:, i]) * A')
    end
    return g_proj
end

function update(p::Vector{<:Real}, g::Vector{<:Real}, eve::Array{<:Real,2}, m::Real)::Vector{<:Real}
    n = size(p, 1) ÷ 2
    opt = Model(Gurobi.Optimizer)
    set_silent(opt)
    @variable(opt, x[1:2*n])
    @constraint(opt, mp, eve * (p[1:n] .+ x[1:n]) .>= 0.0)
    @constraint(opt, dp, eve * (p[n+1:end] .+ x[n+1:end]) .>= 0.0)
    @constraint(opt, ms, sum(x[1:2*n] .^ 2) <= m)
    @objective(opt, Min, g' * x[1:2*n])
    optimize!(opt)
    p .+= value.(x)
    return p
end
