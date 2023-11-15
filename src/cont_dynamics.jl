"""
$(TYPEDSIGNATURES)
"""

function cont_dynamics(model::ContModel,
        p::Real,
        coord::Vector{<:Real},
        tf::Real;
        alg::Union{DEAlgorithm, Nothing} = nothing,
        solve_kwargs::Dict{Symbol, <:Any} = Dict{Symbol, Any}())::ODESolution
    M_const, K_const, A, Af = assemble_matrices_dynamic(model)
    f_old = model.fault
    set_local_disturbance!(model, coord, p)
    f = Af * model.q_proj * (model.p .+ model.fault)
    model.fault = f_old
    u₀ = initial_conditions(model)
    M = M_const + A * spdiagm(model.q_proj * model.m) * A'
    K = K_const - A * spdiagm(model.q_proj * model.d) * A'
    function dif!(du, u, _, _)
        du[:] .= K * u .+ f
    end

    function jac!(J, _, _, _)
        J[:, :] .= K
    end
    rhs = ODEFunction(dif!, mass_matrix = M, jac_prototype = K, jac = jac!)
    problem = ODEProblem(rhs, u₀, (0.0, tf))
    sol_cont = solve(problem, alg; solve_kwargs...)
    return sol_cont
end
