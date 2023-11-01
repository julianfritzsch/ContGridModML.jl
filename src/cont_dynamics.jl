"""
$(TYPEDSIGNATURES)
"""

function cont_dynamics(model::ContModel,
    p::Real,
    coord::Vector{<:Real},
    tf::Real)::ODESolution
    M_const, K_const, A, Af = assemble_matrices_dynamic(model)
    f_old = model.fault
    set_local_disturbance!(model, coord, p)
    f = Af * model.q_proj * (model.p .+ model.fault)
    model.fault = f_old
    u₀ = initial_conditions(model)
    M = M_const + A * spdiagm(model.q_proj * model.m) * A'
    K = K_const - A * spdiagm(model.q_proj * model.d) * A'
    return cont_dyn(M, K, f, u₀, tf)
end
