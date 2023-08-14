export init_model, integrate, interpolate


"""
    get_params(grid::Grid, tf::Real, dm::DiscModel; κ::Real=1.0, u_min::Real=0.1, σ::Real=0.01, bfactor::Real=1.0, bmin::Real=1000.)::ContModel

Create a continuous model from a discrete model by using a diffusion process to distribute the paramters.
"""
function init_model(
    grid::Grid,
    tf::Real,
    dm::DiscModel;
    κ::Real = 1.0,
    u_min::Real = 0.1,
    σ::Real = 0.01,
    bfactor::Real = 1.0,
    bmin::Real = 1.0
)::ContModel

    # Create the dof handler and interpolation functions
    dh₁ = DofHandler(grid)
    push!(dh₁, :u, 1)
    close!(dh₁)
    dh₂ = DofHandler(grid)
    push!(dh₂, :θ, 1)
    push!(dh₂, :ω, 1)
    close!(dh₂)
    ip = Lagrange{2,RefTetrahedron,1}() # 2D tetrahedron -> triangle
    qr = QuadratureRule{2,RefTetrahedron}(2)
    cellvalues = CellScalarValues(qr, ip)


    # Create initial condition functions for diffusion process
    function d₀(x, _)
        re = 0
        for i in 1:dm.Ngen
            if dm.p_gen[i] == 0
                continue
            end
            dif = x .- dm.coord[dm.id_gen[i], :]
            re += dm.d_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
        end
        for i in 1:dm.Nbus
            dif = x .- dm.coord[i, :]
            re += dm.d_load[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
        end
        return max(re, u_min)
    end
    function m₀(x, _)
        re = 0
        for i in 1:dm.Ngen
            if dm.p_gen[i] == 0
                continue
            end
            dif = x .- dm.coord[dm.id_gen[i], :]
            re += dm.m_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
        end
        return max(re, u_min)
    end
    function p₀(x, _)
        re = 0
        for i in 1:dm.Ngen
            dif = x .- dm.coord[dm.id_gen[i], :]
            re += dm.p_gen[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
        end
        for i in 1:dm.Nbus
            dif = x .- dm.coord[i, :]
            re -= dm.p_load[i] / (σ^2 * 2 * π) * exp(-0.5 * (dif' * dif) / σ^2)
        end
        return re
    end
    #function bx₀(x, _)
    #    return bb(dm, x, σ, bfactor)[1]
    #end
    #function by₀(x, _)
    #    return bb(dm, x, σ, bfactor)[2]
    #end
    
    # Get the area of the grid
    area = integrate(dh₁, cellvalues, (x) -> 1)
    
    d = diffusion(dh₁, cellvalues, grid, d₀, tf, κ)
    d = normalize_values!(d, sum(dm.d_load) + sum(dm.d_gen[dm.p_gen.>0]), area, grid, dh₁, cellvalues)
    m = diffusion(dh₁, cellvalues, grid, m₀, tf, κ)
    m = normalize_values!(m, sum(dm.m_gen[dm.p_gen.>0]), area, grid, dh₁, cellvalues)
    p = diffusion(dh₁, cellvalues, grid, p₀, tf, κ)
    p = normalize_values!(p, 0.0, area, grid, dh₁, cellvalues, mode= :off)
    #bx = diffusion(dh₁, cellvalues, grid, bx₀, tf, κ)
    #by = diffusion(dh₁, cellvalues, grid, by₀, tf, κ)
    bx = ones(size(m))
    by = ones(size(m))
    bx .= max.(bx, bmin)
    by .= max.(by, bmin)

    # Create constraint handler for stable solution
    node_coords = [dh₁.grid.nodes[i].x for i in 1:size(dh₁.grid.nodes, 1)]
    disc_slack = [Ferrite.Vec(dm.coord[dm.id_slack, :]...)]
    id_slack = argmin(norm.(node_coords .- disc_slack))
    ch = ConstraintHandler(dh₁)
    gen₀ = Dirichlet(:u, Set([id_slack]), (x, t) -> 0)
    add!(ch, gen₀)
    close!(ch)
    update!(ch, 0)
    θ₀ = zeros(ndofs(dh₁))
    dp = zeros(ndofs(dh₁))

    return ContModel(
        grid,
        dh₁,
        dh₂,
        cellvalues,
        area,
        m,
        d,
        p,
        bx,
        by,
        θ₀,
        dp,
        ch,
    )
end

"""
    diffusion(dh::DofHandler, cellvalues::CellScalarValues, grid::Grid, f::Function, tf::Real, κ::Real)::Vector{<:Real}

Run a diffusion process to distribute the paramters over the grid.
"""
function diffusion(dh::DofHandler, cellvalues::CellScalarValues, grid::Grid, f::Function, tf::Real, κ::Real)::Vector{<:Real}
    # Use the ferrite.jl assemble to assemble the stiffness matrix for the diffusion process.
    function assembleK!(K::SparseMatrixCSC, dh::DofHandler, cellvalues::CellScalarValues)::SparseMatrixCSC
        n_basefuncs = getnbasefunctions(cellvalues)
        Kₑ = zeros(n_basefuncs, n_basefuncs)

        assembler = start_assemble(K)

        for cell in CellIterator(dh)
            fill!(Kₑ, 0)
            Ferrite.reinit!(cellvalues, cell)

            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
                for i in 1:n_basefuncs
                    ∇φᵢ = shape_gradient(cellvalues, q_point, i)
                    for j in 1:n_basefuncs
                        ∇φⱼ = shape_gradient(cellvalues, q_point, j)
                        Kₑ[i, j] -= ∇φᵢ ⋅ ∇φⱼ * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Kₑ)
        end
        return K
    end

    # Use the Ferrite.jl assemble to create the mass matrix for the diffusion proccess.
    function assembleM!(M::SparseMatrixCSC, dh::DofHandler, cellvalues::CellScalarValues)::SparseMatrixCSC
        n_basefuncs = getnbasefunctions(cellvalues)
        Mₑ = zeros(n_basefuncs, n_basefuncs)

        assembler = start_assemble(M)

        for cell in CellIterator(dh)
            fill!(Mₑ, 0)
            Ferrite.reinit!(cellvalues, cell)

            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
                for i in 1:n_basefuncs
                    φᵢ = shape_value(cellvalues, q_point, i)
                    for j in 1:n_basefuncs
                        φⱼ = shape_value(cellvalues, q_point, j)
                        Mₑ[i, j] += φᵢ ⋅ φⱼ * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Mₑ)
        end
        return M
    end
    M = create_sparsity_pattern(dh)
    M = assembleM!(M, dh, cellvalues)
    K = create_sparsity_pattern(dh)
    K = assembleK!(K, dh, cellvalues)

    # Create intital conditions
    ch = ConstraintHandler(dh)
    d = Dirichlet(:u, Set(1:getnnodes(grid)), f)
    add!(ch, d)
    close!(ch)
    update!(ch, 0)
    u₀ = zeros(ndofs(dh))
    apply!(u₀, ch)
    # In a future release of Ferrite this can be replaced by
    # apply_analytical!(u₀, dh, :u, f)

    function dif!(du, u, _, _)
        mul!(du, κ * K, u)
    end

    odefun = ODEFunction(dif!, mass_matrix=M, jac_prototype=K)
    problem = ODEProblem(odefun, u₀, (0.0, tf))
    sol = solve(problem, RadauIIA5(), reltol=1e-5, abstol=1e-7)
    return sol.u[end]
end


"""
    integrate(dh::DofHandler, cellvalues::CellScalarValues, f::Function)::Real

Integrate a function over the whole area of the grid using the finite element method.
"""
function integrate(dh::DofHandler, cellvalues::CellScalarValues, f::Function)::Real
    n_basefuncs = getnbasefunctions(cellvalues)
    int = 0.0
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            x = spatial_coordinate(cellvalues, q_point, getcoordinates(cell))
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues, q_point, i)
                int += f(x) * φᵢ * dΩ
            end
        end
    end
    return int
end

"""
    interpolate(x::Vector{Tensor{1,2,<:Real}}, grid::Grid, dh::DofHandler, u::Vector{<:Real}, fname::Symbol; off::Real=0.0, factor::Real=1.0, extrapolate::Bool=true, warn::Symbol=:semi)::Vector{<:Real}

Interpolate values from the continues model from a  coordinate. If the given coordinate is outside the grid it is replaced by the closed value on the grid.
"""
function interpolate(x::Tensor{1,2,<:Real},
    grid::Grid, dh::DofHandler,
    u::Vector{<:Real},
    fname::Symbol;
    off::Real = 0.0,
    factor::Real = 1.0,
    extrapolate::Bool = true,
    warn::Symbol = :semi
)::Real
    ph = PointEvalHandler(grid, [x], warn=(warn == :all))
    if isnan(get_point_values(ph, dh, u, fname)[1])
        if extrapolate
            if warn in [:all, :semi]
                println("There are points which are outside the grid. There value will be set to the value of the closest grid point.")
            end
            grid_coords = [node.x for node in grid.nodes]
            min_ix = argmin([norm(coord .- x) for coord in grid_coords])
            ph = PointEvalHandler(grid, [grid_coords[min_ix]])
        elseif warn in [:all, :semi]
            println("There are points which are outside the grid. There value will be set to NaN.")
        end
    end
    return factor * get_point_values(ph, dh, u, fname)[1] + off
end

"""
    interpolate(x::Vector{Tensor{1,2,<:Real}}, grid::Grid, dh::DofHandler, u::Vector{<:Real}, fname::Symbol; off::Real=0.0, factor::Real=1.0, extrapolate::Bool=true, warn::Symbol=:semi)::Vector{<:Real}

Interpolate values from the continues model from a vector of coordinates. If a given coordinate is outside the grid it is replaced by the closed value on the grid.
"""
function interpolate(x::Vector{Tensor{1,2,<:Real}},
    grid::Grid,
    dh::DofHandler,
    u::Vector{<:Real},
    fname::Symbol;
    off::Real = 0.0,
    factor::Real = 1.0,
    extrapolate::Bool = true,
    warn::Symbol = :semi
)::Vector{<:Real}
    ph = PointEvalHandler(grid, x, warn=(warn == :all))
    re = get_point_values(ph, dh, u, fname)
    nan_ix = findall(isnan.(re))
    if extrapolate && !isempty(nan_ix)
        if warn in [:all, :semi]
            println("There are points which are outside the grid. There value will be set to the value of the closest grid point.")
        end
        grid_coords = [node.x for node in grid.nodes]
        min_ix = [argmin([norm(coord .- x[j]) for coord in grid_coords]) for j in nan_ix]
        x[nan_ix] = grid_coords[min_ix]
        ph = PointEvalHandler(grid, x)
    elseif !isempty(nan_ix) && warn in [:all, :semi]
        println("There are points which are outside the grid. There value will be set to NaN.")
    end
    return factor * get_point_values(ph, dh, u, fname) .+ off
end

"""
    normalize_values!(u::Vector{<:Real}, value::Real, area::Real, grid::Grid, dh::DofHandler, cellvalues::CellScalarValues[, mode::String="factor"])::Vector{<:Real}

Normalize nodal values over the given area. There are two methods: "factor" rescales everything by a common factor, "off" adds an offset to normalize.
"""
function normalize_values!(
    u::Vector{<:Real},
    value::Real,
    area::Real,
    grid::Grid,
    dh::DofHandler,
    cellvalues::CellScalarValues;
    mode::Symbol = :factor
)::Vector{<:Real}
    utot = integrate(dh, cellvalues, x -> interpolate(x, grid, dh, u, :u))
    ch = ConstraintHandler(dh)
    if mode == :factor
        d = Dirichlet(:u, Set(1:getnnodes(grid)),
            (x, t) -> interpolate(x, grid, dh, u, :u, factor = (value / utot)))
    elseif mode == :off
        d = Dirichlet(:u, Set(1:getnnodes(grid)),
            (x, t) -> interpolate(x, grid, dh, u, :u, off = (value - utot) / area))
    else
        throw(ArgumentError("Mode must be :factor or :off"))
    end
    add!(ch, d)
    close!(ch)
    update!(ch, 0)
    apply!(u, ch)
    return u
end



