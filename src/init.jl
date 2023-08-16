export init_model, integrate, interpolate, distribute_load!, set_slack!

"""
    get_params(grid::Grid, tf::Real, dm::DiscModel; κ::Real=1.0, u_min::Real=0.1, σ::Real=0.01, bfactor::Real=1.0, bmin::Real=1000.)::ContModel

Create a continuous model from a discrete model by using a diffusion process to distribute the paramters.
"""
function init_model(grid::Grid
)::ContModel

    # Create the dof handler and interpolation functions
    dh₁ = DofHandler(grid)
    push!(dh₁, :u, 1)
    close!(dh₁)
    dh₂ = DofHandler(grid)
    push!(dh₂, :θ, 1)
    push!(dh₂, :ω, 1)
    close!(dh₂)
    ip = Lagrange{2, RefTetrahedron, 1}() # 2D tetrahedron -> triangle
    qr = QuadratureRule{2, RefTetrahedron}(2)
    cellvalues = CellScalarValues(qr, ip)

    # Get the area of the grid
    area = integrate(dh₁, cellvalues, (x) -> 1)

    m, d, θ₀ = zeros(ndofs(dh₁)), zeros(ndofs(dh₁)), zeros(ndofs(dh₁))
    p, dp = zeros(ndofs(dh₁)), zeros(ndofs(dh₁))
    bx, by = ones(ndofs(dh₁)), ones(ndofs(dh₁))
    by = ones(ndofs(dh₁))
    id_slack = 0
    disc_proj, q_proj = spzeros(0,0), spzeros(0,0)
    return ContModel(grid, dh₁, dh₂, cellvalues, area, m, d, p, bx, by,
        θ₀, dp, id_slack, disc_proj, q_proj)
end


function set_slack!(cm::ContModel,
    dm::DiscModel
)
    node_coords = [cm.dh₁.grid.nodes[i].x for i in 1:size(cm.dh₁.grid.nodes, 1)]
    disc_slack = [Ferrite.Vec(dm.coord[dm.id_slack, :]...)]
    id_slack = argmin(norm.(node_coords .- disc_slack))
    ch = ConstraintHandler(cm.dh₁)
    gen₀ = Dirichlet(:u, Set([id_slack]), (x, t) -> 0)
    add!(ch, gen₀)
    close!(ch)
    update!(ch, 0)
    cm.id_slack = ch.prescribed_dofs[1]
    nothing
end


function distribute_load!(cm::ContModel,
    dm::DiscModel
)
    cm.p[:] .= 0.0
    grid_coords = [node.x for node in cm.grid.nodes]
    ip = cm.dh₁.field_interpolations[1]
    # assuming that p(x) is a sum of delta functions
    for (i, point) in enumerate(eachrow(dm.coord))
        point = Ferrite.Vec(point...)
        ph = PointEvalHandler(cm.grid, [point], warn = :false)
        # If no cell is found (the point is outside the grid),
        # use the closest grid point instead
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- point)
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh₁, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh₁, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            cm.p[cell_dofs[j]] -= dm.p_load[i] * pv.N[j]  
        end
    end

    for (i, ix) in enumerate(dm.id_gen)
        coord = dm.coord[ix,:]
        point = Ferrite.Vec(coord...)
        ph = PointEvalHandler(cm.grid, [point], warn = :false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- point)
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh₁, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh₁, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            cm.p[cell_dofs[j]] += dm.p_gen[i] * pv.N[j]  
        end
    end
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
function interpolate(x::Tensor{1, 2, <:Real},
    grid::Grid, dh::DofHandler,
    u::Vector{<:Real},
    fname::Symbol;
    off::Real = 0.0,
    factor::Real = 1.0,
    extrapolate::Bool = true,
    warn::Symbol = :semi)::Real
    ph = PointEvalHandler(grid, [x], warn = (warn == :all))
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
function interpolate(x::Vector{Tensor{1, 2, <:Real}},
    grid::Grid,
    dh::DofHandler,
    u::Vector{<:Real},
    fname::Symbol;
    off::Real = 0.0,
    factor::Real = 1.0,
    extrapolate::Bool = true,
    warn::Symbol = :semi)::Vector{<:Real}
    ph = PointEvalHandler(grid, x, warn = (warn == :all))
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
function normalize_values!(u::Vector{<:Real},
    value::Real,
    area::Real,
    grid::Grid,
    dh::DofHandler,
    cellvalues::CellScalarValues;
    mode::Symbol = :factor)::Vector{<:Real}
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
