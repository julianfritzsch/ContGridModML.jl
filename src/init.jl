export init_model, integrate, interpolate, distribute_load!, set_slack!

"""
$(TYPEDSIGNATURES)

Create a continuous model from a discrete model by using a diffusion process to distribute the paramters.
"""
function init_model(grid::Grid)::ContModel

    # Create the dof handler and interpolation functions
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh)
    ip = Lagrange{2, RefTetrahedron, 1}() # 2D tetrahedron -> triangle
    qr = QuadratureRule{2, RefTetrahedron}(2)
    cellvalues = CellScalarValues(qr, ip)

    n_dofs = ndofs(dh)

    # Get the area of the grid

    m, d, θ₀ = zeros(n_dofs), zeros(n_dofs), zeros(n_dofs)
    p, dp = zeros(n_dofs), zeros(n_dofs)
    bx, by = ones(n_dofs), ones(n_dofs)
    id_slack = 0
    disc_proj = spzeros(0, 0)
    q_proj = get_q_proj(dh, cellvalues)
    area = integrate(dh, cellvalues, q_proj, ones(n_dofs))
    return ContModel(grid, dh, cellvalues, area, m, d, p, bx, by,
        θ₀, dp, id_slack, disc_proj, q_proj)
end

function get_q_proj(dh::DofHandler, cellvalues::CellScalarValues)::SparseMatrixCSC
    q_proj = spzeros(size(dh.cell_dofs, 1), ndofs(dh))
    interp_fun = Ferrite.get_func_interpolations(dh, :u)[1]
    ix = 1
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        cell_dofs = celldofs(cell)
        for q in 1:getnquadpoints(cellvalues)
            point = spatial_coordinate(cellvalues,
                q, getcoordinates(cell))
            ph = PointEvalHandler(dh.grid, [Ferrite.Vec(point...)])
            pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], interp_fun)
            for j in 1:getnbasefunctions(cellvalues)
                q_proj[ix, cell_dofs[j]] = pv.N[j]
            end
            ix += 1
        end
    end
    return q_proj
end

"""
$(TYPEDSIGNATURES)
"""
function set_slack!(cm::ContModel,
        dm::DiscModel)
    node_coords = [cm.dh.grid.nodes[i].x for i in 1:size(cm.dh.grid.nodes, 1)]
    disc_slack = [Ferrite.Vec(dm.coord[dm.id_slack, :]...)]
    id_slack = argmin(norm.(node_coords .- disc_slack))
    ch = ConstraintHandler(cm.dh)
    gen₀ = Dirichlet(:u, Set([id_slack]), (x, t) -> 0)
    add!(ch, gen₀)
    close!(ch)
    update!(ch, 0)
    cm.id_slack = ch.prescribed_dofs[1]
    nothing
end

"""
$(TYPEDSIGNATURES)
"""
function distribute_parameters!(cm::ContModel, dm::DiscModel; σ::Real = 0.02,
        bfactor::Real = 0.1, bmin::Real = 0.1)::Nothing
    distribute_inertia!(cm, dm)
    distribute_damping!(cm, dm)
    distribute_load!(cm, dm)
    distribute_susceptances!(cm, dm, σ, bfactor, bmin)
end

"""
$(TYPEDSIGNATURES)
"""
function distribute_susceptances!(
        cm::ContModel, dm::DiscModel, σ::Real, bfactor::Real, bmin::Real)::Nothing
    function bb(dm, x)
        ds = dm.coord[dm.id_line[:, 2], :] .- dm.coord[dm.id_line[:, 1], :]
        re = zeros(2)
        for (i, row) in enumerate(eachrow(ds))
            if norm(row) == 0
                continue
            end
            if dm.b[i] > 1e4
                continue
            end
            # Divide the line parameters into x and y components
            b = dm.b[i] * row / norm(row) * bfactor
            # Where does the straight line from (x, y) intersect the line from bx and by (t = 0 start point, t = 1 end point of line)
            t = (row[1] * (x[1] - dm.coord[dm.id_line[i, 1], 1]) +
                 row[2] * (x[2] - dm.coord[dm.id_line[i, 1], 2])) / norm(row)^2
            dist = 0
            # If shortest path intersects line, distance is the length of shortest path
            if 0 ≤ t ≤ 1
                dist = norm(x .- (dm.coord[dm.id_line[i, 1], :] .+ t * row))
                # Else take distance to closest endpoint
            elseif t < 0
                dist = norm(x .- dm.coord[dm.id_line[i, 1], :])
            else
                dist = norm(x .- dm.coord[dm.id_line[i, 2], :])
            end
            if dist < σ
                re += abs.(b)
            end
        end
        return re
    end

    bx = zeros(ndofs(cm.dh))
    by = zeros(ndofs(cm.dh))
    coords = [cm.dh.grid.nodes[i].x
              for i in unique(reduce(
        vcat, [collect(cm.dh.grid.cells[i].nodes) for i in 1:size(cm.dh.grid.cells, 1)]))]

    for (i, x) in enumerate(coords)
        bx[i], by[i] = bb(dm, x)
    end

    cm.bx[:] .= max.(bx, bmin)
    cm.by[:] .= max.(by, bmin)
    return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function distribute_inertia!(cm::ContModel, dm::DiscModel)::Nothing
    m = zeros(ndofs(cm.dh))
    grid_coords = [node.x for node in cm.grid.nodes]
    ip = cm.dh.field_interpolations[1]
    # assuming that m(x) is a sum of delta functions
    for (i, ix) in enumerate(dm.id_gen)
        if dm.p_gen[i] == 0
            continue
        end
        coord = dm.coord[ix, :]
        point = Ferrite.Vec(coord...)
        ph = PointEvalHandler(cm.grid, [point], warn = :false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- point)
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            m[cell_dofs[j]] += dm.m_gen[i] * pv.N[j]
        end
    end

    # Normalize values
    m .*= sum(dm.m_gen[dm.p_gen .> 0]) / integrate(cm, m)

    cm.m[:] .= m

    return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function distribute_damping!(cm::ContModel,
        dm::DiscModel)::Nothing
    dl, dg = zeros(ndofs(cm.dh)), zeros(ndofs(cm.dh))
    grid_coords = [node.x for node in cm.grid.nodes]
    ip = cm.dh.field_interpolations[1]
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
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            dl[cell_dofs[j]] -= dm.d_load[i] * pv.N[j]
        end
    end

    for (i, ix) in enumerate(dm.id_gen)
        if dm.p_gen[i] == 0
            continue
        end
        coord = dm.coord[ix, :]
        point = Ferrite.Vec(coord...)
        ph = PointEvalHandler(cm.grid, [point], warn = :false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- point)
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            dg[cell_dofs[j]] += dm.d_gen[i] * pv.N[j]
        end
    end

    # Normalize values
    dg .*= sum(dm.d_gen[dm.p_gen .> 0]) / integrate(cm, dg)
    dl .*= sum(dm.d_load) / integrate(cm, dl)

    cm.d[:] .= dg + dl

    return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function distribute_load!(cm::ContModel,
        dm::DiscModel)::Nothing
    pl, pg = zeros(ndofs(cm.dh)), zeros(ndofs(cm.dh))
    grid_coords = [node.x for node in cm.grid.nodes]
    ip = cm.dh.field_interpolations[1]
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
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            pl[cell_dofs[j]] -= dm.p_load[i] * pv.N[j]
        end
    end

    for (i, ix) in enumerate(dm.id_gen)
        coord = dm.coord[ix, :]
        point = Ferrite.Vec(coord...)
        ph = PointEvalHandler(cm.grid, [point], warn = :false)
        if ph.cells[1] === nothing
            min_ix = argmin([norm(coord .- point)
                             for coord in grid_coords])
            ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
        end
        pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
        cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
        Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
        for j in 1:getnbasefunctions(cm.cellvalues)
            pg[cell_dofs[j]] += dm.p_gen[i] * pv.N[j]
        end
    end

    # Normalize values
    pg .*= sum(dm.p_gen) / integrate(cm, pg)
    pl .*= -sum(dm.p_load) / integrate(cm, pl)

    cm.p[:] .= pg + pl

    return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function set_local_disturbance!(cm::ContModel, coord::Vector{<:Real}, dP::Real)::Nothing
    fault = zeros(ndofs(cm.dh))
    if abs(dP) < 1e-10
        cm.fault[:] .= fault
        return nothing
    end
    grid_coords = [node.x for node in cm.grid.nodes]
    ip = cm.dh.field_interpolations[1]
    # assuming that fault(x) is a delta function
    point = Ferrite.Vec(coord...)
    ph = PointEvalHandler(cm.grid, [point], warn = :false)
    # If no cell is found (the point is outside the grid),
    # use the closest grid point instead
    if ph.cells[1] === nothing
        min_ix = argmin([norm(coord .- point)
                         for coord in grid_coords])
        ph = PointEvalHandler(cm.grid, [grid_coords[min_ix]])
    end
    pv = Ferrite.PointScalarValuesInternal(ph.local_coords[1], ip)
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(cm.dh, ph.cells[1]))
    Ferrite.celldofs!(cell_dofs, cm.dh, ph.cells[1])
    for j in 1:getnbasefunctions(cm.cellvalues)
        fault[cell_dofs[j]] += dP * pv.N[j]
    end

    # Normalize values
    fault .*= dP / integrate(cm, fault)

    cm.fault[:] .= fault
    return nothing
end

"""
$(TYPEDSIGNATURES)

Integrate a function over the whole area of the grid using the finite element method.
"""
function integrate(dh::DofHandler,
        cellvalues::CellScalarValues,
        q_proj::SparseMatrixCSC,
        vals::Vector{<:Real})::Real
    q_vals = q_proj * vals
    int = 0.0
    i = 1
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            int += q_vals[i] * dΩ
            i += 1
        end
    end
    return int
end

function integrate(cm::ContModel, vals::Vector{<:Real})::Real
    return integrate(cm.dh, cm.cellvalues, cm.q_proj, vals)
end

"""
$(TYPEDSIGNATURES)

Interpolate values from the continues model from a  coordinate. If the given coordinate is outside the grid it is replaced by the closed value on the grid.
"""
function interpolate(x::Ferrite.Vec{2, T},
        grid::Grid, dh::DofHandler,
        u::Vector{<:Real},
        fname::Symbol;
        off::Real = 0.0,
        factor::Real = 1.0,
        extrapolate::Bool = true,
        warn::Symbol = :semi)::Real where {T <: Real}
    return interpolate([x],
        grid,
        dh,
        u,
        fname,
        off = off,
        factor = factor,
        extrapolate = extrapolate,
        warn = warn)[1]
end

"""
$(TYPEDSIGNATURES)

Interpolate values from the continues model from a vector of coordinates. If a given coordinate is outside the grid it is replaced by the closed value on the grid.
"""
function interpolate(x::Vector{Ferrite.Vec{2, T}},
        grid::Grid,
        dh::DofHandler,
        u::Vector{<:Real},
        fname::Symbol;
        off::Real = 0.0,
        factor::Real = 1.0,
        extrapolate::Bool = true,
        warn::Symbol = :semi)::Vector{<:Real} where {T <: Real}
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
