export nodal_plot


function nodal_plot(
    model::ContModel,
    fieldname::Symbol;
    kwargs...
)::Figure
    val = getfield(model, Symbol(string(fieldname) * "_nodal"))
    nodal_plot(model, val, kwargs...)
end

function nodal_plot(
    model::ContModel,
    val::Vector{<:Real};
    logarithmic::Bool=false,
    colormap::Symbol = :inferno,
    colorbar::Bool = true,
    decorations::Bool = false,
    fig_args::Dict{Symbol,<:Any} = Dict{Symbol,Any}(),
    ax_args::Dict{Symbol,<:Any} = Dict{Symbol,Any}(),
    cbar_args=Dict{Symbol,Any} = Dict{Symbol,Any}()
)::Figure
    f = Figure(; fig_args...)
    ax = Axis(f[1, 1]; ax_args...)
    
    if logarithmic
        sp = solutionplot!(model.dh₁, log10.(val), colormap=colormap)
    else
        sp = solutionplot!(model.dh₁, val, colormap=colormap)
    end
        
    if !decorations
        hidedecorations!(ax)
        hidespines!(ax)
    end
    if colorbar
        if logarithmic
            # This is a crude way to make the values logarithmic
            # Unfortunately Makie has no support for that, so we need to create all ticks and labels by hand
            # Probably, there is a much smarter way to do this
            mi, ma = extrema(log10.(val))
            maj_mi = Int(ceil(mi))
            maj_ma = Int(floor(ma))
            min_mi = Int(ceil(10^(mi - (maj_mi - 1))))
            min_ma = Int(floor(10^(ma - maj_ma)))
            major = maj_mi:maj_ma
            minor_first = [maj_mi - 1 + log10(i) for i in min_mi:9]
            minor_main = [i + log10(j) for i = maj_mi:maj_ma-1 for j = 2:9]
            minor_last = [maj_ma + log10(i) for i in 2:min_ma]
            labels = [L"10^{%$i}" for i in major]
            Colorbar(f[1, 2], sp, ticks=(major, labels), minorticks=[minor_first; minor_main; minor_last], minorticksvisible=true; cbar_args...)
        else
            Colorbar(f[1, 2], sp; cbar_args...)
        end
    end
    return f
end
