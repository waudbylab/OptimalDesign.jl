"""
    plot_design_allocation(candidates, weights; ...)

Plot the weight distribution across candidates for a batch design.
Auto-detects 1D (stem plot) vs 2D (scatter/bubble plot) design variables.
"""
function plot_design_allocation(
    candidates::AbstractVector{<:NamedTuple},
    w::AbstractVector;
    fields::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
)
    ks = keys(first(candidates))
    fs = fields !== nothing ? fields : Tuple(ks)
    ndim = length(fs)

    if ndim == 1
        _plot_design_1d(candidates, w, fs[1])
    elseif ndim == 2
        _plot_design_2d(candidates, w, fs[1], fs[2])
    else
        error("plot_design_allocation supports 1 or 2 design variables, got $ndim")
    end
end

function _plot_design_1d(candidates, w, xf::Symbol)
    x_vals = [getfield(ξ, xf) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel="Weight",
        title="Design Allocation")
    CairoMakie.stem!(ax, x_vals, w, color=:blue)

    fig
end

function _plot_design_2d(candidates, w, xf::Symbol, yf::Symbol)
    x_vals = [getfield(ξ, xf) for ξ in candidates]
    y_vals = [getfield(ξ, yf) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 500))

    # Support points (non-zero weight) as scaled bubbles
    mask = w .> 0
    if any(mask)
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel=string(xf), ylabel=string(yf),
            title="Design Allocation")

        # Bubble area proportional to weight
        w_nz = w[mask]
        ms = 5 .+ 40 .* (w_nz ./ maximum(w_nz))

        CairoMakie.scatter!(ax, x_vals[mask], y_vals[mask];
            markersize=ms, color=w_nz, colormap=:viridis,
            colorrange=(0, maximum(w_nz)))
        CairoMakie.Colorbar(fig[1, 2]; colormap=:viridis,
            colorrange=(0, maximum(w_nz)), label="Weight")

        # Label counts
        for i in findall(mask)
            CairoMakie.text!(ax, x_vals[i], y_vals[i];
                text=string(round(w[i]; digits=3)),
                fontsize=9, align=(:center, :bottom), offset=(0, 5))
        end
    end

    fig
end

"""
    plot_design_allocation(d::ExperimentalDesign, candidates; ...)

Plot the weight distribution for an `ExperimentalDesign`.
"""
function plot_design_allocation(
    d::ExperimentalDesign,
    candidates::AbstractVector{<:NamedTuple};
    kwargs...,
)
    plot_design_allocation(candidates, weights(d, candidates); kwargs...)
end

"""
    plot_gateaux(candidates, gd, p; ...)

Plot the Gateaux derivative at each candidate, with the optimality bound.
Auto-detects 1D (line plot) vs 2D (heatmap/scatter) design variables.
"""
function plot_gateaux(
    candidates::AbstractVector{<:NamedTuple},
    gd::AbstractVector,
    p;
    fields::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
)
    ks = keys(first(candidates))
    fs = fields !== nothing ? fields : Tuple(ks)
    ndim = length(fs)

    if ndim == 1
        _plot_gateaux_1d(candidates, gd, p, fs[1])
    elseif ndim == 2
        _plot_gateaux_2d(candidates, gd, p, fs[1], fs[2])
    else
        error("plot_gateaux supports 1 or 2 design variables, got $ndim")
    end
end

function _plot_gateaux_1d(candidates, gd, p, xf::Symbol)
    x_vals = [getfield(ξ, xf) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel="Gateaux derivative",
        title="Optimality Check")
    CairoMakie.lines!(ax, x_vals, gd, color=:blue, linewidth=1.5)
    CairoMakie.hlines!(ax, [p], color=:red, linestyle=:dash, label="q = $p")
    CairoMakie.axislegend(ax)

    fig
end

function _plot_gateaux_2d(candidates, gd, p, xf::Symbol, yf::Symbol)
    x_vals = [getfield(ξ, xf) for ξ in candidates]
    y_vals = [getfield(ξ, yf) for ξ in candidates]

    crange = (min(minimum(gd), 0.0), max(maximum(gd), p * 1.1))

    fig = CairoMakie.Figure(size=(700, 500))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel=string(yf),
        title="Gateaux Derivative (bound q = $(round(p; digits=1)))")

    CairoMakie.scatter!(ax, x_vals, y_vals;
        color=gd, markersize=8,
        colorrange=crange)
    CairoMakie.Colorbar(fig[1, 2];
        colorrange=crange, label="Gateaux derivative")

    # Mark points exceeding the bound
    above = gd .> p
    if any(above)
        CairoMakie.scatter!(ax, x_vals[above], y_vals[above];
            color=:transparent, strokecolor=:red, strokewidth=2,
            markersize=12, marker=:circle, label="Above bound")
    end

    fig
end

"""
    plot_residuals(log::ExperimentLog)

Plot residual diagnostics from an experiment log.
"""
function plot_residuals(log::ExperimentLog)
    n = length(log)
    steps = 1:n
    residuals = [h.diagnostics.mean_residual for h in log]
    log_evidence = log_evidence_series(log)

    # Handle scalar vs vector residuals
    resid_scalar = [r isa Real ? r : norm(r) for r in residuals]

    fig = CairoMakie.Figure(size=(600, 500))

    ax1 = CairoMakie.Axis(fig[1, 1],
        ylabel="Mean residual",
        title="Observation Diagnostics")
    CairoMakie.scatter!(ax1, steps, resid_scalar, color=:blue, markersize=6)
    CairoMakie.hlines!(ax1, [0], color=:gray, linestyle=:dash)

    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel="Step",
        ylabel="Log marginal likelihood")
    CairoMakie.lines!(ax2, steps, log_evidence, color=:blue, linewidth=1.5)
    CairoMakie.scatter!(ax2, steps, log_evidence, color=:blue, markersize=5)

    fig
end
