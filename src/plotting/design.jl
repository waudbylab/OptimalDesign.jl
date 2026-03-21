"""
    plot_design_allocation(candidates, weights; x_field)

Plot the weight distribution across candidates for a batch design.
"""
function plot_design_allocation(
    candidates::AbstractVector{<:NamedTuple},
    weights::AbstractVector;
    x_field::Symbol=first(keys(first(candidates))),
)
    x_vals = [getfield(ξ, x_field) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Weight",
        title="Design Allocation")

    CairoMakie.stem!(ax, x_vals, weights, color=:blue)

    fig
end

"""
    plot_gateaux(candidates, gd, weights; x_field)

Plot the Gateaux derivative at each candidate, with the optimality bound.
"""
function plot_gateaux(
    candidates::AbstractVector{<:NamedTuple},
    gd::AbstractVector,
    p::Int;
    x_field::Symbol=first(keys(first(candidates))),
)
    x_vals = [getfield(ξ, x_field) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Gateaux derivative",
        title="Optimality Check")

    CairoMakie.lines!(ax, x_vals, gd, color=:blue, linewidth=1.5)
    CairoMakie.hlines!(ax, [p], color=:red, linestyle=:dash, label="p = $p")
    CairoMakie.axislegend(ax)

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
