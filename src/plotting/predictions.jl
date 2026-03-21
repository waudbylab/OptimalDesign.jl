"""
    plot_credible_bands(prob, posterior, ξ_grid; level=0.9, observations=nothing)

Plot posterior credible bands with optional observation overlay.
"""
function plot_credible_bands(
    prob::AbstractDesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    level::Real=0.9,
    observations=nothing,
    x_field::Symbol=first(keys(first(ξ_grid))),
    n_samples::Int=200,
)
    preds = posterior_predictions(prob, posterior, ξ_grid; n_samples=n_samples)
    band = credible_band(preds; level=level)

    x_vals = [getfield(ξ, x_field) for ξ in ξ_grid]

    fig = CairoMakie.Figure(size=(600, 400))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Prediction",
        title="Posterior Credible Band ($(round(Int, level*100))%)")

    CairoMakie.band!(ax, x_vals, band.lower, band.upper, color=(:blue, 0.2))
    CairoMakie.lines!(ax, x_vals, band.median, color=:blue, linewidth=2)

    if observations !== nothing
        obs_x = [getfield(o.ξ, x_field) for o in observations]
        obs_y = [o.y isa NamedTuple ? o.y.value : o.y for o in observations]
        CairoMakie.scatter!(ax, obs_x, obs_y, color=:red, markersize=8)
    end

    fig
end
