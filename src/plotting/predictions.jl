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

"""
    plot_credible_bands(prob, posteriors, ξ_grid; labels, colors, truth, observations, ...)

Multi-panel credible band plot comparing several posteriors (e.g. prior, optimal, uniform).
Creates vertically stacked panels, one per posterior.

# Keyword arguments
- `labels::Vector{String}`: panel titles (default: "1", "2", …)
- `colors`: band colour per panel (default: gray, blue, orange, …)
- `truth`: callable or vector of true y-values to overlay as dashed red line
- `observations`: vector of observation vectors (one per posterior, or `nothing`)
- `level::Real = 0.9`: credible interval level
- `x_field::Symbol`: which field of ξ to use as x-axis
- `n_samples::Int = 200`: posterior samples for prediction
"""
function plot_credible_bands(
    prob::AbstractDesignProblem,
    posteriors::AbstractVector{<:ParticlePosterior},
    ξ_grid::AbstractVector;
    labels::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    colors=nothing,
    truth=nothing,
    observations=nothing,
    level::Real=0.9,
    x_field::Symbol=first(keys(first(ξ_grid))),
    n_samples::Int=200,
)
    n_panels = length(posteriors)
    default_colors = [(:gray, 0.3), (:blue, 0.3), (:orange, 0.3), (:green, 0.3)]
    cs = colors !== nothing ? colors : default_colors[1:min(n_panels, length(default_colors))]
    ls = labels !== nothing ? labels : [string(i) for i in 1:n_panels]

    x_vals = [getfield(ξ, x_field) for ξ in ξ_grid]

    # Compute truth curve if provided
    # truth can be: θ (parameter values → compute predictions), or a pre-computed y vector
    y_true = if truth === nothing
        nothing
    elseif truth isa AbstractVector{<:Real} && length(truth) == length(ξ_grid)
        truth
    else
        # Assume truth is a parameter vector/NamedTuple — compute predictions
        [prob.predict(truth, ξ) for ξ in ξ_grid]
    end

    # Detect whether model is vector-valued
    test_preds = posterior_predictions(prob, first(posteriors), ξ_grid; n_samples=2)
    is_vector = test_preds isa AbstractVector{<:AbstractMatrix}
    n_components = is_vector ? length(test_preds) : 1

    fig = CairoMakie.Figure(size=(600 * n_components, 300 * n_panels))

    for (i, post) in enumerate(posteriors)
        preds = posterior_predictions(prob, post, ξ_grid; n_samples=n_samples)

        for comp in 1:n_components
            comp_preds = is_vector ? preds[comp] : preds
            band = credible_band(comp_preds; level=level)

            ylabel = comp == 1 ? "Prediction" : ""
            xlabel = i == n_panels ? string(x_field) : ""
            comp_label = n_components > 1 ? " [$(comp)]" : ""
            ax = CairoMakie.Axis(fig[i, comp];
                xlabel, ylabel,
                title="$(ls[i])$comp_label ($(round(Int, level*100))% CI)")

            base_color = cs[i] isa Tuple ? cs[i] : (cs[i], 0.3)
            line_color = cs[i] isa Tuple ? cs[i][1] : cs[i]

            CairoMakie.band!(ax, x_vals, band.lower, band.upper, color=base_color)
            CairoMakie.lines!(ax, x_vals, band.median, color=line_color, linewidth=2)

            if y_true !== nothing
                yt = is_vector ? [y[comp] for y in y_true] : y_true
                CairoMakie.lines!(ax, x_vals, yt, color=:red, linewidth=1.5,
                    linestyle=:dash, label="Truth")
            end

            # Observation overlay — only on first component column for scalar obs
            obs = observations !== nothing && i <= length(observations) ? observations[i] : nothing
            if obs !== nothing
                obs_x = [getfield(o.ξ, x_field) for o in obs]
                if is_vector
                    obs_y = [o.y[comp] for o in obs]
                else
                    obs_y = [o.y isa NamedTuple ? o.y.value : o.y for o in obs]
                end
                CairoMakie.scatter!(ax, obs_x, obs_y, color=:black, markersize=5,
                    label="Observations")
            end

            # Hide x decorations on non-bottom panels
            if i < n_panels
                CairoMakie.hidexdecorations!(ax; grid=false)
            end
        end
    end

    fig
end
