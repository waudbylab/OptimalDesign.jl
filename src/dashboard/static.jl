"""
    posterior_predictions(prob, posterior, ξ_grid; n_samples=200, component=0)

Generate posterior predictions over a grid of design points.
Returns a matrix of size (n_samples, length(ξ_grid)).

For vector-valued models, `component` selects which output to extract:
- `component=0` (default): first element (backward compatible)
- `component=k`: k-th element of the prediction vector
"""
function posterior_predictions(
    prob::AbstractDesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    n_samples::Int=200,
    component::Int=0,
)
    particles = sample(posterior, n_samples)
    predictions = Matrix{Float64}(undef, n_samples, length(ξ_grid))

    for (j, ξ) in enumerate(ξ_grid)
        for i in 1:n_samples
            y = prob.predict(particles[i], ξ)
            if y isa Real
                predictions[i, j] = y
            elseif component > 0
                predictions[i, j] = y[component]
            else
                predictions[i, j] = first(y)
            end
        end
    end

    predictions
end

"""
    posterior_predictions_vec(prob, posterior, ξ_grid; n_samples=200)

Generate posterior predictions for vector-valued models, returning all components
from a **single** particle sample for consistency.

Returns a `Vector` of matrices (one per output component), each of size
`(n_samples, length(ξ_grid))`.
"""
function posterior_predictions_vec(
    prob::AbstractDesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    n_samples::Int=200,
)
    particles = sample(posterior, n_samples)

    # Determine number of outputs from first prediction
    y0 = prob.predict(first(particles), first(ξ_grid))
    n_out = y0 isa Real ? 1 : length(y0)

    preds = [Matrix{Float64}(undef, n_samples, length(ξ_grid)) for _ in 1:n_out]

    for (j, ξ) in enumerate(ξ_grid)
        for i in 1:n_samples
            y = prob.predict(particles[i], ξ)
            if y isa Real
                preds[1][i, j] = y
            else
                for c in 1:n_out
                    preds[c][i, j] = y[c]
                end
            end
        end
    end

    preds
end

"""
    credible_band(predictions; level=0.9)

Compute credible band from posterior predictions.
Returns `(lower, median, upper)` vectors.
"""
function credible_band(predictions::AbstractMatrix; level::Real=0.9)
    α = (1 - level) / 2
    n_grid = size(predictions, 2)

    lower = Vector{Float64}(undef, n_grid)
    med = Vector{Float64}(undef, n_grid)
    upper = Vector{Float64}(undef, n_grid)

    for j in 1:n_grid
        col = sort(predictions[:, j])
        lower[j] = quantile(col, α)
        med[j] = quantile(col, 0.5)
        upper[j] = quantile(col, 1 - α)
    end

    (lower=lower, median=med, upper=upper)
end

# --- CairoMakie plotting functions ---
# These use CairoMakie for publication-quality static plots.

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

"""
    plot_posterior_marginals(posterior; params=nothing)

Plot histogram of posterior marginals for each parameter.
Kept for backward compatibility — see `plot_corner` for the full corner plot.
"""
function plot_posterior_marginals(
    posterior::ParticlePosterior;
    params::Union{Nothing,Vector{Symbol}}=nothing,
)
    plot_corner(posterior; params=params)
end

"""
    plot_corner(posteriors...; params, truth, labels, colors, bins, level)

Corner plot (pair plot) showing 1D marginal histograms on the diagonal
and 2D weighted scatter plots on the lower triangle.

Accepts one or more `ParticlePosterior` objects for overlay comparison
(e.g. prior vs posterior, or optimal vs uniform).

# Keyword arguments
- `params::Vector{Symbol}`: which parameters to show (default: all)
- `truth::Union{Nothing, NamedTuple, AbstractVector}`: true parameter values to mark
- `labels::Vector{String}`: legend labels for each posterior (default: "1", "2", …)
- `colors`: color per posterior (default: blue, orange, green, …)
- `bins::Int = 30`: histogram bin count
- `level::Real = 0.9`: credible interval level for 1D marginals
"""
function plot_corner(
    posteriors::ParticlePosterior...;
    params::Union{Nothing,Vector{Symbol}}=nothing,
    truth=nothing,
    labels::Union{Nothing,Vector{String}}=nothing,
    colors=nothing,
    bins::Int=30,
    level::Real=0.9,
)
    n_dist = length(posteriors)
    n_dist >= 1 || error("At least one posterior required")

    # Default parameter names from first posterior
    θ1 = first(first(posteriors).particles)
    names = params !== nothing ? params : collect(keys(θ1))
    d = length(names)

    # Default colours and labels
    default_colors = [(:royalblue, 0.6), (:orange, 0.6), (:green, 0.6), (:purple, 0.6)]
    cs = colors !== nothing ? colors : default_colors[1:min(n_dist, 4)]
    ls = labels !== nothing ? labels : [string(i) for i in 1:n_dist]

    # Extract weighted samples for each posterior
    all_vals = Vector{Vector{Vector{Float64}}}(undef, n_dist)   # [dist][param][particle]
    all_w = Vector{Vector{Float64}}(undef, n_dist)
    for (di, post) in enumerate(posteriors)
        w = exp.(post.log_weights .- logsumexp(post.log_weights))
        all_w[di] = w
        vals = [Float64[getproperty(p, name) for p in post.particles] for name in names]
        all_vals[di] = vals
    end

    # Truth values
    truth_vals = if truth !== nothing
        Float64[getproperty(truth, name) for name in names]
    else
        nothing
    end

    fig = CairoMakie.Figure(size=(250 * d, 250 * d))

    # Store axes for linking
    axes_grid = Matrix{Any}(nothing, d, d)

    for i in 1:d
        for j in 1:d
            if j > i
                # Upper triangle: empty
                continue
            end

            # Axis labels: only on edges
            xlabel = i == d ? string(names[j]) : ""
            ylabel = j == 1 && i > 1 ? string(names[i]) : (j == 1 && i == 1 ? "Density" : "")

            if i == j
                # ── Diagonal: 1D marginal histogram ──
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel,
                    title=i == 1 ? string(names[i]) : "")

                for di in 1:n_dist
                    vals = all_vals[di][i]
                    w = all_w[di]
                    CairoMakie.hist!(ax, vals; weights=w, bins, color=cs[di],
                        label=n_dist > 1 ? ls[di] : nothing,
                        normalization=:pdf)
                end

                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=2, linestyle=:dash)
                end

                if i == 1 && n_dist > 1
                    CairoMakie.axislegend(ax; position=:rt, framevisible=false)
                end

                # Hide y-axis ticks on diagonal except first
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end

            else
                # ── Lower triangle: 2D scatter ──
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel)

                for di in 1:n_dist
                    xv = all_vals[di][j]
                    yv = all_vals[di][i]
                    w = all_w[di]

                    # Resample for visual clarity (weighted scatter)
                    n_draw = min(500, length(xv))
                    idx = _weighted_sample_indices(w, n_draw)

                    base_color = cs[di] isa Tuple ? cs[di][1] : cs[di]
                    CairoMakie.scatter!(ax, xv[idx], yv[idx];
                        color=(base_color, 0.2), markersize=5,
                        label=n_dist > 1 ? ls[di] : nothing)
                end

                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[j]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                    CairoMakie.hlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                end

                # Hide interior tick labels
                if i < d
                    CairoMakie.hidexdecorations!(ax; grid=false)
                end
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end
            end

            axes_grid[i, j] = ax
        end
    end

    # Link axes: same column shares x-limits, same row shares y-limits (for off-diag)
    for j in 1:d
        col_axes = [axes_grid[i, j] for i in j:d if axes_grid[i, j] !== nothing]
        length(col_axes) > 1 && CairoMakie.linkxaxes!(col_axes...)
    end
    for i in 2:d
        row_axes = [axes_grid[i, j] for j in 1:i-1 if axes_grid[i, j] !== nothing]
        length(row_axes) > 1 && CairoMakie.linkyaxes!(row_axes...)
    end

    fig
end

"""Systematic resampling of n indices proportional to weights (for plotting)."""
function _weighted_sample_indices(w::AbstractVector, n::Int)
    cumw = cumsum(w)
    u = rand() / n
    indices = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        target = u + (i - 1) / n
        while j < length(cumw) && cumw[j] < target
            j += 1
        end
        indices[i] = j
    end
    indices
end

# --- Corner plot rendering helper (shared by plot_corner and animation) ---

"""
    _draw_corner_data!(fig, axes_grid, datasets, names, truth_vals; bins, cs, ls)

Render corner plot data into an existing figure/axes grid.
Each dataset is `(vals, weights)` where `vals[param_idx][particle_idx]`.
Clears axes before drawing.
"""
function _draw_corner_data!(fig, axes_grid, datasets, names, truth_vals;
                            bins::Int=30, cs=nothing, ls=nothing)
    d = length(names)
    n_dist = length(datasets)
    default_colors = [(:royalblue, 0.6), (:orange, 0.6), (:green, 0.6), (:purple, 0.6)]
    cs = cs !== nothing ? cs : default_colors[1:min(n_dist, 4)]
    ls = ls !== nothing ? ls : [string(i) for i in 1:n_dist]

    for i in 1:d
        for j in 1:d
            j > i && continue
            ax = axes_grid[i, j]
            ax === nothing && continue

            # Clear previous plot data
            empty!(ax)

            if i == j
                # 1D histogram
                for di in 1:n_dist
                    vals, w = datasets[di]
                    CairoMakie.hist!(ax, vals[i]; weights=w, bins, color=cs[di],
                        label=n_dist > 1 ? ls[di] : nothing,
                        normalization=:pdf)
                end
                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=2, linestyle=:dash)
                end
                if i == 1 && n_dist > 1
                    CairoMakie.axislegend(ax; position=:rt, framevisible=false)
                end
            else
                # 2D scatter
                for di in 1:n_dist
                    vals, w = datasets[di]
                    n_draw = min(500, length(vals[j]))
                    idx = _weighted_sample_indices(w, n_draw)
                    base_color = cs[di] isa Tuple ? cs[di][1] : cs[di]
                    CairoMakie.scatter!(ax, vals[j][idx], vals[i][idx];
                        color=(base_color, 0.2), markersize=5,
                        label=n_dist > 1 ? ls[di] : nothing)
                end
                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[j]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                    CairoMakie.hlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                end
            end
        end
    end
end

"""
    _make_corner_axes(fig, names)

Create the d×d grid of axes for a corner plot. Returns axes_grid matrix.
"""
function _make_corner_axes(fig, names)
    d = length(names)
    axes_grid = Matrix{Any}(nothing, d, d)

    for i in 1:d
        for j in 1:d
            j > i && continue

            xlabel = i == d ? string(names[j]) : ""
            ylabel = j == 1 && i > 1 ? string(names[i]) : (j == 1 && i == 1 ? "Density" : "")

            if i == j
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel,
                    title=i == 1 ? string(names[i]) : "")
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end
            else
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel)
                if i < d
                    CairoMakie.hidexdecorations!(ax; grid=false)
                end
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end
            end

            axes_grid[i, j] = ax
        end
    end

    # Link axes
    for j in 1:d
        col_axes = [axes_grid[i, j] for i in j:d if axes_grid[i, j] !== nothing]
        length(col_axes) > 1 && CairoMakie.linkxaxes!(col_axes...)
    end
    for i in 2:d
        row_axes = [axes_grid[i, j] for j in 1:i-1 if axes_grid[i, j] !== nothing]
        length(row_axes) > 1 && CairoMakie.linkyaxes!(row_axes...)
    end

    axes_grid
end

"""
    _snapshot_to_dataset(snapshot, names)

Convert a posterior snapshot (particles, log_weights) to (vals, weights) for plotting.
"""
function _snapshot_to_dataset(snapshot, names)
    w = exp.(snapshot.log_weights .- logsumexp(snapshot.log_weights))
    vals = [Float64[getproperty(p, name) for p in snapshot.particles] for name in names]
    (vals, w)
end

"""
    record_corner_animation(log, filename; kwargs...)

Record an animation of the posterior evolving from prior through each observation.
Requires `record_posterior=true` in `run_experiment`.

Produces an MP4 or GIF (determined by filename extension) using CairoMakie.

# Keyword arguments
- `params::Vector{Symbol}`: which parameters to show (default: all)
- `truth`: true parameter values to mark with dashed crosshairs
- `step_interval::Int = 1`: show every Nth step (default: every step)
- `framerate::Int = 5`: frames per second
- `bins::Int = 30`: histogram bin count
- `prior_color = (:gray, 0.3)`: colour for the prior overlay
- `posterior_color = (:royalblue, 0.6)`: colour for the current posterior
"""
function record_corner_animation(
    log::ExperimentLog,
    filename::String;
    params::Union{Nothing,Vector{Symbol}}=nothing,
    truth=nothing,
    step_interval::Int=1,
    framerate::Int=5,
    bins::Int=30,
    prior_color=(:gray, 0.3),
    posterior_color=(:royalblue, 0.6),
)
    has_posterior_history(log) || error(
        "ExperimentLog has no posterior snapshots. " *
        "Use record_posterior=true in run_experiment.")

    # Determine parameter names
    θ1 = first(log.prior_snapshot.particles)
    names = params !== nothing ? params : collect(keys(θ1))
    d = length(names)

    # Truth values
    truth_vals = if truth !== nothing
        Float64[getproperty(truth, name) for name in names]
    else
        nothing
    end

    # Build prior dataset (shown faded on every frame)
    prior_data = _snapshot_to_dataset(log.prior_snapshot, names)

    # Build frame indices: 0 = prior only, 1:N = after each observation
    n_steps = length(log)
    all_steps = 0:n_steps
    frame_steps = collect(all_steps[1:step_interval:end])
    if isempty(frame_steps) || last(frame_steps) != n_steps
        push!(frame_steps, n_steps)
    end

    # Create figure and axes
    fig = CairoMakie.Figure(size=(250 * d, 250 * d))
    axes_grid = _make_corner_axes(fig, names)

    # Add a supertitle for step counter
    title_label = CairoMakie.Label(fig[0, 1:d], "Prior (step 0)", fontsize=16)

    cs = [prior_color, posterior_color]
    ls = ["Prior", "Posterior"]

    @info "Recording corner animation: $(length(frame_steps)) frames → $filename"

    CairoMakie.record(fig, filename, frame_steps; framerate=framerate) do step_idx
        # Get current posterior snapshot
        if step_idx == 0
            # Just prior
            datasets = [(prior_data[1], prior_data[2])]
            _draw_corner_data!(fig, axes_grid, datasets, names, truth_vals;
                bins=bins, cs=[posterior_color], ls=["Prior"])
            title_label.text[] = "Prior (step 0 / $n_steps)"
        else
            snapshot = log[step_idx].posterior_snapshot
            post_data = _snapshot_to_dataset(snapshot, names)

            # ESS for display
            lw = snapshot.log_weights .- logsumexp(snapshot.log_weights)
            ess = round(Int, exp(-logsumexp(2 .* lw)))

            datasets = [prior_data, post_data]
            _draw_corner_data!(fig, axes_grid, datasets, names, truth_vals;
                bins=bins, cs=cs, ls=ls)
            title_label.text[] = "Step $step_idx / $n_steps (ESS=$ess)"
        end
    end

    @info "Animation saved: $filename"
    filename
end
