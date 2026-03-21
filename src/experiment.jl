"""
    ExperimentLog

Records the full experiment history for post-hoc analysis, trajectory replay,
and reproducibility.  When `record_posterior=true` is passed to `run_experiment`,
each entry includes a snapshot of the posterior (particles + log_weights) at that step.
"""
struct ExperimentLog
    history::Vector{NamedTuple}
    prior_snapshot::Union{Nothing,NamedTuple}   # (particles, log_weights) before any data
end

ExperimentLog(; prior_snapshot=nothing) = ExperimentLog(NamedTuple[], prior_snapshot)

Base.push!(log::ExperimentLog, entry::NamedTuple) = push!(log.history, entry)
Base.length(log::ExperimentLog) = length(log.history)
Base.getindex(log::ExperimentLog, i) = log.history[i]
Base.iterate(log::ExperimentLog, args...) = iterate(log.history, args...)
Base.lastindex(log::ExperimentLog) = lastindex(log.history)

"""
    design_points(log::ExperimentLog)

Extract all design points from the experiment log.
"""
design_points(log::ExperimentLog) = [h.ξ for h in log.history]

"""
    observations(log::ExperimentLog)

Extract all observations from the experiment log.
"""
observations(log::ExperimentLog) = [h.y for h in log.history]

"""
    cumulative_cost(log::ExperimentLog)

Compute the cumulative cost at each step.
"""
function cumulative_cost(log::ExperimentLog)
    cumsum([h.cost for h in log.history])
end

"""
    log_evidence_series(log::ExperimentLog)

Extract the series of log marginal likelihood values (sequential model checking).
"""
function log_evidence_series(log::ExperimentLog)
    [h.diagnostics.log_marginal for h in log.history]
end

"""
    has_posterior_history(log::ExperimentLog)

Check whether the log contains posterior snapshots for animation/replay.
"""
has_posterior_history(log::ExperimentLog) =
    log.prior_snapshot !== nothing && !isempty(log.history) &&
    hasproperty(first(log.history), :posterior_snapshot)

"""
    posterior_snapshots(log::ExperimentLog)

Return vector of (particles, log_weights) snapshots, one per observation step.
"""
function posterior_snapshots(log::ExperimentLog)
    [h.posterior_snapshot for h in log.history if hasproperty(h, :posterior_snapshot) && h.posterior_snapshot !== nothing]
end

"""
    _snapshot_posterior(posterior::ParticlePosterior)

Deep-copy the current posterior state for history recording.
"""
_snapshot_posterior(posterior::ParticlePosterior) =
    (particles=copy(posterior.particles), log_weights=copy(posterior.log_weights))

# --- Logging helpers ---

"""Format a design point NamedTuple for compact display."""
function _format_design_point(ξ::NamedTuple)
    parts = [string(k, "=", round(v; digits=4)) for (k, v) in pairs(ξ)
             if v isa Real]
    join(parts, ", ")
end

"""Format a parameter ComponentArray for compact display."""
function _format_params(θ)
    parts = [string(k, "=", round(getproperty(θ, k); digits=3))
             for k in keys(θ)]
    join(parts, ", ")
end

"""
    run_experiment(prob, candidates, posterior, acquire; kwargs...)

Run an adaptive experiment: select → acquire → update → display.

# Keyword arguments
- `budget`: total cost budget (required)
- `criterion = DCriterion()`: design criterion
- `posterior_samples = 50`: mini-batch for utility evaluation
- `n_per_step = 1`: measurements per adaptive step
- `headless = false`: suppress GUI for testing
- `prediction_grid = nothing`: dense ξ grid for credible band plots
- `record_posterior = false`: snapshot posterior at every step for animation/replay

Returns `(posterior=posterior, log=ExperimentLog)`.
"""
function run_experiment(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::ParticlePosterior,
    acquire;
    budget::Real,
    criterion::DesignCriterion=DCriterion(),
    posterior_samples::Int=50,
    n_per_step::Int=1,
    headless::Bool=false,
    prediction_grid=nothing,
    record_posterior::Bool=false,
)
    prior_snap = record_posterior ? _snapshot_posterior(posterior) : nothing
    log = ExperimentLog(; prior_snapshot=prior_snap)
    spent = 0.0
    ξ_prev = nothing
    step = 0
    obs_count = 0

    # Dashboard state
    dashboard = if !headless
        _create_dashboard(prob, posterior, prediction_grid, budget)
    else
        nothing
    end

    param_names = keys(prob.parameters)
    @info "Adaptive experiment: budget=$budget, $(length(candidates)) candidates, " *
          "$(length(param_names)) parameters ($(join(param_names, ", ")))"

    while spent < budget
        step += 1

        # Select next design point(s)
        # Pass prior_designs so greedy scorer evaluates marginal gain over
        # accumulated information (essential when n_per_step < p for scalar obs)
        design = select(prob, candidates, posterior;
            n=n_per_step, criterion=criterion,
            posterior_samples=posterior_samples, ξ_prev=ξ_prev,
            budget=budget - spent, #exchange_algorithm=false,
            prior_designs=design_points(log))

        isempty(design) && break

        for (ξ, count) in design
            for _ in 1:count
                c = total_cost(prob, ξ_prev, ξ)
                if spent + c > budget
                    @goto done
                end

                # Acquire observation (on background task if dashboard active)
                y = if dashboard !== nothing
                    # Check for pause/stop
                    _check_controls(dashboard) == :stop && @goto done
                    while _check_controls(dashboard) == :pause
                        sleep(0.1)
                    end
                    fetch(Threads.@spawn acquire(ξ))
                else
                    acquire(ξ)
                end

                spent += c
                obs_count += 1

                # ESS before update (to detect resampling)
                ess_before = effective_sample_size(posterior)

                # Diagnostics before update
                diag = observation_diagnostics(posterior, prob, ξ, y)

                # Update posterior
                update!(posterior, prob, ξ, y)

                ess_after = effective_sample_size(posterior)
                resampled = ess_after > ess_before + 10

                # Snapshot posterior if recording history
                snapshot = record_posterior ? _snapshot_posterior(posterior) : nothing

                # Record
                push!(log, (ξ=ξ, y=y, cost=c, diagnostics=diag, step=step,
                    posterior_snapshot=snapshot))

                # Log: per-step detail at @debug, periodic summaries at @info
                y_str = y isa Real ? round(y; digits=4) : y
                resample_flag = resampled ? " [resampled]" : ""
                @debug "Step $obs_count: ξ=($(_format_design_point(ξ))), " *
                       "y=$y_str, cost=$(round(c; digits=2)), " *
                       "spent=$(round(spent; digits=2))/$budget, " *
                       "ESS=$(round(ess_after; digits=0)), " *
                       "log_ml=$(round(diag.log_marginal; digits=2))$resample_flag"

                if resampled
                    @warn "ESS dropped below threshold, resampled at step $obs_count " *
                          "(ESS: $(round(ess_before; digits=0)) → $(round(ess_after; digits=0)))"
                end

                # Periodic @info summary
                if obs_count % 10 == 0
                    μ = posterior_mean(posterior)
                    @info "  Step $obs_count/$(_est_total_steps(spent, budget, obs_count)): " *
                          "spent=$(round(spent; digits=2))/$budget, " *
                          "ESS=$(round(ess_after; digits=0)), " *
                          "posterior=($(_format_params(μ)))"
                end

                # Update dashboard
                if dashboard !== nothing
                    _update_dashboard!(dashboard, prob, posterior, log,
                        spent, budget, prediction_grid)
                end

                ξ_prev = ξ
            end
        end
    end

    @label done

    if dashboard !== nothing
        _finalize_dashboard(dashboard)
    end

    # Final summary
    μ_final = posterior_mean(posterior)
    ess_final = effective_sample_size(posterior)
    @info "Experiment complete: $obs_count observations, " *
          "cost=$(round(spent; digits=2))/$budget, " *
          "ESS=$(round(ess_final; digits=0))"
    @info "  Final posterior: $(_format_params(μ_final))"

    (posterior=posterior, log=log)
end

"""Estimate total steps from current pace."""
_est_total_steps(spent, budget, obs_count) =
    spent > 0 ? round(Int, obs_count * budget / spent) : "?"
