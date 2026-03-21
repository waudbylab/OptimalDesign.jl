"""
    ExperimentLog

Records the full experiment history for post-hoc analysis, trajectory replay,
and reproducibility.  When `record_posterior=true` is passed to `run_adaptive`,
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
