module OptimalDesign

using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using LogExpFunctions: logsumexp
using Random
using Statistics

import CairoMakie
import GLMakie

export DesignProblem, Identity, DeltaMethod, select,
       DCriterion, ACriterion, ECriterion,
       information, information!, transform, weighted_fim, weighted_fim!,
       expected_utility, score_candidates,
       ParticlePosterior, sample, posterior_mean,
       effective_sample_size, update!, draw,
       loglikelihood,
       # Phase 2
       exchange, apportion, efficiency, uniform_allocation,
       gateaux_derivative, verify_optimality,
       # Phase 3
       ExperimentLog, run_experiment, observation_diagnostics,
       design_points, observations, cumulative_cost, log_evidence_series,
       has_posterior_history, posterior_snapshots,
       # Phase 4
       posterior_predictions, posterior_predictions_vec, credible_band,
       plot_credible_bands, plot_design_allocation,
       plot_gateaux, plot_residuals, plot_posterior_marginals, plot_corner,
       record_corner_animation

# Phase 1: Types and FIM
include("types.jl")
include("information.jl")
include("utility.jl")
include("posteriors/particle.jl")

# Phase 2: Solver
include("select.jl")
include("exchange.jl")
include("gateaux.jl")
include("efficiency.jl")

# Phase 3: Adaptive loop and diagnostics
include("diagnostics.jl")
include("experiment.jl")

# Phase 4: Dashboard
include("dashboard/static.jl")
include("dashboard/live.jl")

end
