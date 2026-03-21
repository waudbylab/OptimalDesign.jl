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

# Problem specification
export DesignProblem, select,
       DCriterion, ACriterion, ECriterion

# Posterior
export ParticlePosterior

# Design workflows
export design, run_batch, run_adaptive

# Inference and analysis
export posterior_mean, effective_sample_size,
       posterior_predictions, credible_band,
       efficiency

# Plotting
export plot_corner, plot_residuals, record_corner_animation

# Types and problem definition
include("types.jl")
include("problem.jl")
include("sampling.jl")
include("information.jl")
include("utility.jl")
include("posteriors/particle.jl")
include("predictions.jl")

# Design optimisation
include("select.jl")
include("sequencing.jl")
include("exchange.jl")
include("gateaux.jl")
include("efficiency.jl")

# Experiment loop and logging
include("log.jl")
include("experiment.jl")

# Plotting
include("plotting/predictions.jl")
include("plotting/design.jl")
include("plotting/posterior.jl")
include("plotting/dashboard.jl")

end
