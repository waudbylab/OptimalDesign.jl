# Example 5: Exponential Decay — Adaptive Design with Simulated Acquisition
#
# Uses Example 1's exponential decay model but runs a full adaptive experiment
# against a simulated ground truth, then compares adaptive vs batch posterior.
#
# Demonstrates:
#   1. Simulated acquisition function
#   2. run_experiment for adaptive sequential design (headless)
#   3. Posterior convergence tracking
#   4. Observation diagnostics (log marginal likelihood, residuals)
#   5. Head-to-head: adaptive vs batch posterior precision
#   6. Design point trajectory (where the adaptive algorithm chooses to measure)


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using GLMakie

ENV["JULIA_DEBUG"] = OptimalDesign
#Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════

prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters=(A=LogUniform(0.1, 10), R₂=Uniform(1, 50)),
    transformation=select(:R₂),
    sigma=(θ, ξ) -> 0.2,
    cost=(prev, ξ) -> ξ.t + 1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]

# Ground truth (unknown to algorithm)
θ_true = ComponentArray(A=2, R₂=42.0)
σ_true = 0.05

# Simulated acquisition function (closure over ground truth)
acquire = let θ = θ_true, σ = σ_true
    ξ -> θ.A * exp(-θ.R₂ * ξ.t) + σ * randn()
end

println("Problem: y = A exp(-R₂ t) + noise")
println("Truth:   A = $(θ_true.A), R₂ = $(θ_true.R₂)")
println("Design:  Adaptive sequential, Ds-optimal for R₂")
println()

# ═══════════════════════════════════════════════
# 2. Run adaptive experiment via run_experiment
# ═══════════════════════════════════════════════

budget = 20.0

println("Running adaptive experiment (budget=$budget)...")
prior_adaptive = ParticlePosterior(prob, 1000)

result = run_experiment(
    prob, candidates, prior_adaptive, acquire;
    budget=budget,
    criterion=DCriterion(),
    # posterior_samples=200,
    n_per_step=1,
    headless=false,
    record_posterior=true,
)

posterior_adaptive = result.posterior
log_adaptive = result.log

n_adaptive = length(log_adaptive)
spent_adaptive = sum(e.cost for e in log_adaptive)
μ_adaptive = posterior_mean(posterior_adaptive)

# run_experiment already logged step-by-step; just print the summary
println("\nAdaptive results:")
println("  Measurements: $n_adaptive")
println("  Budget spent: $(round(spent_adaptive; digits=2)) / $budget")
println("  Posterior mean: A=$(round(μ_adaptive.A; digits=4)), R₂=$(round(μ_adaptive.R₂; digits=2))")

# ═══════════════════════════════════════════════
# 3. Batch design for comparison (same budget → same n)
# ═══════════════════════════════════════════════

println("\n--- Batch design comparison (n=$n_adaptive) ---")
prior_batch = ParticlePosterior(prob, 1000)

batch_design = select(prob, candidates, prior_batch;
    n=n_adaptive, criterion=DCriterion(), exchange_algorithm=true,
    exchange_steps=200)

println("Batch design allocation:")
for (ξ, count) in batch_design
    bar = repeat("█", count)
    println("  t = $(round(ξ.t; digits=4))  ×$(count)  $bar")
end

# Simulate the batch experiment with the same truth
posterior_batch = ParticlePosterior(prob, 1000)
obs_batch = NamedTuple[]

for (ξ, count) in batch_design
    for _ in 1:count
        y = prob.predict(θ_true, ξ) + σ_true * randn()
        OptimalDesign.update!(posterior_batch, prob, ξ, y)
        push!(obs_batch, (ξ=ξ, y=y))
    end
end

μ_batch = posterior_mean(posterior_batch)
println("\nBatch results:")
println("  Posterior mean: A=$(round(μ_batch.A; digits=4)), R₂=$(round(μ_batch.R₂; digits=2))")

# ═══════════════════════════════════════════════
# 4. Comparison summary
# ═══════════════════════════════════════════════

err_adaptive = abs(μ_adaptive.R₂ - θ_true.R₂)
err_batch = abs(μ_batch.R₂ - θ_true.R₂)

println("\n=== Head-to-head comparison ===")
println("  Adaptive |R₂ error|: $(round(err_adaptive; digits=2))")
println("  Batch    |R₂ error|: $(round(err_batch; digits=2))")
println("  (Both use $n_adaptive measurements)")

# ═══════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(t=t,) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid]
y_true = [prob.predict(θ_true, ξ) for ξ in prediction_grid]

# --- Figure 1: Adaptive design trajectory ---
# Where did the adaptive algorithm choose to measure over time?

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Design time t",
    title="Adaptive Design Trajectory")
scatter!(ax1a, 1:n_adaptive, [e.ξ.t for e in log_adaptive],
    color=1:n_adaptive, colormap=:viridis, markersize=8)
lines!(ax1a, 1:n_adaptive, [e.ξ.t for e in log_adaptive],
    color=:gray, linewidth=0.5)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="Step", ylabel="Log marginal likelihood",
    title="Sequential Model Checking")
log_ml = log_evidence_series(log_adaptive)
lines!(ax1b, 1:n_adaptive, log_ml, color=:blue, linewidth=1.5)
scatter!(ax1b, 1:n_adaptive, log_ml, color=:blue, markersize=5)

fig1
# save("ex5_adaptive_trajectory.pdf", fig1)

# --- Figure 2: Adaptive vs Batch posterior credible bands ---

preds_adaptive = posterior_predictions(prob, posterior_adaptive, prediction_grid; n_samples=200)
band_adaptive = credible_band(preds_adaptive; level=0.9)

preds_batch = posterior_predictions(prob, posterior_batch, prediction_grid; n_samples=200)
band_batch = credible_band(preds_batch; level=0.9)

obs_adaptive = [(ξ=e.ξ, y=e.y) for e in log_adaptive]

fig2 = Figure(size=(700, 500))

ax2a = GLMakie.Axis(fig2[1, 1], ylabel="y",
    title="Posterior — Adaptive ($n_adaptive measurements)")
band!(ax2a, x_grid, band_adaptive.lower, band_adaptive.upper, color=(:blue, 0.3))
lines!(ax2a, x_grid, band_adaptive.median, color=:blue, linewidth=2)
lines!(ax2a, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2a, [o.ξ.t for o in obs_adaptive], [o.y for o in obs_adaptive],
    color=:black, markersize=6, label="Observations")
axislegend(ax2a)

ax2b = GLMakie.Axis(fig2[2, 1], xlabel="t", ylabel="y",
    title="Posterior — Batch ($n_adaptive measurements)")
band!(ax2b, x_grid, band_batch.lower, band_batch.upper, color=(:orange, 0.3))
lines!(ax2b, x_grid, band_batch.median, color=:orange, linewidth=2)
lines!(ax2b, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2b, [o.ξ.t for o in obs_batch], [o.y for o in obs_batch],
    color=:black, markersize=6)
for (ξ, count) in batch_design
    vlines!(ax2b, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end

fig2
# save("ex5_adaptive_vs_batch.pdf", fig2)

# --- Figure 3: Corner plot — adaptive vs batch posterior ---

fig3 = plot_corner(posterior_adaptive, posterior_batch;
    params=[:A, :R₂], labels=["Adaptive", "Batch"],
    truth=(A=θ_true.A, R₂=θ_true.R₂))
# save("ex5_corner_adaptive_vs_batch.pdf", fig3)

# --- Figure 4: Observation diagnostics from adaptive run ---

fig4 = plot_residuals(log_adaptive)
# save("ex5_diagnostics.pdf", fig4)

# --- Figure 5: ESS evolution ---

# Track how the effective sample size evolves
# (We need to rerun the adaptive experiment tracking ESS at each step)
println("\nRerunning adaptive experiment to track ESS evolution...")
prior_ess = ParticlePosterior(prob, 1000)
ess_history = Float64[]
r2_history = Float64[]

for entry in log_adaptive
    OptimalDesign.update!(prior_ess, prob, entry.ξ, entry.y)
    push!(ess_history, effective_sample_size(prior_ess))
    push!(r2_history, posterior_mean(prior_ess).R₂)
end

fig5 = Figure(size=(700, 500))

ax5a = GLMakie.Axis(fig5[1, 1], ylabel="R₂ estimate",
    title="Posterior Convergence")
lines!(ax5a, 1:n_adaptive, r2_history, color=:blue, linewidth=2)
hlines!(ax5a, [θ_true.R₂], color=:red, linestyle=:dash, label="Truth")
axislegend(ax5a)

ax5b = GLMakie.Axis(fig5[2, 1], xlabel="Step", ylabel="ESS",
    title="Effective Sample Size")
lines!(ax5b, 1:n_adaptive, ess_history, color=:blue, linewidth=2)
hlines!(ax5b, [100], color=:gray, linestyle=:dash, label="Warning threshold")
axislegend(ax5b)

fig5
# save("ex5_convergence.pdf", fig5)

# --- Figure 6: Animated corner plot ---

if has_posterior_history(log_adaptive)
    println("Recording posterior evolution animation...")
    record_corner_animation(log_adaptive, "ex5_posterior_evolution.mp4";
        params=[:A, :R₂],
        truth=(A=θ_true.A, R₂=θ_true.R₂),
        framerate=5)
end

println("Done. Figures created — uncomment save() calls to export PDFs.")
