# Example 6: Two Decays, Discrete Control — Adaptive with Switching Costs
#
# Uses Example 4's two-decay problem but runs adaptively, demonstrating how
# the selector balances measurements between the two decays with switching costs.
#
# Demonstrates: Adaptive design with discrete control variable, switching cost,
# block-sparse posterior updates, the selector preferring to stay on the current
# decay unless the other is substantially more informative.

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using GLMakie

ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════

θ_true = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)
σ_true = 0.05

acquire = let θ = θ_true, σ = σ_true
    ξ -> (ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t)) + σ * randn()
end

prob = DesignProblem(
    (θ, ξ) -> ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t),
    parameters=(A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> σ_true,
    cost=ξ -> ξ.t + 1,
    switching_cost=(:i, 50.0),
)

candidates = [(i=i, t=t) for i in [1, 2] for t in range(0.001, 0.5, length=200)]

# ═══════════════════════════════════════════════
# 2. Run adaptive experiment
# ═══════════════════════════════════════════════

budget = 200.0

println("Starting adaptive experiment with switching costs (budget=$budget)")
println("True: A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println()

prior_adaptive = ParticlePosterior(prob, 1000)

result = run_adaptive(
    prob, candidates, prior_adaptive, acquire;
    budget=budget,
    n_per_step=1,
    headless=true,
    record_posterior=true,
)

posterior_adaptive = result.posterior
log_adaptive = result.log

# Summary
n_adaptive = length(log_adaptive)
if n_adaptive == 0
    error("Adaptive experiment produced 0 observations — check select/cost configuration")
end
spent_adaptive = sum(e.cost for e in log_adaptive)
μ_adaptive = posterior_mean(posterior_adaptive)

n_decay1 = count(e -> e.ξ.i == 1, log_adaptive)
n_decay2 = count(e -> e.ξ.i == 2, log_adaptive)
n_switches = count(i -> log_adaptive[i].ξ.i != log_adaptive[i-1].ξ.i, 2:n_adaptive)

println("\n--- Adaptive Results ---")
println("Steps: $n_adaptive ($n_decay1 on decay 1, $n_decay2 on decay 2, $n_switches switches)")
println("Budget spent: $(round(spent_adaptive; digits=2)) / $budget")
println("Posterior mean: R₂₁=$(round(μ_adaptive.R₂₁; digits=2)), R₂₂=$(round(μ_adaptive.R₂₂; digits=2))")
println("True values:    R₂₁=$(θ_true.R₂₁), R₂₂=$(θ_true.R₂₂)")

# ═══════════════════════════════════════════════
# 3. Batch design for comparison
# ═══════════════════════════════════════════════

println("\n--- Batch design comparison (n=$n_adaptive) ---")
prior_batch = ParticlePosterior(prob, 1000)

batch_design = design(prob, candidates, prior_batch;
    n=n_adaptive, exchange_steps=200)

posterior_batch = ParticlePosterior(prob, 1000)
result_batch = run_batch(batch_design, prob, posterior_batch, acquire)

μ_batch = posterior_mean(result_batch.posterior)
println("Batch results:")
println("  Posterior mean: R₂₁=$(round(μ_batch.R₂₁; digits=2)), R₂₂=$(round(μ_batch.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 4. Comparison summary
# ═══════════════════════════════════════════════

err_adaptive_1 = abs(μ_adaptive.R₂₁ - θ_true.R₂₁)
err_adaptive_2 = abs(μ_adaptive.R₂₂ - θ_true.R₂₂)
err_batch_1 = abs(μ_batch.R₂₁ - θ_true.R₂₁)
err_batch_2 = abs(μ_batch.R₂₂ - θ_true.R₂₂)

println("\n=== Head-to-head comparison ===")
println("  Adaptive |R₂₁ error|: $(round(err_adaptive_1; digits=2)),  |R₂₂ error|: $(round(err_adaptive_2; digits=2))")
println("  Batch    |R₂₁ error|: $(round(err_batch_1; digits=2)),  |R₂₂ error|: $(round(err_batch_2; digits=2))")
println("  (Both use $n_adaptive measurements, adaptive has switching cost penalty)")

# ═══════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Adaptive trajectory showing decay selection and switching ---

fig1 = Figure(size=(800, 700))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Design time t",
    title="Adaptive Design Trajectory")

steps_1 = [i for i in 1:n_adaptive if log_adaptive[i].ξ.i == 1]
steps_2 = [i for i in 1:n_adaptive if log_adaptive[i].ξ.i == 2]
times_1 = [log_adaptive[i].ξ.t for i in steps_1]
times_2 = [log_adaptive[i].ξ.t for i in steps_2]

scatter!(ax1a, steps_1, times_1, color=:blue, markersize=8, label="Decay 1")
scatter!(ax1a, steps_2, times_2, color=:orange, markersize=8, label="Decay 2")

for i in 2:n_adaptive
    if log_adaptive[i].ξ.i != log_adaptive[i-1].ξ.i
        vlines!(ax1a, [i], color=(:red, 0.3), linewidth=1)
    end
end
axislegend(ax1a)

# Cumulative cost
ax1b = GLMakie.Axis(fig1[2, 1], ylabel="Cumulative cost",
    title="Budget Consumption ($n_switches switches)")
costs = [e.cost for e in log_adaptive]
cumcost = cumsum(costs)
lines!(ax1b, 1:n_adaptive, cumcost, color=:black, linewidth=2)
for i in 2:n_adaptive
    if log_adaptive[i].ξ.i != log_adaptive[i-1].ξ.i
        scatter!(ax1b, [i], [cumcost[i]], color=:red, markersize=10, marker=:diamond)
    end
end
hlines!(ax1b, [budget], color=:gray, linestyle=:dash)

# Log marginal likelihood
ax1c = GLMakie.Axis(fig1[3, 1], xlabel="Step", ylabel="Log marginal likelihood",
    title="Sequential Model Checking")
log_ml = OptimalDesign.log_evidence_series(log_adaptive)
lines!(ax1c, 1:n_adaptive, log_ml, color=:blue, linewidth=1.5)
scatter!(ax1c, 1:n_adaptive, log_ml, color=:blue, markersize=5)

fig1

# --- Figure 2: Adaptive vs batch credible bands (per decay) ---

prediction_grid_1 = [(i=1, t=t) for t in range(0.001, 0.5, length=100)]
prediction_grid_2 = [(i=2, t=t) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid_1]
y_true_1 = [prob.predict(θ_true, ξ) for ξ in prediction_grid_1]
y_true_2 = [prob.predict(θ_true, ξ) for ξ in prediction_grid_2]

# Adaptive posterior bands
pa1 = posterior_predictions(prob, posterior_adaptive, prediction_grid_1; n_samples=200)
pa2 = posterior_predictions(prob, posterior_adaptive, prediction_grid_2; n_samples=200)
ba1 = credible_band(pa1; level=0.9)
ba2 = credible_band(pa2; level=0.9)

# Batch posterior bands
pb1 = posterior_predictions(prob, result_batch.posterior, prediction_grid_1; n_samples=200)
pb2 = posterior_predictions(prob, result_batch.posterior, prediction_grid_2; n_samples=200)
bb1 = credible_band(pb1; level=0.9)
bb2 = credible_band(pb2; level=0.9)

obs_adaptive_1 = [(ξ=e.ξ, y=e.y) for e in log_adaptive if e.ξ.i == 1]
obs_adaptive_2 = [(ξ=e.ξ, y=e.y) for e in log_adaptive if e.ξ.i == 2]
obs_batch_1 = [o for o in result_batch.observations if o.ξ.i == 1]
obs_batch_2 = [o for o in result_batch.observations if o.ξ.i == 2]

fig2 = Figure(size=(900, 600))

ax2a = GLMakie.Axis(fig2[1, 1], ylabel="y",
    title="Adaptive — Decay 1 ($(length(obs_adaptive_1)) obs)")
band!(ax2a, x_grid, ba1.lower, ba1.upper, color=(:blue, 0.3))
lines!(ax2a, x_grid, ba1.median, color=:blue, linewidth=2)
lines!(ax2a, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2a, [o.ξ.t for o in obs_adaptive_1], [o.y for o in obs_adaptive_1],
    color=:black, markersize=6, label="Observations")
axislegend(ax2a)

ax2b = GLMakie.Axis(fig2[1, 2], ylabel="y",
    title="Adaptive — Decay 2 ($(length(obs_adaptive_2)) obs)")
band!(ax2b, x_grid, ba2.lower, ba2.upper, color=(:orange, 0.3))
lines!(ax2b, x_grid, ba2.median, color=:orange, linewidth=2)
lines!(ax2b, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2b, [o.ξ.t for o in obs_adaptive_2], [o.y for o in obs_adaptive_2],
    color=:black, markersize=6)

ax2c = GLMakie.Axis(fig2[2, 1], xlabel="t", ylabel="y",
    title="Batch — Decay 1 ($(length(obs_batch_1)) obs)")
band!(ax2c, x_grid, bb1.lower, bb1.upper, color=(:blue, 0.3))
lines!(ax2c, x_grid, bb1.median, color=:blue, linewidth=2)
lines!(ax2c, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2c, [o.ξ.t for o in obs_batch_1], [o.y for o in obs_batch_1],
    color=:black, markersize=6)

ax2d = GLMakie.Axis(fig2[2, 2], xlabel="t", ylabel="y",
    title="Batch — Decay 2 ($(length(obs_batch_2)) obs)")
band!(ax2d, x_grid, bb2.lower, bb2.upper, color=(:orange, 0.3))
lines!(ax2d, x_grid, bb2.median, color=:orange, linewidth=2)
lines!(ax2d, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2d, [o.ξ.t for o in obs_batch_2], [o.y for o in obs_batch_2],
    color=:black, markersize=6)

fig2

# --- Figure 3: Corner plot — adaptive vs batch posterior ---

fig3 = plot_corner(posterior_adaptive, result_batch.posterior;
    params=[:A₁, :R₂₁, :A₂, :R₂₂], labels=["Adaptive", "Batch"],
    truth=(A₁=θ_true.A₁, R₂₁=θ_true.R₂₁, A₂=θ_true.A₂, R₂₂=θ_true.R₂₂))

# --- Figure 4: Posterior evolution animation ---

if OptimalDesign.has_posterior_history(log_adaptive)
    println("Recording posterior evolution animation...")
    record_corner_animation(log_adaptive, "ex6_posterior_evolution.mp4";
        params=[:R₂₁, :R₂₂],
        truth=(R₂₁=θ_true.R₂₁, R₂₂=θ_true.R₂₂),
        framerate=5)
end

# --- Figure 5: ESS and posterior convergence ---

println("\nRerunning adaptive experiment to track ESS evolution...")
prior_ess = ParticlePosterior(prob, 1000)
ess_history = Float64[]
r21_history = Float64[]
r22_history = Float64[]

for entry in log_adaptive
    OptimalDesign.update!(prior_ess, prob, entry.ξ, entry.y)
    push!(ess_history, effective_sample_size(prior_ess))
    μ = posterior_mean(prior_ess)
    push!(r21_history, μ.R₂₁)
    push!(r22_history, μ.R₂₂)
end

fig5 = Figure(size=(700, 500))

ax5a = GLMakie.Axis(fig5[1, 1], ylabel="R₂ estimate",
    title="Posterior Convergence")
lines!(ax5a, 1:n_adaptive, r21_history, color=:blue, linewidth=2, label="R₂₁")
lines!(ax5a, 1:n_adaptive, r22_history, color=:orange, linewidth=2, label="R₂₂")
hlines!(ax5a, [θ_true.R₂₁], color=:blue, linestyle=:dash, linewidth=1)
hlines!(ax5a, [θ_true.R₂₂], color=:orange, linestyle=:dash, linewidth=1)
axislegend(ax5a)

ax5b = GLMakie.Axis(fig5[2, 1], xlabel="Step", ylabel="ESS",
    title="Effective Sample Size")
lines!(ax5b, 1:n_adaptive, ess_history, color=:blue, linewidth=2)
hlines!(ax5b, [100], color=:gray, linestyle=:dash, label="Warning threshold")
axislegend(ax5b)

fig5

println("\nDone. Figures created.")
