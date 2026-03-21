# Example 3: Two Decays, Vector Observation — Simultaneous Measurement
#
# Two exponential decays observed simultaneously as a vector (y₁, y₂) at
# a single time t. Four parameters (A₁, R₂₁, A₂, R₂₂), interest in both rates.
# The Jacobian is a 2×4 matrix. The FIM sums information from both observables.
#
# Demonstrates: Vector-valued predict, vector sigma, FIM from multiple
# simultaneous observables, batch design where a single time point informs both rates.

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using GLMakie

ENV["JULIA_DEBUG"] = OptimalDesign
# Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════

θ_true = ComponentArray(A₁=7.0, R₂₁=8.0, A₂=1.0, R₂₂=4.0)
σ_true = 0.05
n_obs = 100

prob = DesignProblem(
    (θ, ξ) -> [
        θ.A₁ * exp(-θ.R₂₁ * ξ.t),
        θ.A₂ * exp(-θ.R₂₂ * ξ.t)],
    parameters=(
        A₁=LogUniform(0.1, 10), R₂₁=Uniform(0.1, 100),
        A₂=LogUniform(0.1, 10), R₂₂=Uniform(0.1, 100)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> [σ_true, σ_true],
    cost=ξ -> ξ.t + 1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]

prior = ParticlePosterior(prob, 1000)

println("Problem: y = [A₁ exp(-R₂₁ t), A₂ exp(-R₂₂ t)] + noise")
println("Truth:   A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println("Acquire: $n_obs measurements (vector obs, 2 outputs per measurement)")
println("Goal:    Ds-optimal design for (R₂₁, R₂₂)\n")

# ═══════════════════════════════════════════════
# 2. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("Calculating batch design (n=$n_obs)...")
d = design(prob, candidates, prior;
    n=n_obs,
    criterion=DCriterion(),
    exchange_steps=200)

println("\nOptimal design allocation:")
for (ξ, count) in d
    bar = repeat("█", count)
    println("  t = $(round(ξ.t; digits=4))  ×$(count)  $bar")
end

# ═══════════════════════════════════════════════
# 3. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════

w_opt = zeros(length(candidates))
for (ξ, count) in d
    idx = findfirst(c -> c == ξ, candidates)
    idx !== nothing && (w_opt[idx] = count / n_obs)
end

gd = gateaux_derivative(prob, candidates, prior.particles, w_opt;
    criterion=DCriterion(), posterior_samples=1000)

opt_check = verify_optimality(prob, candidates, prior.particles, w_opt;
    criterion=DCriterion(), posterior_samples=1000)
println("\nOptimality verification:")
println("  Is optimal: $(opt_check.is_optimal)")
println("  Max Gateaux derivative: $(round(opt_check.max_derivative; digits=3))")
println("  Bound (q): $(round(opt_check.dimension; digits=3))")

# ═══════════════════════════════════════════════
# 4. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

uniform = uniform_allocation(candidates, n_obs)
w_unif = zeros(length(candidates))
for (ξ, count) in uniform
    idx = findfirst(c -> c == ξ, candidates)
    idx !== nothing && (w_unif[idx] = count / n_obs)
end

eff = efficiency(w_unif, w_opt, prob, candidates, prior.particles;
    criterion=DCriterion(), posterior_samples=1000)
println("\nD-efficiency of uniform vs optimal: $(round(eff; digits=3))")
println("  Uniform needs ~$(round(1 / eff; digits=1))× more measurements to match")

# ═══════════════════════════════════════════════
# 5. Simulated acquisition — optimal design
# ═══════════════════════════════════════════════

println("\n--- Simulated experiment (optimal design) ---")
posterior_opt = ParticlePosterior(prob, 1000)
obs_opt = NamedTuple[]

for (ξ, count) in d
    for _ in 1:count
        y = prob.predict(θ_true, ξ) .+ σ_true .* randn(2)
        push!(obs_opt, (ξ=ξ, y=y))
    end
end
OptimalDesign.update!(posterior_opt, prob, obs_opt)

μ_opt = posterior_mean(posterior_opt)
println("Posterior mean (optimal):  A₁=$(round(μ_opt.A₁; digits=3)), R₂₁=$(round(μ_opt.R₂₁; digits=2)), " *
        "A₂=$(round(μ_opt.A₂; digits=3)), R₂₂=$(round(μ_opt.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 6. Simulated acquisition — uniform design
# ═══════════════════════════════════════════════

println("\n--- Simulated experiment (uniform design) ---")
posterior_unif = ParticlePosterior(prob, 1000)
obs_unif = NamedTuple[]

for (ξ, count) in uniform
    for _ in 1:count
        y = prob.predict(θ_true, ξ) .+ σ_true .* randn(2)
        push!(obs_unif, (ξ=ξ, y=y))
    end
end
OptimalDesign.update!(posterior_unif, prob, obs_unif)

μ_unif = posterior_mean(posterior_unif)
println("Posterior mean (uniform):  A₁=$(round(μ_unif.A₁; digits=3)), R₂₁=$(round(μ_unif.R₂₁; digits=2)), " *
        "A₂=$(round(μ_unif.A₂; digits=3)), R₂₂=$(round(μ_unif.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(t=t,) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid]

# True predictions: 2-element vectors → separate into components
y_true_vec = [prob.predict(θ_true, ξ) for ξ in prediction_grid]
y_true_1 = [y[1] for y in y_true_vec]
y_true_2 = [y[2] for y in y_true_vec]

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design Allocation (Vector Observation)")
stem!(ax1a, [ξ.t for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="t", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.t for ξ in candidates], gd, color=:blue, linewidth=1.5)
hlines!(ax1b, [opt_check.dimension], color=:red, linestyle=:dash)

fig1
# save("ex3_design.pdf", fig1)

# --- Figure 2: Credible bands — separate panels per component ---

preds_prior = posterior_predictions(prob, prior, prediction_grid; n_samples=200)
preds_prior_1 = preds_prior[1]
preds_prior_2 = preds_prior[2]
band_prior_1 = credible_band(preds_prior_1; level=0.9)
band_prior_2 = credible_band(preds_prior_2; level=0.9)

preds_opt = posterior_predictions(prob, posterior_opt, prediction_grid; n_samples=200)
preds_opt_1 = preds_opt[1]
preds_opt_2 = preds_opt[2]
band_opt_1 = credible_band(preds_opt_1; level=0.9)
band_opt_2 = credible_band(preds_opt_2; level=0.9)

preds_unif = posterior_predictions(prob, posterior_unif, prediction_grid; n_samples=200)
preds_unif_1 = preds_unif[1]
preds_unif_2 = preds_unif[2]
band_unif_1 = credible_band(preds_unif_1; level=0.9)
band_unif_2 = credible_band(preds_unif_2; level=0.9)

# Split observations by component for clarity
obs_t_opt = [o.ξ.t for o in obs_opt]
obs_y1_opt = [o.y[1] for o in obs_opt]
obs_y2_opt = [o.y[2] for o in obs_opt]
obs_t_unif = [o.ξ.t for o in obs_unif]
obs_y1_unif = [o.y[1] for o in obs_unif]
obs_y2_unif = [o.y[2] for o in obs_unif]

# --- Figure 2: Credible bands — separate panels per component ---
# Rows: optimal, uniform.  Columns: y₁ (slow decay), y₂ (fast decay).

fig2 = Figure(size=(900, 800))

# Row 1: Prior
ax2a = GLMakie.Axis(fig2[1, 1], ylabel="y",
    title="Prior — y₁ (slow, R₂₁=$(θ_true.R₂₁))")
band!(ax2a, x_grid, band_prior_1.lower, band_prior_1.upper, color=(:gray, 0.3))
lines!(ax2a, x_grid, band_prior_1.median, color=:gray, linewidth=2)
lines!(ax2a, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash, label="Truth")
axislegend(ax2a)

ax2b = GLMakie.Axis(fig2[1, 2], ylabel="y",
    title="Prior — y₂ (fast, R₂₂=$(θ_true.R₂₂))")
band!(ax2b, x_grid, band_prior_2.lower, band_prior_2.upper, color=(:gray, 0.3))
lines!(ax2b, x_grid, band_prior_2.median, color=:gray, linewidth=2)
lines!(ax2b, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash, label="Truth")
axislegend(ax2b)

# Row 2: Optimal posterior
ax2c = GLMakie.Axis(fig2[2, 1], ylabel="y",
    title="Optimal — y₁ ($n_obs measurements)")
band!(ax2c, x_grid, band_opt_1.lower, band_opt_1.upper, color=(:blue, 0.3))
lines!(ax2c, x_grid, band_opt_1.median, color=:blue, linewidth=2)
lines!(ax2c, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2c, obs_t_opt, obs_y1_opt, color=:black, markersize=4, label="Obs")
for (ξ, count) in d
    vlines!(ax2c, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end
axislegend(ax2c)

ax2d = GLMakie.Axis(fig2[2, 2], ylabel="y",
    title="Optimal — y₂ ($n_obs measurements)")
band!(ax2d, x_grid, band_opt_2.lower, band_opt_2.upper, color=(:orange, 0.3))
lines!(ax2d, x_grid, band_opt_2.median, color=:orange, linewidth=2)
lines!(ax2d, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2d, obs_t_opt, obs_y2_opt, color=:black, markersize=4, label="Obs")
for (ξ, count) in d
    vlines!(ax2d, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end
axislegend(ax2d)

# Row 3: Uniform posterior
ax2e = GLMakie.Axis(fig2[3, 1], xlabel="t", ylabel="y",
    title="Uniform — y₁ ($n_obs measurements)")
band!(ax2e, x_grid, band_unif_1.lower, band_unif_1.upper, color=(:blue, 0.3))
lines!(ax2e, x_grid, band_unif_1.median, color=:blue, linewidth=2)
lines!(ax2e, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2e, obs_t_unif, obs_y1_unif, color=:black, markersize=4)

ax2f = GLMakie.Axis(fig2[3, 2], xlabel="t", ylabel="y",
    title="Uniform — y₂ ($n_obs measurements)")
band!(ax2f, x_grid, band_unif_2.lower, band_unif_2.upper, color=(:orange, 0.3))
lines!(ax2f, x_grid, band_unif_2.median, color=:orange, linewidth=2)
lines!(ax2f, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2f, obs_t_unif, obs_y2_unif, color=:black, markersize=4)

fig2
# save("ex3_credible_bands.pdf", fig2)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, posterior_opt;
    params=[:A₁, :A₂, :R₂₁, :R₂₂], labels=["Prior", "Optimal"],
    truth=(A₁=θ_true.A₁, A₂=θ_true.A₂, R₂₁=θ_true.R₂₁, R₂₂=θ_true.R₂₂))
# save("ex3_corner_prior_vs_opt.pdf", fig3)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(posterior_opt, posterior_unif;
    params=[:A₁, :A₂, :R₂₁, :R₂₂], labels=["Optimal", "Uniform"],
    truth=(A₁=θ_true.A₁, A₂=θ_true.A₂, R₂₁=θ_true.R₂₁, R₂₂=θ_true.R₂₂))
# save("ex3_corner_opt_vs_unif.pdf", fig4)

println("Done. Figures created — uncomment save() calls to export PDFs.")
