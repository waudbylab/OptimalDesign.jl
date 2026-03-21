# Example 1: Exponential Decay — Batch Design, Simulated Experiment, Posterior
#
# Simplest case. One design variable (time t), two parameters (A, R₂),
# interest in R₂ via Ds-optimality (DeltaMethod transformation).
#
# Demonstrates the full workflow:
#   1. Problem setup and prior
#   2. Batch design via exchange algorithm
#   3. Optimality verification (Gateaux derivative)
#   4. Efficiency comparison against uniform spacing
#   5. Simulated acquisition using the batch design
#   6. Posterior credible bands and marginals
#   7. Comparison: optimal vs uniform posterior precision


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using GLMakie

ENV["JULIA_DEBUG"] = OptimalDesign
# Random.seed!(42)

# ═══════════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════════

n_obs = 20
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters=(A=LogUniform(0.1, 10), R₂=Uniform(1, 50)),
    transformation=select(:R₂),
    sigma=(θ, ξ) -> 0.1,
    cost=Returns(1.0),
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]
prior = ParticlePosterior(prob, 1000)

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A=1.0, R₂=25.0)
σ_true = 0.1

println("Problem: y = A exp(-R₂ t) + noise")
println("Truth:   A = $(θ_true.A), R₂ = $(θ_true.R₂)")
println("Acquire: $n_obs measurements")
println("Goal:    Ds-optimal design for R₂\n")

# ═══════════════════════════════════════════════════
# 2. Batch design via exchange algorithm
# ═══════════════════════════════════════════════════

println("Calculating batch design (n=$n_obs)...")
design = select(prob, candidates, prior;
    n=n_obs,
    criterion=DCriterion(),
    # posterior_samples=200,
    # exchange_algorithm=f,
    exchange_steps=200)

println("\nOptimal design allocation:")
for (ξ, count) in design
    bar = repeat("█", count)
    println("  t = $(round(ξ.t; digits=4))  ×$(count)  $bar")
end

# ═══════════════════════════════════════════════════
# 3. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════════

# Build weight vector from design
w_opt = zeros(length(candidates))
for (ξ, count) in design
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

# ═══════════════════════════════════════════════════
# 4. Efficiency comparison against uniform
# ═══════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════
# 5. Simulated acquisition — optimal design
# ═══════════════════════════════════════════════════

println("\n--- Simulated experiment (optimal design) ---")
posterior_opt = ParticlePosterior(prob, 1000)
obs_opt = NamedTuple[]

for (ξ, count) in design
    for _ in 1:count
        y = prob.predict(θ_true, ξ) + σ_true * randn()
        OptimalDesign.update!(posterior_opt, prob, ξ, y)
        push!(obs_opt, (ξ=ξ, y=y))
    end
end

μ_opt = posterior_mean(posterior_opt)
println("Posterior mean (optimal):  A = $(round(μ_opt.A; digits=3)), R₂ = $(round(μ_opt.R₂; digits=2))")

# ═══════════════════════════════════════════════════
# 6. Simulated acquisition — uniform design
# ═══════════════════════════════════════════════════

println("\n--- Simulated experiment (uniform design) ---")
posterior_unif = ParticlePosterior(prob, 1000)
obs_unif = NamedTuple[]

for (ξ, count) in uniform
    for _ in 1:count
        y = prob.predict(θ_true, ξ) + σ_true * randn()
        OptimalDesign.update!(posterior_unif, prob, ξ, y)
        push!(obs_unif, (ξ=ξ, y=y))
    end
end

μ_unif = posterior_mean(posterior_unif)
println("Posterior mean (uniform):  A = $(round(μ_unif.A; digits=3)), R₂ = $(round(μ_unif.R₂; digits=2))")

# ═══════════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(t=t,) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid]
y_true = [prob.predict(θ_true, ξ) for ξ in prediction_grid]

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design Allocation")
stem!(ax1a, [ξ.t for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="t", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.t for ξ in candidates], gd, color=:blue, linewidth=1.5)
hlines!(ax1b, [opt_check.dimension], color=:red, linestyle=:dash)

fig1
# save("ex1_design.pdf", fig1)

# --- Figure 2: Prior → Posterior credible bands ---

preds_prior = posterior_predictions(prob, prior, prediction_grid; n_samples=200)
band_prior = credible_band(preds_prior; level=0.9)

preds_opt = posterior_predictions(prob, posterior_opt, prediction_grid; n_samples=200)
band_opt = credible_band(preds_opt; level=0.9)

preds_unif = posterior_predictions(prob, posterior_unif, prediction_grid; n_samples=200)
band_unif = credible_band(preds_unif; level=0.9)

fig2 = Figure(size=(700, 700))

ax2a = GLMakie.Axis(fig2[1, 1], ylabel="y", title="Prior (before experiment)")
band!(ax2a, x_grid, band_prior.lower, band_prior.upper, color=(:gray, 0.3))
lines!(ax2a, x_grid, band_prior.median, color=:gray, linewidth=2)
lines!(ax2a, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash, label="Truth")
axislegend(ax2a)

ax2b = GLMakie.Axis(fig2[2, 1], ylabel="y",
    title="Posterior — Optimal design ($n_obs measurements)")
band!(ax2b, x_grid, band_opt.lower, band_opt.upper, color=(:blue, 0.3))
lines!(ax2b, x_grid, band_opt.median, color=:blue, linewidth=2)
lines!(ax2b, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2b, [o.ξ.t for o in obs_opt], [o.y for o in obs_opt],
    color=:black, markersize=6, label="Observations")
fig2
# Mark design support points
for (ξ, count) in design
    vlines!(ax2b, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end
axislegend(ax2b)

ax2c = GLMakie.Axis(fig2[3, 1], xlabel="t", ylabel="y",
    title="Posterior — Uniform design ($n_obs measurements)")
band!(ax2c, x_grid, band_unif.lower, band_unif.upper, color=(:orange, 0.3))
lines!(ax2c, x_grid, band_unif.median, color=:orange, linewidth=2)
lines!(ax2c, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2c, [o.ξ.t for o in obs_unif], [o.y for o in obs_unif],
    color=:black, markersize=6)
fig2
# save("ex1_credible_bands.pdf", fig2)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, posterior_opt;
    params=[:A, :R₂], labels=["Prior", "Optimal"],
    truth=(A=θ_true.A, R₂=θ_true.R₂))
# save("ex1_corner_prior_vs_opt.pdf", fig3)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(posterior_opt, posterior_unif;
    params=[:A, :R₂], labels=["Optimal", "Uniform"],
    truth=(A=θ_true.A, R₂=θ_true.R₂))
# save("ex1_corner_opt_vs_unif.pdf", fig4)

println("Done. Figures created — uncomment save() calls to export PDFs.")
