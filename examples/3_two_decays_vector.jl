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

# ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════

θ_true = ComponentArray(A₁=7.0, R₂₁=8.0, A₂=1.0, R₂₂=80.0)
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

acquire = let θ = θ_true, σ = σ_true
    ξ -> [θ.A₁ * exp(-θ.R₂₁ * ξ.t), θ.A₂ * exp(-θ.R₂₂ * ξ.t)] .+ σ .* randn(2)
end

println("Problem: y = [A₁ exp(-R₂₁ t), A₂ exp(-R₂₂ t)] + noise")
println("Truth:   A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println("Acquire: $n_obs measurements (vector obs, 2 outputs per measurement)")
println("Goal:    Ds-optimal design for (R₂₁, R₂₂)\n")

# ═══════════════════════════════════════════════
# 2. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("Calculating batch design (n=$n_obs)...")
d = design(prob, candidates, prior; n=n_obs, exchange_steps=200)
display(d)

# ═══════════════════════════════════════════════
# 3. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════

opt_check = OptimalDesign.verify_optimality(prob, candidates, prior, d;
    posterior_samples=1000)
println("\nOptimality verification:")
println("  Is optimal: $(opt_check.is_optimal)")
println("  Max Gateaux derivative: $(round(opt_check.max_derivative; digits=3))")
println("  Bound (q): $(round(opt_check.dimension; digits=3))")

# ═══════════════════════════════════════════════
# 4. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

u = OptimalDesign.uniform_allocation(candidates, n_obs)

eff = efficiency(u, d, prob, candidates, prior; posterior_samples=1000)
println("\nD-efficiency of uniform vs optimal: $(round(eff; digits=3))")
println("  Uniform needs ~$(round(1 / eff; digits=1))× more measurements to match")

# ═══════════════════════════════════════════════
# 5. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════

println("\n--- Simulated experiments ---")

posterior_opt = ParticlePosterior(prob, 1000)
result_opt = run_batch(d, prob, posterior_opt, acquire)

posterior_unif = ParticlePosterior(prob, 1000)
result_unif = run_batch(u, prob, posterior_unif, acquire)

μ_opt = posterior_mean(result_opt.posterior)
μ_unif = posterior_mean(result_unif.posterior)
println("Posterior mean (optimal):  A₁=$(round(μ_opt.A₁; digits=3)), R₂₁=$(round(μ_opt.R₂₁; digits=2)), " *
        "A₂=$(round(μ_opt.A₂; digits=3)), R₂₂=$(round(μ_opt.R₂₂; digits=2))")
println("Posterior mean (uniform):  A₁=$(round(μ_unif.A₁; digits=3)), R₂₁=$(round(μ_unif.R₂₁; digits=2)), " *
        "A₂=$(round(μ_unif.A₂; digits=3)), R₂₂=$(round(μ_unif.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(t=t,) for t in range(0.001, 0.5, length=100)]

# --- Figure 1: Design allocation + Gateaux derivative ---

gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d;
    posterior_samples=1000)
w_opt = OptimalDesign.weights(d, candidates)

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design Allocation (Vector Observation)")
stem!(ax1a, [ξ.t for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="t", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.t for ξ in candidates], gd, color=:blue, linewidth=1.5)
hlines!(ax1b, [opt_check.dimension], color=:red, linestyle=:dash)

fig1

# --- Figure 2: Credible bands — vector model auto-splits by component ---

fig2 = OptimalDesign.plot_credible_bands(prob,
    [prior, result_opt.posterior, result_unif.posterior],
    prediction_grid;
    labels=["Prior", "Optimal ($n_obs obs)", "Uniform ($n_obs obs)"],
    truth=θ_true,
    observations=[nothing, result_opt.observations, result_unif.observations])

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, result_opt.posterior;
    params=[:A₁, :A₂, :R₂₁, :R₂₂], labels=["Prior", "Optimal"],
    truth=(A₁=θ_true.A₁, A₂=θ_true.A₂, R₂₁=θ_true.R₂₁, R₂₂=θ_true.R₂₂))

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif.posterior, result_opt.posterior;
    params=[:A₁, :A₂, :R₂₁, :R₂₂], labels=["Uniform", "Optimal"],
    truth=(A₁=θ_true.A₁, A₂=θ_true.A₂, R₂₁=θ_true.R₂₁, R₂₂=θ_true.R₂₂))

println("Done. Figures created.")
