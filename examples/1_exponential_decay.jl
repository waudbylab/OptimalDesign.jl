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
#   5. Simulated acquisition using run_batch
#   6. Posterior credible bands and corner plots


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random
using GLMakie

ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════════
# 1. Problem setup
# ═══════════════════════════════════════════════════

n_obs = 20
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters=(A=LogUniform(0.1, 10), R₂=Uniform(1, 50)),
    transformation=select(:R₂),
    sigma=(θ, ξ) -> 0.1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]
prior = ParticlePosterior(prob, 1000)

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A=1.0, R₂=25.0)
σ_true = 0.1

# Simulated acquisition function
acquire = let θ = θ_true, σ = σ_true
    ξ -> θ.A * exp(-θ.R₂ * ξ.t) + σ * randn()
end

println("Problem: y = A exp(-R₂ t) + noise")
println("Truth:   A = $(θ_true.A), R₂ = $(θ_true.R₂)")
println("Acquire: $n_obs measurements")
println("Goal:    Ds-optimal design for R₂\n")

# ═══════════════════════════════════════════════════
# 2. Batch design via exchange algorithm
# ═══════════════════════════════════════════════════

println("Calculating batch design (n=$n_obs)...")
d = design(prob, candidates, prior; n=n_obs, exchange_steps=200)
display(d)

# ═══════════════════════════════════════════════════
# 3. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════════

opt_check = OptimalDesign.verify_optimality(prob, candidates, prior, d;
    posterior_samples=1000)
println("\nOptimality verification:")
println("  Is optimal: $(opt_check.is_optimal)")
println("  Max Gateaux derivative: $(round(opt_check.max_derivative; digits=3))")
println("  Bound (q): $(round(opt_check.dimension; digits=3))")

# ═══════════════════════════════════════════════════
# 4. Efficiency comparison against uniform
# ═══════════════════════════════════════════════════

u = OptimalDesign.uniform_allocation(candidates, n_obs)

eff = efficiency(u, d, prob, candidates, prior; posterior_samples=1000)
println("\nD-efficiency of uniform vs optimal: $(round(eff; digits=3))")
println("  Uniform needs ~$(round(1 / eff; digits=1))× more measurements to match")

# ═══════════════════════════════════════════════════
# 5. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════════

println("\n--- Simulated experiments ---")

posterior_opt = ParticlePosterior(prob, 1000)
result_opt = run_batch(d, prob, posterior_opt, acquire)

posterior_unif = ParticlePosterior(prob, 1000)
result_unif = run_batch(u, prob, posterior_unif, acquire)

μ_opt = posterior_mean(result_opt.posterior)
μ_unif = posterior_mean(result_unif.posterior)
println("Posterior mean (optimal):  A = $(round(μ_opt.A; digits=3)), R₂ = $(round(μ_opt.R₂; digits=2))")
println("Posterior mean (uniform):  A = $(round(μ_unif.A; digits=3)), R₂ = $(round(μ_unif.R₂; digits=2))")

# ═══════════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(t=t,) for t in range(0.001, 0.5, length=100)]

# --- Figure 1: Design allocation + Gateaux derivative ---

gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d;
    posterior_samples=1000)
w_opt = OptimalDesign.weights(d, candidates)

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design Allocation")
stem!(ax1a, [ξ.t for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="t", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.t for ξ in candidates], gd, color=:blue, linewidth=1.5)
hlines!(ax1b, [opt_check.dimension], color=:red, linestyle=:dash)

fig1

# --- Figure 2: Prior → Posterior credible bands ---

fig2 = OptimalDesign.plot_credible_bands(prob,
    [prior, result_opt.posterior, result_unif.posterior],
    prediction_grid;
    labels=["Prior", "Optimal ($n_obs obs)", "Uniform ($n_obs obs)"],
    truth=θ_true,
    observations=[nothing, result_opt.observations, result_unif.observations])

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, result_opt.posterior;
    params=[:A, :R₂], labels=["Prior", "Optimal"],
    truth=(A=θ_true.A, R₂=θ_true.R₂))

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif.posterior, result_opt.posterior;
    params=[:A, :R₂], labels=["Uniform", "Optimal"],
    truth=(A=θ_true.A, R₂=θ_true.R₂))

println("Done. Figures created.")
