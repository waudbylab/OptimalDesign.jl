# Example 4: Two Decays, Discrete Control Variable — Selective Measurement
#
# Same two decays as Example 3, but now a control variable i ∈ {1, 2} selects
# which one is observed. Each measurement returns a scalar. The parameter space
# is the same, but only one decay contributes to the FIM per measurement.
#
# Demonstrates: Discrete control variable as a candidate field, block-sparse
# Jacobian, batch design choosing how to allocate between decays,
# block independence in the posterior.

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

θ_true = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)
σ_true = 0.05
n_obs = 20

prob = DesignProblem(
    (θ, ξ) -> if ξ.i == 1
        θ.A₁ * exp(-θ.R₂₁ * ξ.t)
    else
        θ.A₂ * exp(-θ.R₂₂ * ξ.t)
    end,
    parameters=(A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> σ_true,
    cost=ξ -> ξ.t + 0.1,
)

candidates = [
    (i=i, t=t)
    for i in [1, 2]
    for t in range(0.001, 0.5, length=200)
]

prior = ParticlePosterior(prob, 1000)

acquire = let θ = θ_true, σ = σ_true
    ξ -> (ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t)) + σ * randn()
end

println("Problem: y = Aᵢ exp(-R₂ᵢ t)  where i ∈ {1,2} selects the decay")
println("Truth:   A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println("Acquire: $n_obs measurements, choosing which decay to measure")
println("Goal:    Ds-optimal design for (R₂₁, R₂₂)\n")

# ═══════════════════════════════════════════════
# 2. Examine block-sparse FIM structure
# ═══════════════════════════════════════════════

M1 = OptimalDesign.information(prob, θ_true, (i=1, t=0.1))
M2 = OptimalDesign.information(prob, θ_true, (i=2, t=0.05))

println("FIM measuring decay 1 (i=1, t=0.1):")
display(round.(M1, digits=4))
println("\nFIM measuring decay 2 (i=2, t=0.05):")
display(round.(M2, digits=4))

println("\nBlock sparsity verification:")
println("  M1 rows 3:4 norm: ", round(norm(M1[3:4, :]), digits=10), " (should be ≈ 0)")
println("  M2 rows 1:2 norm: ", round(norm(M2[1:2, :]), digits=10), " (should be ≈ 0)")

M_combined = M1 + M2
println("\nCombined FIM rank: ", rank(M_combined))

# ═══════════════════════════════════════════════
# 3. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("\nCalculating batch design (n=$n_obs)...")
d = design(prob, candidates, prior; n=n_obs, exchange_steps=200)
display(d)

n_decay1 = sum(c for (ξ, c) in d if ξ.i == 1; init=0)
n_decay2 = sum(c for (ξ, c) in d if ξ.i == 2; init=0)
println("  Allocation: $n_decay1 on decay 1, $n_decay2 on decay 2")

# ═══════════════════════════════════════════════
# 4. Optimality verification
# ═══════════════════════════════════════════════

opt_check = OptimalDesign.verify_optimality(prob, candidates, prior, d;
    posterior_samples=1000)
println("\nOptimality verification:")
println("  Is optimal: $(opt_check.is_optimal)")
println("  Max Gateaux derivative: $(round(opt_check.max_derivative; digits=3))")
println("  Bound (q): $(round(opt_check.dimension; digits=3))")

# ═══════════════════════════════════════════════
# 5. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

u = OptimalDesign.uniform_allocation(candidates, n_obs)

eff = efficiency(u, d, prob, candidates, prior; posterior_samples=1000)
println("\nD-efficiency of uniform vs optimal: $(round(eff; digits=3))")
println("  Uniform needs ~$(round(1 / eff; digits=1))× more measurements to match")

# ═══════════════════════════════════════════════
# 6. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════

println("\n--- Simulated experiments ---")

posterior_opt = ParticlePosterior(prob, 1000)
result_opt = run_batch(d, prob, posterior_opt, acquire)

posterior_unif = ParticlePosterior(prob, 1000)
result_unif = run_batch(u, prob, posterior_unif, acquire)

μ_opt = posterior_mean(result_opt.posterior)
μ_unif = posterior_mean(result_unif.posterior)
println("Posterior mean (optimal):  R₂₁=$(round(μ_opt.R₂₁; digits=2)), R₂₂=$(round(μ_opt.R₂₂; digits=2))")
println("Posterior mean (uniform):  R₂₁=$(round(μ_unif.R₂₁; digits=2)), R₂₂=$(round(μ_unif.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Design allocation + Gateaux derivative ---

gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d;
    posterior_samples=1000)
w_opt = OptimalDesign.weights(d, candidates)

# Separate candidates by decay
times_1 = [c.t for c in candidates if c.i == 1]
times_2 = [c.t for c in candidates if c.i == 2]
gd_1 = [gd[k] for k in eachindex(candidates) if candidates[k].i == 1]
gd_2 = [gd[k] for k in eachindex(candidates) if candidates[k].i == 2]
w_1 = [w_opt[k] for k in eachindex(candidates) if candidates[k].i == 1]
w_2 = [w_opt[k] for k in eachindex(candidates) if candidates[k].i == 2]

fig1 = Figure(size=(700, 700))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight",
    title="Design Allocation — Decay 1 (R₂₁=$(θ_true.R₂₁))")
stem!(ax1a, times_1, w_1, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], ylabel="Weight",
    title="Design Allocation — Decay 2 (R₂₂=$(θ_true.R₂₂))")
stem!(ax1b, times_2, w_2, color=:orange)

ax1c = GLMakie.Axis(fig1[3, 1], xlabel="t", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1c, times_1, gd_1, color=:blue, linewidth=1.5, label="Decay 1")
lines!(ax1c, times_2, gd_2, color=:orange, linewidth=1.5, label="Decay 2")
hlines!(ax1c, [opt_check.dimension], color=:red, linestyle=:dash)
axislegend(ax1c)

fig1

# --- Figure 2: Credible bands for each decay ---

prediction_grid_1 = [(i=1, t=t) for t in range(0.001, 0.5, length=100)]
prediction_grid_2 = [(i=2, t=t) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid_1]

obs_1_opt = [(ξ=o.ξ, y=o.y) for o in result_opt.observations if o.ξ.i == 1]
obs_2_opt = [(ξ=o.ξ, y=o.y) for o in result_opt.observations if o.ξ.i == 2]
obs_1_unif = [(ξ=o.ξ, y=o.y) for o in result_unif.observations if o.ξ.i == 1]
obs_2_unif = [(ξ=o.ξ, y=o.y) for o in result_unif.observations if o.ξ.i == 2]

fig2 = OptimalDesign.plot_credible_bands(prob,
    [prior, result_opt.posterior, result_unif.posterior],
    prediction_grid_1;
    labels=["Prior", "Optimal", "Uniform"],
    truth=θ_true)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, result_opt.posterior;
    params=[:A₁, :R₂₁, :A₂, :R₂₂], labels=["Prior", "Optimal"],
    truth=(A₁=θ_true.A₁, R₂₁=θ_true.R₂₁, A₂=θ_true.A₂, R₂₂=θ_true.R₂₂))

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif.posterior, result_opt.posterior;
    params=[:A₁, :R₂₁, :A₂, :R₂₂], labels=["Uniform", "Optimal"],
    truth=(A₁=θ_true.A₁, R₂₁=θ_true.R₂₁, A₂=θ_true.A₂, R₂₂=θ_true.R₂₂))

# --- Comparison with Example 3 ---
println("\n=== Comparison ===")
println("Example 3 (vector obs): Each measurement observes BOTH decays simultaneously")
println("Example 4 (selective):  Each measurement observes ONE decay (must choose)")
println("The selective design must allocate between decays;")
println("R₂₂=$(θ_true.R₂₂) (fast) needs shorter measurement times than R₂₁=$(θ_true.R₂₁) (slow).")
println("Optimal allocation: $n_decay1 on decay 1, $n_decay2 on decay 2")

println("\nDone. Figures created.")
