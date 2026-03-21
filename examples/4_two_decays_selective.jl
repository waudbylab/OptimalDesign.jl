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

ENV["JULIA_DEBUG"] = OptimalDesign
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

println("Problem: y = Aᵢ exp(-R₂ᵢ t)  where i ∈ {1,2} selects the decay")
println("Truth:   A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println("Acquire: $n_obs measurements, choosing which decay to measure")
println("Goal:    Ds-optimal design for (R₂₁, R₂₂)\n")

# ═══════════════════════════════════════════════
# 2. Examine block-sparse FIM structure
# ═══════════════════════════════════════════════

M1 = information(prob, θ_true, (i=1, t=0.1))
M2 = information(prob, θ_true, (i=2, t=0.05))

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
d = design(prob, candidates, prior;
    n=n_obs,
    criterion=DCriterion(),
    exchange_steps=200)

println("\nOptimal design allocation:")
for (ξ, count) in d
    label = ξ.i == 1 ? "decay 1" : "decay 2"
    bar = repeat("█", count)
    println("  i=$(ξ.i) ($label), t=$(round(ξ.t; digits=4))  ×$(count)  $bar")
end

n_decay1 = sum(c for (ξ, c) in d if ξ.i == 1; init=0)
n_decay2 = sum(c for (ξ, c) in d if ξ.i == 2; init=0)
println("\n  Allocation: $n_decay1 on decay 1, $n_decay2 on decay 2")

# ═══════════════════════════════════════════════
# 4. Optimality verification (Gateaux derivative)
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
# 5. Efficiency comparison against uniform
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
# 6. Simulated acquisition — optimal design
# ═══════════════════════════════════════════════

println("\n--- Simulated experiment (optimal design) ---")
posterior_opt = ParticlePosterior(prob, 1000)
obs_opt = NamedTuple[]

for (ξ, count) in d
    for _ in 1:count
        y = prob.predict(θ_true, ξ) + σ_true * randn()
        push!(obs_opt, (ξ=ξ, y=y))
    end
end
OptimalDesign.update!(posterior_opt, prob, obs_opt)

μ_opt = posterior_mean(posterior_opt)
println("Posterior mean (optimal):  R₂₁=$(round(μ_opt.R₂₁; digits=2)), R₂₂=$(round(μ_opt.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 7. Simulated acquisition — uniform design
# ═══════════════════════════════════════════════

println("\n--- Simulated experiment (uniform design) ---")
posterior_unif = ParticlePosterior(prob, 1000)
obs_unif = NamedTuple[]

for (ξ, count) in uniform
    for _ in 1:count
        y = prob.predict(θ_true, ξ) + σ_true * randn()
        push!(obs_unif, (ξ=ξ, y=y))
    end
end
OptimalDesign.update!(posterior_unif, prob, obs_unif)

μ_unif = posterior_mean(posterior_unif)
println("Posterior mean (uniform):  R₂₁=$(round(μ_unif.R₂₁; digits=2)), R₂₂=$(round(μ_unif.R₂₂; digits=2))")

# ═══════════════════════════════════════════════
# 8. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# Separate candidates by decay
times_1 = [c.t for c in candidates if c.i == 1]
times_2 = [c.t for c in candidates if c.i == 2]
gd_1 = [gd[k] for k in eachindex(candidates) if candidates[k].i == 1]
gd_2 = [gd[k] for k in eachindex(candidates) if candidates[k].i == 2]
w_1 = [w_opt[k] for k in eachindex(candidates) if candidates[k].i == 1]
w_2 = [w_opt[k] for k in eachindex(candidates) if candidates[k].i == 2]

# --- Figure 1: Design allocation + Gateaux derivative (both decays) ---

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
# save("ex4_design.pdf", fig1)

# --- Figure 2: Credible bands for each decay ---

# Prediction grids for each decay
prediction_grid_1 = [(i=1, t=t) for t in range(0.001, 0.5, length=100)]
prediction_grid_2 = [(i=2, t=t) for t in range(0.001, 0.5, length=100)]
x_grid = [ξ.t for ξ in prediction_grid_1]
y_true_1 = [prob.predict(θ_true, ξ) for ξ in prediction_grid_1]
y_true_2 = [prob.predict(θ_true, ξ) for ξ in prediction_grid_2]

preds_opt_1 = posterior_predictions(prob, posterior_opt, prediction_grid_1; n_samples=200)
preds_opt_2 = posterior_predictions(prob, posterior_opt, prediction_grid_2; n_samples=200)
band_opt_1 = credible_band(preds_opt_1; level=0.9)
band_opt_2 = credible_band(preds_opt_2; level=0.9)

preds_unif_1 = posterior_predictions(prob, posterior_unif, prediction_grid_1; n_samples=200)
preds_unif_2 = posterior_predictions(prob, posterior_unif, prediction_grid_2; n_samples=200)
band_unif_1 = credible_band(preds_unif_1; level=0.9)
band_unif_2 = credible_band(preds_unif_2; level=0.9)

obs_1_opt = [(ξ=o.ξ, y=o.y) for o in obs_opt if o.ξ.i == 1]
obs_2_opt = [(ξ=o.ξ, y=o.y) for o in obs_opt if o.ξ.i == 2]
obs_1_unif = [(ξ=o.ξ, y=o.y) for o in obs_unif if o.ξ.i == 1]
obs_2_unif = [(ξ=o.ξ, y=o.y) for o in obs_unif if o.ξ.i == 2]

fig2 = Figure(size=(900, 600))

# Optimal design
ax2a = GLMakie.Axis(fig2[1, 1], ylabel="y",
    title="Optimal — Decay 1 ($(length(obs_1_opt)) obs)")
band!(ax2a, x_grid, band_opt_1.lower, band_opt_1.upper, color=(:blue, 0.3))
lines!(ax2a, x_grid, band_opt_1.median, color=:blue, linewidth=2)
lines!(ax2a, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2a, [o.ξ.t for o in obs_1_opt], [o.y for o in obs_1_opt],
    color=:black, markersize=6, label="Observations")
for (ξ, count) in d
    ξ.i == 1 && vlines!(ax2a, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end
axislegend(ax2a)

ax2b = GLMakie.Axis(fig2[1, 2], ylabel="y",
    title="Optimal — Decay 2 ($(length(obs_2_opt)) obs)")
band!(ax2b, x_grid, band_opt_2.lower, band_opt_2.upper, color=(:orange, 0.3))
lines!(ax2b, x_grid, band_opt_2.median, color=:orange, linewidth=2)
lines!(ax2b, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2b, [o.ξ.t for o in obs_2_opt], [o.y for o in obs_2_opt],
    color=:black, markersize=6)
for (ξ, count) in d
    ξ.i == 2 && vlines!(ax2b, [ξ.t], color=(:green, 0.3), linewidth=count * 2)
end

# Uniform design
ax2c = GLMakie.Axis(fig2[2, 1], xlabel="t", ylabel="y",
    title="Uniform — Decay 1 ($(length(obs_1_unif)) obs)")
band!(ax2c, x_grid, band_unif_1.lower, band_unif_1.upper, color=(:blue, 0.3))
lines!(ax2c, x_grid, band_unif_1.median, color=:blue, linewidth=2)
lines!(ax2c, x_grid, y_true_1, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2c, [o.ξ.t for o in obs_1_unif], [o.y for o in obs_1_unif],
    color=:black, markersize=6)

ax2d = GLMakie.Axis(fig2[2, 2], xlabel="t", ylabel="y",
    title="Uniform — Decay 2 ($(length(obs_2_unif)) obs)")
band!(ax2d, x_grid, band_unif_2.lower, band_unif_2.upper, color=(:orange, 0.3))
lines!(ax2d, x_grid, band_unif_2.median, color=:orange, linewidth=2)
lines!(ax2d, x_grid, y_true_2, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2d, [o.ξ.t for o in obs_2_unif], [o.y for o in obs_2_unif],
    color=:black, markersize=6)

fig2
# save("ex4_credible_bands.pdf", fig2)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(prior, posterior_opt;
    params=[:A₁, :R₂₁, :A₂, :R₂₂], labels=["Prior", "Optimal"],
    truth=(A₁=θ_true.A₁, R₂₁=θ_true.R₂₁, A₂=θ_true.A₂, R₂₂=θ_true.R₂₂))
# save("ex4_corner_prior_vs_opt.pdf", fig3)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(posterior_opt, posterior_unif;
    params=[:A₁, :R₂₁, :A₂, :R₂₂], labels=["Optimal", "Uniform"],
    truth=(A₁=θ_true.A₁, R₂₁=θ_true.R₂₁, A₂=θ_true.A₂, R₂₂=θ_true.R₂₂))
# save("ex4_corner_opt_vs_unif.pdf", fig4)

# --- Comparison with Example 3 ---
println("\n=== Comparison ===")
println("Example 3 (vector obs): Each measurement observes BOTH decays simultaneously")
println("Example 4 (selective):  Each measurement observes ONE decay (must choose)")
println("The selective design must allocate between decays;")
println("R₂₂=$(θ_true.R₂₂) (fast) needs shorter measurement times than R₂₁=$(θ_true.R₂₁) (slow).")
println("Optimal allocation: $n_decay1 on decay 1, $n_decay2 on decay 2")

println("\nDone. Figures created — uncomment save() calls to export PDFs.")
