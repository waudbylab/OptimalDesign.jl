# Example 2: Inversion Recovery — Analytic Jacobian, Batch Design, Transformation
#
# Model: y = A - B exp(-R₁ τ)   (inversion recovery in NMR/MRI)
# Three parameters (A, B, R₁), Ds-optimality for R₁ via DeltaMethod.
# Analytic Jacobian supplied for performance and validated against ForwardDiff.
#
# Demonstrates:
#   1. Analytic Jacobian with ForwardDiff validation
#   2. Ds-optimality via DeltaMethod transformation
#   3. Batch design via exchange algorithm
#   4. Optimality verification (Gateaux derivative)
#   5. Efficiency comparison against uniform spacing
#   6. Simulated acquisition and posterior inference
#   7. Validation: optimal delays near τ/T₁ ≈ 1.2

ENV["JULIA_DEBUG"] = OptimalDesign

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using GLMakie

Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. Problem setup — with analytic Jacobian
# ═══════════════════════════════════════════════


predict = (θ, ξ) -> θ.A - θ.B * exp(-θ.R₁ * ξ.τ)

jac = (θ, ξ) -> begin
    e = exp(-θ.R₁ * ξ.τ)
    # ∂y/∂A = 1, ∂y/∂B = -e, ∂y/∂R₁ = B τ e
    [1.0 -e θ.B * ξ.τ * e]
end

prob = DesignProblem(
    predict,
    jacobian=jac,
    parameters=(A=Normal(1, 0.5), B=Normal(2, 0.5), R₁=Uniform(0.1, 5)),
    transformation=select(:R₁),
    sigma=(θ, ξ) -> 0.05,
    cost=Returns(1.0),
)

# Candidate delay times: 0.01 to 5.0 seconds
candidates = [(τ=τ,) for τ in range(0.01, 5.0, length=200)]
prior = ParticlePosterior(prob, 1000)

# Ground truth (unknown to design algorithm)
θ_true = ComponentArray(A=1.0, B=2.0, R₁=1.3)
σ_true = 0.05
T₁_true = 1.0 / θ_true.R₁   # = 1.0 s

println("Problem: y = A - B exp(-R₁ τ)  (inversion recovery)")
println("Truth:   A = $(θ_true.A), B = $(θ_true.B), R₁ = $(θ_true.R₁)  (T₁ = $(T₁_true) s)")
println("Goal:    Ds-optimal design for R₁\n")

# ═══════════════════════════════════════════════
# 2. Validate analytic Jacobian against ForwardDiff
# ═══════════════════════════════════════════════

θ_test = draw(prob.parameters)
ξ_test = candidates[50]

J_analytic = prob.jacobian(θ_test, ξ_test)
J_ad = ForwardDiff.jacobian(θ_ -> [prob.predict(θ_, ξ_test)], θ_test)

println("Jacobian validation:")
println("  Analytic:    ", round.(J_analytic, digits=6))
println("  ForwardDiff: ", round.(J_ad, digits=6))
println("  Max error:   ", round(maximum(abs.(J_analytic .- J_ad)); sigdigits=3))

# Also compare FIM: create an equivalent problem without analytic Jacobian
prob_ad = DesignProblem(
    predict,
    parameters=(A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogUniform(0.1, 5)),
    transformation=select(:R₁),
    sigma=(θ, ξ) -> 0.05,
    cost=Returns(1.0),
)

θ_eval = ComponentArray(A=1.0, B=2.0, R₁=1.0)
ξ_eval = (τ=1.0,)
M_analytic = information(prob, θ_eval, ξ_eval)
M_ad = information(prob_ad, θ_eval, ξ_eval)
println("  FIM agreement: ", isapprox(M_analytic, M_ad, atol=1e-10), "\n")

# ═══════════════════════════════════════════════
# 3. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("Running exchange algorithm for batch design (n=20)...")
d = design(prob, candidates, prior;
    n=20, criterion=DCriterion(), posterior_samples=1000, exchange_algorithm=true,
    exchange_steps=200)

println("\nOptimal design allocation:")
for (ξ, count) in d
    bar = repeat("█", count)
    τ_over_T1 = ξ.τ * θ_true.R₁   # τ/T₁ ratio (for known truth)
    println("  τ = $(round(ξ.τ; digits=4))  (τ/T₁≈$(round(τ_over_T1; digits=2)))  ×$(count)  $bar")
end
println("  (Known result: optimal delays cluster near τ/T₁ ≈ 1.2)")

# ═══════════════════════════════════════════════
# 4. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════

w_opt = zeros(length(candidates))
for (ξ, count) in d
    idx = findfirst(c -> c == ξ, candidates)
    idx !== nothing && (w_opt[idx] = count / 20)
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

uniform = uniform_allocation(candidates, 20)
w_unif = zeros(length(candidates))
for (ξ, count) in uniform
    idx = findfirst(c -> c == ξ, candidates)
    idx !== nothing && (w_unif[idx] = count / 20)
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
        OptimalDesign.update!(posterior_opt, prob, ξ, y)
        push!(obs_opt, (ξ=ξ, y=y))
    end
end

μ_opt = posterior_mean(posterior_opt)
println("Posterior mean (optimal):  A=$(round(μ_opt.A; digits=3)), B=$(round(μ_opt.B; digits=3)), R₁=$(round(μ_opt.R₁; digits=3))")

# ═══════════════════════════════════════════════
# 7. Simulated acquisition — uniform design
# ═══════════════════════════════════════════════

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
println("Posterior mean (uniform):  A=$(round(μ_unif.A; digits=3)), B=$(round(μ_unif.B; digits=3)), R₁=$(round(μ_unif.R₁; digits=3))")

# ═══════════════════════════════════════════════
# 8. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(τ=τ,) for τ in range(0.01, 5.0, length=100)]
x_grid = [ξ.τ for ξ in prediction_grid]
y_true = [prob.predict(θ_true, ξ) for ξ in prediction_grid]

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design for R₁ (Inversion Recovery)")
stem!(ax1a, [ξ.τ for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="τ (s)", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.τ for ξ in candidates], gd, color=:blue, linewidth=1.5)
hlines!(ax1b, [opt_check.dimension], color=:red, linestyle=:dash)

fig1
# save("ex2_design.pdf", fig1)

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
    title="Posterior — Optimal design (20 measurements)")
band!(ax2b, x_grid, band_opt.lower, band_opt.upper, color=(:blue, 0.3))
lines!(ax2b, x_grid, band_opt.median, color=:blue, linewidth=2)
lines!(ax2b, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2b, [o.ξ.τ for o in obs_opt], [o.y for o in obs_opt],
    color=:black, markersize=6, label="Observations")
for (ξ, count) in d
    vlines!(ax2b, [ξ.τ], color=(:green, 0.3), linewidth=count * 2)
end
axislegend(ax2b)

ax2c = GLMakie.Axis(fig2[3, 1], xlabel="τ (s)", ylabel="y",
    title="Posterior — Uniform design (20 measurements)")
band!(ax2c, x_grid, band_unif.lower, band_unif.upper, color=(:orange, 0.3))
lines!(ax2c, x_grid, band_unif.median, color=:orange, linewidth=2)
lines!(ax2c, x_grid, y_true, color=:red, linewidth=1.5, linestyle=:dash)
scatter!(ax2c, [o.ξ.τ for o in obs_unif], [o.y for o in obs_unif],
    color=:black, markersize=6)

fig2
# save("ex2_credible_bands.pdf", fig2)

# --- Figure 3: Corner plot — prior vs optimal posterior (3 params) ---

fig3 = plot_corner(prior, posterior_opt;
    params=[:A, :B, :R₁], labels=["Prior", "Optimal"],
    truth=(A=θ_true.A, B=θ_true.B, R₁=θ_true.R₁))
# save("ex2_corner_prior_vs_opt.pdf", fig3)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(posterior_opt, posterior_unif;
    params=[:A, :B, :R₁], labels=["Optimal", "Uniform"],
    truth=(A=θ_true.A, B=θ_true.B, R₁=θ_true.R₁))
# save("ex2_corner_opt_vs_unif.pdf", fig4)

println("Done. Figures created — uncomment save() calls to export PDFs.")
