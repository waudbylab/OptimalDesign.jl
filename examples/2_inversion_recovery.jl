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


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using GLMakie

# ENV["JULIA_DEBUG"] = OptimalDesign
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

θ_test = OptimalDesign.draw(prob.parameters)
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
)

θ_eval = ComponentArray(A=1.0, B=2.0, R₁=1.0)
ξ_eval = (τ=1.0,)
M_analytic = OptimalDesign.information(prob, θ_eval, ξ_eval)
M_ad = OptimalDesign.information(prob_ad, θ_eval, ξ_eval)
println("  FIM agreement: ", isapprox(M_analytic, M_ad, atol=1e-10), "\n")

# ═══════════════════════════════════════════════
# 3. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

n_obs = 20
println("Running exchange algorithm for batch design (n=$n_obs)...")
d = design(prob, candidates, prior; n=n_obs, posterior_samples=1000, exchange_steps=200)
display(d)

# ═══════════════════════════════════════════════
# 4. Optimality verification (Gateaux derivative)
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

acquire = let θ = θ_true, σ = σ_true
    ξ -> θ.A - θ.B * exp(-θ.R₁ * ξ.τ) + σ * randn()
end

posterior_opt = ParticlePosterior(prob, 1000)
result_opt = run_batch(d, prob, posterior_opt, acquire)

posterior_unif = ParticlePosterior(prob, 1000)
result_unif = run_batch(u, prob, posterior_unif, acquire)

μ_opt = posterior_mean(result_opt.posterior)
μ_unif = posterior_mean(result_unif.posterior)
println("Posterior mean (optimal):  A=$(round(μ_opt.A; digits=3)), B=$(round(μ_opt.B; digits=3)), R₁=$(round(μ_opt.R₁; digits=3))")
println("Posterior mean (uniform):  A=$(round(μ_unif.A; digits=3)), B=$(round(μ_unif.B; digits=3)), R₁=$(round(μ_unif.R₁; digits=3))")

# ═══════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")
prediction_grid = [(τ=τ,) for τ in range(0.01, 5.0, length=100)]

# --- Figure 1: Design allocation + Gateaux derivative ---

gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d;
    posterior_samples=1000)
w_opt = OptimalDesign.weights(d, candidates)

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Weight", title="Ds-Optimal Design for R₁ (Inversion Recovery)")
stem!(ax1a, [ξ.τ for ξ in candidates], w_opt, color=:blue)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="τ (s)", ylabel="Gateaux derivative",
    title="Optimality Check (GEQ bound = $(round(Int, opt_check.dimension)))")
lines!(ax1b, [ξ.τ for ξ in candidates], gd, color=:blue, linewidth=1.5)
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
    params=[:A, :B, :R₁], labels=["Prior", "Optimal"],
    truth=(A=θ_true.A, B=θ_true.B, R₁=θ_true.R₁))

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_opt.posterior, result_unif.posterior;
    params=[:A, :B, :R₁], labels=["Optimal", "Uniform"],
    truth=(A=θ_true.A, B=θ_true.B, R₁=θ_true.R₁))

println("Done. Figures created.")
