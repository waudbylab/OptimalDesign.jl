# Example 8: R1ρ Dispersion — Adaptive Design with Two Continuous Design Variables
#
# Model: I = I₀ exp(-R₁ρ(νSL) · tSL)
# where  R₁ρ(νSL) = R₂₀ + A · K² / (4π² νSL² + K²)
#
# Four parameters (I₀, R₂₀, A, K), two continuous design variables (tSL, νSL).
# Interest in K via Ds-optimality.
#
# Demonstrates:
#   1. Two-dimensional design space (spin-lock time × spin-lock frequency)
#   2. Adaptive experiment with 2D candidates
#   3. Batch design for comparison (matched observation count)
#   4. 2D design allocation (bubble plot) and Gateaux derivative
#   5. Posterior evolution animation
#   6. Corner plots for dispersion parameters

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

# R1ρ dispersion: R₁ρ(ν) = R₂₀ + A K² / (4π² ν² + K²)
R1rho(R₂₀, A, K, ν) = R₂₀ + A * K^2 / (4π^2 * ν^2 + K^2)

predict = (θ, ξ) -> θ.I₀ * exp(-R1rho(θ.R₂₀, θ.A, θ.K, ξ.νSL) * ξ.tSL)

# Ground truth
θ_true = ComponentArray(I₀=2.0, R₂₀=5.0, A=20.0, K=15000.0)
σ_true = 0.2

prob = DesignProblem(
    predict,
    parameters=(
        I₀=LogUniform(0.01, 100),
        R₂₀=Uniform(0, 20),
        A=Uniform(0, 100),
        K=LogUniform(100, 100_000),
    ),
    transformation=select(:K),
    sigma=(θ, ξ) -> σ_true,
)

# 2D candidate grid: tSL × νSL
tSL_vals = range(0.001, 0.08, length=20)
νSL_vals = range(300, 15_000, length=25)
candidates = [(tSL=t, νSL=ν) for t in tSL_vals for ν in νSL_vals]

acquire = let θ = θ_true, σ = σ_true
    ξ -> predict(θ, ξ) + σ * randn()
end

println("Problem: I = I₀ exp(-R₁ρ(νSL) · tSL)")
println("         R₁ρ(ν) = R₂₀ + A K² / (4π² ν² + K²)")
println("Truth:   I₀=$(θ_true.I₀), R₂₀=$(θ_true.R₂₀), A=$(θ_true.A), K=$(θ_true.K)")
println("Design:  $(length(candidates)) candidates on $(length(tSL_vals))×$(length(νSL_vals)) grid")
println("Goal:    Ds-optimal for K\n")

# ═══════════════════════════════════════════════
# 2. Inspect the R1ρ dispersion curve
# ═══════════════════════════════════════════════

println("R₁ρ dispersion at true parameters:")
for ν in [300, 500, 1000, 2000, 5000, 10000, 15000]
    r = R1rho(θ_true.R₂₀, θ_true.A, θ_true.K, ν)
    bar = repeat("█", round(Int, r))
    println("  νSL=$(lpad(ν, 5))  R₁ρ=$(round(r; digits=2))  $bar")
end

# ═══════════════════════════════════════════════
# 3. Adaptive experiment (run first to determine n)
# ═══════════════════════════════════════════════

budget = 100.0

println("\n--- Adaptive experiment (budget=$budget) ---")
prior_adaptive = ParticlePosterior(prob, 1000)

result_adaptive = run_adaptive(
    prob, candidates, prior_adaptive, acquire;
    budget=budget,
    n_per_step=1,
    headless=true,
    record_posterior=true,
)

log_adaptive = result_adaptive.log
n_adaptive = length(log_adaptive)
μ_adaptive = posterior_mean(result_adaptive.posterior)

println("Steps: $n_adaptive")
println("Posterior mean (adaptive): I₀=$(round(μ_adaptive.I₀; digits=2)), " *
        "R₂₀=$(round(μ_adaptive.R₂₀; digits=2)), " *
        "A=$(round(μ_adaptive.A; digits=2)), K=$(round(μ_adaptive.K; digits=0))")

# ═══════════════════════════════════════════════
# 4. Batch design for comparison (same n)
# ═══════════════════════════════════════════════

println("\n--- Batch design (n=$n_adaptive, matching adaptive count) ---")
prior_batch = ParticlePosterior(prob, 1000)

d = design(prob, candidates, prior_batch; n=n_adaptive, exchange_steps=200)
display(d)

posterior_batch = ParticlePosterior(prob, 1000)
result_batch = run_batch(d, prob, posterior_batch, acquire)

μ_batch = posterior_mean(result_batch.posterior)
println("Posterior mean (batch):  I₀=$(round(μ_batch.I₀; digits=2)), " *
        "R₂₀=$(round(μ_batch.R₂₀; digits=2)), " *
        "A=$(round(μ_batch.A; digits=2)), K=$(round(μ_batch.K; digits=0))")

# ═══════════════════════════════════════════════
# 5. Optimality verification (batch design)
# ═══════════════════════════════════════════════

opt_check = OptimalDesign.verify_optimality(prob, candidates, prior_batch, d;
    posterior_samples=500)
println("\nOptimality verification (batch):")
println("  Is optimal: $(opt_check.is_optimal)")
println("  Max Gateaux derivative: $(round(opt_check.max_derivative; digits=3))")
println("  Bound (q): $(round(opt_check.dimension; digits=3))")

# ═══════════════════════════════════════════════
# 6. Comparison summary
# ═══════════════════════════════════════════════

println("\n=== Head-to-head ($n_adaptive observations each) ===")
for (name, μ) in [("Batch   ", μ_batch), ("Adaptive", μ_adaptive)]
    err_R = abs(μ.R₂₀ - θ_true.R₂₀)
    err_A = abs(μ.A - θ_true.A)
    err_K = abs(μ.K - θ_true.K)
    println("  $name:  |ΔR₂₀|=$(round(err_R; digits=2)), |ΔA|=$(round(err_A; digits=2)), |ΔK|=$(round(err_K; digits=0))")
end

# ═══════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Adaptive trajectory in 2D design space ---

fig1 = Figure(size=(700, 500))

ax1 = GLMakie.Axis(fig1[1, 1],
    xlabel="tSL (s)", ylabel="νSL (Hz)",
    title="Adaptive Design Trajectory ($n_adaptive steps)")

scatter!(ax1, [e.ξ.tSL for e in log_adaptive], [e.ξ.νSL for e in log_adaptive];
    color=1:n_adaptive, colormap=:viridis, markersize=10)
lines!(ax1, [e.ξ.tSL for e in log_adaptive], [e.ξ.νSL for e in log_adaptive];
    color=:gray, linewidth=0.5)

CairoMakie.Colorbar(fig1[1, 2]; colormap=:viridis,
    colorrange=(1, n_adaptive), label="Step")

fig1

# --- Figure 2: 2D batch design allocation (bubble plot) ---

fig2 = OptimalDesign.plot_design_allocation(d, candidates)

# --- Figure 3: Gateaux derivative over 2D candidate space ---

gd = OptimalDesign.gateaux_derivative(prob, candidates, prior_batch, d;
    posterior_samples=500)

fig3 = OptimalDesign.plot_gateaux(candidates, gd, opt_check.dimension)

# --- Figure 4: Dispersion curve — prior vs posterior credible bands ---
# Slice: fix tSL at a short time to see the dispersion shape

ν_grid = range(300, 15_000, length=100)
t_slice = 0.01
slice_grid = [(tSL=t_slice, νSL=ν) for ν in ν_grid]

preds_prior = posterior_predictions(prob, prior_batch, slice_grid; n_samples=200)
preds_batch = posterior_predictions(prob, result_batch.posterior, slice_grid; n_samples=200)
preds_adaptive = posterior_predictions(prob, result_adaptive.posterior, slice_grid; n_samples=200)

band_prior = credible_band(preds_prior; level=0.9)
band_batch = credible_band(preds_batch; level=0.9)
band_adaptive = credible_band(preds_adaptive; level=0.9)

y_true_slice = [predict(θ_true, ξ) for ξ in slice_grid]

fig4 = Figure(size=(700, 700))

ax4a = GLMakie.Axis(fig4[1, 1], ylabel="I(tSL=$t_slice)",
    title="Prior (90% CI)")
band!(ax4a, collect(ν_grid), band_prior.lower, band_prior.upper, color=(:gray, 0.3))
lines!(ax4a, collect(ν_grid), band_prior.median, color=:gray, linewidth=2)
lines!(ax4a, collect(ν_grid), y_true_slice, color=:red, linewidth=1.5, linestyle=:dash)

ax4b = GLMakie.Axis(fig4[2, 1], ylabel="I(tSL=$t_slice)",
    title="Adaptive ($n_adaptive obs, 90% CI)")
band!(ax4b, collect(ν_grid), band_adaptive.lower, band_adaptive.upper, color=(:orange, 0.3))
lines!(ax4b, collect(ν_grid), band_adaptive.median, color=:orange, linewidth=2)
lines!(ax4b, collect(ν_grid), y_true_slice, color=:red, linewidth=1.5, linestyle=:dash)
adaptive_obs_near = [(ξ=e.ξ, y=e.y) for e in log_adaptive if abs(e.ξ.tSL - t_slice) < 0.005]
if !isempty(adaptive_obs_near)
    scatter!(ax4b, [o.ξ.νSL for o in adaptive_obs_near], [o.y for o in adaptive_obs_near],
        color=:black, markersize=6)
end

ax4c = GLMakie.Axis(fig4[3, 1], xlabel="νSL (Hz)", ylabel="I(tSL=$t_slice)",
    title="Batch ($n_adaptive obs, 90% CI)")
band!(ax4c, collect(ν_grid), band_batch.lower, band_batch.upper, color=(:blue, 0.3))
lines!(ax4c, collect(ν_grid), band_batch.median, color=:blue, linewidth=2)
lines!(ax4c, collect(ν_grid), y_true_slice, color=:red, linewidth=1.5, linestyle=:dash)
batch_obs_near = [o for o in result_batch.observations if abs(o.ξ.tSL - t_slice) < 0.005]
if !isempty(batch_obs_near)
    scatter!(ax4c, [o.ξ.νSL for o in batch_obs_near], [o.y for o in batch_obs_near],
        color=:black, markersize=6)
end

CairoMakie.hidexdecorations!(ax4a; grid=false)
CairoMakie.hidexdecorations!(ax4b; grid=false)

fig4

# --- Figure 5: Corner plot — adaptive vs batch posterior ---

fig5 = plot_corner(result_adaptive.posterior, result_batch.posterior;
    params=[:R₂₀, :A, :K], labels=["Adaptive", "Batch"],
    truth=(R₂₀=θ_true.R₂₀, A=θ_true.A, K=θ_true.K))

# --- Figure 6: ESS and posterior convergence ---

prior_ess = ParticlePosterior(prob, 1000)
ess_history = Float64[]
r20_history = Float64[]
A_history = Float64[]
K_history = Float64[]

for entry in log_adaptive
    OptimalDesign.update!(prior_ess, prob, entry.ξ, entry.y)
    push!(ess_history, effective_sample_size(prior_ess))
    μ = posterior_mean(prior_ess)
    push!(r20_history, μ.R₂₀)
    push!(A_history, μ.A)
    push!(K_history, μ.K)
end

fig6 = Figure(size=(700, 700))

ax6a = GLMakie.Axis(fig6[1, 1], ylabel="R₂₀", title="Posterior Convergence")
lines!(ax6a, 1:n_adaptive, r20_history, color=:blue, linewidth=2)
hlines!(ax6a, [θ_true.R₂₀], color=:red, linestyle=:dash)

ax6b = GLMakie.Axis(fig6[2, 1], ylabel="A")
lines!(ax6b, 1:n_adaptive, A_history, color=:blue, linewidth=2)
hlines!(ax6b, [θ_true.A], color=:red, linestyle=:dash)

ax6c = GLMakie.Axis(fig6[3, 1], ylabel="K")
lines!(ax6c, 1:n_adaptive, K_history, color=:blue, linewidth=2)
hlines!(ax6c, [θ_true.K], color=:red, linestyle=:dash)

ax6d = GLMakie.Axis(fig6[4, 1], xlabel="Step", ylabel="ESS")
lines!(ax6d, 1:n_adaptive, ess_history, color=:blue, linewidth=2)
hlines!(ax6d, [100], color=:gray, linestyle=:dash)

CairoMakie.hidexdecorations!(ax6a; grid=false)
CairoMakie.hidexdecorations!(ax6b; grid=false)
CairoMakie.hidexdecorations!(ax6c; grid=false)

fig6

# --- Figure 7: Posterior evolution animation ---

if OptimalDesign.has_posterior_history(log_adaptive)
    println("Recording posterior evolution animation...")
    record_corner_animation(log_adaptive, "ex8_posterior_evolution.mp4";
        params=[:R₂₀, :A, :K],
        truth=(R₂₀=θ_true.R₂₀, A=θ_true.A, K=θ_true.K),
        framerate=5)
end

println("\nDone. Figures created.")
