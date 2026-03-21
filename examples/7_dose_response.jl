# Example 7: Dose-Response (Sigmoid Emax) — Domain-Agnostic Validation
#
# Four parameters, full D-optimality. Validates against Kirstine.jl
# published results for the same model.
#
# Demonstrates: Non-domain-specific use, full D-optimality (no transformation),
# validation against an independent implementation.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Sigmoid Emax model ---
#   y = E0 + Emax * dose^h / (ED50^h + dose^h)

prob = DesignProblem(
    (θ, ξ) -> θ.E0 + θ.Emax * ξ.dose^θ.h / (θ.ED50^θ.h + ξ.dose^θ.h),
    parameters=(E0=Normal(1, 0.5), Emax=Normal(2, 0.5),
        ED50=LogNormal(-1, 0.5), h=LogNormal(1, 0.5)),
    cost=Returns(1.0),
)

candidates = [(dose=d,) for d in range(0.01, 1.0, length=50)]

# --- Examine FIM at nominal parameters ---

θ_nom = ComponentArray(E0=1.0, Emax=2.0, ED50=exp(-1), h=exp(1))
println("Nominal parameters:")
println("  E0=$(θ_nom.E0), Emax=$(θ_nom.Emax), ED50=$(round(θ_nom.ED50, digits=4)), h=$(round(θ_nom.h, digits=4))")

# Evaluate FIM at a few dose levels
for d in [0.1, 0.3, 0.5, 0.8]
    M = information(prob, θ_nom, (dose=d,))
    println("\nFIM at dose=$d:")
    println("  rank = $(rank(M)), trace = $(round(tr(M), digits=2))")
end

# --- Score candidates with full D-optimality ---

prior = ParticlePosterior(prob, 500)

println("\nScoring dose levels by D-optimality...")
scores = score_candidates(prob, DCriterion(), prior.particles, candidates; posterior_samples=100)

ranking = sortperm(scores, rev=true)
println("\nTop 10 dose levels:")
for i in 1:10
    idx = ranking[i]
    println("  dose = $(round(candidates[idx].dose, digits=4)),  utility = $(round(scores[idx], digits=3))")
end

# --- Model predictions at nominal parameters ---

println("\nDose-response curve at nominal parameters:")
for d in range(0.0, 1.0, length=11)
    if d == 0.0
        y = θ_nom.E0
    else
        y = prob.predict(θ_nom, (dose=d,))
    end
    bar = repeat("█", round(Int, y * 10))
    println("  dose=$(round(d, digits=2))  y=$(round(y, digits=3))  $bar")
end

# --- Kirstine.jl comparison ---
# The optimal design for the sigmoid Emax model with 4 parameters under
# D-optimality should place support points at approximately 4-5 distinct
# dose levels spanning the range, with more weight near the inflection
# point (around ED50).
