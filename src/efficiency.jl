"""
    efficiency(weights_a, weights_b, prob, candidates, particles; kwargs...)

Relative efficiency of design_a vs design_b.

For D-optimality: (det M_a / det M_b)^(1/q) where q = dimension of interest.
Efficiency > 1 means design_a is better; < 1 means design_b is better.
"""
function efficiency(
    weights_a::AbstractVector,
    weights_b::AbstractVector,
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector;
    criterion::DesignCriterion=DCriterion(),
    posterior_samples::Int=50,
)
    # Compute average criterion value for each design
    Φ_a = _average_criterion(prob, candidates, particles, weights_a;
        criterion=criterion, posterior_samples=posterior_samples)
    Φ_b = _average_criterion(prob, candidates, particles, weights_b;
        criterion=criterion, posterior_samples=posterior_samples)

    _efficiency(criterion, Φ_a, Φ_b, _transformed_dimension(prob))
end

"""
Compute the average criterion value E_θ[Φ(M_τ(w,θ))] for a given weight vector.
"""
function _average_criterion(
    prob, candidates, particles, weights;
    criterion, posterior_samples,
)
    n_particles = length(particles)
    bs = min(posterior_samples, n_particles)
    idx = randperm(n_particles)[1:bs]

    total = 0.0
    count = 0

    for j in idx
        θ = particles[j]
        M_w = _particle_weighted_fim(prob, θ, candidates, weights)
        Mt = transform(prob, M_w, θ)
        val = safe_criterion(criterion, Mt)
        if isfinite(val)
            total += val
            count += 1
        end
    end

    count == 0 ? -Inf : total / count
end

function _efficiency(::DCriterion, Φ_a, Φ_b, q)
    # Φ = log det M, so (det_a / det_b)^(1/q) = exp((Φ_a - Φ_b) / q)
    exp((Φ_a - Φ_b) / q)
end

function _efficiency(::ACriterion, Φ_a, Φ_b, q)
    # Φ = -tr(M⁻¹), so efficiency = Φ_b / Φ_a (both negative, ratio > 1 means a is better)
    Φ_b / Φ_a
end

function _efficiency(::ECriterion, Φ_a, Φ_b, q)
    # Φ = λ_min, efficiency = Φ_a / Φ_b
    Φ_a / Φ_b
end

"""
    apportion(weights, n)

Convert continuous weights to integer counts summing to n.

Uses the largest remainder method (Hamilton's method):
1. Floor each weight * n
2. Distribute remaining counts to candidates with largest fractional parts.
"""
function apportion(weights::AbstractVector{<:Real}, n::Int)
    scaled = weights .* n
    floored = floor.(Int, scaled)
    remainders = scaled .- floored
    deficit = n - sum(floored)

    # Distribute deficit to candidates with largest remainders
    if deficit > 0
        order = sortperm(remainders, rev=true)
        for i in 1:deficit
            floored[order[i]] += 1
        end
    end

    floored
end

"""
    apportion(weights, budget, costs)

Budget-aware apportionment: convert continuous budget-fraction weights to
integer measurement counts given per-point costs.

Point k gets `budget * weights[k] / costs[k]` ideal measurements.
Uses largest-remainder rounding, checking that each additional measurement
fits within the budget.
"""
function apportion(weights::AbstractVector{<:Real}, budget::Real,
                   costs::AbstractVector{<:Real})
    ideal = [weights[k] > 1e-10 ? budget * weights[k] / costs[k] : 0.0
             for k in eachindex(weights)]
    floored = floor.(Int, ideal)
    remainders = ideal .- floored

    # Distribute remaining budget to candidates with largest remainders
    used = sum(floored[k] * costs[k] for k in eachindex(costs))
    order = sortperm(remainders, rev=true)
    for k in order
        if remainders[k] > 0 && used + costs[k] <= budget + 1e-10
            floored[k] += 1
            used += costs[k]
        end
    end
    floored
end
