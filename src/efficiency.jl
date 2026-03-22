"""
    efficiency(d_a, d_b, prob, candidates, posterior; kwargs...)

Relative efficiency of design_a vs design_b.

For D-optimality: (det M_a / det M_b)^(1/q) where q = dimension of interest.
Efficiency > 1 means design_a is better; < 1 means design_b is better.
"""
function efficiency(
    d_a::ExperimentalDesign,
    d_b::ExperimentalDesign,
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::ParticlePosterior;
    posterior_samples::Int=50,
)
    particles = _get_particles(posterior)
    efficiency(weights(d_a, candidates), weights(d_b, candidates),
        prob, candidates, particles; posterior_samples=posterior_samples)
end

function efficiency(
    weights_a::AbstractVector,
    weights_b::AbstractVector,
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector;
    posterior_samples::Int=50,
)
    criterion = prob.criterion
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
