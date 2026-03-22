"""
    safe_criterion(criterion, M) -> Float64

Evaluate the design criterion on M, returning -Inf for singular/non-posdef
matrices instead of throwing. Avoids try/catch in hot loops.
"""
function safe_criterion(::DCriterion, M::AbstractMatrix)
    S = Symmetric(M)
    C = cholesky(S; check=false)
    issuccess(C) ? logdet(C) : -Inf
end

function safe_criterion(::ACriterion, M::AbstractMatrix)
    S = Symmetric(M)
    C = cholesky(S; check=false)
    issuccess(C) ? -tr(inv(C)) : -Inf
end

function safe_criterion(::ECriterion, M::AbstractMatrix)
    S = Symmetric(M)
    C = cholesky(S; check=false)
    issuccess(C) ? eigmin(S) : -Inf
end

"""
    expected_utility(prob, particles, ξ; posterior_samples=50)

Compute the expected utility of design point ξ by Monte Carlo over posterior particles.
Uses the criterion from `prob.criterion`.
"""
function expected_utility(prob::AbstractDesignProblem, particles::AbstractVector, ξ; posterior_samples::Int=50)
    criterion = prob.criterion
    n = length(particles)
    bs = min(posterior_samples, n)
    idx = randperm(n)[1:bs]
    total = 0.0
    count = 0
    for i in idx
        θ = particles[i]
        M = information(prob, θ, ξ)
        Mt = transform(prob, M, θ)
        val = safe_criterion(criterion, Mt)
        if isfinite(val)
            total += val
            count += 1
        end
    end
    count == 0 ? -Inf : total / count
end

"""
    score_candidates(prob, posterior, candidates; posterior_samples=50)

Score all candidates by expected utility. Returns a vector of scores.
"""
function score_candidates(prob::AbstractDesignProblem, posterior::ParticlePosterior, candidates::AbstractVector; posterior_samples::Int=50)
    score_candidates(prob, posterior.particles, candidates; posterior_samples=posterior_samples)
end

function score_candidates(prob::AbstractDesignProblem, particles::AbstractVector, candidates::AbstractVector; posterior_samples::Int=50)
    [expected_utility(prob, particles, ξ; posterior_samples=posterior_samples) for ξ in candidates]
end
