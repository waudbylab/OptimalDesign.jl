"""
    ParticlePosterior{T}

Weighted particle approximation to a posterior distribution.
Particles are ComponentArrays, weights are in log-space.
"""
struct ParticlePosterior{T}
    particles::Vector{T}
    log_weights::Vector{Float64}
end

"""
    ParticlePosterior(prob::AbstractDesignProblem, n::Int)

Construct a ParticlePosterior by drawing n particles from the prior.
All particles have equal weight.
"""
function ParticlePosterior(prob::AbstractDesignProblem, n::Int)
    particles = draw(prob.parameters, n)
    log_weights = fill(-log(n), n)
    ParticlePosterior(particles, log_weights)
end

"""
    sample(posterior::ParticlePosterior, n::Int)

Draw n particles from the posterior (with replacement, proportional to weights).
"""
function sample(posterior::ParticlePosterior, n::Int)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)
    posterior.particles[indices]
end

"""
    posterior_mean(posterior::ParticlePosterior)

Compute the weighted mean of the posterior particles.
"""
function posterior_mean(posterior::ParticlePosterior)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    result = sum(w[i] * posterior.particles[i] for i in eachindex(posterior.particles))
    result
end

"""
    effective_sample_size(posterior::ParticlePosterior)

Compute the effective sample size (ESS) of the weighted particles.
"""
function effective_sample_size(posterior::ParticlePosterior)
    lw = posterior.log_weights .- logsumexp(posterior.log_weights)
    exp(-logsumexp(2 .* lw))
end

"""
    loglikelihood(prob::AbstractDesignProblem, θ, ξ, y)

Log-likelihood of observation y at (θ, ξ) under the noise model defined by prob.sigma.

Handles scalar, vector, and structured observations (NamedTuple with :value and :σ).
"""
function loglikelihood(prob::AbstractDesignProblem, θ, ξ, y)
    ŷ = prob.predict(θ, ξ)
    # Structured observation: use realised noise
    if y isa NamedTuple && haskey(y, :value) && haskey(y, :σ)
        return _loglikelihood_gaussian(y.value, ŷ, y.σ)
    end
    σ = prob.sigma(θ, ξ)
    _loglikelihood_gaussian(y, ŷ, σ)
end

function _loglikelihood_gaussian(y::Real, ŷ::Real, σ::Real)
    -0.5 * log(2π) - log(σ) - 0.5 * ((y - ŷ) / σ)^2
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::AbstractVector, σ::AbstractVector)
    n = length(y)
    -0.5 * n * log(2π) - sum(log.(σ)) - 0.5 * sum(((y .- ŷ) ./ σ) .^ 2)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::AbstractVector, Σ::AbstractMatrix)
    n = length(y)
    r = y .- ŷ
    -0.5 * n * log(2π) - 0.5 * logdet(Σ) - 0.5 * r' * inv(Σ) * r
end

# Scalar y with vector prediction or vice versa — promote
function _loglikelihood_gaussian(y::Real, ŷ::AbstractVector, σ)
    _loglikelihood_gaussian([y], ŷ, σ)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::Real, σ)
    _loglikelihood_gaussian(y, [ŷ], σ isa Real ? [σ] : σ)
end

"""
    update!(posterior::ParticlePosterior, prob::AbstractDesignProblem, ξ, y; ess_threshold=0.5, a=0.95)

Incorporate observation y at design point ξ. Delegates to the batch method
with adaptive tempering, so even a single highly informative observation
is tempered in gracefully.
"""
function update!(posterior::ParticlePosterior, prob::AbstractDesignProblem, ξ, y;
                 ess_threshold::Float64=0.5, a::Float64=0.95)
    update!(posterior, prob, [(ξ=ξ, y=y)]; ess_threshold=ess_threshold, a=a)
end

"""
    update!(posterior::ParticlePosterior, prob::AbstractDesignProblem, data::AbstractVector{<:NamedTuple};
            ess_threshold=0.5, a=0.95)

Batch update using adaptive likelihood tempering (SMC sampler).

Computes each particle's total log-likelihood across all data, then raises
the tempering exponent β from 0 → 1 in adaptive steps. At each step, the
step size Δβ is chosen by bisection so that the ESS stays just above
`ess_threshold × n`, then particles are resampled with Liu-West jittering.

Each element of `data` must have fields `ξ` (design point) and `y` (observation).
"""
function update!(posterior::ParticlePosterior, prob::AbstractDesignProblem,
                 data::AbstractVector{<:NamedTuple};
                 ess_threshold::Float64=0.5, a::Float64=0.95)
    n = length(posterior.particles)
    target_ess = ess_threshold * n

    # Compute total log-likelihood for each particle
    total_ll = _compute_total_ll(posterior, prob, data)

    β = 0.0
    step = 0
    while β < 1.0
        step += 1
        remaining = 1.0 - β

        # Find largest Δβ ∈ (0, remaining] keeping trial ESS ≥ target
        Δβ = _bisect_Δβ(posterior.log_weights, total_ll, remaining, target_ess)

        # Apply the step
        for i in 1:n
            posterior.log_weights[i] += Δβ * total_ll[i]
        end
        lse = logsumexp(posterior.log_weights)
        posterior.log_weights .-= lse
        β += Δβ

        ess = effective_sample_size(posterior)
        @debug "Tempering step $step: Δβ=$(round(Δβ; digits=4)), β=$(round(β; digits=4)), ESS=$(round(ess; digits=1))"

        if ess < target_ess
            resample!(posterior; prob=prob, a=a)
            total_ll = _compute_total_ll(posterior, prob, data)
        end
    end
    @debug "Tempering complete in $step steps"
    posterior
end

"""Compute total log-likelihood of all data for each particle."""
function _compute_total_ll(posterior::ParticlePosterior, prob::AbstractDesignProblem,
                           data::AbstractVector{<:NamedTuple})
    n = length(posterior.particles)
    total_ll = Vector{Float64}(undef, n)
    for i in 1:n
        θ = posterior.particles[i]
        ll = 0.0
        for d in data
            ll += loglikelihood(prob, θ, d.ξ, d.y)
        end
        total_ll[i] = ll
    end
    total_ll
end

"""
Bisect for the largest Δβ ∈ (0, remaining] such that trial ESS ≥ target.
If even Δβ = remaining keeps ESS above target, return remaining (finish in one step).
"""
function _bisect_Δβ(log_weights::Vector{Float64}, total_ll::Vector{Float64},
                    remaining::Float64, target_ess::Float64;
                    max_iter::Int=30, tol::Float64=1e-6)
    # First check: can we take the full remaining step?
    trial_ess = _trial_ess(log_weights, total_ll, remaining)
    trial_ess >= target_ess && return remaining

    # Bisect between lo (safe) and hi (too aggressive)
    lo = 0.0
    hi = remaining
    for _ in 1:max_iter
        mid = (lo + hi) / 2
        (hi - lo) < tol && break
        if _trial_ess(log_weights, total_ll, mid) >= target_ess
            lo = mid
        else
            hi = mid
        end
    end
    # Return lo (the safe side); but ensure we make some progress
    max(lo, remaining * 1e-6)
end

"""Compute ESS that would result from adding Δβ * total_ll to log_weights, without modifying them."""
function _trial_ess(log_weights::Vector{Float64}, total_ll::Vector{Float64}, Δβ::Float64)
    n = length(log_weights)
    trial = Vector{Float64}(undef, n)
    for i in 1:n
        trial[i] = log_weights[i] + Δβ * total_ll[i]
    end
    lse = logsumexp(trial)
    trial .-= lse
    exp(-logsumexp(2 .* trial))
end

"""
    systematic_resample(weights, n)

Systematic resampling: returns n indices sampled proportional to weights.
"""
function systematic_resample(weights::AbstractVector, n::Int)
    cumw = cumsum(weights)
    u = rand() / n
    indices = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        target = u + (i - 1) / n
        while j < length(cumw) && cumw[j] < target
            j += 1
        end
        indices[i] = j
    end
    indices
end

# --- Parameter transforms for bound-respecting jitter ---

"""
    _param_transforms(parameters::NamedTuple)

Derive (forward, inverse) transform pairs from prior distributions.
Forward maps to unconstrained space; inverse maps back.
"""
function _param_transforms(parameters::NamedTuple)
    map(parameters) do dist
        sup = support(dist)
        lo = minimum(sup)
        hi = maximum(sup)
        if lo == -Inf && hi == Inf
            # Unbounded (e.g., Normal)
            (forward=identity, inverse=identity)
        elseif isfinite(lo) && hi == Inf
            # Lower-bounded (e.g., LogUniform, Exponential)
            (forward=x -> log(x - lo + eps()), inverse=z -> lo + exp(z))
        elseif lo == -Inf && isfinite(hi)
            # Upper-bounded (rare)
            (forward=x -> -log(hi - x + eps()), inverse=z -> hi - exp(-z))
        else
            # Bounded [lo, hi] (e.g., Uniform, Beta)
            (forward=x -> log((x - lo + eps()) / (hi - x + eps())),
             inverse=z -> lo + (hi - lo) / (1 + exp(-z)))
        end
    end
end

"""
    resample!(posterior; prob=nothing, a=0.98)

Systematic resampling with Liu-West kernel jittering.

The Liu-West filter shrinks each resampled particle toward the ensemble mean
and adds correlated noise, calibrated to preserve the posterior's first two moments.
When `prob` is provided, jittering operates in a transformed (unconstrained) space
derived from the prior bounds, preventing particles from leaving the support.

# Arguments
- `prob`: DesignProblem (optional). Provides prior distributions for bound-aware transforms.
- `a`: shrinkage parameter (default 0.98). Controls jitter magnitude: h² = 1 - a².
"""
function resample!(posterior::ParticlePosterior; prob::Union{DesignProblem,Nothing}=nothing, a::Float64=0.98)
    n = length(posterior.particles)
    d = length(first(posterior.particles))
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)

    new_particles = posterior.particles[indices]

    # Liu-West kernel: shrink + correlated noise preserving moments
    h² = 1 - a^2
    h = sqrt(h²)

    # Get parameter transforms (identity if no prob)
    transforms = if prob !== nothing
        _param_transforms(prob.parameters)
    else
        nothing
    end

    pnames = keys(first(posterior.particles))

    # Transform particles to unconstrained space
    Z = Matrix{Float64}(undef, d, n)  # columns are particles in transformed space
    for i in 1:n
        θ = posterior.particles[i]
        for (ki, k) in enumerate(pnames)
            val = getproperty(θ, k)
            Z[ki, i] = transforms !== nothing ? transforms[ki].forward(val) : val
        end
    end

    # Weighted mean and covariance in transformed space
    μ_z = Z * w
    Z_centered = Z .- μ_z
    Σ_z = (Z_centered .* w') * Z_centered'

    # Cholesky with regularisation
    C = cholesky(Symmetric(Σ_z + 1e-8 * I); check=false)
    if issuccess(C)
        L = C.L
    else
        # Fallback: diagonal jitter
        @warn "Liu-West: covariance Cholesky failed, falling back to diagonal jitter"
        L = Diagonal(sqrt.(max.(diag(Σ_z), 1e-20)))
    end

    # Apply Liu-West to resampled particles
    for i in 1:n
        # Get resampled particle in transformed space
        θ_old = new_particles[i]
        z_i = Vector{Float64}(undef, d)
        for (ki, k) in enumerate(pnames)
            z_i[ki] = transforms !== nothing ? transforms[ki].forward(getproperty(θ_old, k)) : getproperty(θ_old, k)
        end

        # Shrink toward mean + correlated noise
        m_i = a .* z_i .+ (1 - a) .* μ_z
        z_new = m_i .+ h .* (L * randn(d))

        # Back-transform to original space
        vals = ntuple(d) do ki
            z = z_new[ki]
            transforms !== nothing ? transforms[ki].inverse(z) : z
        end
        new_particles[i] = ComponentArray(NamedTuple{pnames}(vals))
    end

    copyto!(posterior.particles, new_particles)
    fill!(posterior.log_weights, -log(n))

    @debug "Resampled with Liu-West kernel (a=$a, h=$(round(h; digits=4)))"
    posterior
end
