"""
    gateaux_derivative(prob, candidates, posterior, design; kwargs...)

Compute the Gateaux derivative of the expected design criterion at each
candidate, for a design with the given weights.

For D-optimality (Identity):  d(x) = tr(M⁻¹ M(x))
For Ds-optimality (DeltaMethod): d(x) = tr(C M(x)) where C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹

For A/E criteria, numerical differentiation is used.

Returns a vector of derivatives (one per candidate).
"""
function gateaux_derivative(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::Particles,
    ξ::ExperimentalDesign;
    kwargs...,
)
    particles = _get_particles(posterior)
    gateaux_derivative(prob, candidates, particles, weights(ξ, candidates); kwargs...)
end

function gateaux_derivative(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector,
    weights::AbstractVector;
    posterior_samples::Int=50,
    costs::Union{Nothing,AbstractVector{<:Real}}=nothing,
    M_prior=nothing,
)
    criterion = prob.criterion
    K = length(candidates)
    n_particles = length(particles)
    bs = min(posterior_samples, n_particles)
    idx = bs >= n_particles ? (1:n_particles) : randperm(n_particles)[1:bs]
    p = length(first(particles))

    gd = zeros(K)
    count = 0

    for j in idx
        θ = particles[j]
        cache = GradientCache(θ, prob.predict, first(candidates))

        # Build weighted FIM in full parameter space
        M_w = _particle_weighted_fim(prob, θ, candidates, weights; cache=cache, costs=costs)
        if M_prior !== nothing
            M_w .+= M_prior[j]
        end

        C = cholesky(Symmetric(M_w); check=false)
        if !issuccess(C)
            continue
        end

        count += 1

        # Compute per-candidate Gateaux derivatives for this particle
        gd .+= _gateaux_for_particle(criterion, prob, θ, M_w, candidates, cache; costs=costs)
    end

    count == 0 ? fill(-Inf, K) : gd ./ count
end


# --- D-criterion: analytical Gateaux derivative ---

function _gateaux_for_particle(::DCriterion, prob, θ, M_w, candidates,
                               cache::Union{Nothing, GradientCache}=nothing;
                               costs::Union{Nothing,AbstractVector{<:Real}}=nothing)
    p = size(M_w, 1)
    M_w_inv = inv(Symmetric(M_w))

    # Precompute the "sensitivity" matrix C such that d_k = tr(C M_k)
    C = _d_sensitivity_matrix(prob, M_w_inv, θ)::Matrix{Float64}

    # Pre-allocate buffer for the inner loop
    M_k = zeros(p, p)
    K = length(candidates)
    result = Vector{Float64}(undef, K)

    @inbounds for k in 1:K
        information!(M_k, prob, θ, candidates[k]; cache=cache)
        # tr(C * M_k) without allocating the product
        s = 0.0
        for j in 1:p, i in 1:p
            s += C[i, j] * M_k[j, i]
        end
        # Scale by 1/cost: moving weight to candidate k yields info at rate M_k/c_k
        result[k] = costs === nothing ? s : s / costs[k]
    end
    result
end

"""
For D-optimality with Identity: C = M⁻¹
For Ds-optimality with DeltaMethod: C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹

In both cases, d(x) = tr(C M(x)).
"""
function _d_sensitivity_matrix(prob, M_w_inv, θ)
    _d_sensitivity_matrix(prob.transformation, M_w_inv, θ)
end

function _d_sensitivity_matrix(::Identity, M_w_inv, θ)
    Matrix{Float64}(M_w_inv)
end

function _d_sensitivity_matrix(dm::DeltaMethod, M_w_inv, θ)
    ∇τ = ForwardDiff.jacobian(dm.f, θ)
    # Mτ = (∇τ M⁻¹ ∇τ')⁻¹
    Mt = inv(Symmetric(∇τ * M_w_inv * ∇τ'))
    # C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹
    Matrix{Float64}(M_w_inv * ∇τ' * Mt * ∇τ * M_w_inv)
end

# --- A-criterion and E-criterion: numerical Gateaux derivative ---

function _gateaux_for_particle(criterion::DesignCriterion, prob, θ, M_w, candidates,
                               cache::Union{Nothing, GradientCache}=nothing;
                               costs::Union{Nothing,AbstractVector{<:Real}}=nothing)
    p = size(M_w, 1)
    Mt = transform(prob, M_w, θ)
    Φ0 = safe_criterion(criterion, Mt)
    isfinite(Φ0) || return fill(-Inf, length(candidates))

    ε = 1e-6
    M_k = zeros(p, p)
    M_pert = zeros(p, p)
    K = length(candidates)
    result = Vector{Float64}(undef, K)

    @inbounds for k in 1:K
        information!(M_k, prob, θ, candidates[k]; cache=cache)
        # Perturbation scaled by 1/cost when costs are provided
        ε_k = costs === nothing ? ε : ε / costs[k]
        # M_pert = M_w + ε_k * M_k (no allocation)
        for j in 1:p, i in 1:p
            M_pert[i, j] = M_w[i, j] + ε_k * M_k[i, j]
        end
        Mt_ε = transform(prob, M_pert, θ)
        Φ_ε = safe_criterion(criterion, Mt_ε)
        result[k] = isfinite(Φ_ε) ? (Φ_ε - Φ0) / ε : -Inf
    end
    result
end

# --- Optimality verification ---

"""
    verify_optimality(prob, candidates, posterior, ξ; kwargs...) → OptimalityResult

Check the General Equivalence Theorem: at an optimal design, the
Gateaux derivative should be ≤ q (dimension of interest) at all candidates,
with equality at support points.

Returns an `OptimalityResult` with fields `is_optimal`, `max_derivative`,
`dimension`, `gateaux` (full derivative vector), and `candidates`.
"""
function verify_optimality(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::Particles,
    ξ::ExperimentalDesign;
    posterior_samples::Int=50,
    tol::Float64=0.05,
    costs::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    particles = _get_particles(posterior)
    gd = gateaux_derivative(prob, candidates, particles, weights(ξ, candidates);
        posterior_samples=posterior_samples, costs=costs)

    q = _transformed_dimension(prob)
    max_gd = maximum(gd)

    OptimalityResult(max_gd ≤ q + tol, max_gd, q, gd, candidates)
end
