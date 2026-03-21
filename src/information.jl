"""
    weighted_fim(J, σ)

Compute J'Σ⁻¹J from Jacobian J and noise specification σ.

- Scalar σ: F = J'J / σ²
- Vector σ: F = J' diag(1 ./ σ.²) J
- Matrix Σ: F = J' Σ⁻¹ J
"""
function weighted_fim(J::AbstractMatrix, σ::Real)
    return J' * J / σ^2
end

function weighted_fim(J::AbstractMatrix, σ::AbstractVector)
    W = Diagonal(1 ./ σ .^ 2)
    return J' * W * J
end

function weighted_fim(J::AbstractMatrix, Σ::AbstractMatrix)
    Σ_inv = inv(Σ)
    return J' * Σ_inv * J
end

# Handle scalar predict (1D Jacobian comes back as a vector, reshape to 1×p)
function weighted_fim(J::AbstractVector, σ)
    return weighted_fim(reshape(J, 1, length(J)), σ)
end

"""
    information(prob, θ, ξ)

Compute the Fisher Information Matrix at (θ, ξ) for the given DesignProblem.

Dispatches on whether an analytic Jacobian is provided or ForwardDiff is used.
"""
function information(prob::DesignProblem, θ, ξ)
    J = if prob.jacobian === nothing
        # ForwardDiff: differentiate predict w.r.t. θ
        y = prob.predict(θ, ξ)
        if y isa Real
            # Scalar observation: gradient -> reshape to 1×p
            g = ForwardDiff.gradient(θ_ -> prob.predict(θ_, ξ), θ)
            reshape(g, 1, length(g))
        else
            ForwardDiff.jacobian(θ_ -> prob.predict(θ_, ξ), θ)
        end
    else
        prob.jacobian(θ, ξ)
    end
    σ = prob.sigma(θ, ξ)
    weighted_fim(J, σ)
end

"""
    GradientCache(p)

Pre-allocated buffers for repeated `information!` calls.
Avoids creating a new GradientConfig on every ForwardDiff.gradient! call.
"""
struct GradientCache
    g_buf::Vector{Float64}
    cfg::ForwardDiff.GradientConfig
end

function GradientCache(θ::AbstractVector, predict, ξ)
    p = length(θ)
    g_buf = zeros(p)
    f = θ_ -> predict(θ_, ξ)
    cfg = ForwardDiff.GradientConfig(f, θ)
    GradientCache(g_buf, cfg)
end

"""
    information!(M, prob, θ, ξ; cache=nothing)

In-place version: compute the FIM and write result into pre-allocated matrix M.
Accepts an optional `GradientCache` to avoid allocating ForwardDiff config each call.
"""
function information!(M::AbstractMatrix, prob::DesignProblem, θ, ξ;
                      cache::Union{Nothing, GradientCache}=nothing)
    p = size(M, 1)

    if prob.jacobian === nothing
        y = prob.predict(θ, ξ)
        if y isa Real
            # Scalar observation: gradient into buffer, then outer product
            f = θ_ -> prob.predict(θ_, ξ)
            if cache !== nothing
                ForwardDiff.gradient!(cache.g_buf, f, θ, cache.cfg, Val{false}())
                g = cache.g_buf
            else
                g = ForwardDiff.gradient(f, θ)
            end
            σ = prob.sigma(θ, ξ)
            σ² = σ^2
            @inbounds for j in 1:p, i in 1:p
                M[i, j] = g[i] * g[j] / σ²
            end
        else
            J = ForwardDiff.jacobian(θ_ -> prob.predict(θ_, ξ), θ)
            σ = prob.sigma(θ, ξ)
            weighted_fim!(M, J, σ)
        end
    else
        J = prob.jacobian(θ, ξ)
        σ = prob.sigma(θ, ξ)
        weighted_fim!(M, J, σ)
    end
    M
end

"""
    weighted_fim!(M, J, σ)

In-place computation of J'Σ⁻¹J, writing result into M.
"""
function weighted_fim!(M::AbstractMatrix, J::AbstractMatrix, σ::Real)
    σ² = σ^2
    mul!(M, J', J)
    M ./= σ²
    M
end

function weighted_fim!(M::AbstractMatrix, J::AbstractMatrix, σ::AbstractVector)
    p = size(J, 2)
    n = size(J, 1)
    fill!(M, 0.0)
    @inbounds for i in 1:n
        inv_σ² = 1 / σ[i]^2
        for c in 1:p, r in 1:p
            M[r, c] += J[i, r] * J[i, c] * inv_σ²
        end
    end
    M
end

function weighted_fim!(M::AbstractMatrix, J::AbstractVector, σ)
    weighted_fim!(M, reshape(J, 1, length(J)), σ)
end

"""
    transform(prob, M)

Apply the transformation to the information matrix.

- Identity: returns M unchanged.
- DeltaMethod: computes [∇τ M⁻¹ ∇τ']⁻¹ via ForwardDiff.
"""
function transform(prob::DesignProblem, M::AbstractMatrix, θ)
    transform(prob.transformation, M, θ)
end

function transform(::Identity, M::AbstractMatrix, θ)
    M
end

function transform(dm::DeltaMethod, M::AbstractMatrix, θ)
    # M may be singular (e.g. rank-deficient FIM from too few observations).
    # Use cholesky to check before inverting; return a zero matrix on failure
    # so that downstream safe_criterion correctly returns -Inf.
    C = cholesky(Symmetric(M); check=false)
    issuccess(C) || return zeros(eltype(M), size(M))
    M_inv = inv(C)
    ∇τ = ForwardDiff.jacobian(dm.f, θ)
    R = ∇τ * M_inv * ∇τ'
    C2 = cholesky(Symmetric(R); check=false)
    issuccess(C2) ? Matrix(inv(C2)) : zeros(eltype(M), size(R))
end
