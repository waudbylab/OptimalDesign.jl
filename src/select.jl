"""
    select(problem, candidates, posterior; kwargs...)

Unified interface for both batch and adaptive design.

Returns `Vector{Tuple{NamedTuple, Int}}` — (ξ, count) pairs.

**Batch design** (`n` large, prior as posterior): optimises weights via exchange algorithm.
**Adaptive design** (`n = 1` or small): scores by utility/cost, greedy multi-point selection.

# Keyword arguments
- `n = 1`: number of measurements to allocate
- `criterion = DCriterion()`: design criterion
- `budget = Inf`: total cost budget
- `posterior_samples = 0`: number of posterior samples for utility evaluation
- `ξ_prev = nothing`: previous design point (for cost computation)
- `exchange_algorithm = (n > 5)`: if true, use exchange algorithm for weight optimisation
- `exchange_steps = 100`: iterations for exchange algorithm
"""
function select(
    problem::DesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior;
    n::Int=1,
    criterion::DesignCriterion=DCriterion(),
    budget::Real=Inf,
    posterior_samples::Int=0,
    ξ_prev=nothing,
    exchange_algorithm::Bool=n > 5,
    exchange_steps::Int=100,
    prior_designs::AbstractVector=NamedTuple[],
)
    all_particles = _get_particles(posterior)

    if posterior_samples ≤ 0
        posterior_samples = length(all_particles)
    end
    if posterior_samples > length(all_particles)
        posterior_samples = length(all_particles)
    end

    # Draw a weighted subsample so uniform averaging in scoring is correct.
    # For ParticlePosterior this resamples proportional to weights;
    # for plain Vector it just returns the full set.
    particles = _get_particles(posterior; n=posterior_samples)

    if exchange_algorithm
        _select_batch(problem, candidates, particles, n;
            criterion, posterior_samples, exchange_steps)
    else
        _select_greedy(problem, candidates, particles, n;
            criterion, posterior_samples, budget, ξ_prev, prior_designs)
    end
end

"""
Extract particles from posterior (supports ParticlePosterior or plain Vector).
When `resample=true`, draws from the posterior proportional to weights
so that a uniform average over the returned particles is correct.
"""
function _get_particles(post::ParticlePosterior; n::Int=0)
    if n > 0
        # Weighted subsample: draw n particles proportional to weights
        # so that uniform averaging in the scoring loop is correct
        sample(post, n)
    else
        post.particles
    end
end
_get_particles(particles::AbstractVector; n::Int=0) = particles

"""
Greedy sequential selection: pick the best candidate, add its FIM to a
running total, then pick the next based on marginal gain.

When `n=1` (adaptive case), this is a simple argmax of utility/cost.
When `n>1`, the accumulated FIM ensures each successive pick adds
complementary information rather than repeating the same candidate.

Precomputes all per-particle FIMs to avoid redundant ForwardDiff calls.
"""
function _select_greedy(
    prob, candidates, particles, n;
    criterion, posterior_samples, budget, ξ_prev,
    prior_designs=NamedTuple[],
)
    selected = NamedTuple[]
    remaining_budget = budget
    prev = ξ_prev
    np = length(particles)
    p = length(first(particles))
    K = length(candidates)

    # Precompute FIM for each (particle, candidate) pair — the expensive part
    @debug "Precomputing FIM cache: $np particles × $K candidates"
    M_cache = Vector{Vector{Matrix{Float64}}}(undef, np)
    M_buf = zeros(p, p)
    for ji in 1:np
        θ = particles[ji]
        cache = GradientCache(θ, prob.predict, first(candidates))
        M_cache[ji] = Vector{Matrix{Float64}}(undef, K)
        for k in 1:K
            information!(M_buf, prob, θ, candidates[k]; cache=cache)
            M_cache[ji][k] = copy(M_buf)
        end
    end

    # Running FIM per particle: accumulates information from already-selected points
    # Initialise with FIMs from previous observations so that the greedy scorer
    # evaluates Φ(M_accumulated + M_new) — critical for n=1 adaptive steps where
    # a single scalar observation gives a rank-deficient FIM.
    M_running = [zeros(p, p) for _ in 1:np]
    if !isempty(prior_designs)
        for ji in 1:np
            θ = particles[ji]
            cache = GradientCache(θ, prob.predict, first(prior_designs))
            for ξ_old in prior_designs
                information!(M_buf, prob, θ, ξ_old; cache=cache)
                M_running[ji] .+= M_buf
            end
        end
        @debug "Initialised M_running from $(length(prior_designs)) prior designs"
    end

    M_trial = zeros(p, p)

    for step in 1:n
        # Compute baseline E[Φ(M_running)] — the criterion without any new point.
        # Scoring by marginal gain (score - baseline) / cost avoids the sign
        # inversion that occurs when Φ is negative and cost divides the total.
        baseline_total = 0.0
        baseline_count = 0
        for ji in 1:np
            Mt = transform(prob, M_running[ji], particles[ji])
            val = safe_criterion(criterion, Mt)
            if isfinite(val)
                baseline_total += val
                baseline_count += 1
            end
        end
        baseline = baseline_count == 0 ? -Inf : baseline_total / baseline_count

        # Score each candidate by E[Φ(M_running + M_k) - Φ(M_running)] / cost
        scores = fill(-Inf, K)
        for k in 1:K
            c = prob.cost(prev, candidates[k])
            if c > remaining_budget
                continue
            end

            total = 0.0
            count = 0
            for ji in 1:np
                # M_trial = M_running[ji] + M_cache[ji][k] (no allocation)
                @inbounds for col in 1:p, row in 1:p
                    M_trial[row, col] = M_running[ji][row, col] + M_cache[ji][k][row, col]
                end
                Mt = transform(prob, M_trial, particles[ji])
                val = safe_criterion(criterion, Mt)
                if isfinite(val)
                    total += val
                    count += 1
                end
            end
            score = count == 0 ? -Inf : total / count
            gain = isfinite(baseline) ? score - baseline : score
            scores[k] = gain / max(c, eps())
        end

        # Fallback: if all transform-based scores are -Inf (e.g., FIM still singular
        # for DeltaMethod), score by tr(M) which works on rank-deficient matrices.
        # This selects the most informative candidates until M_running reaches full rank.
        if all(==(-Inf), scores)
            @debug "All transform-based scores are -Inf; falling back to tr(FIM) scoring"
            for k in 1:K
                c = prob.cost(prev, candidates[k])
                c > remaining_budget && continue
                total = 0.0
                for ji in 1:np
                    @inbounds for col in 1:p, row in 1:p
                        M_trial[row, col] = M_running[ji][row, col] + M_cache[ji][k][row, col]
                    end
                    total += tr(M_trial)
                end
                scores[k] = (total / np) / max(c, eps())
            end
        end

        best_idx = argmax(scores)
        best_score = scores[best_idx]

        best_score == -Inf && break

        ξ = candidates[best_idx]
        push!(selected, ξ)
        remaining_budget -= prob.cost(prev, ξ)
        prev = ξ

        # Update running FIM from cache (no recomputation)
        for ji in 1:np
            M_running[ji] .+= M_cache[ji][best_idx]
        end

        @debug "Greedy step $step: ξ=$(ξ), score=$(round(best_score; digits=4)), budget_left=$(round(remaining_budget; digits=2))"
    end

    _compress(selected)
end

"""
Batch selection via exchange algorithm.
"""
function _select_batch(
    prob, candidates, particles, n;
    criterion, posterior_samples, exchange_steps,
)
    @info "Running exchange algorithm for batch design..."
    weights = exchange(prob, candidates, particles;
        criterion=criterion,
        posterior_samples=posterior_samples,
        max_iter=exchange_steps)

    counts = apportion(weights, n)

    result = Tuple{eltype(candidates),Int}[]
    for k in eachindex(candidates)
        if counts[k] > 0
            push!(result, (candidates[k], counts[k]))
        end
    end
    result
end

"""
Compress a list of selected candidates into (ξ, count) pairs.
"""
function _compress(selected::Vector{<:NamedTuple})
    isempty(selected) && return Tuple{NamedTuple,Int}[]

    result = Tuple{eltype(selected),Int}[]
    current = selected[1]
    count = 1

    for i in 2:length(selected)
        if selected[i] == current
            count += 1
        else
            push!(result, (current, count))
            current = selected[i]
            count = 1
        end
    end
    push!(result, (current, count))
    result
end

"""
    uniform_allocation(candidates, n)

Allocate n measurements uniformly spaced across the candidate list.
Selects n evenly-spaced indices from the candidate vector.
Returns Vector{Tuple{NamedTuple, Int}}.
"""
function uniform_allocation(candidates::AbstractVector{<:NamedTuple}, n::Int)
    K = length(candidates)
    if n >= K
        # More measurements than candidates: distribute evenly
        counts = apportion(fill(1.0 / K, K), n)
        result = Tuple{eltype(candidates),Int}[]
        for k in eachindex(candidates)
            if counts[k] > 0
                push!(result, (candidates[k], counts[k]))
            end
        end
        return result
    end

    # Select n evenly-spaced indices
    indices = round.(Int, range(1, K, length=n))

    # Count duplicates (can happen when n is close to K)
    result = Tuple{eltype(candidates),Int}[]
    i = 1
    while i <= length(indices)
        idx = indices[i]
        c = 1
        while i + c <= length(indices) && indices[i+c] == idx
            c += 1
        end
        push!(result, (candidates[idx], c))
        i += c
    end
    result
end
