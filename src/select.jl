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
    problem::AbstractDesignProblem,
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
            criterion, posterior_samples, exchange_steps, budget, ξ_prev)
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
            c = total_cost(prob, prev, candidates[k])
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
                c = total_cost(prob, prev, candidates[k])
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
        remaining_budget -= total_cost(prob, prev, ξ)
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
Batch selection via exchange algorithm with cost awareness.

When per-measurement costs vary, the FIM is weighted by `wₖ/cₖ` so the
exchange algorithm prefers candidates that are informative per unit cost.
For `SwitchingDesignProblem`, the output is sequenced to minimise switching.
"""
function _select_batch(
    prob, candidates, particles, n;
    criterion, posterior_samples, exchange_steps,
    budget=Inf, ξ_prev=nothing,
)
    # Per-measurement costs (1-arg cost function)
    costs_vec = [prob.cost(ξ) for ξ in candidates]
    has_uniform_cost = all(c -> c ≈ costs_vec[1], costs_vec)
    costs = has_uniform_cost ? nothing : costs_vec

    @info "Running exchange algorithm for batch design..."
    weights = exchange(prob, candidates, particles;
        criterion=criterion,
        posterior_samples=posterior_samples,
        max_iter=exchange_steps,
        costs=costs)

    # Apportion: budget-aware when costs vary, count-based otherwise
    if costs !== nothing && budget < Inf
        # Deduct switching overhead for SwitchingDesignProblem
        measurement_budget = _deduct_switching_overhead(prob, weights, candidates, budget)
        counts = apportion(weights, measurement_budget, costs_vec)
    else
        counts = apportion(weights, n)
    end

    result = Tuple{eltype(candidates),Int}[]
    for k in eachindex(candidates)
        if counts[k] > 0
            push!(result, (candidates[k], counts[k]))
        end
    end

    # Sequence to minimise switching cost
    _sequence_design(prob, result, ξ_prev)
end

# --- Switching cost helpers ---

"""No switching overhead for plain DesignProblem."""
_deduct_switching_overhead(::DesignProblem, weights, candidates, budget) = budget

"""Estimate switching overhead and deduct from budget."""
function _deduct_switching_overhead(prob::SwitchingDesignProblem, weights, candidates, budget)
    support_idx = findall(weights .> 1e-10)
    length(support_idx) <= 1 && return budget

    # Count distinct groups of the switching parameter
    support_points = candidates[support_idx]
    param = prob.switching_param
    groups = unique(getfield(ξ, param) for ξ in support_points)
    n_switches = length(groups) - 1

    overhead = n_switches * prob.switching_cost
    measurement_budget = max(budget - overhead, 0.0)
    if overhead > 0
        @debug "Switching overhead: $n_switches switches × $(prob.switching_cost) = $overhead, " *
               "measurement budget: $(round(measurement_budget; digits=1))/$budget"
    end
    measurement_budget
end

# --- Design sequencing ---

"""No-op sequencing for plain DesignProblem."""
_sequence_design(::DesignProblem, result, ξ_prev) = result

"""Reorder (ξ, count) pairs to minimise total switching cost."""
function _sequence_design(prob::SwitchingDesignProblem, result, ξ_prev)
    length(result) <= 1 && return result

    param = prob.switching_param
    points = [r[1] for r in result]
    counts = [r[2] for r in result]

    best_order = _min_switching_order(prob, points, ξ_prev)
    [(points[best_order[i]], counts[best_order[i]]) for i in eachindex(best_order)]
end

"""
Find the permutation of support points that minimises total switching cost.
Brute-force for n ≤ 8 (≤ 40320 perms), nearest-neighbour for larger.
"""
function _min_switching_order(prob::SwitchingDesignProblem, points, ξ_prev)
    n = length(points)
    param = prob.switching_param

    if n <= 8
        # Brute-force: try all permutations
        best_cost = Inf
        best_perm = collect(1:n)
        _tsp_brute_force!(best_perm, Ref(best_cost), Int[], trues(n),
                          prob, points, ξ_prev)
        return best_perm
    end

    # Nearest-neighbour for larger support sets
    visited = falses(n)
    order = Int[]
    prev_val = ξ_prev === nothing ? nothing : getfield(ξ_prev, param)

    for _ in 1:n
        best_k = 0
        best_c = Inf
        for k in 1:n
            visited[k] && continue
            sc = (prev_val !== nothing && getfield(points[k], param) != prev_val) ?
                 prob.switching_cost : 0.0
            c = prob.cost(points[k]) + sc
            if c < best_c
                best_c = c
                best_k = k
            end
        end
        push!(order, best_k)
        visited[best_k] = true
        prev_val = getfield(points[best_k], param)
    end
    order
end

"""Recursive brute-force TSP over support points."""
function _tsp_brute_force!(best_perm, best_cost, current, available,
                           prob, points, ξ_prev)
    param = prob.switching_param
    n = length(points)

    if length(current) == n
        # Compute total switching cost for this permutation
        cost = 0.0
        prev_val = ξ_prev === nothing ? nothing : getfield(ξ_prev, param)
        for i in current
            if prev_val !== nothing && getfield(points[i], param) != prev_val
                cost += prob.switching_cost
            end
            prev_val = getfield(points[i], param)
        end
        if cost < best_cost[]
            best_cost[] = cost
            best_perm .= current
        end
        return
    end

    for k in 1:n
        available[k] || continue
        available[k] = false
        push!(current, k)
        _tsp_brute_force!(best_perm, best_cost, current, available,
                          prob, points, ξ_prev)
        pop!(current)
        available[k] = true
    end
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
