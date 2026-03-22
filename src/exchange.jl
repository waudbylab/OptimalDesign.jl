"""
    exchange(prob, candidates, particles; kwargs...)

Two-phase Frank-Wolfe algorithm for batch weight optimisation on a discrete
candidate grid, following the approach of Yang, Biedermann & Tang (2013).

**Phase 1 (Discovery):** Pairwise Frank-Wolfe to find approximate support.
Uses adaptive smoothness estimation for step sizes (à la Kirstine.jl).

**Phase 2 (Refinement):** Fix the support set found in Phase 1 and run
another round of pairwise FW restricted to those points.

Returns a weight vector (length = number of candidates) that sums to 1.
"""
function exchange(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector;
    posterior_samples::Int=0,
    max_iter::Int=200,
    tol::Float64=1e-3,
    costs::Union{Nothing,AbstractVector{<:Real}}=nothing,
)
    K = length(candidates)
    q = _transformed_dimension(prob)

    # Fix a particle subset for the entire run so the objective is deterministic
    n_particles = length(particles)
    if posterior_samples ≤ 0
        bs = n_particles
    else
        bs = min(posterior_samples, n_particles)
    end
    fixed_particles = particles[randperm(n_particles)[1:bs]]

    @info "Exchange: K=$K candidates, q=$q, $bs fixed particles, max_iter=$max_iter"

    # Initialise with evenly-spread support
    weights = _initialise_weights(prob, candidates, fixed_particles, bs, q)
    n_support = count(>(1e-10), weights)
    @info "  Init: $n_support support points"

    # ═══════════════════════════════════════════
    # Phase 1: Pairwise FW — discover support
    # ═══════════════════════════════════════════
    phase1_iters = max_iter ÷ 2
    best_weights = copy(weights)
    best_gap = Inf
    best_obj = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)

    # Adaptive smoothness estimate (like Kirstine's adaptive step)
    L = max(q, 1.0)

    for iter in 1:phase1_iters
        gd = gateaux_derivative(prob, candidates, fixed_particles, weights;
            posterior_samples=bs, costs=costs)

        if all(isinf, gd)
            weights = _initialise_weights(prob, candidates, fixed_particles,
                bs, q)
            @warn "Phase1 iter $iter: FIM singular, reinitialised"
            L = max(q, 1.0)
            continue
        end

        k_max = argmax(gd)
        d_max = gd[k_max]
        fw_gap = d_max - dot(weights, gd)
        n_support = count(>(1e-10), weights)

        if fw_gap < best_gap
            best_gap = fw_gap
            best_weights .= weights
            best_obj = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
        end

        if iter <= 3 || iter % 10 == 0 || fw_gap < tol
            support_idx = findall(weights .> 1e-10)
            d_min_s = isempty(support_idx) ? NaN : minimum(gd[support_idx])
            @debug "P1 iter $iter" fw_gap=round(fw_gap; digits=4) n_support d_min=round(d_min_s; digits=4) L=round(L; digits=2)
        end

        if fw_gap < tol
            @info "  Phase 1 converged at iter $iter (fw_gap=$(round(fw_gap; digits=5)))"
            break
        end

        # Pairwise FW: away vertex = support point with lowest derivative
        support = findall(weights .> 1e-10)
        k_min = support[argmin(gd[support])]
        d_min = gd[k_min]

        if k_min == k_max
            # All support points have the same derivative — use standard FW
            γ = 2.0 / (iter + 2)
            weights .*= (1 - γ)
            weights[k_max] += γ
        else
            # Pairwise step with adaptive smoothness
            γ_max = weights[k_min]
            dir_deriv = d_max - d_min  # directional derivative along pairwise direction

            # Adaptive step: γ = dir_deriv / (2L), capped by γ_max
            γ = min(γ_max, dir_deriv / (2.0 * L))
            γ = max(γ, 1e-6)  # floor to avoid stalling

            # Compute CURRENT objective for comparison
            obj_current = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)

            # Trial step
            w_trial = copy(weights)
            w_trial[k_min] -= γ
            w_trial[k_max] += γ
            w_trial[w_trial .< 1e-10] .= 0.0
            s = sum(w_trial); s > 0 && (w_trial ./= s)

            obj_trial = _weighted_objective(prob, fixed_particles, candidates, w_trial; costs=costs)

            # Backtracking: if objective didn't improve vs CURRENT, increase L and shrink step
            backtracks = 0
            while obj_trial < obj_current - 1e-10 && backtracks < 6
                L *= 2.0
                γ = min(γ_max, dir_deriv / (2.0 * L))
                γ = max(γ, 1e-6)
                w_trial .= weights
                w_trial[k_min] -= γ
                w_trial[k_max] += γ
                w_trial[w_trial .< 1e-10] .= 0.0
                s = sum(w_trial); s > 0 && (w_trial ./= s)
                obj_trial = _weighted_objective(prob, fixed_particles, candidates, w_trial; costs=costs)
                backtracks += 1
            end

            if obj_trial >= obj_current - 1e-10
                # Accept step, gently reduce L for next iteration
                weights .= w_trial
                L *= 0.9
                @debug "  P1 transfer" iter α=round(γ; digits=4) from=k_min to=k_max L=round(L; digits=2) backtracks
            else
                @debug "  P1 step rejected after $backtracks backtracks" iter L=round(L; digits=2)
            end
        end

        # Prune and renormalise
        weights[weights .< 1e-10] .= 0.0
        s = sum(weights)
        s > 0 && (weights ./= s)
    end

    # Restore best if we overshot
    final_obj = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
    if best_obj > final_obj + 1e-6
        @debug "Restoring best Phase 1 weights" best_gap=round(best_gap; digits=4)
        weights .= best_weights
    end

    # Merge nearby support points before Phase 2
    n_before = count(>(1e-10), weights)
    _merge_nearby!(weights, candidates)
    weights[weights .< 1e-6] .= 0.0
    s = sum(weights); s > 0 && (weights ./= s)
    n_after = count(>(1e-10), weights)
    n_before != n_after && @debug "Merge" before=n_before after=n_after

    # ═══════════════════════════════════════════
    # Phase 2: Refine weights on fixed support
    # ═══════════════════════════════════════════
    support_idx = findall(weights .> 1e-10)
    n_sup = length(support_idx)
    @info "  Phase 2: refining weights on $n_sup support points"

    if n_sup >= 2
        phase2_iters = max_iter - (max_iter ÷ 2)
        best_obj2 = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
        best_weights2 = copy(weights)
        best_gap2 = Inf
        L2 = max(q, 1.0)

        for iter in 1:phase2_iters
            gd = gateaux_derivative(prob, candidates, fixed_particles, weights;
                posterior_samples=bs, costs=costs)

            if all(isinf, gd)
                @warn "Phase2 iter $iter: FIM singular"
                break
            end

            d_max_all = maximum(gd)
            fw_gap = d_max_all - dot(weights, gd)

            if fw_gap < best_gap2
                best_gap2 = fw_gap
                best_weights2 .= weights
                best_obj2 = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
            end

            if iter == 1 || iter % 10 == 0 || fw_gap < tol
                @debug "P2 iter $iter" fw_gap=round(fw_gap; digits=4) max_gd=round(d_max_all; digits=4) L=round(L2; digits=2)
            end

            if fw_gap < tol
                @info "  Phase 2 converged at iter $iter (fw_gap=$(round(fw_gap; digits=5)))"
                break
            end

            # Pairwise step restricted to support
            gd_sup = gd[support_idx]
            k_max_local = argmax(gd_sup)
            k_min_local = argmin(gd_sup)

            if k_max_local == k_min_local
                # Check if a non-support point is much better
                k_max_global = argmax(gd)
                if k_max_global in support_idx
                    @debug "P2: support optimal, stopping"
                    break
                else
                    # Add non-support point
                    k_target = k_max_global
                    k_min_sup = support_idx[k_min_local]
                    dir_deriv = gd[k_target] - gd[k_min_sup]
                    γ_max = weights[k_min_sup]
                    γ = min(γ_max, dir_deriv / (2.0 * L2))
                    γ = max(γ, 1e-6)

                    weights[k_min_sup] -= γ
                    weights[k_target] += γ

                    if weights[k_min_sup] < 1e-10
                        support_idx = setdiff(support_idx, [k_min_sup])
                    end
                    if !(k_target in support_idx)
                        push!(support_idx, k_target)
                        sort!(support_idx)
                    end
                end
            else
                k_max_sup = support_idx[k_max_local]
                k_min_sup = support_idx[k_min_local]
                d_max_sup = gd[k_max_sup]
                d_min_sup = gd[k_min_sup]

                # Check if any non-support point is substantially better
                k_max_global = argmax(gd)
                if gd[k_max_global] > d_max_sup + 1e-4 && !(k_max_global in support_idx)
                    k_target = k_max_global
                    if !(k_target in support_idx)
                        push!(support_idx, k_target)
                        sort!(support_idx)
                    end
                else
                    k_target = k_max_sup
                end

                dir_deriv = gd[k_target] - d_min_sup
                γ_max = weights[k_min_sup]
                γ = min(γ_max, dir_deriv / (2.0 * L2))
                γ = max(γ, 1e-6)

                # Trial step with backtracking against CURRENT objective
                obj_current = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
                w_trial = copy(weights)
                w_trial[k_min_sup] -= γ
                w_trial[k_target] += γ
                w_trial[w_trial .< 1e-10] .= 0.0
                s = sum(w_trial); s > 0 && (w_trial ./= s)
                obj_trial = _weighted_objective(prob, fixed_particles, candidates, w_trial; costs=costs)

                backtracks = 0
                while obj_trial < obj_current - 1e-10 && backtracks < 4
                    L2 *= 2.0
                    γ = min(γ_max, dir_deriv / (2.0 * L2))
                    γ = max(γ, 1e-6)
                    w_trial .= weights
                    w_trial[k_min_sup] -= γ
                    w_trial[k_target] += γ
                    w_trial[w_trial .< 1e-10] .= 0.0
                    s = sum(w_trial); s > 0 && (w_trial ./= s)
                    obj_trial = _weighted_objective(prob, fixed_particles, candidates, w_trial; costs=costs)
                    backtracks += 1
                end

                if obj_trial >= obj_current - 1e-10
                    weights .= w_trial
                    L2 *= 0.9
                end

                if weights[k_min_sup] < 1e-10
                    support_idx = setdiff(support_idx, [k_min_sup])
                end
            end

            weights[weights .< 1e-10] .= 0.0
            s = sum(weights)
            s > 0 && (weights ./= s)
        end

        # Restore best Phase 2 weights if we overshot
        final_obj2 = _weighted_objective(prob, fixed_particles, candidates, weights; costs=costs)
        if best_obj2 > final_obj2 + 1e-6
            weights .= best_weights2
        end
    end

    # Final cleanup
    weights[weights .< 1e-6] .= 0.0
    s = sum(weights)
    s > 0 && (weights ./= s)

    n_final = count(>(1e-10), weights)
    final_gap = _compute_fw_gap(prob, candidates, fixed_particles, weights;
        posterior_samples=bs, costs=costs)
    @info "  Exchange complete: $n_final support points, fw_gap=$(round(final_gap; digits=4))"

    weights
end

"""Compute the weighted objective: E_θ[Φ(Σ_k (w_k/c_k) M_k(θ))]."""
function _weighted_objective(prob, particles, candidates, weights;
                             costs=nothing)
    criterion = prob.criterion
    total = 0.0
    count = 0
    for θ in particles
        M_w = _particle_weighted_fim(prob, θ, candidates, weights; costs=costs)
        Mt = transform(prob, M_w, θ)
        val = safe_criterion(criterion, Mt)
        if isfinite(val)
            total += val
            count += 1
        end
    end
    count > 0 ? total / count : -Inf
end

"""Compute FW gap without side effects."""
function _compute_fw_gap(prob, candidates, particles, weights;
    posterior_samples=50, costs=nothing)
    gd = gateaux_derivative(prob, candidates, particles, weights;
        posterior_samples=posterior_samples, costs=costs)
    all(isinf, gd) && return Inf
    maximum(gd) - dot(weights, gd)
end

"""
    _merge_nearby!(weights, candidates; rtol=0.05)

Merge nearby support points based on normalised distance in design space.

For each pair of support points closer than `rtol` (relative to the range
of each continuous dimension), consolidate weight onto the heavier point.
Discrete (non-numeric) fields must match exactly — points differing in any
discrete field are never merged regardless of distance.

Works correctly for:
- Non-uniform grids (logarithmic, etc.)
- Multidimensional design spaces (e.g., `(t=..., dose=...)`)
- Mixed continuous/discrete variables
"""
function _merge_nearby!(weights, candidates; rtol::Float64=0.05)
    support = findall(weights .> 1e-10)
    length(support) < 2 && return weights

    # Classify fields as numeric or discrete, compute ranges for numeric fields
    example = first(candidates)
    field_names = keys(example)
    numeric_fields = Symbol[]
    discrete_fields = Symbol[]
    ranges = Dict{Symbol,Float64}()

    for fn in field_names
        vals = [getfield(c, fn) for c in candidates]
        if eltype(vals) <: Number
            push!(numeric_fields, fn)
            lo, hi = extrema(vals)
            ranges[fn] = hi - lo
        else
            push!(discrete_fields, fn)
        end
    end

    # Greedy agglomerative merge: iterate support, absorb nearby lighter points
    merged = falses(length(support))
    for i in eachindex(support)
        merged[i] && continue
        for j in (i+1):length(support)
            merged[j] && continue
            ci = candidates[support[i]]
            cj = candidates[support[j]]

            # Discrete fields must match exactly
            discrete_match = all(fn -> getfield(ci, fn) == getfield(cj, fn), discrete_fields)
            discrete_match || continue

            # Normalised Euclidean distance over continuous fields
            dist_sq = 0.0
            for fn in numeric_fields
                r = ranges[fn]
                r < 1e-15 && continue  # constant field, skip
                d = (Float64(getfield(ci, fn)) - Float64(getfield(cj, fn))) / r
                dist_sq += d * d
            end
            dist = sqrt(dist_sq)

            if dist < rtol
                # Absorb j into i (keep the heavier one's index)
                if weights[support[j]] > weights[support[i]]
                    # Swap: absorb i into j
                    weights[support[j]] += weights[support[i]]
                    weights[support[i]] = 0.0
                    # Mark i as merged, but j continues absorbing
                    merged[i] = true
                    break  # i is gone, move on
                else
                    weights[support[i]] += weights[support[j]]
                    weights[support[j]] = 0.0
                    merged[j] = true
                end
            end
        end
    end

    weights
end

"""
Initialise exchange weights with a robust set of support points.

Falls back to evenly-spaced points when single-point utilities are
not computable (e.g., rank-deficient FIM under DeltaMethod).
"""
function _initialise_weights(prob, candidates, particles, posterior_samples, q)
    K = length(candidates)
    p = length(keys(prob.parameters))
    n_init = min(K, max(2p + 1, round(Int, q) + 4, round(Int, sqrt(K))))

    scores = score_candidates(prob, particles, candidates;
        posterior_samples=posterior_samples)

    if all(s -> !isfinite(s), scores)
        indices = unique(round.(Int, range(1, K, length=n_init)))
    else
        indices = sortperm(scores, rev=true)[1:n_init]
    end

    weights = zeros(K)
    for k in indices
        weights[k] = 1.0 / length(indices)
    end

    weights
end
