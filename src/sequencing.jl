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
