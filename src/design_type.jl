# --- ExperimentalDesign: iteration, display, and utility methods ---

# Iteration protocol
Base.iterate(ξ::ExperimentalDesign, args...) = iterate(ξ.allocation, args...)
Base.length(ξ::ExperimentalDesign) = length(ξ.allocation)
Base.getindex(ξ::ExperimentalDesign, i) = ξ.allocation[i]
Base.lastindex(ξ::ExperimentalDesign) = lastindex(ξ.allocation)
Base.eltype(::Type{ExperimentalDesign{T}}) where {T} = Tuple{T, Int}
Base.keys(ξ::ExperimentalDesign) = keys(ξ.allocation)
Base.isempty(ξ::ExperimentalDesign) = isempty(ξ.allocation)

"""
    n_obs(ξ::ExperimentalDesign)

Total number of measurements in the design.
"""
n_obs(ξ::ExperimentalDesign) = isempty(ξ.allocation) ? 0 : sum(c for (_, c) in ξ.allocation)

"""
    weights(ξ::ExperimentalDesign, candidates)

Convert design allocation to a weight vector over the candidate set.
Returns a vector of length `length(candidates)` summing to 1.
"""
function weights(ξ::ExperimentalDesign, candidates::AbstractVector)
    n = n_obs(ξ)
    w = zeros(length(candidates))
    for (x, count) in ξ.allocation
        idx = findfirst(c -> c == x, candidates)
        idx !== nothing && (w[idx] = count / n)
    end
    w
end

# --- Display ---

function Base.show(io::IO, ξ::ExperimentalDesign)
    n = n_obs(ξ)
    print(io, "ExperimentalDesign($n measurements, $(length(ξ.allocation)) support points)")
end

function Base.show(io::IO, ::MIME"text/plain", ξ::ExperimentalDesign)
    n = n_obs(ξ)
    println(io, "ExperimentalDesign: $n measurements at $(length(ξ.allocation)) support points")
    isempty(ξ.allocation) && return
    rows = [(join(["$k=$(round(v; digits=4))" for (k, v) in pairs(x)], ", "), count)
            for (x, count) in ξ.allocation]
    max_val = maximum(length(r[1]) for r in rows)
    max_cnt = maximum(ndigits(r[2]) for r in rows)
    for (vals, count) in rows
        bar = repeat("█", count)
        println(io, "  ", rpad(vals, max_val), "  ×", lpad(string(count), max_cnt), "  ", bar)
    end
end

# --- Slicing ---

"""
    _take_first(ξ::ExperimentalDesign, n::Int; switching_param=nothing) → ExperimentalDesign

Extract the first `n` measurements from a design.

When `switching_param` is provided, measurements are apportioned proportionally
within each contiguous group block (identified by `switching_param` value),
preserving the time-point diversity of the exchange algorithm's allocation.
Without `switching_param`, simply takes the first `n` in sequence order.
"""
function _take_first(ξ::ExperimentalDesign{T}, n::Int;
                     switching_param::Union{Symbol,Nothing}=nothing) where T
    if switching_param === nothing
        # Simple sequential take
        result = Tuple{T, Int}[]
        remaining = n
        for (x, count) in ξ
            remaining <= 0 && break
            take = min(count, remaining)
            push!(result, (x, take))
            remaining -= take
        end
        return ExperimentalDesign(result)
    end

    # Group-aware take: apportion within each contiguous group block
    # so that within-group time diversity is preserved.
    blocks = Vector{Vector{Tuple{T, Int}}}()
    current_group = nothing
    for (x, count) in ξ
        g = getfield(x, switching_param)
        if g != current_group
            push!(blocks, Tuple{T, Int}[])
            current_group = g
        end
        push!(blocks[end], (x, count))
    end

    result = Tuple{T, Int}[]
    remaining = n
    for block in blocks
        remaining <= 0 && break
        block_total = sum(c for (_, c) in block)
        take = min(block_total, remaining)

        # Apportion proportionally across the block's design points
        block_weights = [c / block_total for (_, c) in block]
        counts = apportion(block_weights, take)

        for (idx, (x, _)) in enumerate(block)
            counts[idx] > 0 && push!(result, (x, counts[idx]))
        end
        remaining -= take
    end
    ExperimentalDesign(result)
end


