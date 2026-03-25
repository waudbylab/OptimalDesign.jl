# --- Criterion callables ---

(::DCriterion)(M::AbstractMatrix) = logdet(Symmetric(M))
(::ACriterion)(M::AbstractMatrix) = -tr(inv(Symmetric(M)))
(::ECriterion)(M::AbstractMatrix) = eigmin(Symmetric(M))

# --- DeltaMethod convenience constructor ---

"""
    select(names::Symbol...)

Convenience constructor for DeltaMethod that selects named parameters.
Returns a function that extracts the named components from a ComponentArray.
"""
function select(names::Symbol...)
    DeltaMethod(θ -> ComponentArray(NamedTuple{names}(ntuple(i -> getproperty(θ, names[i]), length(names)))), names)
end

# --- DesignProblem factory constructor ---

"""
    DesignProblem(predict; kwargs...)

Construct a design problem. Returns `DesignProblem` or `SwitchingDesignProblem`
depending on whether `switching_cost` is provided.

- `jacobian`: (θ, x) -> J matrix, or `nothing` for ForwardDiff (default: `nothing`)
- `sigma`: (θ, x) -> noise (default: `Returns(1.0)`). For constant noise, use `Returns(σ)`
- `parameters`: NamedTuple of prior distributions (required)
- `transformation`: defaults to `Identity()`
- `criterion`: design criterion (default: `DCriterion()`)
- `cost`: x -> Real, per-measurement cost (default: `Returns(1.0)`)
- `switching_cost`: `nothing` or `(:param, value)` — fixed cost when switching `param` (default: `nothing`)
- `constraint`: (x, θ) -> Bool (default: `(x, θ) -> true`)
"""
function DesignProblem(
    predict;
    jacobian=nothing,
    sigma=Returns(1.0),
    parameters::NamedTuple,
    transformation::Transformation=Identity(),
    criterion::DesignCriterion=DCriterion(),
    cost=Returns(1.0),
    switching_cost=nothing,
    constraint=(x, θ) -> true,
)
    if switching_cost === nothing
        DesignProblem(predict, jacobian, sigma, parameters, transformation, criterion, cost, constraint)
    else
        param, sc = switching_cost
        @warn "Problems with switching costs are experimental and may not be fully supported."
        SwitchingDesignProblem(predict, jacobian, sigma, parameters, transformation,
            criterion, cost, param, Float64(sc), constraint)
    end
end

"""
    selected_parameters(prob) → Union{Nothing, Tuple{Symbol...}}

Return the parameter names selected for estimation (via `select(...)`), or `nothing` if full D-optimality.
"""
selected_parameters(prob::AbstractDesignProblem) = _selected_parameters(prob.transformation)
_selected_parameters(::Identity) = nothing
_selected_parameters(dm::DeltaMethod) = dm.selected

# --- Cost helpers ---

"""
    total_cost(prob, prev, x)

Total cost of measuring at `x` after `prev`, including any switching penalty.
"""
total_cost(prob::DesignProblem, prev, x) = prob.cost(x)

function total_cost(prob::SwitchingDesignProblem, prev, x)
    c = prob.cost(x)
    if prev !== nothing && getfield(prev, prob.switching_param) != getfield(x, prob.switching_param)
        c += prob.switching_cost
    end
    c
end

