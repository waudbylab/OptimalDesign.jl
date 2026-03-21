# --- Transformation ---

abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F
end

"""
    select(names::Symbol...)

Convenience constructor for DeltaMethod that selects named parameters.
Returns a function that extracts the named components from a ComponentArray.
"""
function select(names::Symbol...)
    DeltaMethod(θ -> ComponentArray(NamedTuple{names}(ntuple(i -> getproperty(θ, names[i]), length(names)))))
end

# --- Design Criteria ---

abstract type DesignCriterion end

struct DCriterion <: DesignCriterion end
struct ACriterion <: DesignCriterion end
struct ECriterion <: DesignCriterion end

(::DCriterion)(M::AbstractMatrix) = logdet(Symmetric(M))
(::ACriterion)(M::AbstractMatrix) = -tr(inv(Symmetric(M)))
(::ECriterion)(M::AbstractMatrix) = eigmin(Symmetric(M))

# --- DesignProblem ---

abstract type AbstractDesignProblem end

struct DesignProblem{F,J,S,T,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    cost::C              # ξ -> Real (per-measurement cost)
    constraint::K
end

struct SwitchingDesignProblem{F,J,S,T,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    cost::C              # ξ -> Real (per-measurement cost)
    switching_param::Symbol
    switching_cost::Float64
    constraint::K
end

"""
    DesignProblem(predict; kwargs...)

Construct a design problem. Returns `DesignProblem` or `SwitchingDesignProblem`
depending on whether `switching_cost` is provided.

- `jacobian`: (θ, ξ) -> J matrix, or `nothing` for ForwardDiff (default: `nothing`)
- `sigma`: (θ, ξ) -> noise (default: `Returns(1.0)`)
- `parameters`: NamedTuple of prior distributions (required)
- `transformation`: defaults to `Identity()`
- `cost`: ξ -> Real, per-measurement cost (default: `Returns(1.0)`)
- `switching_cost`: `nothing` or `(:param, value)` — fixed cost when switching `param` (default: `nothing`)
- `constraint`: (ξ, θ) -> Bool (default: `(ξ, θ) -> true`)
"""
function DesignProblem(
    predict;
    jacobian = nothing,
    sigma = Returns(1.0),
    parameters::NamedTuple,
    transformation::Transformation = Identity(),
    cost = Returns(1.0),
    switching_cost = nothing,
    constraint = (ξ, θ) -> true,
)
    if switching_cost === nothing
        DesignProblem(predict, jacobian, sigma, parameters, transformation, cost, constraint)
    else
        param, sc = switching_cost
        SwitchingDesignProblem(predict, jacobian, sigma, parameters, transformation,
            cost, param, Float64(sc), constraint)
    end
end

"""
    total_cost(prob, prev, ξ)

Total cost of measuring at `ξ` after `prev`, including any switching penalty.
"""
total_cost(prob::DesignProblem, prev, ξ) = prob.cost(ξ)

function total_cost(prob::SwitchingDesignProblem, prev, ξ)
    c = prob.cost(ξ)
    if prev !== nothing && getfield(prev, prob.switching_param) != getfield(ξ, prob.switching_param)
        c += prob.switching_cost
    end
    c
end

# --- Parameter utilities ---

"""
    draw(parameters::NamedTuple)

Draw a single sample from the prior, returning a ComponentArray.
"""
function draw(parameters::NamedTuple)
    vals = map(rand, parameters)
    ComponentArray(vals)
end

"""
    draw(parameters::NamedTuple, n::Int)

Draw n samples from the prior, returning a Vector of ComponentArrays.
"""
function draw(parameters::NamedTuple, n::Int)
    [draw(parameters) for _ in 1:n]
end
