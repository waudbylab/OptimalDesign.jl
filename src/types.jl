# --- Transformation ---

abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F
end

# --- Design Criteria ---

abstract type DesignCriterion end

struct DCriterion <: DesignCriterion end
struct ACriterion <: DesignCriterion end
struct ECriterion <: DesignCriterion end

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
