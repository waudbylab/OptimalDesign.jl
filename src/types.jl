# --- Transformation ---

abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F
    selected::Union{Nothing,Tuple{Vararg{Symbol}}}  # parameter names, if constructed via select()
end
DeltaMethod(f) = DeltaMethod(f, nothing)

# --- Design Criteria ---

abstract type DesignCriterion end

"""
    DCriterion()

D-optimal design criterion: maximises `log(det(M))`, the log-determinant of the
(transformed) Fisher information matrix. Equivalent to minimising the volume of
the confidence ellipsoid.
"""
struct DCriterion <: DesignCriterion end

"""
    ACriterion()

A-optimal design criterion: maximises `-tr(M⁻¹)`, i.e. minimises the average
variance of the parameter estimates.
"""
struct ACriterion <: DesignCriterion end

"""
    ECriterion()

E-optimal design criterion: maximises the minimum eigenvalue of `M`, i.e.
minimises the worst-case variance direction.
"""
struct ECriterion <: DesignCriterion end

# --- DesignProblem ---

abstract type AbstractDesignProblem end

struct DesignProblem{F,J,S,T,CR<:DesignCriterion,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    criterion::CR
    cost::C              # x -> Real (per-measurement cost)
    constraint::K
end

"""
    SwitchingDesignProblem

A design problem with switching costs between levels of a discrete control variable.
Created automatically by `DesignProblem(...)` when `switching_cost` is specified.

!!! warning "Experimental"
    The switching-cost API may change in future releases.

See also: [`DesignProblem`](@ref)
"""
struct SwitchingDesignProblem{F,J,S,T,CR<:DesignCriterion,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    criterion::CR
    cost::C              # x -> Real (per-measurement cost)
    switching_param::Symbol
    switching_cost::Float64
    constraint::K
end

# --- ExperimentalDesign ---

"""
    ExperimentalDesign{T}

An experimental design: a list of `(design_point, count)` pairs specifying how many
measurements to make at each design point. Iterable, indexable, and displayable.

Construct via [`design`](@ref) or directly from a vector of `(NamedTuple, Int)` pairs.
"""
struct ExperimentalDesign{T<:NamedTuple}
    allocation::Vector{Tuple{T, Int}}
end

# --- Abstract posterior ---

"""
    AbstractPosterior

Supertype for posterior representations. Currently the only concrete subtype is
[`Particles`](@ref).
"""
abstract type AbstractPosterior end

# --- Particles ---

struct Particles{T} <: AbstractPosterior
    particles::Vector{T}
    log_weights::Vector{Float64}
end

# --- ExperimentLog ---

"""
    ExperimentLog

Stores the history of an adaptive experiment: design points, observations,
costs, and diagnostics for each step. Created internally by [`run_adaptive`](@ref).
"""
struct ExperimentLog
    history::Vector{NamedTuple}
    prior_snapshot::Union{Nothing,NamedTuple}   # (particles, log_weights) before any data
end

# --- GradientCache ---

struct GradientCache
    g_buf::Vector{Float64}
    cfg::ForwardDiff.GradientConfig
end

# --- LiveDashboard ---

mutable struct LiveDashboard
    fig::Any                    # GLMakie.Figure
    screen::Any                 # GLMakie.Screen
    # Design panel
    ax_design::Any              # GLMakie.Axis for design points
    obs_design_x::Any           # Observable: x coords
    obs_design_y::Any           # Observable: y coords
    obs_design_color::Any       # Observable: step number (for coloring)
    design_fields::Tuple{Vararg{Symbol}}  # which NamedTuple fields to plot
    # Posterior panel
    ax_posterior::Vector{Any}   # GLMakie.Axis per interest parameter
    interest_params::Vector{Symbol}
    # Log marginal likelihood panel
    ax_logml::Any               # GLMakie.Axis
    obs_logml::Any              # Observable: log marginal per step
    # Budget panel
    obs_budget_spent::Any       # Observable: budget spent
    budget_total::Float64
    # Controls
    control_state::Ref{Symbol}  # :running, :paused, :stopped
end

# --- Experiment results ---

"""
    AbstractExperimentResult

Supertype for experiment results. Subtypes: [`BatchResult`](@ref), [`AdaptiveResult`](@ref).
All results carry `.prior`, `.posterior`, and `.observations`.
"""
abstract type AbstractExperimentResult end

"""
    BatchResult

Result of [`run_batch`](@ref). Contains `prior`, `posterior`, `design`, and `observations`.
"""
struct BatchResult{P,D<:ExperimentalDesign} <: AbstractExperimentResult
    prior::P
    posterior::P
    design::D
    observations::Vector{NamedTuple}
end

"""
    AdaptiveResult

Result of [`run_adaptive`](@ref). Contains `prior`, `posterior`, `log`, and `observations`.
The `log` field is an [`ExperimentLog`](@ref) with per-step diagnostics.
"""
struct AdaptiveResult{P} <: AbstractExperimentResult
    prior::P
    posterior::P
    log::ExperimentLog
    observations::Vector{NamedTuple}
end

# --- Optimality verification ---

"""
    OptimalityResult

Result of [`verify_optimality`](@ref). Fields: `is_optimal`, `max_derivative`,
`dimension`, `gateaux` (derivative at each candidate), and `candidates`.
"""
struct OptimalityResult{T<:NamedTuple}
    is_optimal::Bool
    max_derivative::Float64
    dimension::Float64
    gateaux::Vector{Float64}
    candidates::Vector{T}
end
