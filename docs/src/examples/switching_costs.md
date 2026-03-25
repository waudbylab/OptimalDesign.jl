# Switching Costs

!!! warning "Experimental"
    The switching-cost API is experimental and may change in future releases.

In many experiments, changing measurement configuration is expensive — switching an instrument channel, moving a sample, or recalibrating. OptimalDesign.jl models this with a **switching cost**: a fixed penalty incurred whenever a discrete design variable changes value between consecutive measurements.

## The model

Two exponential decays, selectively measured via a discrete control variable ``i \in \{1, 2\}``:

```math
y = A_i \exp(-k_i \, t) + \varepsilon
```

Each measurement observes **one** decay (chosen by ``i``). Switching between decays costs 20 budget units.

```@example switching
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    if x.i == 1
        θ.A₁ * exp(-θ.k₁ * x.t)
    else
        θ.A₂ * exp(-θ.k₂ * x.t)
    end
end

θ_true = ComponentArray(A₁ = 0.7, k₁ = 10.0, A₂ = 1.5, k₂ = 40.0)
σ = 0.05
acquire(x) = model(θ_true, x) + σ * randn()
nothing # hide
```

## Defining the design problem

The `switching_cost = (:i, 20.0)` tells the algorithm that changing the value of `x.i` between consecutive measurements costs 20, on top of the per-measurement `cost`. The candidates span both channels and a range of times:

```@example switching
prob = DesignProblem(
    model,
    parameters = (
        A₁ = Normal(1, 0.1),    k₁ = LogUniform(1, 50),
        A₂ = Normal(1, 0.1),    k₂ = LogUniform(1, 50)),
    transformation = select(:k₁, :k₂),
    sigma = Returns(σ),
    cost = x -> x.t + 1,
    switching_cost = (:i, 20.0),
)

candidates = candidate_grid(i = [1, 2], t = range(0.001, 0.5, length = 200))
nothing # hide
```

## Running the adaptive experiment

With `n_per_step > 1`, the algorithm uses a **receding-horizon** strategy:

1. **Plan**: compute an optimal batch design for the *entire remaining budget* using the exchange algorithm, accounting for accumulated information from prior observations
2. **Sequence**: reorder the design to minimise switching costs (via TSP over the support points)
3. **Execute**: take the first `n_per_step` measurements, apportioned proportionally within each group to preserve time-point diversity
4. **Update**: update the posterior with the new observations, then repeat from step 1

This amortises the switching cost — the algorithm allocates a block of measurements on one channel before switching, rather than reconsidering the channel at every single step. We also use a larger number of particles for the prior/posterior reflecting the increased number of parameters:

```@example switching
prior = Particles(prob, 10000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 200.0,
    n_per_step = 10,
    headless = true, # hide
)

record_dashboard(result, prob; filename="switching_dashboard.gif") # hide
nothing # hide
```

A live dashboard is displayed of the acquisition process and evolution of parameter estimates:

![Acquisition dashboard](switching_dashboard.gif)


## Convergence

The convergence plot shows both rates being learned simultaneously, with uncertainty decreasing as data accumulates:

```@example switching
plot_convergence(result; truth = θ_true, params = [:k₁, :k₂])
```

## Corner plot

```@example switching
plot_corner(result; truth = θ_true)
```

## Posterior evolution

```@example switching
record_corner_animation(result.log, "switching_posterior.gif";
    params = [:k₁, :k₂],
    truth = θ_true, framerate = 3)
nothing # hide
```

![Posterior evolution](switching_posterior.gif)
