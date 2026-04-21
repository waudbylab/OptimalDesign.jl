using Pkg
Pkg.activate(@__DIR__)  # Activate the docs environment

using Documenter
using OptimalDesign

DocMeta.setdocmeta!(OptimalDesign, :DocTestSetup,
    :(using OptimalDesign, ComponentArrays, Distributions, Statistics, Random);
    recursive=true)

makedocs(;
    sitename="OptimalDesign.jl",
    format=Documenter.HTML(),
    modules=[OptimalDesign],
    doctest=true,
    warnonly=[:cross_references],
    pages=[
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "Guide" => [
            "Workflows" => "guide/workflow.md",
            "Defining Problems" => "guide/problems.md",
            "Posterior Inference" => "guide/posteriors.md",
            "Plotting" => "guide/plotting.md",
        ],
        "Theory" => "theory.md",
        "Examples" => [
            "Batch Design" => "examples/batch_design.md",
            "Vector Observations" => "examples/vector_observation.md",
            "Adaptive Design" => "examples/adaptive_design.md",
            "Switching Costs" => "examples/switching_costs.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/waudbylab/OptimalDesign.jl",
    devbranch="main",
)
