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
