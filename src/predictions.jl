"""
    posterior_predictions(prob, posterior, ξ_grid; n_samples=200)

Generate posterior predictions over a grid of design points.

For scalar models, returns a `Matrix{Float64}` of size `(n_samples, length(ξ_grid))`.
For vector models, returns a `Vector{Matrix{Float64}}` — one matrix per output component.
"""
function posterior_predictions(
    prob::AbstractDesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    n_samples::Int=200,
)
    particles = sample(posterior, n_samples)
    y0 = prob.predict(first(particles), first(ξ_grid))

    if y0 isa Real
        # Scalar model → single Matrix
        predictions = Matrix{Float64}(undef, n_samples, length(ξ_grid))
        for (j, ξ) in enumerate(ξ_grid)
            for i in 1:n_samples
                predictions[i, j] = prob.predict(particles[i], ξ)
            end
        end
        return predictions
    else
        # Vector model → Vector{Matrix}, one per output component
        n_out = length(y0)
        preds = [Matrix{Float64}(undef, n_samples, length(ξ_grid)) for _ in 1:n_out]
        for (j, ξ) in enumerate(ξ_grid)
            for i in 1:n_samples
                y = prob.predict(particles[i], ξ)
                for c in 1:n_out
                    preds[c][i, j] = y[c]
                end
            end
        end
        return preds
    end
end

"""
    credible_band(predictions; level=0.9)

Compute credible band from posterior predictions.
Returns `(lower, median, upper)` vectors.
"""
function credible_band(predictions::AbstractMatrix; level::Real=0.9)
    α = (1 - level) / 2
    n_grid = size(predictions, 2)

    lower = Vector{Float64}(undef, n_grid)
    med = Vector{Float64}(undef, n_grid)
    upper = Vector{Float64}(undef, n_grid)

    for j in 1:n_grid
        col = sort(predictions[:, j])
        lower[j] = quantile(col, α)
        med[j] = quantile(col, 0.5)
        upper[j] = quantile(col, 1 - α)
    end

    (lower=lower, median=med, upper=upper)
end
