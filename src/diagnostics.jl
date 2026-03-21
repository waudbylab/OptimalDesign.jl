"""
    observation_diagnostics(posterior, prob, ξ, y)

Score an observation against the current posterior to detect model deviations.

Returns `(mean_residual, log_marginal)`:
- `mean_residual`: posterior-weighted mean residual (y - ŷ)
- `log_marginal`: log marginal likelihood p(y | data so far)

A running series of `log_marginal` values constitutes sequential Bayesian
model checking. Sharp drops indicate observations surprising under the current model.
"""
function observation_diagnostics(posterior::ParticlePosterior, prob::AbstractDesignProblem, ξ, y)
    n = length(posterior.particles)

    # Log marginal likelihood: logsumexp of weighted log-likelihoods
    ll_terms = [
        posterior.log_weights[i] + loglikelihood(prob, posterior.particles[i], ξ, y)
        for i in 1:n
    ]
    log_ml = logsumexp(ll_terms)

    # Posterior-weighted mean residual
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    y_scalar = y isa NamedTuple ? y.value : y

    mean_pred = sum(
        w[i] * prob.predict(posterior.particles[i], ξ)
        for i in 1:n
    )
    mean_residual = y_scalar .- mean_pred

    (mean_residual=mean_residual, log_marginal=log_ml)
end
