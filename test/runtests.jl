using OptimalDesign
using Test
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

Random.seed!(42)

# Resolve name conflicts with Distributions.jl
const od_sample = OptimalDesign.sample
const od_loglikelihood = OptimalDesign.loglikelihood

@testset "OptimalDesign.jl" begin

    @testset "draw" begin
        params = (A=Normal(1, 0.1), R₂=LogNormal(2, 0.5))
        θ = draw(params)
        @test θ isa ComponentArray
        @test haskey(θ, :A)
        @test haskey(θ, :R₂)

        θs = draw(params, 100)
        @test length(θs) == 100
        @test all(θ -> θ isa ComponentArray, θs)
    end

    @testset "DesignProblem construction" begin
        # Minimal construction
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
        )
        @test prob.predict isa Function
        @test prob.jacobian === nothing
        @test prob.transformation isa OptimalDesign.Identity
        @test prob.cost((t=0.1,), (t=0.2,)) == 1.0
        @test prob.constraint((t=0.1,), nothing) == true

        # Full construction
        prob2 = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            jacobian=(θ, x) -> [exp(-θ.R₂ * x.t) -θ.A * x.t * exp(-θ.R₂ * x.t)],
            sigma=(θ, x) -> 0.05,
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            transformation=select(:R₂),
            cost=x -> x.t + 0.1,
        )
        @test prob2.jacobian !== nothing
    end

    @testset "Criteria" begin
        M = [2.0 0.5; 0.5 3.0]

        d = DCriterion()(M)
        @test d ≈ log(det(M))

        a = ACriterion()(M)
        @test a ≈ -tr(inv(M))

        e = ECriterion()(Symmetric(M))
        @test e ≈ eigmin(Symmetric(M))
    end

    @testset "weighted_fim" begin
        J = [1.0 2.0; 3.0 4.0]

        # Scalar sigma
        F1 = OptimalDesign.weighted_fim(J, 0.5)
        @test F1 ≈ J' * J / 0.25

        # Vector sigma
        F2 = OptimalDesign.weighted_fim(J, [0.5, 1.0])
        W = Diagonal([1 / 0.25, 1 / 1.0])
        @test F2 ≈ J' * W * J

        # Matrix sigma
        Σ = [1.0 0.2; 0.2 0.5]
        F3 = OptimalDesign.weighted_fim(J, Σ)
        @test F3 ≈ J' * inv(Σ) * J

        # Vector J (scalar observation)
        Jv = [1.0, 2.0]
        F4 = OptimalDesign.weighted_fim(Jv, 1.0)
        @test F4 ≈ reshape(Jv, 1, 2)' * reshape(Jv, 1, 2)
    end

    @testset "Example 1: Exponential decay — FIM" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)
        x = (t=0.1,)

        M = information(prob, θ, x)
        @test size(M) == (2, 2)
        @test issymmetric(M) || M ≈ M'
        # Rank 1 from single scalar observation — one eigenvalue is zero (up to float)
        @test all(eigvals(Symmetric(M)) .>= -1e-10)

        # Verify FIM by hand:
        # y = A * exp(-R₂*t)
        # ∂y/∂A = exp(-R₂*t) = exp(-1) ≈ 0.3679
        # ∂y/∂R₂ = -A*t*exp(-R₂*t) = -0.1*exp(-1) ≈ -0.03679
        e = exp(-10.0 * 0.1)
        J_expected = [e -1.0 * 0.1 * e]
        F_expected = J_expected' * J_expected / 0.05^2
        @test M ≈ F_expected atol = 1e-10
    end

    @testset "Example 2: Inversion recovery — analytic vs ForwardDiff Jacobian" begin
        predict = (θ, x) -> θ.A - θ.B * exp(-θ.R₁ * x.τ)
        jac = (θ, x) -> begin
            e = exp(-θ.R₁ * x.τ)
            [1.0 -e θ.B * x.τ * e]
        end

        prob_ad = DesignProblem(
            predict,
            parameters=(A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogNormal(0, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        prob_analytic = DesignProblem(
            predict,
            jacobian=jac,
            parameters=(A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogNormal(0, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        θ = ComponentArray(A=1.0, B=2.0, R₁=1.0)
        x = (τ=0.5,)

        M_ad = information(prob_ad, θ, x)
        M_analytic = information(prob_analytic, θ, x)

        @test M_ad ≈ M_analytic atol = 1e-10
        @test size(M_ad) == (3, 3)
    end

    @testset "DeltaMethod transformation" begin
        # Use a full-rank FIM by summing over multiple design points
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            transformation=select(:R₂),
            sigma=(θ, x) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)

        # Sum FIM over two well-separated time points for full rank
        M = information(prob, θ, (t=0.05,)) + information(prob, θ, (t=0.2,))
        @test isposdef(Symmetric(M))

        Mt = OptimalDesign.transform(prob, M, θ)

        # Transformed matrix should be 1×1 for single parameter of interest
        @test size(Mt) == (1, 1)
        @test Mt[1, 1] > 0
    end

    @testset "Particles" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        post = Particles(prob, 500)
        @test length(post.particles) == 500
        @test length(post.log_weights) == 500
        @test all(isfinite, post.log_weights)

        # ESS should be n for uniform weights
        ess = effective_sample_size(post)
        @test ess ≈ 500.0 atol = 1.0

        # Posterior mean should be close to prior mean
        μ = mean(post)
        @test μ isa ComponentArray
        @test abs(μ.A - 1.0) < 0.1  # prior mean of A is 1.0

        # Sample (qualified to avoid conflict with Distributions.sample)
        s = od_sample(post, 10)
        @test length(s) == 10
    end

    @testset "loglikelihood" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        θ = ComponentArray(A=1.0, R₂=10.0)
        x = (t=0.1,)
        ŷ = prob.predict(θ, x)

        # Perfect observation: highest likelihood
        ll_perfect = od_loglikelihood(prob, θ, x, ŷ)
        ll_noisy = od_loglikelihood(prob, θ, x, ŷ + 0.1)
        @test ll_perfect > ll_noisy

        # Structured observation
        ll_struct = od_loglikelihood(prob, θ, x, (value=ŷ, σ=0.05))
        @test ll_struct ≈ ll_perfect
    end

    @testset "update! posterior" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        post = Particles(prob, 1000)
        θ_true = ComponentArray(A=1.0, R₂=10.0)
        x = (t=0.1,)
        y = prob.predict(θ_true, x) + 0.05 * randn()

        update!(post, prob, x, y)

        μ = mean(post)
        @test μ isa ComponentArray
    end

    @testset "expected_utility" begin
        # Use vector observation so single-point FIM is full rank (2 obs, 2 params)
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
        )

        particles = draw(prob.parameters, 100)
        x = (t=0.1,)

        u = OptimalDesign.expected_utility(prob, particles, x; posterior_samples=50)
        @test isfinite(u)

        # Score multiple candidates
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        scores = score_candidates(prob, particles, candidates; posterior_samples=50)
        @test length(scores) == 20
        @test all(isfinite, scores)

        # Scalar observation (rank-deficient FIM) — should not error
        prob_scalar = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns(0.05),
        )
        u_scalar = OptimalDesign.expected_utility(prob_scalar, particles, x; posterior_samples=50)
        @test !isnan(u_scalar)
    end

    @testset "Example 3: Vector observation" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A₁ * exp(-θ.R₂₁ * x.t),
                θ.A₂ * exp(-θ.R₂₂ * x.t)],
            parameters=(A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
        )

        θ = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=25.0)
        x = (t=0.1,)

        M = information(prob, θ, x)
        @test size(M) == (4, 4)
        @test all(eigvals(Symmetric(M)) .>= -1e-10)
    end

    @testset "Example 4: Selective observation — block sparsity" begin
        prob = DesignProblem(
            (θ, x) -> if x.i == 1
                θ.A₁ * exp(-θ.R₂₁ * x.t)
            else
                θ.A₂ * exp(-θ.R₂₂ * x.t)
            end,
            parameters=(A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        θ = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=25.0)

        # Measuring decay 1: only A₁, R₂₁ contribute
        M1 = information(prob, θ, (i=1, t=0.1))
        @test size(M1) == (4, 4)
        # Columns/rows for A₂, R₂₂ (indices 3,4) should be zero
        @test norm(M1[3:4, :]) < 1e-10
        @test norm(M1[:, 3:4]) < 1e-10

        # Measuring decay 2: only A₂, R₂₂ contribute
        M2 = information(prob, θ, (i=2, t=0.1))
        @test norm(M2[1:2, :]) < 1e-10
        @test norm(M2[:, 1:2]) < 1e-10

        # Sum gives full-rank (if time chosen well)
        M_sum = M1 + M2
        @test rank(M_sum) == 2  # rank 2: each measurement contributes rank 1
    end

    # =============================================
    # Phase 2: Solver tests
    # =============================================

    @testset "apportion" begin
        # Basic rounding
        counts = OptimalDesign.apportion([0.5, 0.3, 0.2], 10)
        @test sum(counts) == 10
        @test counts == [5, 3, 2]

        # Handles remainders correctly
        counts2 = OptimalDesign.apportion([1 / 3, 1 / 3, 1 / 3], 10)
        @test sum(counts2) == 10

        # Edge case: all weight on one candidate
        counts3 = OptimalDesign.apportion([1.0, 0.0, 0.0], 5)
        @test counts3 == [5, 0, 0]
    end

    @testset "design — greedy" begin
        # Vector observation for full-rank FIM
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
            cost=Returns(1.0),
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        prior = Particles(prob, 100)

        # Select 3 points greedily
        ξ = design(prob, candidates, prior;
            n=3, posterior_samples=50)

        @test !isempty(ξ)
        total_count = sum(last.(ξ))
        @test total_count == 3
        # All selected candidates should be from the candidate list
        for (x, count) in ξ
            @test x in candidates
            @test count >= 1
        end
    end

    @testset "design — greedy with budget" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
            cost=x -> x.t + 0.5,
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        prior = Particles(prob, 100)

        # Budget of 2.0 should limit selections
        ξ = design(prob, candidates, prior;
            n=100, posterior_samples=50,
            budget=2.0)

        total_cost = sum(prob.cost(x) * count for (x, count) in ξ)
        @test total_cost <= 2.0
    end

    @testset "exchange algorithm" begin
        # Vector observation for well-conditioned FIM
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        particles = draw(prob.parameters, 100)

        weights = OptimalDesign.exchange(prob, candidates, particles;
            posterior_samples=50, max_iter=50)

        @test length(weights) == 20
        @test sum(weights) ≈ 1.0 atol = 1e-6
        @test all(weights .>= 0)
        # Should concentrate on a few support points
        @test count(weights .> 0.01) < 15
    end

    @testset "gateaux_derivative" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> [0.05, 0.05],
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        particles = draw(prob.parameters, 100)

        weights = OptimalDesign.exchange(prob, candidates, particles;
            posterior_samples=50, max_iter=50)

        gd = gateaux_derivative(prob, candidates, particles, weights;
            posterior_samples=50)

        @test length(gd) == 20
        @test all(isfinite, gd)
        # GEQ: at optimum, d(x) ≤ p for all candidates
        @test maximum(gd) <= 2.0 + 0.1  # p=2 with some tolerance
    end

    @testset "gateaux_derivative — DeltaMethod" begin
        # D_s-optimality: interest in R₂ only
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
            transformation=select(:R₂),
            sigma=(θ, x) -> 0.05,
        )
        candidates = [(t=t,) for t in range(0.001, 0.5, length=30)]
        particles = draw(prob.parameters, 100)

        weights = OptimalDesign.exchange(prob, candidates, particles;
            posterior_samples=50, max_iter=100)

        gd = gateaux_derivative(prob, candidates, particles, weights;
            posterior_samples=50)

        @test length(gd) == 30
        @test all(isfinite, gd)
        # GEQ: at optimum, d(x) ≤ q=1 for D_s with 1 parameter of interest
        @test maximum(gd) <= 1.0 + 0.2  # q=1 with tolerance
    end

    @testset "uniform_allocation" begin
        candidates = [(t=t,) for t in range(0.01, 0.5, length=5)]
        alloc = uniform_allocation(candidates, 10)
        total = sum(last.(alloc))
        @test total == 10
    end

    # =============================================
    # Phase 3: Diagnostics and adaptive loop tests
    # =============================================

    @testset "observation_diagnostics" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        post = Particles(prob, 500)
        θ_true = ComponentArray(A=1.0, R₂=10.0)
        x = (t=0.1,)
        y = prob.predict(θ_true, x)

        diag = observation_diagnostics(post, prob, x, y)
        @test haskey(diag, :mean_residual)
        @test haskey(diag, :log_marginal)
        @test isfinite(diag.log_marginal)
    end

    @testset "ExperimentLog" begin
        log = ExperimentLog()
        @test length(log) == 0

        push!(log, (x=(t=0.1,), y=0.5, cost=1.0,
            diagnostics=(mean_residual=0.01, log_marginal=-3.0)))
        push!(log, (x=(t=0.2,), y=0.3, cost=1.0,
            diagnostics=(mean_residual=-0.02, log_marginal=-2.5)))

        @test length(log) == 2
        @test design_points(log) == [(t=0.1,), (t=0.2,)]
        @test OptimalDesign.observations(log) == [0.5, 0.3]
        @test cumulative_cost(log) == [1.0, 2.0]
        @test log_evidence_series(log) == [-3.0, -2.5]
    end

    @testset "run_adaptive — headless" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns([0.05, 0.05]),
            cost=Returns(1.0),
        )

        θ_true = ComponentArray(A=1.0, R₂=10.0)
        acquire(x) = [θ_true.A * exp(-θ_true.R₂ * x.t) + 0.05 * randn(),
                      θ_true.A * exp(-θ_true.R₂ * x.t * 2) + 0.05 * randn()]

        candidates = candidate_grid(t=range(0.01, 0.5, length=20))
        prior = Particles(prob, 200)

        result = run_adaptive(
            prob, candidates, prior, acquire;
            budget=5.0,
            posterior_samples=50,
            n_per_step=1,
            headless=true,
        )

        @test result isa AdaptiveResult
        @test result.posterior isa Particles
        @test result.prior isa Particles
        @test result.log isa ExperimentLog
        @test length(result.log) >= 1
        @test length(result.log) <= 5  # budget=5, cost=1 per step
        @test length(result.observations) == length(result.log)

        # Prior should not have been mutated
        ess_prior = effective_sample_size(prior)
        @test ess_prior ≈ 200.0 atol=1.0

        # Check posterior has been updated
        μ = mean(result.posterior)
        @test μ isa ComponentArray
    end

    @testset "run_batch — returns BatchResult, non-mutating" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns(0.05),
        )

        θ_true = ComponentArray(A=1.0, R₂=10.0)
        acquire(x) = prob.predict(θ_true, x) + 0.05 * randn()

        candidates = candidate_grid(t=range(0.01, 0.5, length=20))
        prior = Particles(prob, 200)
        ξ = design(prob, candidates, prior; n=5, posterior_samples=50)

        result = run_batch(ξ, prob, prior, acquire)

        @test result isa BatchResult
        @test result.posterior isa Particles
        @test result.prior isa Particles
        @test length(result.observations) == 5
        @test result.design === ξ

        # Prior should not have been mutated
        ess_prior = effective_sample_size(prior)
        @test ess_prior ≈ 200.0 atol=1.0
    end

    @testset "verify_optimality — returns OptimalityResult" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns([0.05, 0.05]),
        )

        candidates = candidate_grid(t=range(0.01, 0.5, length=20))
        prior = Particles(prob, 100)
        ξ = design(prob, candidates, prior; n=10, posterior_samples=50)

        opt = verify_optimality(prob, candidates, prior, ξ; posterior_samples=50)
        @test opt isa OptimalityResult
        @test opt.is_optimal isa Bool
        @test isfinite(opt.max_derivative)
        @test opt.dimension > 0
        @test length(opt.gateaux) == length(candidates)
        @test opt.candidates === candidates
    end

    @testset "candidate_grid" begin
        g1 = candidate_grid(t=range(0, 1, length=5))
        @test length(g1) == 5
        @test g1[1] == (t=0.0,)
        @test g1[end] == (t=1.0,)

        g2 = candidate_grid(i=[1, 2], t=[0.1, 0.2, 0.3])
        @test length(g2) == 6
        @test all(x -> haskey(x, :i) && haskey(x, :t), g2)
    end

    # =============================================
    # Phase 4: ExperimentalDesign utilities
    # =============================================

    @testset "ExperimentalDesign basics" begin
        ξ = ExperimentalDesign([((t=0.1,), 3), ((t=0.2,), 5), ((t=0.3,), 2)])
        @test n_obs(ξ) == 10
        @test length(ξ) == 3
        @test !isempty(ξ)
        @test ξ[1] == ((t=0.1,), 3)
        @test ξ[end] == ((t=0.3,), 2)

        # Iteration
        xc = collect(ξ)
        @test length(xc) == 3
        @test xc[2] == ((t=0.2,), 5)

        # Weights
        candidates = [(t=0.1,), (t=0.2,), (t=0.3,), (t=0.4,)]
        w = weights(ξ, candidates)
        @test length(w) == 4
        @test sum(w) ≈ 1.0
        @test w[4] == 0.0
        @test w[1] ≈ 0.3  # 3/10

        # Empty design
        ξ_empty = ExperimentalDesign(Tuple{@NamedTuple{t::Float64}, Int}[])
        @test n_obs(ξ_empty) == 0
        @test isempty(ξ_empty)
    end

    @testset "_take_first — sequential" begin
        ξ = ExperimentalDesign([((t=0.1,), 5), ((t=0.2,), 5), ((t=0.3,), 5)])

        ξ3 = OptimalDesign._take_first(ξ, 3)
        @test n_obs(ξ3) == 3
        @test ξ3[1] == ((t=0.1,), 3)

        ξ7 = OptimalDesign._take_first(ξ, 7)
        @test n_obs(ξ7) == 7
        @test length(ξ7) == 2  # spans first two points

        ξ15 = OptimalDesign._take_first(ξ, 15)
        @test n_obs(ξ15) == 15  # takes all
    end

    @testset "_take_first — switching_param aware" begin
        # Group 1: two time points, group 2: two time points
        ξ = ExperimentalDesign([
            ((i=1, t=0.1), 6), ((i=1, t=0.2), 4),
            ((i=2, t=0.1), 8), ((i=2, t=0.3), 2),
        ])

        # Take 5 from group 1 (total=10): should apportion across both time points
        ξ5 = OptimalDesign._take_first(ξ, 5; switching_param=:i)
        @test n_obs(ξ5) == 5
        # All should be from group 1
        @test all(x.i == 1 for (x, _) in ξ5)
        # Should have measurements at both time points (proportional to 6:4)
        @test length(ξ5) == 2

        # Take 10 (all of group 1): should get all of group 1
        ξ10 = OptimalDesign._take_first(ξ, 10; switching_param=:i)
        @test n_obs(ξ10) == 10
        @test all(x.i == 1 for (x, _) in ξ10)

        # Take 15: should get all of group 1 + 5 from group 2
        ξ15 = OptimalDesign._take_first(ξ, 15; switching_param=:i)
        @test n_obs(ξ15) == 15
        g1 = sum(c for (x, c) in ξ15 if x.i == 1)
        g2 = sum(c for (x, c) in ξ15 if x.i == 2)
        @test g1 == 10
        @test g2 == 5
        # Group 2 should be spread across both time points
        g2_points = [(x, c) for (x, c) in ξ15 if x.i == 2]
        @test length(g2_points) == 2
    end

    @testset "efficiency" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns([0.05, 0.05]),
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        prior = Particles(prob, 100)

        ξ_opt = design(prob, candidates, prior; n=10, posterior_samples=100)
        ξ_unif = uniform_allocation(candidates, 10)

        # Optimal vs itself ≈ 1.0 (uses all particles to reduce sampling noise)
        eff_self = efficiency(ξ_opt, ξ_opt, prob, candidates, prior; posterior_samples=100)
        @test eff_self ≈ 1.0 atol=0.1

        # Uniform vs optimal should be < 1
        eff = efficiency(ξ_unif, ξ_opt, prob, candidates, prior; posterior_samples=100)
        @test 0.0 < eff < 1.1
    end

    # =============================================
    # Phase 4b: SwitchingDesignProblem (experimental)
    # =============================================

    @testset "SwitchingDesignProblem construction" begin
        prob = DesignProblem(
            (θ, x) -> x.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * x.t) : θ.A₂ * exp(-θ.R₂₂ * x.t),
            parameters=(A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                        A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma=Returns(0.05),
            cost=x -> x.t + 1,
            switching_cost=(:i, 10.0),
        )

        @test prob isa OptimalDesign.SwitchingDesignProblem
        @test prob.switching_param == :i
        @test prob.switching_cost == 10.0

        # total_cost with no switching
        c1 = OptimalDesign.total_cost(prob, (i=1, t=0.1), (i=1, t=0.2))
        @test c1 == prob.cost((i=1, t=0.2))

        # total_cost with switching
        c2 = OptimalDesign.total_cost(prob, (i=1, t=0.1), (i=2, t=0.1))
        @test c2 == prob.cost((i=2, t=0.1)) + 10.0
    end

    @testset "sequencing" begin
        prob = DesignProblem(
            (θ, x) -> x.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * x.t) : θ.A₂ * exp(-θ.R₂₂ * x.t),
            parameters=(A₁=Normal(1, 0.1), R₂₁=LogNormal(2, 0.5),
                        A₂=Normal(1, 0.1), R₂₂=LogNormal(2, 0.5)),
            sigma=Returns(0.05),
            cost=Returns(1.0),
            switching_cost=(:i, 10.0),
        )

        # Interleaved input: should be reordered to group by :i
        result = [((i=1, t=0.1), 3), ((i=2, t=0.1), 2), ((i=1, t=0.2), 4), ((i=2, t=0.2), 1)]
        sequenced = OptimalDesign._sequence_design(prob, result, nothing)

        # Should group all i=1 together and all i=2 together (or vice versa)
        groups = [x.i for (x, _) in sequenced]
        switches = sum(groups[k] != groups[k-1] for k in 2:length(groups))
        @test switches <= 1  # at most one switch
    end

    @testset "design with variable costs" begin
        prob = DesignProblem(
            (θ, x) -> [θ.A * exp(-θ.R₂ * x.t), θ.A * exp(-θ.R₂ * x.t * 2)],
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=Returns([0.05, 0.05]),
            cost=x -> x.t + 0.5,
        )
        candidates = [(t=t,) for t in range(0.01, 0.5, length=20)]
        prior = Particles(prob, 100)

        ξ = design(prob, candidates, prior; budget=3.0, posterior_samples=50)
        total = sum(prob.cost(x) * count for (x, count) in ξ)
        @test total <= 3.0
        @test n_obs(ξ) >= 1
    end

    # =============================================
    # Phase 5: Plotting building blocks
    # =============================================

    @testset "posterior_predictions and credible_band" begin
        prob = DesignProblem(
            (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
            parameters=(A=Normal(1, 0.1), R₂=LogNormal(2, 0.5)),
            sigma=(θ, x) -> 0.05,
        )

        post = Particles(prob, 200)
        grid = [(t=t,) for t in range(0.01, 0.5, length=50)]

        preds = posterior_predictions(prob, post, grid; n_samples=100)
        @test size(preds) == (100, 50)
        @test all(isfinite, preds)

        band = credible_band(preds; level=0.9)
        @test length(band.lower) == 50
        @test length(band.median) == 50
        @test length(band.upper) == 50
        @test all(band.lower .<= band.median)
        @test all(band.median .<= band.upper)
    end

end
