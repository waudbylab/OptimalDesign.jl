# Live dashboard using GLMakie Observables for reactive updates.
# Panels: design points, posterior marginals, information gain per step,
# budget tracker, pause/resume/stop controls.

"""
Dashboard state holding Observables and figure references.
"""
mutable struct LiveDashboard
    fig::Any                    # GLMakie.Figure
    screen::Any                 # GLMakie.Screen
    obs_design_x::Any           # Observable: x coords of design points
    obs_design_y::Any           # Observable: y coords (observations)
    obs_posterior_vals::Any      # Observable: Dict of param name => values
    obs_posterior_weights::Any   # Observable: weights
    obs_info_gain::Any          # Observable: info gain per step
    obs_budget_spent::Any        # Observable: budget spent
    obs_budget_total::Any        # budget total (constant)
    obs_pred_lower::Any          # Observable: credible band lower
    obs_pred_median::Any         # Observable: credible band median
    obs_pred_upper::Any          # Observable: credible band upper
    control_state::Ref{Symbol}  # :running, :paused, :stopped
end

function _create_dashboard(prob, posterior, prediction_grid, budget)
    try
        fig = GLMakie.Figure(size=(1200, 800),
            figure_padding=20)

        # --- Design points panel ---
        ax_design = GLMakie.Axis(fig[1, 1],
            title="Design Points",
            xlabel="Measurement",
            ylabel="Observation")

        obs_design_x = GLMakie.Observable(Float64[])
        obs_design_y = GLMakie.Observable(Float64[])
        GLMakie.scatter!(ax_design, obs_design_x, obs_design_y,
            color=:red, markersize=8)

        # --- Posterior marginals panel ---
        ax_post = GLMakie.Axis(fig[1, 2],
            title="Posterior (param 1)",
            xlabel="Value",
            ylabel="Count")

        obs_posterior_vals = GLMakie.Observable(Dict{Symbol, Vector{Float64}}())
        obs_posterior_weights = GLMakie.Observable(Float64[])

        # --- Information gain per step ---
        ax_info = GLMakie.Axis(fig[2, 1],
            title="Log Marginal Likelihood",
            xlabel="Step",
            ylabel="Log p(y)")

        obs_info_gain = GLMakie.Observable(Float64[])
        GLMakie.scatter!(ax_info, GLMakie.@lift(collect(1:length($obs_info_gain))),
            obs_info_gain, color=:blue, markersize=5)
        GLMakie.lines!(ax_info, GLMakie.@lift(collect(1:length($obs_info_gain))),
            obs_info_gain, color=:blue)

        # --- Budget tracker ---
        ax_budget = GLMakie.Axis(fig[2, 2],
            title="Budget",
            xlabel="", ylabel="")

        obs_budget_spent = GLMakie.Observable(0.0)

        GLMakie.barplot!(ax_budget,
            [1, 2],
            GLMakie.@lift([$obs_budget_spent, budget - $obs_budget_spent]),
            color=[:orange, :lightgray],
            bar_labels=GLMakie.@lift(["Spent: $(round($obs_budget_spent, digits=1))",
                                      "Remaining: $(round(budget - $obs_budget_spent, digits=1))"]))

        # --- Prediction bands (if grid provided) ---
        obs_pred_lower = GLMakie.Observable(Float64[])
        obs_pred_median = GLMakie.Observable(Float64[])
        obs_pred_upper = GLMakie.Observable(Float64[])

        if prediction_grid !== nothing
            ax_pred = GLMakie.Axis(fig[3, 1:2],
                title="Posterior Predictions",
                xlabel=string(first(keys(first(prediction_grid)))),
                ylabel="Prediction")

            x_vals = [getfield(ξ, first(keys(first(prediction_grid))))
                      for ξ in prediction_grid]

            GLMakie.band!(ax_pred, x_vals, obs_pred_lower, obs_pred_upper,
                color=(:blue, 0.2))
            GLMakie.lines!(ax_pred, x_vals, obs_pred_median,
                color=:blue, linewidth=2)
            GLMakie.scatter!(ax_pred, obs_design_x, obs_design_y,
                color=:red, markersize=8)
        end

        # --- Controls ---
        control_state = Ref(:running)

        btn_layout = fig[4, 1:2] = GLMakie.GridLayout()
        btn_pause = GLMakie.Button(btn_layout[1, 1], label="Pause")
        btn_resume = GLMakie.Button(btn_layout[1, 2], label="Resume")
        btn_stop = GLMakie.Button(btn_layout[1, 3], label="Stop")

        GLMakie.on(btn_pause.clicks) do _
            control_state[] = :paused
        end
        GLMakie.on(btn_resume.clicks) do _
            control_state[] = :running
        end
        GLMakie.on(btn_stop.clicks) do _
            control_state[] = :stopped
        end

        screen = GLMakie.display(fig)

        LiveDashboard(
            fig, screen,
            obs_design_x, obs_design_y,
            obs_posterior_vals, obs_posterior_weights,
            obs_info_gain, obs_budget_spent, budget,
            obs_pred_lower, obs_pred_median, obs_pred_upper,
            control_state,
        )
    catch e
        @warn "Could not create live dashboard: $e"
        nothing
    end
end

function _check_controls(dashboard::LiveDashboard)
    dashboard.control_state[]
end

function _check_controls(::Nothing)
    :running
end

function _update_dashboard!(dashboard::LiveDashboard, prob, posterior, log, spent, budget, prediction_grid)
    # Update design points
    if !isempty(log.history)
        last_entry = log.history[end]
        ξ = last_entry.ξ
        y = last_entry.y
        # Use first field of ξ as x coordinate
        x_val = first(values(ξ))
        y_val = y isa NamedTuple ? y.value : (y isa Real ? y : first(y))

        push!(dashboard.obs_design_x[], x_val)
        push!(dashboard.obs_design_y[], y_val)
        GLMakie.notify(dashboard.obs_design_x)
        GLMakie.notify(dashboard.obs_design_y)

        # Update info gain
        push!(dashboard.obs_info_gain[], last_entry.diagnostics.log_marginal)
        GLMakie.notify(dashboard.obs_info_gain)
    end

    # Update budget
    dashboard.obs_budget_spent[] = spent

    # Update prediction bands
    if prediction_grid !== nothing && !isempty(prediction_grid)
        try
            preds = posterior_predictions(prob, posterior, prediction_grid; n_samples=100)
            band = credible_band(preds; level=0.9)
            dashboard.obs_pred_lower[] = band.lower
            dashboard.obs_pred_median[] = band.median
            dashboard.obs_pred_upper[] = band.upper
        catch
        end
    end
end

function _update_dashboard!(::Nothing, args...)
    nothing
end

function _finalize_dashboard(dashboard::LiveDashboard)
    # Keep the window open — user can close manually
    nothing
end

function _finalize_dashboard(::Nothing)
    nothing
end
