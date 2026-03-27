"""
burnout.jl
Counterfactual burnout attribution algorithm — Section 7 of formulation.
Computes policy-attributable burnout risk Δ^B_H(π) with confidence intervals.

Key insight: PAIRED rollouts sharing initial state AND noise sequence.
This cancels out 'patient was having a bad week' from 'system caused burnout'.
Tighter CIs than unpaired — common noise variance cancels out.

Decision rule: block action if upper_ci > ε_burn
regardless of glycemic benefit.
"""

using Statistics

# ─────────────────────────────────────────────────────────────────
# Burnout threshold — clinical burnout definition
# B_t >= B_thresh means patient has stopped engaging meaningfully
# ─────────────────────────────────────────────────────────────────

const B_THRESH = 0.70

function _matching_null_action(action::AbstractAction) :: AbstractAction
    if action isa ScheduledAction
        return ScheduledAction(
            1,
            parameter_adjustment,
            deepcopy(action.segments),
            SegmentDelta[],
            StructureEdit[],
        )
    end
    return CandidateAction(Dict{Symbol, Float64}())
end

# ─────────────────────────────────────────────────────────────────
# Check if burnout threshold was crossed in a rollout trajectory
# Z^π_i = 𝟙[∃k : B^π_(i,k) >= B_thresh]
# ─────────────────────────────────────────────────────────────────

function crossed_burnout_threshold(rollout::RolloutResult) :: Bool
    for psy in rollout.psy_trajectory
        psy.burnout.value >= B_THRESH && return true
    end
    # also check terminal state
    return rollout.terminal_psy.burnout.value >= B_THRESH
end

function crossed_burnout_threshold(
    rollout :: RolloutResult,
    H_days  :: Int
) :: Bool
    _ = H_days
    return crossed_burnout_threshold(rollout)
end

function crossed_burnout_threshold(rollout::LatentRolloutResult) :: Bool
    return any(
        decode_latent_summary(b).burnout >= B_THRESH
        for b in (rollout.short_belief, rollout.med_belief, rollout.long_belief)
    )
end

function crossed_burnout_threshold(
    rollout :: LatentRolloutResult,
    H_days  :: Int
) :: Bool
    beliefs =
        H_days <= 14 ? (rollout.short_belief,) :
        H_days <= 30 ? (rollout.short_belief, rollout.med_belief) :
                       (rollout.short_belief, rollout.med_belief, rollout.long_belief)

    return any(decode_latent_summary(b).burnout >= B_THRESH for b in beliefs)
end

# ─────────────────────────────────────────────────────────────────
# Counterfactual Burnout Attribution
# Full algorithm from Section 7 of the formulation.
#
# For i = 1..N:
#   1. Draw shared initial state x0 ~ b_t
#   2. Draw shared noise sequence ξ_{0:H-1} ~ p(ξ)
#   3. Treated path: run under active policy π
#   4. Baseline path: run under null policy π⁰ (same x0, same ξ)
#   5. Record burnout events Z^π_i, Z^0_i
#
# Compute:
#   P^π = mean(Z^π)
#   P^0 = mean(Z^0)
#   Δ^B_H = P^π - P^0
#   SE_paired = std(Z^π - Z^0) / sqrt(N)
#   CI = Δ^B_H ± t_{0.025, N-1} * SE_paired
# ─────────────────────────────────────────────────────────────────

function attribute_burnout(
    belief      :: AbstractBeliefState,
    action      :: AbstractAction,        # active policy π
    twin        :: DigitalTwin,
    sim         :: AbstractSimulator,
    noise       :: RolloutNoise,
    config      :: ConfiguratorState;
    N           :: Int = 100,             # number of pairs
    H           :: Union{Int, Nothing} = nothing  # horizon in days
) :: BurnoutAttribution

    H_burn = isnothing(H) ? config.φ_cost.H_burn : H
    null_action = _matching_null_action(action)

    # run paired rollouts
    treated, baseline = run_paired_rollouts(
        belief, action, null_action,
        twin, sim, noise, H_burn, N
    )

    # compute burnout events for each pair
    Z_treated  = [crossed_burnout_threshold(r, H_burn) ? 1.0 : 0.0 for r in treated]
    Z_baseline = [crossed_burnout_threshold(r, H_burn) ? 1.0 : 0.0 for r in baseline]

    # attributable risk
    P_treated  = mean(Z_treated)
    P_baseline = mean(Z_baseline)
    Δ_hat      = P_treated - P_baseline

    # paired differences — D_i = Z^π_i - Z^0_i
    D = Z_treated .- Z_baseline
    D_bar = mean(D)
    s_D   = std(D) + 1e-10   # avoid division by zero

    # paired standard error
    se_paired = s_D / sqrt(N)

    # t critical value for 95% CI with N-1 degrees of freedom
    # approximation: t_{0.025, N-1} ≈ 1.96 for large N, 2.0 for small N
    t_crit = N >= 30 ? 1.96 : 2.045

    ci_lower = Δ_hat - t_crit * se_paired
    upper_ci = Δ_hat + t_crit * se_paired

    # sensitivity analysis across multiple horizons
    horizon_sensitivity = _compute_horizon_sensitivity(
        belief, action, twin, sim, noise, config, N
    )

    return BurnoutAttribution(
        Δ_hat      = Δ_hat,
        P_treated  = P_treated,
        P_baseline = P_baseline,
        se_paired  = se_paired,
        ci_lower   = ci_lower,
        upper_ci   = upper_ci,
        n_pairs    = N,
        horizon    = H_burn,
        horizon_sensitivity = horizon_sensitivity
    )
end

function attribute_burnout(
    belief      :: JEPABeliefState,
    action      :: AbstractAction,
    twin        :: DigitalTwin,
    sim         :: AbstractSimulator,
    noise       :: RolloutNoise,
    config      :: ConfiguratorState;
    N           :: Int = 100,
    H           :: Union{Int, Nothing} = nothing
) :: BurnoutAttribution
    _ = twin
    _ = sim
    _ = noise

    H_burn = isnothing(H) ? config.φ_cost.H_burn : H
    null_action = _matching_null_action(action)

    treated, baseline = run_paired_latent_rollouts(
        belief, action, null_action, JEPA_PREDICTOR, config; N=N
    )

    Z_treated  = [crossed_burnout_threshold(r) ? 1.0 : 0.0 for r in treated]
    Z_baseline = [crossed_burnout_threshold(r) ? 1.0 : 0.0 for r in baseline]

    P_treated  = mean(Z_treated)
    P_baseline = mean(Z_baseline)
    Δ_hat      = P_treated - P_baseline

    D = Z_treated .- Z_baseline
    s_D = std(D) + 1e-10
    se_paired = s_D / sqrt(N)
    t_crit = N >= 30 ? 1.96 : 2.045
    ci_lower = Δ_hat - t_crit * se_paired
    upper_ci = Δ_hat + t_crit * se_paired

    horizon_sensitivity = _compute_horizon_sensitivity(
        belief, action, twin, sim, noise, config, N
    )

    return BurnoutAttribution(
        Δ_hat      = Δ_hat,
        P_treated  = P_treated,
        P_baseline = P_baseline,
        se_paired  = se_paired,
        ci_lower   = ci_lower,
        upper_ci   = upper_ci,
        n_pairs    = N,
        horizon    = H_burn,
        horizon_sensitivity = horizon_sensitivity
    )
end

# ─────────────────────────────────────────────────────────────────
# Horizon Sensitivity Analysis
# Compute Δ^B_H at multiple horizons and check sign stability.
# If sign changes across horizons → estimate is unreliable.
# From Section 7: H ∈ {14, 21, 30, 45, 60} days
# ─────────────────────────────────────────────────────────────────

function _compute_horizon_sensitivity(
    belief  :: AbstractBeliefState,
    action  :: AbstractAction,
    twin    :: DigitalTwin,
    sim     :: AbstractSimulator,
    noise   :: RolloutNoise,
    config  :: ConfiguratorState,
    N       :: Int
) :: Vector{NamedTuple{(:H, :Δ), Tuple{Int, Float64}}}

    horizons = [14, 21, 30, 45, 60]
    results  = NamedTuple{(:H, :Δ), Tuple{Int, Float64}}[]

    null_action = _matching_null_action(action)

    for H in horizons
        # use fewer pairs for sensitivity — just need sign
        n_pairs = max(20, N ÷ 5)

        treated, baseline = run_paired_rollouts(
            belief, action, null_action,
            twin, sim, noise, H, n_pairs
        )

        Z_t = [crossed_burnout_threshold(r, H) ? 1.0 : 0.0 for r in treated]
        Z_b = [crossed_burnout_threshold(r, H) ? 1.0 : 0.0 for r in baseline]

        push!(results, (H=H, Δ=mean(Z_t) - mean(Z_b)))
    end

    return results
end

function _compute_horizon_sensitivity(
    belief  :: JEPABeliefState,
    action  :: AbstractAction,
    twin    :: DigitalTwin,
    sim     :: AbstractSimulator,
    noise   :: RolloutNoise,
    config  :: ConfiguratorState,
    N       :: Int
) :: Vector{NamedTuple{(:H, :Δ), Tuple{Int, Float64}}}
    _ = twin
    _ = sim
    _ = noise

    horizons = [14, 21, 30, 45, 60]
    results  = NamedTuple{(:H, :Δ), Tuple{Int, Float64}}[]
    null_action = _matching_null_action(action)

    for H in horizons
        n_pairs = max(20, N ÷ 5)
        treated, baseline = run_paired_latent_rollouts(
            belief, action, null_action, JEPA_PREDICTOR, config; N=n_pairs
        )

        Z_t = [crossed_burnout_threshold(r) ? 1.0 : 0.0 for r in treated]
        Z_b = [crossed_burnout_threshold(r) ? 1.0 : 0.0 for r in baseline]

        push!(results, (H=H, Δ=mean(Z_t) - mean(Z_b)))
    end

    return results
end

# ─────────────────────────────────────────────────────────────────
# Check sign stability across horizons
# If sign of Δ^B_H varies → estimate is unreliable → flag
# ─────────────────────────────────────────────────────────────────

function is_attribution_stable(
    attribution :: BurnoutAttribution
) :: Bool

    sensitivity = attribution.horizon_sensitivity
    isempty(sensitivity) && return true

    signs = [sign(r.Δ) for r in sensitivity if abs(r.Δ) > 1e-6]
    isempty(signs) && return true

    # stable if all signs agree
    return length(unique(signs)) == 1
end

# ─────────────────────────────────────────────────────────────────
# Format attribution result for logging and UI
# ─────────────────────────────────────────────────────────────────

function summarize_attribution(attribution::BurnoutAttribution) :: String
    stable = is_attribution_stable(attribution)
    stability_str = stable ? "stable" : "UNSTABLE across horizons"

    string(
        "Burnout Attribution (H=$(attribution.horizon) days, N=$(attribution.n_pairs) pairs):\n",
        "  Treated:  $(round(attribution.P_treated * 100, digits=1))% burnout probability\n",
        "  Baseline: $(round(attribution.P_baseline * 100, digits=1))% burnout probability\n",
        "  Δ^B_H:    $(round(attribution.Δ_hat * 100, digits=1))% ",
        "[$(round(attribution.ci_lower * 100, digits=1))%, $(round(attribution.upper_ci * 100, digits=1))%]\n",
        "  Stability: $stability_str"
    )
end
