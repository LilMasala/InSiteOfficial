"""
InSiteDomainAdapter.jl

InSite-domain adapter for Chamelia.

This file is the single place that owns InSite/diabetes-specific semantics
that are NOT part of Chamelia's domain-agnostic core.

InSite-domain concepts owned here:
  - Physical cost weight names: w_low, w_high, w_tir, w_var
  - Mapping of hypoglycemia_fear preference → w_low weight
  - Interpretation of aggressiveness → action bounds is shared with core
    but the diabetes-specific physical weight scaling is here

Chamelia core is NOT allowed to reference these signal names directly.
If a future domain (e.g., cardiovascular rehab, AID advisory) is added,
it provides its own AbstractDomainAdapter without touching core code.
"""

using Random
using Distributions

using Main: AbstractDomainAdapter, UserPreferences,
            RegimeDetectionResult, ConnectedAppState, MemoryBuffer,
            TwinPosterior, TwinPrior
import Main: detect_regime, calibrate_posterior!

struct InSiteDomainAdapter <: AbstractDomainAdapter end

"""
    default_physical_weights(::InSiteDomainAdapter, prefs) → Dict{Symbol, Float64}

InSite-specific physical cost weights.

Signal semantics (InSite/T1D domain):
  :w_low  — weight on percent-time-low (hypoglycemia). Scaled by hypoglycemia_fear.
             Range [3, 7]; higher → system is more conservative about low BG risk.
  :w_high — weight on percent-time-high (hyperglycemia). Fixed at 1.0.
  :w_tir  — weight on time-in-range. Fixed at 1.0.
  :w_var  — weight on BG coefficient of variation. Fixed at 0.5.

These keys are matched by InSiteSimulator.compute_physical_cost.
A future domain adapter would define its own signal names here.
"""
function default_physical_weights(
    :: InSiteDomainAdapter,
    prefs :: UserPreferences
) :: Dict{Symbol, Float64}
    # w_low: hypoglycemia_fear ∈ [0,1] → w_low ∈ [3, 7]
    # Higher fear → heavier penalty on %low → system avoids actions that increase lows
    w_low = 3.0 + prefs.hypoglycemia_fear * 4.0

    # w_high: aggressiveness ∈ [0,1] → w_high ∈ [1.0, 1.5]
    # Higher aggressiveness → heavier penalty on %high → system is more motivated to reduce
    # persistent highs. Combined with w_low, this expresses the user's preferred tradeoff:
    # aggressive + low hypo fear → accept some low risk to fix highs
    # conservative + high hypo fear → protect against lows even if highs persist
    w_high = 1.0 + prefs.aggressiveness * 0.5

    return Dict{Symbol, Float64}(
        :w_low  => w_low,
        :w_high => w_high,
        :w_tir  => 1.0,
        :w_var  => 0.5
    )
end

function domain_name(:: InSiteDomainAdapter) :: String
    return "insite_t1d"
end

# -------------------------------------------------------------------
# Regime Detection — InSite / T1D domain
#
# Chamelia core calls detect_regime(adapter, signals, app_state, memory)
# and gets back a RegimeDetectionResult without knowing anything about
# T1D signal names, menstrual cycles, or day-of-week concepts.
#
# Regime priority order (highest clinical confidence first):
#   1. menstrual_phase    — cycle_phase_menstrual > 0.5
#   2. luteal_phase       — cycle_phase_luteal > 0.5
#   3. high_activity_day  — exercise_mins >= 60
#   4. weekend            — day_of_week ∈ {0, 6}  (Sun=0, Sat=6)
#
# Scope logic:
#   - If an existing profile's name contains the regime label → patch_existing
#   - Else if ≥1 other profile exists → create_new (from active profile base)
#   - Else → patch_current (only one profile; regime suggestion deferred)
# -------------------------------------------------------------------

function _t1d_regime_label(signals :: Dict{Symbol, Any}) :: Union{String, Nothing}
    # Menstrual phase (direct binary signal from CGM-adjacent cycle tracker)
    v = get(signals, :cycle_phase_menstrual, 0.0)
    v isa Number && Float64(v) > 0.5 && return "menstrual_phase"

    # Luteal phase (elevated insulin resistance window)
    v = get(signals, :cycle_phase_luteal, 0.0)
    v isa Number && Float64(v) > 0.5 && return "luteal_phase"

    # High-activity / training day
    v = get(signals, :exercise_mins, 0.0)
    v isa Number && Float64(v) >= 60.0 && return "high_activity_day"

    # Weekend (day_of_week sent by app: 0=Sunday … 6=Saturday)
    v = get(signals, :day_of_week, nothing)
    if v isa Number
        d = Int(round(Float64(v)))
        (d == 0 || d == 6) && return "weekend"
    end

    return nothing
end

function detect_regime(
    ::    InSiteDomainAdapter,
    signals   :: Dict{Symbol, Any},
    app_state :: ConnectedAppState,
    memory    :: MemoryBuffer
) :: RegimeDetectionResult
    regime = _t1d_regime_label(signals)
    isnothing(regime) && return RegimeDetectionResult(nothing, "patch_current", nothing)

    # Check whether an existing profile already targets this regime
    # (simple name-match heuristic — future: embed regime metadata on profiles)
    match_idx = findfirst(
        p -> occursin(lowercase(regime), lowercase(p.name)),
        app_state.available_profiles
    )
    if !isnothing(match_idx)
        return RegimeDetectionResult(
            regime,
            "patch_existing",
            app_state.available_profiles[match_idx].id
        )
    end

    # At least one other profile exists → propose creating a regime-specific profile
    if !isempty(app_state.available_profiles)
        return RegimeDetectionResult(regime, "create_new", app_state.active_profile_id)
    end

    # No other profiles yet — still surface the regime label but patch current for now
    return RegimeDetectionResult(regime, "patch_current", nothing)
end

# -------------------------------------------------------------------
# Cold-Start Twin Calibration — InSite / T1D domain
#
# Importance-sampling calibration from self-reported glycemic metrics.
# Called once at patient initialization when calibration_targets are present.
# Updates posterior.physical[:isf_multiplier] and [:basal_multiplier] with
# soft regularization toward the prior mean.
# -------------------------------------------------------------------

function calibrate_posterior!(
    :: InSiteDomainAdapter,
    posterior :: TwinPosterior,
    prior     :: TwinPrior,
    targets   :: Dict{String, Float64}
) :: Nothing
    tir_obs      = get(targets, "recent_tir",      NaN)
    pct_low_obs  = get(targets, "recent_pct_low",  NaN)
    pct_high_obs = get(targets, "recent_pct_high", NaN)

    # bail out if no usable targets
    (isnan(tir_obs) && isnan(pct_low_obs) && isnan(pct_high_obs)) && return nothing

    # impossible self-reports should not drag the posterior toward a bogus fit
    if !isnan(pct_low_obs) && !isnan(pct_high_obs) && (pct_low_obs + pct_high_obs > 1.0)
        return nothing
    end
    if !isnan(tir_obs) && !isnan(pct_low_obs) && !isnan(pct_high_obs)
        total = tir_obs + pct_low_obs + pct_high_obs
        abs(total - 1.0) > 0.15 && return nothing
    end

    seed = UInt32(abs(hash(tir_obs)) % typemax(UInt32))
    rng = Random.MersenneTwister(seed)

    isf_dist   = get(prior.physical_priors, :isf_multiplier,   Normal(1.0, 0.12))
    basal_dist = get(prior.physical_priors, :basal_multiplier, Normal(1.0, 0.10))

    N = 200
    isf_particles   = Float64[clamp(rand(rng, isf_dist),   0.5, 1.8) for _ in 1:N]
    basal_particles = Float64[clamp(rand(rng, basal_dist), 0.5, 1.8) for _ in 1:N]

    σ = 0.08
    log_weights = zeros(N)
    for i in 1:N
        isf, basal = isf_particles[i], basal_particles[i]
        tir_hat      = clamp(0.50 + 0.35*(isf - 1.0) + 0.15*(basal - 1.0), 0.05, 0.98)
        pct_low_hat  = clamp(0.08 - 0.12*(isf - 1.0) - 0.04*(basal - 1.0), 0.0,  0.40)
        pct_high_hat = clamp(1.0 - tir_hat - pct_low_hat,                   0.0,  0.95)
        if !isnan(tir_obs)
            log_weights[i] -= (tir_hat - tir_obs)^2 / (2σ^2)
        end
        if !isnan(pct_low_obs)
            log_weights[i] -= (pct_low_hat - pct_low_obs)^2 / (2σ^2)
        end
        if !isnan(pct_high_obs)
            log_weights[i] -= (pct_high_hat - pct_high_obs)^2 / (2σ^2)
        end
    end

    log_weights .-= maximum(log_weights)  # log-sum-exp stability
    weights = exp.(log_weights)
    w_sum = sum(weights)
    w_sum < 1e-12 && return nothing  # degenerate — bail

    weights ./= w_sum

    # effective sample size check
    n_eff = 1.0 / sum(weights .^ 2)
    n_eff < 5.0 && return nothing   # targets inconsistent with prior — don't update

    isf_est   = sum(weights[i] * isf_particles[i]   for i in 1:N)
    basal_est = sum(weights[i] * basal_particles[i] for i in 1:N)

    regularization = 10.0
    α = n_eff / (n_eff + regularization)  # soft weight toward calibrated estimate

    posterior.physical[:isf_multiplier]   = α * isf_est   + (1 - α) * get(posterior.physical, :isf_multiplier,   1.0)
    posterior.physical[:basal_multiplier] = α * basal_est + (1 - α) * get(posterior.physical, :basal_multiplier,  1.0)

    return nothing
end
