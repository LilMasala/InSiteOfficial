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

using Main: AbstractDomainAdapter, UserPreferences,
            RegimeDetectionResult, ConnectedAppState, MemoryBuffer
import Main: detect_regime

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
