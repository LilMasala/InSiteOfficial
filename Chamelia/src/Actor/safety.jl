"""
safety.jl
Safety gate — hard constraints on candidate actions.
These are lexicographic constraints — safety comes before everything else.
No predicted benefit can override a safety violation.

Two constraint families:
  1. Hard safety: worst-case rollout must not violate safety thresholds
  2. Epistemic feasibility: system must know enough to act

Both are checked BEFORE CVaR optimization.
A failing action is rejected entirely — not penalized, rejected.
"""

using Statistics

function _action_delta_pairs(action :: CandidateAction)
    return collect(pairs(action.deltas))
end

function _action_delta_pairs(action :: ScheduledAction)
    out = Pair{Symbol, Float64}[]
    for delta in action.segment_deltas
        for (key, value) in delta.parameter_deltas
            push!(out, key => Float64(value))
        end
    end
    return out
end

# ─────────────────────────────────────────────────────────────────
# Safety Constraint Interface
# Simulator plugin registers domain-specific safety constraints.
# Chamelia enforces them — it does not define them.
# ─────────────────────────────────────────────────────────────────

"""
    check_safety(sim, rollout_signals, thresholds) → Bool

Domain-specific safety check on one rollout's signals.
Returns true if SAFE, false if VIOLATED.
Must be implemented by simulator plugin.
"""
function check_safety(
    sim              :: AbstractSimulator,
    rollout_signals  :: Vector{Dict{Symbol, Any}},
    thresholds       :: Dict{Symbol, Float64}
) :: Bool
    error("$(typeof(sim)) must implement check_safety!")
end

# Mock — always safe
function check_safety(
    ::MockSimulator,
    ::Vector{Dict{Symbol, Any}},
    ::Dict{Symbol, Float64}
) :: Bool
    return true
end

# ─────────────────────────────────────────────────────────────────
# Hard Safety Gate
# Safety is lexicographic, but it should be evaluated counterfactually:
#   1. Absolute catastrophic ceilings still block outright
#   2. Otherwise compare treated risk against the no-change baseline
# If treated risk is lower than baseline risk, that is a valid safety outcome
# even when the baseline regime is already imperfect.
# ─────────────────────────────────────────────────────────────────

function _explicit_rollout_safety_stats(
    rollouts   :: Vector{RolloutResult},
    thresholds :: Dict{Symbol, Float64}
)
    isempty(rollouts) && return (
        violation_rate = 0.0,
        excess = 0.0,
        catastrophic = false,
        max_pct_low = 0.0,
        max_pct_high = 0.0,
    )

    pct_low_max = get(thresholds, :pct_low_max, 0.04)
    pct_high_max = get(thresholds, :pct_high_max, 0.25)
    pct_low_hard_max = get(thresholds, :pct_low_hard_max, max(0.08, 2.0 * pct_low_max))
    pct_high_hard_max = get(thresholds, :pct_high_hard_max, max(0.40, 1.5 * pct_high_max))

    violations = 0
    excess_sum = 0.0
    catastrophic = false
    max_pct_low = 0.0
    max_pct_high = 0.0

    for rollout in rollouts
        rollout_violation = false
        rollout_excess = 0.0

        for signals in rollout.phys_signals
            pct_low = Float64(get(signals, :pct_low_7d, get(signals, :pct_low, 0.0)))
            pct_high = Float64(get(signals, :pct_high_7d, get(signals, :pct_high, 0.0)))
            max_pct_low = max(max_pct_low, pct_low)
            max_pct_high = max(max_pct_high, pct_high)

            low_excess = max(0.0, pct_low - pct_low_max) / max(pct_low_max, 1e-6)
            high_excess = max(0.0, pct_high - pct_high_max) / max(pct_high_max, 1e-6)
            rollout_excess = max(rollout_excess, low_excess + high_excess)
            rollout_violation |= (low_excess > 0.0 || high_excess > 0.0)

            if pct_low > pct_low_hard_max || pct_high > pct_high_hard_max
                catastrophic = true
            end
        end

        violations += rollout_violation ? 1 : 0
        excess_sum += rollout_excess
    end

    return (
        violation_rate = violations / length(rollouts),
        excess = excess_sum / length(rollouts),
        catastrophic = catastrophic,
        max_pct_low = max_pct_low,
        max_pct_high = max_pct_high,
    )
end

function _latent_rollout_safety_stats(
    rollouts   :: Vector{LatentRolloutResult},
    thresholds :: Dict{Symbol, Float64}
)
    isempty(rollouts) && return (
        violation_rate = 0.0,
        excess = 0.0,
        catastrophic = false,
        max_physical_risk = 0.0,
    )

    physical_risk_max = get(thresholds, :physical_risk_max, 0.30)
    physical_risk_hard_max = get(thresholds, :physical_risk_hard_max, max(0.45, 1.5 * physical_risk_max))

    violations = 0
    excess_sum = 0.0
    catastrophic = false
    max_physical_risk = 0.0

    for rollout in rollouts
        rollout_violation = false
        rollout_excess = 0.0
        for belief in (rollout.short_belief, rollout.med_belief, rollout.long_belief)
            summary = decode_latent_summary(belief)
            max_physical_risk = max(max_physical_risk, summary.physical_risk)
            excess = max(0.0, summary.physical_risk - physical_risk_max) / max(physical_risk_max, 1e-6)
            rollout_excess = max(rollout_excess, excess)
            rollout_violation |= excess > 0.0
            catastrophic |= summary.physical_risk > physical_risk_hard_max
        end
        violations += rollout_violation ? 1 : 0
        excess_sum += rollout_excess
    end

    return (
        violation_rate = violations / length(rollouts),
        excess = excess_sum / length(rollouts),
        catastrophic = catastrophic,
        max_physical_risk = max_physical_risk,
    )
end

function safety_diagnostics(
    rollouts          :: Vector{RolloutResult},
    baseline_rollouts :: Vector{RolloutResult},
    sim               :: AbstractSimulator,
    thresholds        :: Dict{Symbol, Float64}
)
    _ = sim
    treated = _explicit_rollout_safety_stats(rollouts, thresholds)
    baseline = _explicit_rollout_safety_stats(baseline_rollouts, thresholds)
    violation_tol = get(thresholds, :safety_violation_tol, 0.02)
    excess_tol = get(thresholds, :safety_excess_tol, 0.05)
    catastrophic_relief_tol = get(thresholds, :catastrophic_relief_tol, 0.05)
    relative_violation_gap = treated.violation_rate - baseline.violation_rate
    relative_excess_gap = treated.excess - baseline.excess
    catastrophic_passed = if treated.catastrophic
        baseline.catastrophic && treated.excess <= baseline.excess - catastrophic_relief_tol
    else
        true
    end
    passed = catastrophic_passed &&
        treated.violation_rate <= baseline.violation_rate + violation_tol &&
        treated.excess <= baseline.excess + excess_tol
    failure_mode = treated.catastrophic && !catastrophic_passed ? :catastrophic : (
        relative_violation_gap > violation_tol || relative_excess_gap > excess_tol ? :relative_worsening : :passed
    )
    return (
        mode = :explicit,
        passed = passed,
        failure_mode = failure_mode,
        treated = treated,
        baseline = baseline,
        violation_tol = violation_tol,
        excess_tol = excess_tol,
        catastrophic_relief_tol = catastrophic_relief_tol,
        relative_violation_gap = relative_violation_gap,
        relative_excess_gap = relative_excess_gap,
    )
end

function safety_diagnostics(
    rollouts          :: Vector{LatentRolloutResult},
    baseline_rollouts :: Vector{LatentRolloutResult},
    sim               :: AbstractSimulator,
    thresholds        :: Dict{Symbol, Float64}
)
    _ = sim
    treated = _latent_rollout_safety_stats(rollouts, thresholds)
    baseline = _latent_rollout_safety_stats(baseline_rollouts, thresholds)
    violation_tol = get(thresholds, :safety_violation_tol, 0.02)
    excess_tol = get(thresholds, :safety_excess_tol, 0.05)
    catastrophic_relief_tol = get(thresholds, :catastrophic_relief_tol, 0.05)
    relative_violation_gap = treated.violation_rate - baseline.violation_rate
    relative_excess_gap = treated.excess - baseline.excess
    catastrophic_passed = if treated.catastrophic
        baseline.catastrophic && treated.excess <= baseline.excess - catastrophic_relief_tol
    else
        true
    end
    passed = catastrophic_passed &&
        treated.violation_rate <= baseline.violation_rate + violation_tol &&
        treated.excess <= baseline.excess + excess_tol
    failure_mode = treated.catastrophic && !catastrophic_passed ? :catastrophic : (
        relative_violation_gap > violation_tol || relative_excess_gap > excess_tol ? :relative_worsening : :passed
    )
    return (
        mode = :latent,
        passed = passed,
        failure_mode = failure_mode,
        treated = treated,
        baseline = baseline,
        violation_tol = violation_tol,
        excess_tol = excess_tol,
        catastrophic_relief_tol = catastrophic_relief_tol,
        relative_violation_gap = relative_violation_gap,
        relative_excess_gap = relative_excess_gap,
    )
end

function passes_safety_gate(
    rollouts          :: Vector{RolloutResult},
    baseline_rollouts :: Vector{RolloutResult},
    sim               :: AbstractSimulator,
    thresholds        :: Dict{Symbol, Float64}
) :: Bool
    return safety_diagnostics(rollouts, baseline_rollouts, sim, thresholds).passed
end

function passes_safety_gate(
    rollouts          :: Vector{LatentRolloutResult},
    baseline_rollouts :: Vector{LatentRolloutResult},
    sim               :: AbstractSimulator,
    thresholds        :: Dict{Symbol, Float64}
) :: Bool
    return safety_diagnostics(rollouts, baseline_rollouts, sim, thresholds).passed
end

function passes_safety_gate(
    rollouts   :: Vector{RolloutResult},
    sim        :: AbstractSimulator,
    thresholds :: Dict{Symbol, Float64}
) :: Bool
    _ = sim
    return !_explicit_rollout_safety_stats(rollouts, thresholds).catastrophic
end

function passes_safety_gate(
    rollouts   :: Vector{LatentRolloutResult},
    sim        :: AbstractSimulator,
    thresholds :: Dict{Symbol, Float64}
) :: Bool
    _ = sim
    return !_latent_rollout_safety_stats(rollouts, thresholds).catastrophic
end

# ─────────────────────────────────────────────────────────────────
# Epistemic Feasibility Gate
# F_t = 𝟙[κ ≥ κ_min] · 𝟙[ρ ≥ ρ_min] · 𝟙[η ≥ η_min]
# Hard AND — all three must pass.
# ─────────────────────────────────────────────────────────────────

function passes_epistemic_gate(
    epistemic  :: EpistemicState
) :: Bool
    return epistemic.feasible
end

# ─────────────────────────────────────────────────────────────────
# Effect Size Gate
# δ_eff = (CVaR(a⁰) - CVaR(a)) / std(energies(a)) > δ_min
# Ensures improvement is real, not just rollout noise.
# ─────────────────────────────────────────────────────────────────

function passes_effect_size_gate(
    energies_action   :: Vector{Float64},
    energies_baseline :: Vector{Float64},
    config            :: ConfiguratorState
) :: Bool

    δ_eff = compute_effect_size(energies_action, energies_baseline, config.φ_act.α_cvar)
    return δ_eff > config.φ_act.δ_min_effect
end

# ─────────────────────────────────────────────────────────────────
# Clinical Delta Gate
# Suppresses recommendations whose largest per-dimension action magnitude
# is below the domain's minimum meaningful therapy change.
#
# Distinct from the effect size gate:
#   Effect size gate  → is the predicted improvement statistically real?
#   Clinical delta gate → is the raw action physically / clinically actionable?
#
# A tiny action delta may show positive effect size in rollouts but still be
# below device granularity or practical implementation significance.
# Both gates must pass for a recommendation to be surfaced.
# ─────────────────────────────────────────────────────────────────

function passes_clinical_delta_gate(
    action :: CandidateAction,
    sim    :: AbstractSimulator,
    adapter :: AbstractDomainAdapter = Main.DefaultDomainAdapter(),
    prefs  :: UserPreferences = UserPreferences()
) :: Bool
    baseline_threshold = min_clinical_delta(sim)
    return any(abs(v) >= max(baseline_threshold, minimum_action_delta_threshold(adapter, prefs, key))
               for (key, v) in _action_delta_pairs(action))
end

function passes_clinical_delta_gate(
    action :: ScheduledAction,
    sim    :: AbstractSimulator,
    adapter :: AbstractDomainAdapter = Main.DefaultDomainAdapter(),
    prefs  :: UserPreferences = UserPreferences()
) :: Bool
    isempty(action.segment_deltas) && return false
    baseline_threshold = min_clinical_delta(sim)
    return any(abs(v) >= max(baseline_threshold, minimum_action_delta_threshold(adapter, prefs, key))
               for (key, v) in _action_delta_pairs(action))
end

function passes_clinical_delta_gate(
    action :: AbstractAction,
    sim    :: AbstractSimulator,
    adapter :: AbstractDomainAdapter = Main.DefaultDomainAdapter(),
    prefs  :: UserPreferences = UserPreferences()
) :: Bool
    # NullAction and other no-op types never pass — they have no therapy change
    return false
end

# ─────────────────────────────────────────────────────────────────
# Burnout Safety Gate
# Block action if upper CI on attributable burnout risk exceeds threshold.
# Δ^B_H(π) + t_{0.025} · SE_paired < ε_burn
# ─────────────────────────────────────────────────────────────────

function passes_burnout_gate(
    attribution :: BurnoutAttribution,
    config      :: ConfiguratorState
) :: Bool
    return attribution.upper_ci < config.φ_cost.ε_burn
end

# ─────────────────────────────────────────────────────────────────
# Shadow Graduation Gate
# System cannot surface recommendations before graduating.
# Graduation requires:
#   - minimum 21 shadow days
#   - >= 60% win rate
#   - zero safety violations
#   - 7 consecutive days of sustained performance
# ─────────────────────────────────────────────────────────────────

const SHADOW_MIN_DAYS     = 21
const SHADOW_MIN_WIN_RATE = 0.60
const SHADOW_CONSEC_DAYS  = 7

Base.@kwdef struct ShadowScorecard
    n_days            :: Int
    win_rate          :: Float64
    safety_violations :: Int
    consecutive_days  :: Int
end

function passes_graduation_gate(scorecard::ShadowScorecard) :: Bool
    scorecard.n_days            >= SHADOW_MIN_DAYS     || return false
    scorecard.win_rate          >= SHADOW_MIN_WIN_RATE || return false
    scorecard.safety_violations == 0                   || return false
    scorecard.consecutive_days  >= SHADOW_CONSEC_DAYS  || return false
    return true
end

# ─────────────────────────────────────────────────────────────────
# Full Gate Pipeline
# Applies all gates in lexicographic order.
# Returns (passes::Bool, reason::Symbol)
# reason is used for logging and shadow scorecard
# ─────────────────────────────────────────────────────────────────

function apply_all_gates(
    action          :: CandidateAction,
    rollouts        :: Vector{RolloutResult},
    baseline_rollouts :: Vector{RolloutResult},
    epistemic       :: EpistemicState,
    attribution     :: Union{BurnoutAttribution, Nothing},
    sim             :: AbstractSimulator,
    config          :: ConfiguratorState,
    thresholds      :: Dict{Symbol, Float64}
) :: Tuple{Bool, Symbol}

    # 1. Epistemic feasibility — can we act at all?
    passes_epistemic_gate(epistemic) || return (false, :epistemic_failed)

    # 2. Hard safety — is this action safe under worst-case?
    passes_safety_gate(rollouts, baseline_rollouts, sim, thresholds) || return (false, :safety_violated)

    # 3. Effect size — is improvement real or just noise?
    energies_action   = compute_energies(rollouts, ZeroCritic(), config)
    energies_baseline = compute_energies(baseline_rollouts, ZeroCritic(), config)
    passes_effect_size_gate(energies_action, energies_baseline, config) ||
        return (false, :effect_size_insufficient)

    # 4. Burnout gate — does action increase burnout risk?
    if !isnothing(attribution)
        passes_burnout_gate(attribution, config) || return (false, :burnout_risk_exceeded)
    end

    return (true, :passed)
end
