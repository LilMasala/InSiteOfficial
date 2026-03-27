"""
rules.jl
Rule-based configurator — v1.1
Rules derived from mathematical constraints, not intuition.

Every parameter bound has a justification from the formulation.
Every rule has a clear mathematical rationale.
No arbitrary numbers — only system-derived constraints.
"""

using Statistics

# ─────────────────────────────────────────────────────────────────
# Local memory summaries
# Configurator only needs lightweight statistics from MemoryBuffer.
# Use Main.Memory helpers when that module is already loaded; otherwise
# fall back to local summaries so Configurator remains loadable standalone.
# ─────────────────────────────────────────────────────────────────

Base.@kwdef struct ScorecardSummary
    win_rate          :: Float64 = 0.0
    safety_violations :: Int = 0
    consecutive_days  :: Int = 0
end

function _recent_records(mem::MemoryBuffer, n::Int)
    if isdefined(Main, :Memory) && isdefined(Main.Memory, :recent_records)
        return Main.Memory.recent_records(mem, n)
    end

    return last(mem.records, min(n, length(mem.records)))
end

function _compute_scorecard(mem::MemoryBuffer) :: ScorecardSummary
    if isdefined(Main, :Memory) && isdefined(Main.Memory, :compute_scorecard)
        scorecard = Main.Memory.compute_scorecard(mem)
        return ScorecardSummary(
            win_rate = scorecard.win_rate,
            safety_violations = scorecard.safety_violations,
            consecutive_days = scorecard.consecutive_days
        )
    end

    scored = filter(r -> !isnothing(r.shadow_delta_score), mem.records)
    isempty(scored) && return ScorecardSummary()

    wins = count(r -> something(r.shadow_delta_score, 0.0) > 0.0, scored)
    win_rate = wins / length(scored)

    recent_costs = [r.realized_cost for r in scored if !isnothing(r.realized_cost)]
    safety_violations = if isempty(recent_costs)
        0
    else
        cost_threshold = mean(recent_costs) + 2.0 * std(recent_costs)
        count(
            r -> !isnothing(r.realized_cost) &&
                 something(r.realized_cost, -Inf) > cost_threshold &&
                 !is_null(r.action),
            scored
        )
    end

    ScorecardSummary(
        win_rate = win_rate,
        safety_violations = safety_violations,
        consecutive_days = _compute_consecutive_days(scored)
    )
end

function _compute_consecutive_days(
    records;
    threshold :: Float64 = 0.60
) :: Int

    isempty(records) && return 0

    days = sort(unique(r.day for r in records))
    isempty(days) && return 0

    max_consecutive = 0
    current_streak  = 0

    for day in days
        scored_today = filter(
            r -> r.day == day && !isnothing(r.shadow_delta_score),
            records
        )

        if isempty(scored_today)
            current_streak = 0
            continue
        end

        daily_wins = count(r -> something(r.shadow_delta_score, 0.0) > 0.0, scored_today)
        daily_win_rate = daily_wins / length(scored_today)

        if daily_win_rate >= threshold
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
        else
            current_streak = 0
        end
    end

    return max_consecutive
end

# ─────────────────────────────────────────────────────────────────
# System-derived parameter bounds
# These come from the math — not from guessing
# ─────────────────────────────────────────────────────────────────

"""
Minimum N_roll for CVaR to be statistically meaningful.
With alpha = 0.8, CVaR averages ceil((1-alpha)*N) samples.
Need at least 5 samples in the tail for SE < 30%.
ceil((1-0.8)*N) >= 5 → N >= 25
We use 30 as practical minimum (SE ≈ 18% with 5 tail samples).
"""
function min_n_roll(α_cvar::Float64) :: Int
    min_tail_samples = 5
    return ceil(Int, min_tail_samples / (1.0 - α_cvar))
end

"""
Minimum H_med for psychological dynamics to manifest.
Trust dynamics have decay rate ~0.05/day → half-life ~14 days.
But meaningful trust signal requires at least 3 days of observations.
Burnout dynamics operate on ~7 day timescale minimum.
"""
const H_MED_MIN = 3   # days — minimum for any meaningful rollout

"""
Maximum H_med during drift.
During drift, twin reliability degrades.
Cap at days_since_last_stable to not simulate beyond known regime.
"""
function max_h_med_during_drift(days_since_drift::Int) :: Int
    return max(H_MED_MIN, min(7, days_since_drift))
end

"""
Δ_max safety scaling from Section 9 of formulation:
Δ_max(t) = Δ_base · f_aggr · f_trust · f_score · f_drift
"""
function compute_delta_max(
    prefs       :: UserPreferences,
    trust       :: Float64,
    win_rate    :: Float64,
    drift       :: Bool,
    n_records   :: Int
) :: Float64

    Δ_base = 0.10   # hard ceiling — 10% max change ever

    # aggressiveness scale [0.3, 1.0] from user preference
    f_aggr = 0.3 + prefs.aggressiveness * 0.7

    # trust scale — can't act aggressively without patient trust
    # requires trust > 0.3 to act at all, scales linearly to 1.0 at trust = 0.8
    τ_min = 0.30
    τ_req = 0.80
    f_trust = clamp((trust - τ_min) / (τ_req - τ_min), 0.0, 1.0)

    # scorecard scale — must prove performance before expanding
    # requires win_rate > 0.5 to get any scale, full scale at 0.65
    if n_records < 10
        f_score = 0.3   # very conservative until we have data
    else
        f_score = clamp((win_rate - 0.50) / (0.65 - 0.50), 0.3, 1.0)
    end

    # drift scale — half the bounds during drift
    f_drift = drift ? 0.5 : 1.0

    return Δ_base * f_aggr * f_trust * f_score * f_drift
end

# ─────────────────────────────────────────────────────────────────
# MetaState — summary of current system health
# ─────────────────────────────────────────────────────────────────

Base.@kwdef struct MetaState
    belief_entropy    :: Float64
    κ_familiarity     :: Float64
    ρ_concordance     :: Float64
    η_calibration     :: Float64
    win_rate          :: Float64
    safety_violations :: Int
    consecutive_days  :: Int
    trust_level       :: Float64
    burnout_level     :: Float64
    drift_detected    :: Bool
    days_since_drift  :: Int
    n_records         :: Int
    current_day       :: Int
    graduated         :: Bool = false
    no_surface_streak :: Int = 0
    last_decision_reason :: Symbol = :initialized
end

function meta_to_features(m :: MetaState) :: Vector{Float64}
    [
        m.belief_entropy,
        m.κ_familiarity,
        m.ρ_concordance,
        m.η_calibration,
        m.win_rate,
        Float64(m.safety_violations),
        Float64(m.consecutive_days),
        m.trust_level,
        m.burnout_level,
        m.drift_detected ? 1.0 : 0.0,
        Float64(m.days_since_drift),
        Float64(m.n_records)
    ]
end

function compute_meta_state(
    belief      :: AbstractBeliefState,
    epistemic   :: EpistemicState,
    mem         :: MemoryBuffer,
    psy         :: PsyState,
    current_day :: Int;
    graduated   :: Bool = false,
    last_decision_reason :: Symbol = :initialized,
) :: MetaState

    scorecard = _compute_scorecard(mem)
    drift_detected, days_since_drift = _detect_drift(mem, belief, current_day)
    no_surface_streak = graduated ? _compute_postgrad_no_surface_streak(mem, current_day) : 0

    MetaState(
        belief_entropy    = belief.entropy,
        κ_familiarity     = epistemic.κ_familiarity,
        ρ_concordance     = epistemic.ρ_concordance,
        η_calibration     = epistemic.η_calibration,
        win_rate          = scorecard.win_rate,
        safety_violations = scorecard.safety_violations,
        consecutive_days  = scorecard.consecutive_days,
        trust_level       = psy.trust.value,
        burnout_level     = psy.burnout.value,
        drift_detected    = drift_detected,
        days_since_drift  = days_since_drift,
        n_records         = length(mem.records),
        current_day       = current_day,
        graduated         = graduated,
        no_surface_streak = no_surface_streak,
        last_decision_reason = last_decision_reason,
    )
end

const POSTGRAD_NO_SURFACE_OFFSET_DAYS = 21
const POSTGRAD_ADAPT_STREAK_MIN = 4

function _compute_trailing_null_streak(
    mem         :: MemoryBuffer,
    current_day :: Int
) :: Int
    streak = 0
    for day in current_day:-1:1
        day_records = filter(r -> r.day == day, mem.records)
        isempty(day_records) && break
        all(r -> is_null(r.action), day_records) || break
        streak += 1
    end
    return streak
end

function _compute_postgrad_no_surface_streak(
    mem         :: MemoryBuffer,
    current_day :: Int
) :: Int
    trailing_null = _compute_trailing_null_streak(mem, current_day)
    return max(0, trailing_null - POSTGRAD_NO_SURFACE_OFFSET_DAYS)
end

function _apply_postgrad_no_surface_adaptation(
    config :: ConfiguratorState,
    meta   :: MetaState
) :: ConfiguratorState
    !meta.graduated && return config
    meta.no_surface_streak < POSTGRAD_ADAPT_STREAK_MIN && return config
    meta.safety_violations > 0 && return config

    adapt_strength = clamp(
        (meta.no_surface_streak - POSTGRAD_ADAPT_STREAK_MIN + 1) / 10,
        0.0,
        1.0
    )

    H_perc = config.φ_perc.H_perc
    Δ_max = config.φ_act.Δ_max
    δ_min_effect = config.φ_act.δ_min_effect
    α_cvar = config.φ_act.α_cvar
    N_search = config.φ_act.N_search
    H_burn = config.φ_cost.H_burn
    ε_burn = config.φ_cost.ε_burn
    N_roll = config.φ_world.N_roll
    H_med = config.φ_world.H_med

    if meta.last_decision_reason == :effect_size_insufficient
        δ_min_effect = max(0.15, δ_min_effect * (1.0 - 0.45 * adapt_strength))
        Δ_max = min(0.10, max(0.03, Δ_max * (1.0 + 0.25 * adapt_strength)))
        N_search = min(81, round(Int, N_search * (1.0 + 1.0 * adapt_strength)))
    elseif meta.last_decision_reason == :burnout_risk_exceeded
        H_burn = max(14, round(Int, H_burn * (1.0 - 0.50 * adapt_strength)))
        ε_burn = min(0.10, ε_burn * (1.0 + 0.25 * adapt_strength))
        δ_min_effect = max(0.20, δ_min_effect * (1.0 - 0.20 * adapt_strength))
        N_search = min(81, round(Int, N_search * (1.0 + 0.50 * adapt_strength)))
    elseif meta.last_decision_reason == :safety_violated
        Δ_max = max(0.02, Δ_max * (1.0 - 0.25 * adapt_strength))
        N_roll = min(150, round(Int, N_roll * (1.0 + 0.50 * adapt_strength)))
        N_search = min(81, round(Int, N_search * (1.0 + 0.50 * adapt_strength)))
    elseif meta.last_decision_reason == :epistemic_failed
        N_roll = min(150, round(Int, N_roll * (1.0 + 0.50 * adapt_strength)))
        H_med = min(14, H_med + round(Int, 2 * adapt_strength))
        H_perc = min(24, H_perc + round(Int, 4 * adapt_strength))
    else
        δ_min_effect = max(0.20, δ_min_effect * (1.0 - 0.20 * adapt_strength))
        N_search = min(81, round(Int, N_search * (1.0 + 0.50 * adapt_strength)))
    end

    return ConfiguratorState(
        φ_perc = PercConfig(
            config.φ_perc.signal_mask,
            H_perc,
            config.φ_perc.δ_anomaly
        ),
        φ_world = WorldConfig(
            config.φ_world.H_short,
            H_med,
            N_roll
        ),
        φ_cost = CostConfig(
            config.φ_cost.weights,
            config.φ_cost.thresholds,
            H_burn,
            ε_burn,
            config.φ_cost.γ_discount
        ),
        φ_act = ActConfig(
            Δ_max,
            δ_min_effect,
            α_cvar,
            N_search
        ),
        last_update_day = meta.current_day
    )
end

# ─────────────────────────────────────────────────────────────────
# Rule-based adaptation
# Each parameter derived from system constraints.
# Each rule has explicit mathematical justification.
# ─────────────────────────────────────────────────────────────────

function adapt_rule_based(
    config :: ConfiguratorState,
    meta   :: MetaState,
    prefs  :: UserPreferences
) :: ConfiguratorState

    α_cvar = config.φ_act.α_cvar

    # ── Δ_max: mathematically derived from Section 9 ──────────────
    Δ_max = compute_delta_max(
        prefs,
        meta.trust_level,
        meta.win_rate,
        meta.drift_detected,
        meta.n_records
    )

    # ── N_roll: derived from CVaR statistical requirements ────────
    # Base: enough for meaningful CVaR
    N_roll_base = min_n_roll(α_cvar)

    # Scale up when belief is uncertain — more rollouts needed
    # to characterize the distribution of outcomes
    # Justification: Var(CVaR) ∝ 1/N, so cap the entropy multiplier at 2x
    entropy_ratio  = clamp(meta.belief_entropy, 0.0, 1.0)
    N_roll = round(Int, clamp(
        N_roll_base * (1.0 + entropy_ratio),
        N_roll_base,
        150   # computational budget ceiling
    ))

    # During drift: more rollouts to characterize uncertainty
    meta.drift_detected && (N_roll = min(150, round(Int, N_roll * 1.5)))

    # After safety violation: maximum rollouts
    meta.safety_violations > 0 && (N_roll = 100)

    # ── H_med: derived from dynamics timescales ───────────────────
    # Base horizon from config
    H_med_base = config.φ_world.H_med

    if meta.drift_detected
        # cap at drift timescale — don't simulate beyond known regime
        H_med = max_h_med_during_drift(meta.days_since_drift)
    elseif meta.consecutive_days >= 30 && meta.win_rate >= 0.65
        # proven stability — can extend horizon to see longer dynamics
        H_med = min(14, H_med_base + 1)
    else
        H_med = H_med_base
    end

    # ── H_perc: lookback window ───────────────────────────────────
    # During drift: look further back to understand the regime change
    H_perc = meta.drift_detected ?
             min(24, config.φ_perc.H_perc + 6) :
             config.φ_perc.H_perc

    # ── δ_anomaly: anomaly detection sensitivity ──────────────────
    # Tighter during drift — we want to catch signals early
    # Looser during stable periods — reduce false positives
    δ_anomaly = meta.drift_detected ? 2.0 :
                meta.consecutive_days >= 14 ? 3.0 :
                2.5   # default

    # ── Cost weights: burnout priority scaling ────────────────────
    # Scale burnout weight by proximity to burnout threshold
    # Justification: convex penalty approaching threshold
    B_thresh = 0.70
    burnout_urgency = clamp(meta.burnout_level / B_thresh, 0.0, 1.0)
    burnout_weight_scale = 1.0 + 2.0 * burnout_urgency^2

    new_weights = CostWeights(
        config.φ_cost.weights.c_burden  * burnout_weight_scale,
        config.φ_cost.weights.c_trust,
        config.φ_cost.weights.c_burnout * burnout_weight_scale,
        config.φ_cost.weights.γ_β,
        config.φ_cost.weights.physical
    )

    # ── ε_burn: burnout risk tolerance ───────────────────────────
    # Tighter when already close to burnout
    ε_burn = clamp(
        config.φ_cost.ε_burn * (1.0 - 0.5 * burnout_urgency),
        0.02,
        0.10
    )

    # ── Safety override: always apply after violation ─────────────
    if meta.safety_violations > 0
        Δ_max  = 0.02   # hard floor — absolute minimum
        N_roll = 100
        H_med  = H_MED_MIN
    end

    ConfiguratorState(
        φ_perc = PercConfig(
            config.φ_perc.signal_mask,
            H_perc,
            δ_anomaly
        ),
        φ_world = WorldConfig(
            config.φ_world.H_short,
            H_med,
            N_roll
        ),
        φ_cost = CostConfig(
            new_weights,
            config.φ_cost.thresholds,
            config.φ_cost.H_burn,
            ε_burn,
            config.φ_cost.γ_discount
        ),
        φ_act = ActConfig(
            Δ_max,
            config.φ_act.δ_min_effect,
            α_cvar,
            config.φ_act.N_search
        ),
        last_update_day = meta.current_day
    )
end

# ─────────────────────────────────────────────────────────────────
# Drift detection
# Statistical test: is current entropy significantly higher
# than recent history? Uses z-score with adaptive threshold.
# ─────────────────────────────────────────────────────────────────

function _detect_drift(
    mem         :: MemoryBuffer,
    belief      :: AbstractBeliefState,
    current_day :: Int,
    window      :: Int = 7,
    threshold   :: Float64 = 2.5
) :: Tuple{Bool, Int}

    recent = _recent_records(mem, window)
    isempty(recent) && return false, 0

    recent_entropies = [r.belief_entropy for r in recent]
    μ = mean(recent_entropies)
    σ = std(recent_entropies) + 1e-6

    z = (belief.entropy - μ) / σ
    drift_detected = z > threshold

    if drift_detected
        days_since = current_day - maximum(r.day for r in recent)
        return true, days_since
    end

    return false, 0
end
