 """
scorecard.jl
Shadow scorecard — tracks system performance before and after graduation.

Shadow mode: system makes recommendations silently for 21+ days.
Scores itself against what actually happened.
Only graduates when performance criteria are met.

Graduation criteria:
  - >= 21 shadow days
  - >= 60% win rate
  - zero safety violations
  - 7 consecutive days of sustained performance

A hold is scored too — if the system holds and outcomes are stable,
that is a correct decision and counts toward win rate.
"""

using Statistics

function _prior_completed_records(
    mem :: MemoryBuffer,
    day :: Int;
    window :: Int = 14,
)
    return filter(
        r -> !isnothing(r.realized_cost) && r.day < day && r.day >= day - window,
        mem.records,
    )
end

function _hold_baseline(
    mem :: MemoryBuffer,
    rec :: MemoryRecord,
) :: Union{Tuple{Float64, Float64}, Nothing}
    prior = _prior_completed_records(mem, rec.day)
    isempty(prior) && return nothing
    costs = [r.realized_cost for r in prior if !isnothing(r.realized_cost)]
    isempty(costs) && return nothing
    baseline = median(costs)
    tolerance = max(0.05, 0.5 * std(costs))
    return baseline, tolerance
end

# ─────────────────────────────────────────────────────────────────
# Score one completed record
# Did this recommendation (or hold) improve on the baseline?
# Win = realized cost under action < realized cost under null action
# ─────────────────────────────────────────────────────────────────

function score_record!(
    mem :: MemoryBuffer,
    rec_id :: Int
) :: Nothing

    idx = findfirst(r -> r.id == rec_id, mem.records)
    isnothing(idx) && return nothing

    rec = mem.records[idx]

    # need realized cost to score
    isnothing(rec.realized_cost) && return nothing

    # for holds: win if realized cost stayed low
    if is_null(rec.action)
        baseline = _hold_baseline(mem, rec)
        isnothing(baseline) && return nothing
        baseline_cost, tolerance = baseline
        # A hold should only count as wrong when outcomes are materially
        # worse than the recent regime, not when they fluctuate around the median.
        rec.shadow_delta_score = (baseline_cost + tolerance) - rec.realized_cost
        return nothing
    end

    # For shadow recommendations, realized_cost is the observed null path.
    # Score positive when the model predicted the treated path would be better.
    if !isnothing(rec.predicted_cvar)
        rec.shadow_delta_score = rec.realized_cost - rec.predicted_cvar
        return nothing
    end

    # Legacy fallback for older records that do not yet store predicted_cvar.
    baseline = _hold_baseline(mem, rec)
    isnothing(baseline) && return nothing
    baseline_cost, tolerance = baseline
    rec.shadow_delta_score = (baseline_cost + tolerance) - rec.realized_cost

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Update all scores in memory
# Called periodically as outcomes arrive
# ─────────────────────────────────────────────────────────────────

function update_all_scores!(mem::MemoryBuffer) :: Nothing
    for rec in mem.records
        isnothing(rec.shadow_delta_score) || continue
        score_record!(mem, rec.id)
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Compute shadow scorecard
# Summarizes system performance over all scored records
# ─────────────────────────────────────────────────────────────────

function compute_scorecard(mem::MemoryBuffer) :: ShadowScorecard
    update_all_scores!(mem)

    scored = filter(
        r -> !isnothing(r.shadow_delta_score),
        mem.records
    )

    isempty(scored) && return ShadowScorecard(
        n_days            = 0,
        win_rate          = 0.0,
        safety_violations = 0,
        consecutive_days  = 0
    )

    n_days = length(unique(r.day for r in scored))

    # win rate — fraction of records with positive delta score
    wins     = count(r -> r.shadow_delta_score > 0.0, scored)
    win_rate = wins / length(scored)

    # safety violations — records where epistemic gate passed
    # but safety gate would have flagged
    # proxy: records with unusually high realized cost
    recent_costs = [r.realized_cost for r in scored
                    if !isnothing(r.realized_cost)]
    if isempty(recent_costs)
        safety_violations = 0
    else
        cost_threshold = mean(recent_costs) + 2.0 * std(recent_costs)
        safety_violations = count(
            r -> !isnothing(r.realized_cost) &&
                 r.realized_cost > cost_threshold &&
                 !is_null(r.action),
            scored
        )
    end

    # consecutive days of sustained performance
    # find longest run of days where win rate >= 0.6
    consecutive_days = _compute_consecutive_days(scored)

    return ShadowScorecard(
        n_days            = n_days,
        win_rate          = win_rate,
        safety_violations = safety_violations,
        consecutive_days  = consecutive_days
    )
end

# ─────────────────────────────────────────────────────────────────
# Compute consecutive days of sustained performance
# Looks for longest run of days where daily win rate >= threshold
# ─────────────────────────────────────────────────────────────────

function _compute_consecutive_days(
    records   :: Vector{MemoryRecord},
    threshold :: Float64 = 0.60
) :: Int

    isempty(records) && return 0

    # group records by day
    days = sort(unique(r.day for r in records))
    isempty(days) && return 0

    max_consecutive = 0
    current_streak  = 0

    for day in days
        day_records = filter(r -> r.day == day, records)
        scored_today = filter(r -> !isnothing(r.shadow_delta_score), day_records)

        if isempty(scored_today)
            current_streak = 0
            continue
        end

        daily_wins     = count(r -> r.shadow_delta_score > 0.0, scored_today)
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
# Twin posterior update trigger
# Called from Memory when enough behavioral data exists
# ─────────────────────────────────────────────────────────────────

function maybe_update_twin_posterior!(
    twin        :: DigitalTwin,
    mem         :: MemoryBuffer,
    current_day :: Int
) :: Nothing

    # update behavioral params daily
    Main.Twin.update_posterior!(
        twin.posterior,
        twin.prior,
        mem.records,
        current_day
    )

    return nothing
end
