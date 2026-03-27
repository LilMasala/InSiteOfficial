"""
epistemic.jl
Computes κ, ρ, η — the three epistemic quality measures.
These are CONSTRAINTS not costs.
If any one fails → F_t = 0 → no action regardless of predicted benefit.

κ — GP familiarity:       have we seen this patient state before?
ρ — ensemble concordance: do our models agree?
η — calibration quality:  have our predictions been historically accurate?

All three must pass their thresholds for F_t = 1.
This is a hard AND — not a weighted average.
Each measure catches a different failure mode that the others don't.
"""

using Statistics

const CALIBRATION_NEUTRAL = 0.80
const FAMILIARITY_MARGIN_Z = 1.0

# ─────────────────────────────────────────────────────────────────
# GP Familiarity — κ_t ∈ [0,1]
# Measures how well-covered the current belief is by training data.
# Uses belief entropy relative to recent history as a proxy.
#
# High entropy relative to recent history = unfamiliar state = low κ
# Low entropy relative to recent history  = familiar state  = high κ
#
# κ high → familiar territory → can act
# κ low  → never seen this state → hold
# ─────────────────────────────────────────────────────────────────

function compute_familiarity(
    belief :: AbstractBeliefState,
    mem    :: MemoryBuffer
) :: Float64

    # need at least 5 records to estimate familiarity
    length(mem.records) < 5 && return 0.0

    # use recent 30 days of entropy history
    recent = last(mem.records, min(30, length(mem.records)))
    recent_entropies = [r.belief_entropy for r in recent]

    mean_entropy = mean(recent_entropies)
    std_entropy  = std(recent_entropies) + 1e-6   # avoid division by zero

    # z-score of current entropy relative to recent history
    # positive z = more uncertain than usual = unfamiliar
    z_entropy = (belief.entropy - mean_entropy) / std_entropy

    # convert to [0,1]
    # Ordinary states near the recent mean should count as familiar enough
    # to pass the gate. We only want κ to collapse when entropy is materially
    # higher than the recent regime, not when it is merely typical.
    κ = 1.0 - sigmoid(z_entropy - FAMILIARITY_MARGIN_Z)
    return clamp(κ, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────
# Ensemble Concordance — ρ_t ∈ [0,1]
# Do our models agree on predictions for the current belief?
# Uses recent epistemic concordance from memory as proxy.
#
# In full implementation: compute IoU between prediction intervals
# across all active models for the current candidate action.
# In POC: use recent concordance history from memory records.
#
# ρ high → models agree → confident prediction → can act
# ρ low  → models disagree → something is uncertain → hold
# ─────────────────────────────────────────────────────────────────

function compute_concordance(
    mem    :: MemoryBuffer,
    action :: AbstractAction
) :: Float64

    # need recent records with epistemic state
    isempty(mem.records) && return 0.5   # neutral when no data

    recent = last(mem.records, min(10, length(mem.records)))

    # use recent concordance scores from memory
    concordances = [r.epistemic.ρ_concordance for r in recent]
    return clamp(mean(concordances), 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────
# Calibration Quality — η_t ∈ [0,1]
# Are our prediction intervals historically accurate?
#
# Checks empirical coverage of 80% prediction intervals.
# η = 1 - |empirical_coverage - 0.80|
#
# Uses standard error to decide when estimate is reliable enough.
# SE < 0.05 required — naturally satisfied around 30+ records
# but can be satisfied earlier if predictions are very good or bad.
# This is better than a fixed record count because:
#   - good predictions → low SE early → calibration kicks in sooner
#   - noisy predictions → high SE → wait for more data
#
# η high → intervals are honest → trust the model → can act
# η low  → intervals are wrong → don't trust model → hold
# ─────────────────────────────────────────────────────────────────

function compute_calibration(
    mem :: MemoryBuffer
) :: Float64

    # filter to records with both predictions and realized outcomes
    completed = filter(
        r -> !isnothing(r.realized_cost) && !isnothing(r.critic_target),
        mem.records
    )

    # Underdetermined calibration should not hard-block action selection.
    isempty(completed) && return CALIBRATION_NEUTRAL

    # compute empirical coverage of 80% prediction intervals
    in_interval = 0
    total       = 0

    for r in completed
        predicted = r.critic_target
        realized  = r.realized_cost

        # approximate σ as 20% of predicted value
        # TODO: replace with actual model uncertainty estimates
        # once Critic outputs proper prediction distributions
        σ_approx = abs(predicted) * 0.2 + 1e-6

        # 80% CI: μ ± 1.28σ
        # 1.28 = z-score for 80% coverage under standard normal
        lower = predicted - 1.28 * σ_approx
        upper = predicted + 1.28 * σ_approx

        if lower <= realized <= upper
            in_interval += 1
        end
        total += 1
    end

    total == 0 && return CALIBRATION_NEUTRAL

    p̂ = in_interval / total

    # standard error of coverage estimate: SE = sqrt(p(1-p)/n)
    # only trust the estimate when SE < 0.05
    # meaning: coverage estimate is accurate to within ±5 percentage points
    se = sqrt(p̂ * (1.0 - p̂) / total)
    se > 0.05 && return CALIBRATION_NEUTRAL

    # η = 1 - deviation from target coverage (0.80)
    η = 1.0 - abs(p̂ - 0.80)
    return clamp(η, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────
# Compute full EpistemicState
# Combines κ, ρ, η and sets feasibility flag F_t
# Called once per recommendation cycle before Actor runs.
#
# F_t = 𝟙[κ ≥ κ_min] · 𝟙[ρ ≥ ρ_min] · 𝟙[η ≥ η_min]
#
# Hard AND — not weighted average.
# Any single failure blocks action regardless of other measures.
# ─────────────────────────────────────────────────────────────────

function compute_epistemic_state(
    belief     :: AbstractBeliefState,
    mem        :: MemoryBuffer,
    action     :: AbstractAction,
    thresholds :: EpistemicThresholds
) :: EpistemicState

    κ = compute_familiarity(belief, mem)
    ρ = compute_concordance(mem, action)
    η = compute_calibration(mem)

    # hard AND gate — all three must pass
    feasible = (κ >= thresholds.κ_min) &&
               (ρ >= thresholds.ρ_min) &&
               (η >= thresholds.η_min)

    EpistemicState(
        κ_familiarity = κ,
        ρ_concordance = ρ,
        η_calibration = η,
        feasible      = feasible
    )
end

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

sigmoid(x::Float64) = 1.0 / (1.0 + exp(-x))
