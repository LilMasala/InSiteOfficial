"""
behavioral.jl
Psychological state dynamics during rollouts.
These are Chamelia's hardwired math — not the simulator's responsibility.

Implements Section 6 of the formulation:
  τ_{t+1} = trust dynamics
  B_{t+1} = burnout dynamics  
  ω_{t+1} = engagement dynamics
  β_{t+1} = burden accumulation
"""

using Statistics

# ─────────────────────────────────────────────────────────────────
# Trust Dynamics
# τ_{t+1} = clip[τ_t + κ⁺·q⁺·𝟙[accept] - κ⁻·q⁻·𝟙[accept] - λ·d(a,a⁰), 0, 1]
#
# Trust builds slowly from good outcomes after acceptance.
# Trust decays faster from bad outcomes or large surprising actions.
# Asymmetry is crucial — one bad rec can undo weeks of trust building.
# ─────────────────────────────────────────────────────────────────

function update_trust(
    τ          :: Float64,          # current trust ∈ [0,1]
    action     :: AbstractAction,   # proposed action
    response   :: UserResponse,     # did patient accept?
    outcome_quality :: Float64,     # glycemic improvement (+ good, - bad)
    posterior  :: TwinPosterior,    # patient-specific trust rates
    ξ          :: Dict{Symbol, Float64}  # noise
) :: Float64

    # relative action magnitude — how surprising is this change?
    d = magnitude(action)

    # trust gain — only from accepted recommendations with good outcomes
    q_plus  = max(0.0, outcome_quality)   # glycemic improvement
    q_minus = max(0.0, -outcome_quality)  # glycemic deterioration

    Δτ = 0.0
    if response == Accept
        Δτ += posterior.trust_growth_rate * q_plus
        Δτ -= posterior.trust_decay_rate  * q_minus
    end

    # trust cost of large action — independent of acceptance
    Δτ -= posterior.trust_decay_rate * 0.5 * d

    # add noise
    Δτ += get(ξ, :trust, 0.0)

    return clamp(τ + Δτ, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────
# Burden Accumulation
# β_{t+1} = γ_β · β_t + 𝟙[a ≠ a⁰] · magnitude(a)
# Decaying sum of recent recommendations.
# Half-life ≈ 13.5 days at γ_β = 0.95
# ─────────────────────────────────────────────────────────────────

function update_burden(
    β      :: Float64,          # current burden ≥ 0
    action :: AbstractAction,   # proposed action
    γ_β    :: Float64 = 0.95    # decay rate — configurable
) :: Float64

    # decay existing burden
    β_new = γ_β * β

    # add new burden if recommendation was made
    if !is_null(action)
        β_new += magnitude(action)
    end

    return max(0.0, β_new)
end

# ─────────────────────────────────────────────────────────────────
# Burnout Dynamics
# B_{t+1} = σ(σ⁻¹(B_t) + ΔB^endo + ΔB^policy + η^B)
# Additive in log-odds space — keeps B_t ∈ [0,1] without clipping
#
# Endogenous: would occur even without system recommendations
# Policy-induced: attributable to system's actions
# ─────────────────────────────────────────────────────────────────

function update_burnout(
    B           :: Float64,         # current burnout ∈ [0,1]
    β           :: Float64,         # current burden
    τ           :: Float64,         # current trust
    ω           :: Float64,         # current engagement
    action      :: AbstractAction,
    response    :: UserResponse,
    frustration :: Float64,    # clip(%low_7d + 0.5*(1-TIR_7d), 0, 1)
    sleep_debt  :: Float64,         # days of accumulated sleep deficit
    stress      :: Float64,         # acute stress [0,1]
    posterior   :: TwinPosterior,
    ξ           :: Dict{Symbol, Float64}
) :: Float64

    # work in log-odds space
    logit_B = log(B / (1.0 - B + 1e-8) + 1e-8)

    # ── Endogenous drivers (would happen without system) ──────────
    ΔB_endo = 0.08 * frustration +
              0.05 * sleep_debt +
              0.06 * stress +
              0.04 * (1.0 - posterior.burnout_sensitivity)

    # ── Policy-induced drivers (attributable to system) ───────────
    bad_outcome_after_accept = (response == Accept && frustration > 0.5) ? 1.0 : 0.0
    cognitive_load = magnitude(action) * 2.0   # larger changes = more cognitive load

    ΔB_policy = 0.07 * β +
                0.10 * bad_outcome_after_accept +
                0.05 * cognitive_load

    # ── Noise ─────────────────────────────────────────────────────
    η_B = get(ξ, :burnout, 0.0)

    # update in log-odds space then map back to [0,1]
    logit_B_new = logit_B + ΔB_endo + ΔB_policy + η_B
    return 1.0 / (1.0 + exp(-logit_B_new))
end

# ─────────────────────────────────────────────────────────────────
# Engagement Dynamics
# ω_{t+1} = clip[ω_t - decay·B_t + recovery·τ_t + noise, 0, 1]
# Decays with burnout, recovers slowly with trust
# ─────────────────────────────────────────────────────────────────

function update_engagement(
    ω         :: Float64,   # current engagement ∈ [0,1]
    B         :: Float64,   # current burnout
    τ         :: Float64,   # current trust
    posterior :: TwinPosterior,
    ξ         :: Dict{Symbol, Float64}
) :: Float64

    Δω = -posterior.engagement_decay * B +    # burnout kills engagement
          0.02 * τ +                           # trust slowly recovers engagement
          get(ξ, :engagement, 0.0)             # noise

    return clamp(ω + Δω, 0.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────
# Physiological Frustration
# A domain-specific signal that feeds into burnout dynamics.
# The simulator registers what "frustration" means for its domain.
# InSite: glycemic frustration = %low + 0.5*(1-TIR)
# Cardiac: could be symptom burden, medication side effects, etc.
# ─────────────────────────────────────────────────────────────────

"""
    compute_frustration(sim, signals) → Float64

Domain-specific frustration signal ∈ [0,1].
Must be registered by the simulator plugin.
Higher = more frustrated = higher burnout risk.
"""
function compute_frustration(
    sim     :: AbstractSimulator,
    signals :: Dict{Symbol, Any}
) :: Float64
    error("$(typeof(sim)) must implement compute_frustration!")
end

# Mock implementation — neutral frustration
function compute_frustration(
    sim     :: MockSimulator,
    signals :: Dict{Symbol, Any}
) :: Float64
    return 0.0   # mock has no frustration model
end
# ─────────────────────────────────────────────────────────────────
# Full psychological state update — one timestep
# Combines all four dynamics into one call
# Called by rollout.jl for each step of each rollout
# ─────────────────────────────────────────────────────────────────

function update_psy_state(
    psy              :: PsyState,
    action           :: AbstractAction,
    response         :: UserResponse,
    outcome_quality  :: Float64,
    frustration :: Float64,
    sleep_debt       :: Float64,
    stress           :: Float64,
    posterior        :: TwinPosterior,
    ξ                :: Dict{Symbol, Float64},
    γ_β              :: Float64 = 0.95
) :: PsyState

    τ = psy.trust.value
    B = psy.burnout.value
    ω = psy.engagement.value
    β = psy.burden.value

    # update in correct order — burden before burnout, trust before engagement
    β_new = update_burden(β, action, γ_β)
    τ_new = update_trust(τ, action, response, outcome_quality, posterior, ξ)
    B_new = update_burnout(B, β_new, τ_new, ω, action, response,
                           frustration, sleep_debt, stress, posterior, ξ)
    ω_new = update_engagement(ω, B_new, τ_new, posterior, ξ)

    PsyState(
        trust      = ScalarTrust(τ_new),
        burden     = ScalarBurden(β_new),
        engagement = ScalarEngagement(ω_new),
        burnout    = ScalarBurnout(B_new)
    )
end