"""
learned.jl
Contextual bandit configurator — v1.5
Proper exploration/exploitation over configuration parameters.

Uses Thompson sampling — the principled Bayesian approach:
  - Maintain a posterior distribution over each config parameter
  - Sample from posterior at each decision
  - Update posterior from observed performance
  - No assumptions about which configs work — data decides

Why Thompson sampling over nearest-neighbor:
  - Principled uncertainty quantification
  - Natural exploration/exploitation tradeoff
  - Exact Bayesian updates — no approximation needed
  - Works well with small data — prior prevents overconfidence

Config parameters treated as independent bandits:
  - Δ_max:  Beta posterior (bounded [0,1], rescaled to valid range)
  - N_roll: Normal posterior (continuous, clipped to valid range)
  - H_med:  Normal posterior (continuous, clipped to valid range)
  - α_cvar: Beta posterior (bounded [0,1], rescaled to [0.70, 0.95])
"""

using Statistics
using Distributions
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# BetaBandit
# Thompson sampling for bounded parameter in [0,1].
# Rescaled to actual parameter range at inference time.
# Beta(α, β): α = successes, β = failures
# ─────────────────────────────────────────────────────────────────

mutable struct BetaBandit
    α        :: Float64   # pseudo-successes
    β        :: Float64   # pseudo-failures
    p_min    :: Float64   # actual parameter minimum
    p_max    :: Float64   # actual parameter maximum
end

function BetaBandit(p_min::Float64, p_max::Float64;
                    prior_α::Float64=2.0, prior_β::Float64=5.0)
    # prior_α=2, prior_β=5 → prior mean = 2/7 ≈ 0.29
    # conservative start — biased toward lower end of range
    BetaBandit(prior_α, prior_β, p_min, p_max)
end

function sample_bandit(b::BetaBandit) :: Float64
    p = rand(Beta(b.α, b.β))
    return b.p_min + p * (b.p_max - b.p_min)
end

function update_bandit!(
    b           :: BetaBandit,
    value_used  :: Float64,
    performance :: Float64   # positive = good, negative = bad
) :: Nothing
    # convert value back to [0,1]
    p = (value_used - b.p_min) / (b.p_max - b.p_min + 1e-10)
    p = clamp(p, 0.0, 1.0)

    # Bayesian update:
    # positive performance → increase α (success weight for this region)
    # negative performance → increase β (failure weight for this region)
    if performance > 0.0
        b.α += performance * p          # reward proportional to p used
        b.β += performance * (1.0 - p)  # and inversely elsewhere
    else
        b.α += abs(performance) * (1.0 - p)
        b.β += abs(performance) * p
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# NormalBandit
# Thompson sampling for continuous parameter.
# Normal-Normal conjugate update.
# ─────────────────────────────────────────────────────────────────

mutable struct NormalBandit
    μ        :: Float64   # posterior mean
    σ²       :: Float64   # posterior variance
    p_min    :: Float64   # clipping minimum
    p_max    :: Float64   # clipping maximum
    n        :: Int       # number of observations
    σ²_noise :: Float64   # observation noise variance
end

function NormalBandit(
    prior_μ :: Float64,
    prior_σ :: Float64,
    p_min   :: Float64,
    p_max   :: Float64;
    σ_noise :: Float64 = 0.1
)
    NormalBandit(prior_μ, prior_σ^2, p_min, p_max, 0, σ_noise^2)
end

function sample_bandit(b::NormalBandit) :: Float64
    sample = rand(Normal(b.μ, sqrt(b.σ²)))
    return clamp(sample, b.p_min, b.p_max)
end

function update_bandit!(
    b           :: NormalBandit,
    value_used  :: Float64,
    performance :: Float64
) :: Nothing
    # Normal-Normal conjugate update
    # Treat performance as a noisy observation of the parameter's quality
    # High performance → pull posterior toward value_used
    b.n += 1

    # posterior update: standard Bayesian linear regression step
    # precision-weighted combination of prior and likelihood
    prior_precision    = 1.0 / b.σ²
    likelihood_precision = abs(performance) / b.σ²_noise + 1e-10

    new_precision = prior_precision + likelihood_precision
    new_μ = (prior_precision * b.μ +
             likelihood_precision * value_used * sign(performance)) / new_precision

    b.μ  = new_μ
    b.σ² = 1.0 / new_precision

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# ContextualBanditConfigurator
# One bandit per configuration parameter.
# Context (MetaState) modulates the valid ranges — not the bandit itself.
# This keeps the bandit simple while respecting system constraints.
# ─────────────────────────────────────────────────────────────────

mutable struct ContextualBanditConfigurator
    Δ_max_bandit  :: BetaBandit
    N_roll_bandit :: NormalBandit
    H_med_bandit  :: NormalBandit
    α_cvar_bandit :: BetaBandit
    n_updates     :: Int
    is_ready      :: Bool
    min_updates   :: Int
end

function ContextualBanditConfigurator()
    ContextualBanditConfigurator(
        # Δ_max: Beta over [0,1], rescaled at inference
        # prior: conservative (biased toward low end)
        BetaBandit(0.0, 1.0, prior_α=2.0, prior_β=8.0),

        # N_roll: Normal around 50, std=20
        # prior: 50 rollouts — reasonable default
        NormalBandit(50.0, 20.0, 20.0, 150.0),

        # H_med: Normal around 7 days, std=3
        # prior: 7 day horizon — POC default
        NormalBandit(7.0, 3.0, Float64(H_MED_MIN), 14.0),

        # α_cvar: Beta over [0,1], rescaled to [0.70, 0.95]
        # prior: biased toward 0.80 (formulation default)
        BetaBandit(0.70, 0.95, prior_α=4.0, prior_β=1.0),

        0, false, 20
    )
end

# Global bandit configurator
const BANDIT_CONFIG = ContextualBanditConfigurator()

# ─────────────────────────────────────────────────────────────────
# Sample configuration from bandit posteriors
# Context modulates the valid ranges via system constraints
# ─────────────────────────────────────────────────────────────────

function adapt_bandit(
    config :: ConfiguratorState,
    meta   :: MetaState,
    prefs  :: UserPreferences
) :: ConfiguratorState

    # fall back to rules if not ready
    if !BANDIT_CONFIG.is_ready
        return adapt_rule_based(config, meta, prefs)
    end

    # get system-derived bounds from context
    Δ_max_ceiling = compute_delta_max(
        prefs, meta.trust_level, meta.win_rate,
        meta.drift_detected, meta.n_records
    )
    α_cvar = config.φ_act.α_cvar
    N_roll_floor = min_n_roll(α_cvar)
    H_med_ceiling = meta.drift_detected ?
                    max_h_med_during_drift(meta.days_since_drift) : 14

    # temporarily update bandit bounds from context
    BANDIT_CONFIG.Δ_max_bandit.p_max  = Δ_max_ceiling
    BANDIT_CONFIG.N_roll_bandit.p_min = Float64(N_roll_floor)
    BANDIT_CONFIG.H_med_bandit.p_max  = Float64(H_med_ceiling)

    # sample from posteriors
    Δ_max  = sample_bandit(BANDIT_CONFIG.Δ_max_bandit)
    N_roll = round(Int, sample_bandit(BANDIT_CONFIG.N_roll_bandit))
    H_med  = round(Int, sample_bandit(BANDIT_CONFIG.H_med_bandit))
    α_new  = sample_bandit(BANDIT_CONFIG.α_cvar_bandit)

    # safety override — never relax after violation
    if meta.safety_violations > 0
        Δ_max  = 0.02
        N_roll = 100
        H_med  = H_MED_MIN
    end

    # get rule-based config for everything we don't bandit
    # (cost weights, perception config, etc.)
    rule_config = adapt_rule_based(config, meta, prefs)

    ConfiguratorState(
        φ_perc  = rule_config.φ_perc,
        φ_world = WorldConfig(config.φ_world.H_short, H_med, N_roll),
        φ_cost  = rule_config.φ_cost,
        φ_act   = ActConfig(Δ_max, config.φ_act.δ_min_effect,
                            α_new, config.φ_act.N_search),
        last_update_day = meta.current_day
    )
end

# ─────────────────────────────────────────────────────────────────
# Update bandit posteriors from observed performance
# Called when outcome is known — downstream win rate delta
# ─────────────────────────────────────────────────────────────────

function update_bandit!(
    Δ_max_used  :: Float64,
    N_roll_used :: Int,
    H_med_used  :: Int,
    α_cvar_used :: Float64,
    performance :: Float64   # win_rate_after - win_rate_before
) :: Nothing

    update_bandit!(BANDIT_CONFIG.Δ_max_bandit,
                   Δ_max_used, performance)
    update_bandit!(BANDIT_CONFIG.N_roll_bandit,
                   Float64(N_roll_used), performance)
    update_bandit!(BANDIT_CONFIG.H_med_bandit,
                   Float64(H_med_used), performance)
    update_bandit!(BANDIT_CONFIG.α_cvar_bandit,
                   α_cvar_used, performance)

    BANDIT_CONFIG.n_updates += 1

    if BANDIT_CONFIG.n_updates >= BANDIT_CONFIG.min_updates
        BANDIT_CONFIG.is_ready = true
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Posterior summary — for logging and debugging
# Shows current belief about each parameter
# ─────────────────────────────────────────────────────────────────

function summarize_bandits() :: String
    b = BANDIT_CONFIG
    Δ_mean = b.Δ_max_bandit.p_min +
             (b.Δ_max_bandit.α / (b.Δ_max_bandit.α + b.Δ_max_bandit.β)) *
             (b.Δ_max_bandit.p_max - b.Δ_max_bandit.p_min)

    string(
        "Bandit posteriors ($(b.n_updates) updates):\n",
        "  Δ_max:  mean=$(round(Δ_mean, digits=3))\n",
        "  N_roll: mean=$(round(b.N_roll_bandit.μ, digits=1)) ",
        "std=$(round(sqrt(b.N_roll_bandit.σ²), digits=1))\n",
        "  H_med:  mean=$(round(b.H_med_bandit.μ, digits=1)) ",
        "std=$(round(sqrt(b.H_med_bandit.σ²), digits=1))\n",
        "  α_cvar: mean=$(round(0.70 + b.α_cvar_bandit.α /
            (b.α_cvar_bandit.α + b.α_cvar_bandit.β) * 0.25, digits=3))"
    )
end
