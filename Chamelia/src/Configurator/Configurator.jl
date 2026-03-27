"""
Configurator.jl
Configurator module — meta-policy over all other modules.

Three implementations, running simultaneously:
  v1.1 → RuleBasedConfigurator    (mathematically derived rules)
  v1.5 → LearnedRuleConfigurator  (contextual bandit, Thompson sampling)
  v2.0 → MetaPolicyConfigurator   (offline RL, CQL)

Same pattern as belief estimators — epistemic layer decides which to trust.
All three run every cycle. The one with highest demonstrated reliability leads.

The Configurator is the only module that reads from ALL other modules.
It is the system's self-awareness — knowing when it knows enough to act.
"""

include("../types.jl")

module Configurator

using Statistics
using Distributions
using LinearAlgebra
using Flux

using Main: AbstractBeliefState, AbstractDomainAdapter, PsyState,
            MemoryBuffer, ConfiguratorState,
            PercConfig, WorldConfig, CostConfig, ActConfig,
            CostWeights, EpistemicThresholds, EpistemicState,
            UserPreferences, default_physical_weights, is_null, META_FEATURE_DIM

include("preferences.jl")
include("rules.jl")
include("learned.jl")
include("meta_policy.jl")

# ─────────────────────────────────────────────────────────────────
# Competence tracking
# Per-mode win rates from memory — basis for routing decisions.
# Richer modes must earn their position through demonstrated performance,
# not just by being "ready". Each mode is only preferred over rules when
# it has enough samples and its win rate is within tolerance of rules.
# ─────────────────────────────────────────────────────────────────

const MODE_COMPETENCE_MIN_SAMPLES = 15   # records per mode before judgment
const MODE_COMPETENCE_TOLERANCE   = 0.05 # richer mode may trail rules by at most 5pp

# Tracks the mode actually selected in the most recent routing decision.
# Updated by _route_adaptation; read by active_mode() and store calls.
const _LAST_ACTIVE_MODE = Ref{Symbol}(:rules)

function _mode_sample_counts(mem :: MemoryBuffer) :: Dict{Symbol, Int}
    counts = Dict{Symbol, Int}()
    for rec in mem.records
        isnothing(rec.shadow_delta_score) && continue
        mode = rec.configurator_mode
        counts[mode] = get(counts, mode, 0) + 1
    end
    return counts
end

function _mode_win_rates(mem :: MemoryBuffer) :: Dict{Symbol, Float64}
    wins   = Dict{Symbol, Int}()
    counts = Dict{Symbol, Int}()
    for rec in mem.records
        isnothing(rec.shadow_delta_score) && continue
        mode = rec.configurator_mode
        counts[mode] = get(counts, mode, 0) + 1
        rec.shadow_delta_score > 0.0 && (wins[mode] = get(wins, mode, 0) + 1)
    end
    return Dict(mode => get(wins, mode, 0) / n
                for (mode, n) in counts if n > 0)
end

# ─────────────────────────────────────────────────────────────────
# Main adaptation entry point
# Computes meta-state, routes to appropriate configurator,
# records experience for learning.
# ─────────────────────────────────────────────────────────────────

function adapt(
    config      :: ConfiguratorState,
    belief      :: AbstractBeliefState,
    epistemic   :: EpistemicState,
    mem         :: MemoryBuffer,
    psy         :: PsyState,
    prefs       :: UserPreferences,
    current_day :: Int;
    graduated   :: Bool = false,
    last_decision_reason :: Symbol = :initialized,
) :: ConfiguratorState

    # only update at configured cadence
    days_since_update = current_day - config.last_update_day
    days_since_update < 1 && return config

    # compute meta-state — the system's self-assessment
    meta = compute_meta_state(
        belief,
        epistemic,
        mem,
        psy,
        current_day;
        graduated=graduated,
        last_decision_reason=last_decision_reason,
    )

    # route to appropriate configurator based on data availability
    new_config = _route_adaptation(config, meta, prefs, mem)
    return _apply_postgrad_no_surface_adaptation(new_config, meta)
end

# ─────────────────────────────────────────────────────────────────
# Route to correct configurator
# CQL if ready → Bandit if ready → Rules always
# Multiple dispatch on configurator model type
# ─────────────────────────────────────────────────────────────────

function _route_adaptation(
    config :: ConfiguratorState,
    meta   :: MetaState,
    prefs  :: UserPreferences,
    mem    :: MemoryBuffer
) :: ConfiguratorState

    win_rates  = _mode_win_rates(mem)
    counts     = _mode_sample_counts(mem)
    rules_rate = get(win_rates, :rules, nothing)

    # ── CQL: use if ready and competence-verified ─────────────────
    if Q_NET.is_ready
        cql_n    = get(counts, :cql, 0)
        cql_rate = get(win_rates, :cql, nothing)

        # give benefit of the doubt until enough CQL records exist
        cql_competent = cql_n < MODE_COMPETENCE_MIN_SAMPLES ||
                        isnothing(rules_rate)               ||
                        something(cql_rate, 0.0) >= rules_rate - MODE_COMPETENCE_TOLERANCE

        if cql_competent
            _LAST_ACTIVE_MODE[] = :cql
            return adapt_cql(config, meta, prefs)
        end

        @info "[Configurator] CQL underperforming rules (cql=$(round(something(cql_rate,0.0),digits=2)), rules=$(round(rules_rate,digits=2))); downgrading"
    end

    # ── Bandit: use if ready and competence-verified ──────────────
    if BANDIT_CONFIG.is_ready
        bandit_n    = get(counts, :bandit, 0)
        bandit_rate = get(win_rates, :bandit, nothing)

        bandit_competent = bandit_n < MODE_COMPETENCE_MIN_SAMPLES ||
                           isnothing(rules_rate)                  ||
                           something(bandit_rate, 0.0) >= rules_rate - MODE_COMPETENCE_TOLERANCE

        if bandit_competent
            _LAST_ACTIVE_MODE[] = :bandit
            return adapt_bandit(config, meta, prefs)
        end

        @info "[Configurator] Bandit underperforming rules (bandit=$(round(something(bandit_rate,0.0),digits=2)), rules=$(round(rules_rate,digits=2))); falling back"
    end

    # ── Rules: always available, the floor ───────────────────────
    _LAST_ACTIVE_MODE[] = :rules
    return adapt_rule_based(config, meta, prefs)
end

# ─────────────────────────────────────────────────────────────────
# Record configuration experience
# Called after outcome is known — what was the performance?
# Feeds both bandit and CQL learning pipelines
# ─────────────────────────────────────────────────────────────────

function record_outcome!(
    config      :: ConfiguratorState,
    meta        :: MetaState,
    performance :: Float64,
    _prefs      :: UserPreferences
) :: Nothing

    # update bandit posteriors
    update_bandit!(
        config.φ_act.Δ_max,
        config.φ_world.N_roll,
        config.φ_world.H_med,
        config.φ_act.α_cvar,
        performance
    )

    # CQL history is provided explicitly to maybe_train_cql!;
    # this hook only updates the online bandit state.
    _ = meta

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Maybe train CQL
# Triggered when enough experiences have accumulated
# ─────────────────────────────────────────────────────────────────

function maybe_train_cql!(
    meta_history        :: Vector{MetaState},
    config_history      :: Vector{ConfiguratorState},
    performance_history :: Vector{Float64},
    prefs               :: UserPreferences,
    current_day         :: Int
) :: Nothing

    length(meta_history) < Q_NET.min_samples && return nothing

    # only retrain periodically — expensive
    days_since_train = current_day - Q_NET.last_trained
    days_since_train < 7 && Q_NET.is_ready && return nothing

    train_cql!(
        current_day,
        meta_history,
        config_history,
        performance_history,
        prefs
    )

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Configurator status summary
# Shows which configurator is active and why
# ─────────────────────────────────────────────────────────────────

function status() :: String
    active_sym = _LAST_ACTIVE_MODE[]
    active_str = active_sym == :cql    ? "CQL (v2.0)"    :
                 active_sym == :bandit ? "Bandit (v1.5)" :
                                         "Rules (v1.1)"

    string(
        "Configurator Status:\n",
        "  Active: $active_str\n",
        "  Bandit updates: $(BANDIT_CONFIG.n_updates)\n",
        "  CQL samples: $(Q_NET.n_trained)\n",
        Q_NET.is_ready ? "" : BANDIT_CONFIG.is_ready ?
            summarize_bandits() * "\n" : ""
    )
end

function active_mode() :: Symbol
    return _LAST_ACTIVE_MODE[]
end

# ─────────────────────────────────────────────────────────────────
# Mode competence diagnostics
# Structured summary of which configurator and belief modes are
# performing best — used for logging and surfacing in status reports.
# ─────────────────────────────────────────────────────────────────

"""
    mode_competence_diagnostics(mem) → NamedTuple

Returns per-mode win rates, sample counts, active mode, and belief-mode
(JEPA vs explicit rollout) diagnostics derived from memory records.
"""
function mode_competence_diagnostics(mem :: MemoryBuffer)
    win_rates = _mode_win_rates(mem)
    counts    = _mode_sample_counts(mem)

    # belief-mode split: JEPA records have latent_μ_at_rec; explicit do not
    jepa_recs = filter(r -> !isnothing(r.latent_μ_at_rec) && !isnothing(r.shadow_delta_score), mem.records)
    expl_recs = filter(r ->  isnothing(r.latent_μ_at_rec) && !isnothing(r.shadow_delta_score), mem.records)

    jepa_win_rate = isempty(jepa_recs) ? nothing :
        count(r -> r.shadow_delta_score > 0.0, jepa_recs) / length(jepa_recs)
    expl_win_rate = isempty(expl_recs) ? nothing :
        count(r -> r.shadow_delta_score > 0.0, expl_recs) / length(expl_recs)

    jepa_preferred = !isnothing(jepa_win_rate) && !isnothing(expl_win_rate) &&
                     jepa_win_rate > expl_win_rate

    return (
        active_mode             = _LAST_ACTIVE_MODE[],
        configurator_win_rates  = win_rates,
        configurator_counts     = counts,
        jepa_win_rate           = jepa_win_rate,
        explicit_win_rate       = expl_win_rate,
        jepa_preferred          = jepa_preferred,
        cql_ready               = Q_NET.is_ready,
        bandit_ready            = BANDIT_CONFIG.is_ready,
    )
end

export adapt, initialize_config, record_outcome!,
       maybe_train_cql!, status, active_mode,
       compute_meta_state, MetaState,
       adapt_rule_based, adapt_bandit, adapt_cql,
       BANDIT_CONFIG, Q_NET,
       mode_competence_diagnostics

end # module Configurator
