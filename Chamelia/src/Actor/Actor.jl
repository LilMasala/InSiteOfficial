"""
Actor.jl
Actor module — constrained CVaR-minimizing action search.

The full decision cycle:
  1. Epistemic gate    — can we act at all?
  2. Generate candidates — search strategy produces candidate actions
  3. Safety gate       — reject unsafe actions
  4. Effect size gate  — reject insignificant improvements
  5. CVaR selection    — pick best surviving action
  6. Burnout attribution — does winner increase burnout risk?
  7. Burnout gate      — block if upper CI > ε_burn
  8. Package result    — return RecommendationPackage or hold

Hold is an active decision — logged, scored, contributes to scorecard.
"""

include("../types.jl")

module Actor

using Statistics
using Flux

using Main: AbstractSimulator, AbstractAction, AbstractBeliefState,
            AbstractBeliefEstimator, AbstractCriticModel,
            AbstractSearchStrategy, GridSearch, BeamSearch,
            GradientSearch, OfflineRLPolicy,
            PatientState, PhysState, PsyState,
            TwinPrior, TwinPosterior, DigitalTwin,
            RolloutNoise, RolloutResult, LatentRolloutResult, Observation,
            GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            EpistemicState, EpistemicThresholds,
            MemoryBuffer, MemoryRecord, ConfiguratorState,
            CostWeights, UserResponse, BurnoutAttribution,
            RecommendationPackage, AnomalyResult,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            NullAction, Accept, Reject, Partial,
            sample_noise, CRITIC_FEATURE_DIM,
            ActionFamily, parameter_adjustment, structure_edit, continuous_schedule,
            ConnectedAppCapabilities, ConnectedAppState,
            SegmentSurface, SegmentDelta, StructureEdit, ScheduledAction,
            AbstractDomainAdapter, RegimeDetectionResult, UserPreferences,
            BridgeProposalAdvisory

import Main: is_null, magnitude, detect_regime, minimum_action_delta_threshold

using Main.WorldModule: MockSimulator, run_rollouts, run_paired_rollouts,
                        JEPA_PREDICTOR, run_latent_rollouts, run_paired_latent_rollouts,
                        action_to_features, min_clinical_delta, action_dimensions
using Main.Cost: compute_energies, compute_cvar, compute_effect_size,
                 ZeroCritic, decode_latent_summary

include("search.jl")
include("actor_training.jl")
include("safety.jl")
include("burnout.jl")

# ─────────────────────────────────────────────────────────────────
# Main Actor entry point
# Returns RecommendationPackage if action found, nothing if hold
# reason symbol explains why hold was chosen (for logging)
# ─────────────────────────────────────────────────────────────────

function decide(
    belief      :: AbstractBeliefState,
    twin        :: DigitalTwin,
    sim         :: AbstractSimulator,
    noise       :: RolloutNoise,
    critic      :: AbstractCriticModel,
    epistemic   :: EpistemicState,
    config      :: ConfiguratorState,
    dimensions  :: Vector{Symbol},        # action dimensions from simulator
    thresholds  :: Dict{Symbol, Float64}, # safety thresholds from simulator
    current_day :: Int = 0,
    capabilities::ConnectedAppCapabilities = ConnectedAppCapabilities(),
    app_state   :: ConnectedAppState = ConnectedAppState(),
    adapter     :: AbstractDomainAdapter = Main.DefaultDomainAdapter(),
    prefs       :: UserPreferences = UserPreferences(),
    signals     :: Dict{Symbol, Any} = Dict{Symbol, Any}(),
    memory      :: MemoryBuffer = MemoryBuffer(),
    ;
    proposal_actions::Union{Nothing, Vector{AbstractAction}} = nothing,
    proposal_advisories::Union{Nothing, Vector{BridgeProposalAdvisory}} = nothing,
) :: Tuple{Union{RecommendationPackage, Nothing}, Symbol, Any}

    # ── 1. Epistemic gate ─────────────────────────────────────────
    if !passes_epistemic_gate(epistemic)
        return nothing, :epistemic_failed, nothing
    end

    # ── 2. Generate and evaluate candidates ───────────────────────
    strategy = _select_strategy(belief, config)
    use_schedule_surface = capabilities.level_1_enabled && !isempty(app_state.current_segments)
    candidates = if !isnothing(proposal_actions) && !isempty(proposal_actions)
        _evaluate_provided_actions(proposal_actions, belief, twin, sim, noise, critic, config; advisories=proposal_advisories)
    else
        use_schedule_surface ?
            search_scheduled_actions(strategy, belief, twin, sim, noise, critic, config, capabilities, app_state; current_day=current_day) :
            search_actions(strategy, belief, twin, sim, noise, critic, config, dimensions)
    end

    isempty(candidates) && return nothing, :no_candidates, nothing

    # ── 3. Baseline rollouts (null action) ────────────────────────
    null_action = use_schedule_surface ?
        ScheduledAction(1, parameter_adjustment, deepcopy(app_state.current_segments), SegmentDelta[], StructureEdit[]) :
        CandidateAction(Dict(d => 0.0 for d in dimensions))
    baseline_rollouts = run_rollouts(belief, null_action, twin, sim, noise, config)
    baseline_energies = compute_energies(baseline_rollouts, critic, config)

    # ── 4. Apply safety + effect size gates ───────────────────────
    survivors = eltype(candidates)[]
    safe_results = eltype(candidates)[]
    best_safety_diagnostics = nothing
    best_safety_score = Inf

    for result in candidates
        diagnostics = safety_diagnostics(result.rollouts, baseline_rollouts, sim, thresholds)
        if !diagnostics.passed
            safety_score = diagnostics.relative_excess_gap + diagnostics.relative_violation_gap +
                (diagnostics.failure_mode == :catastrophic ? 10.0 : 0.0)
            if safety_score < best_safety_score
                best_safety_score = safety_score
                best_safety_diagnostics = diagnostics
            end
            continue
        end
        push!(safe_results, result)

        # effect size gate: is the predicted improvement statistically real?
        passes_effect_size_gate(
            result.energies,
            baseline_energies,
            config
        ) || continue

        # clinical delta gate: is the action magnitude physically meaningful?
        passes_clinical_delta_gate(result.action, sim, adapter, prefs) || continue

        push!(survivors, result)
    end

    selection_reason = :recommended
    best = isempty(survivors) ? nothing : survivors[1]
    if _shadow_exploration_due(current_day, config)
        exploratory = _pick_shadow_exploration_candidate(
            safe_results,
            baseline_energies,
            config,
            current_day,
        )
        if !isnothing(exploratory)
            if isnothing(best)
                push!(survivors, exploratory)
                best = exploratory
            elseif best.action !== exploratory.action
                push!(survivors, exploratory)
                best = exploratory
            end
            selection_reason = :shadow_explore
        end
    end
    if isnothing(best) && _postgrad_probe_due(current_day, config)
        probe = _pick_postgrad_probe_candidate(
            safe_results,
            baseline_energies,
            config,
            current_day,
        )
        if !isnothing(probe)
            push!(survivors, probe)
            best = probe
            selection_reason = :postgrad_probe
        end
    end

    if isnothing(best)
        isempty(safe_results) && return nothing, :safety_violated, best_safety_diagnostics
        isempty(survivors) && return nothing, :effect_size_insufficient, nothing
        return nothing, :no_survivors, nothing
    end

    # ── 5. Pick best by CVaR ──────────────────────────────────────
    sort!(survivors, by = _candidate_order_key)

    # ── 6. Burnout attribution ────────────────────────────────────
    burnout_horizon = selection_reason == :postgrad_probe ? min(14, config.φ_cost.H_burn) : nothing
    attribution = isnothing(burnout_horizon) ?
        attribute_burnout(belief, best.action, twin, sim, noise, config) :
        attribute_burnout(belief, best.action, twin, sim, noise, config; H=burnout_horizon, N=50)

    # ── 7. Burnout gate ───────────────────────────────────────────
    if !passes_burnout_gate(attribution, config)
        # search for alternative with lower burnout risk
        alt = _find_burnout_safe_alternative(survivors, belief, twin,
                                              sim, noise, config)
        if isnothing(alt)
            return nothing, :burnout_risk_exceeded, nothing
        end
        best        = alt.result
        attribution = alt.attribution
    end

    # ── 8. Package result ─────────────────────────────────────────
    baseline_cvar     = compute_cvar(baseline_energies, config.φ_act.α_cvar)

    alternatives = AbstractAction[]
    for result in survivors
        result.action === best.action && continue
        push!(alternatives, result.action)
        length(alternatives) >= 2 && break
    end

    confidence_breakdown = _confidence_breakdown(epistemic, best, config, selection_reason)
    confidence = confidence_breakdown.final_confidence

    # ── 9. Regime detection ───────────────────────────────────────
    # Domain adapter classifies current context; Chamelia core attaches result.
    regime = detect_regime(adapter, signals, app_state, memory)

    pkg = RecommendationPackage(
        action                = best.action,
        predicted_improvement = baseline_cvar - best.cvar,
        confidence            = confidence,
        confidence_breakdown  = confidence_breakdown,
        alternatives          = alternatives,
        effect_size           = compute_effect_size(
                                    best.energies,
                                    baseline_energies,
                                    config.φ_act.α_cvar),
        cvar_value            = best.cvar,
        burnout_attribution   = attribution,
        predicted_outcomes    = _predicted_outcome_summary(
                                    best.rollouts,
                                    baseline_rollouts,
                                    best.energies,
                                    baseline_energies,
                                    best.cvar,
                                    baseline_cvar,
                                ),
        predicted_uncertainty = _predicted_uncertainty_summary(
                                    best.rollouts,
                                    baseline_rollouts,
                                    best.energies,
                                    baseline_energies,
                                ),
        action_level          = _action_level(best.action),
        action_family         = _action_family(best.action),
        segment_summaries     = _segment_summaries(best.action, app_state),
        structure_summaries   = _structure_summaries(best.action),
        recommendation_scope  = regime.scope,
        target_profile_id     = regime.target_profile_id,
        detected_regime       = regime.regime_label,
    )

    return pkg, selection_reason, nothing
end

function decide(
    belief      :: JEPABeliefState,
    twin        :: DigitalTwin,
    sim         :: AbstractSimulator,
    noise       :: RolloutNoise,
    critic      :: AbstractCriticModel,
    epistemic   :: EpistemicState,
    config      :: ConfiguratorState,
    dimensions  :: Vector{Symbol},
    thresholds  :: Dict{Symbol, Float64},
    current_day :: Int = 0,
    capabilities::ConnectedAppCapabilities = ConnectedAppCapabilities(),
    app_state   :: ConnectedAppState = ConnectedAppState(),
    adapter     :: AbstractDomainAdapter = Main.DefaultDomainAdapter(),
    prefs       :: UserPreferences = UserPreferences(),
    signals     :: Dict{Symbol, Any} = Dict{Symbol, Any}(),
    memory      :: MemoryBuffer = MemoryBuffer(),
    ;
    proposal_actions::Union{Nothing, Vector{AbstractAction}} = nothing,
    proposal_advisories::Union{Nothing, Vector{BridgeProposalAdvisory}} = nothing,
) :: Tuple{Union{RecommendationPackage, Nothing}, Symbol, Any}

    if !passes_epistemic_gate(epistemic)
        return nothing, :epistemic_failed, nothing
    end

    strategy = _select_strategy(belief, config)
    use_schedule_surface = capabilities.level_1_enabled && !isempty(app_state.current_segments)
    candidates = if !isnothing(proposal_actions) && !isempty(proposal_actions)
        _evaluate_provided_actions(proposal_actions, belief, twin, sim, noise, critic, config; advisories=proposal_advisories)
    else
        use_schedule_surface ?
            search_scheduled_actions(strategy, belief, twin, sim, noise, critic, config, capabilities, app_state; current_day=current_day) :
            search_actions(strategy, belief, twin, sim, noise, critic, config, dimensions)
    end
    isempty(candidates) && return nothing, :no_candidates, nothing

    null_action = use_schedule_surface ?
        ScheduledAction(1, parameter_adjustment, deepcopy(app_state.current_segments), SegmentDelta[], StructureEdit[]) :
        CandidateAction(Dict(d => 0.0 for d in dimensions))
    baseline_rollouts = run_latent_rollouts(belief, null_action, JEPA_PREDICTOR, config)
    baseline_energies = compute_energies(baseline_rollouts, critic, config)

    survivors = eltype(candidates)[]
    safe_results = eltype(candidates)[]
    best_safety_diagnostics = nothing
    best_safety_score = Inf
    for result in candidates
        diagnostics = safety_diagnostics(result.rollouts, baseline_rollouts, sim, thresholds)
        if !diagnostics.passed
            safety_score = diagnostics.relative_excess_gap + diagnostics.relative_violation_gap +
                (diagnostics.failure_mode == :catastrophic ? 10.0 : 0.0)
            if safety_score < best_safety_score
                best_safety_score = safety_score
                best_safety_diagnostics = diagnostics
            end
            continue
        end
        push!(safe_results, result)
        passes_effect_size_gate(result.energies, baseline_energies, config) || continue
        passes_clinical_delta_gate(result.action, sim, adapter, prefs) || continue
        push!(survivors, result)
    end

    selection_reason = :recommended
    best = isempty(survivors) ? nothing : survivors[1]
    if _shadow_exploration_due(current_day, config)
        exploratory = _pick_shadow_exploration_candidate(
            safe_results,
            baseline_energies,
            config,
            current_day,
        )
        if !isnothing(exploratory)
            if isnothing(best)
                push!(survivors, exploratory)
                best = exploratory
            elseif best.action !== exploratory.action
                push!(survivors, exploratory)
                best = exploratory
            end
            selection_reason = :shadow_explore
        end
    end
    if isnothing(best) && _postgrad_probe_due(current_day, config)
        probe = _pick_postgrad_probe_candidate(
            safe_results,
            baseline_energies,
            config,
            current_day,
        )
        if !isnothing(probe)
            push!(survivors, probe)
            best = probe
            selection_reason = :postgrad_probe
        end
    end

    if isnothing(best)
        isempty(safe_results) && return nothing, :safety_violated, best_safety_diagnostics
        isempty(survivors) && return nothing, :effect_size_insufficient, nothing
        return nothing, :no_survivors, nothing
    end
    sort!(survivors, by = _candidate_order_key)
    burnout_horizon = selection_reason == :postgrad_probe ? min(14, config.φ_cost.H_burn) : nothing
    attribution = isnothing(burnout_horizon) ?
        attribute_burnout(belief, best.action, twin, sim, noise, config) :
        attribute_burnout(belief, best.action, twin, sim, noise, config; H=burnout_horizon, N=50)

    if !passes_burnout_gate(attribution, config)
        alt = _find_burnout_safe_alternative(survivors, belief, twin,
                                             sim, noise, config)
        if isnothing(alt)
            return nothing, :burnout_risk_exceeded, nothing
        end
        best        = alt.result
        attribution = alt.attribution
    end

    baseline_cvar = compute_cvar(baseline_energies, config.φ_act.α_cvar)
    alternatives = AbstractAction[]
    for result in survivors
        result.action === best.action && continue
        push!(alternatives, result.action)
        length(alternatives) >= 2 && break
    end

    confidence_breakdown = _confidence_breakdown(epistemic, best, config, selection_reason)
    confidence = confidence_breakdown.final_confidence

    # ── 9. Regime detection ───────────────────────────────────────
    regime = detect_regime(adapter, signals, app_state, memory)

    pkg = RecommendationPackage(
        action                = best.action,
        predicted_improvement = baseline_cvar - best.cvar,
        confidence            = confidence,
        confidence_breakdown  = confidence_breakdown,
        alternatives          = alternatives,
        effect_size           = compute_effect_size(
                                    best.energies,
                                    baseline_energies,
                                    config.φ_act.α_cvar),
        cvar_value            = best.cvar,
        burnout_attribution   = attribution,
        predicted_outcomes    = nothing,
        predicted_uncertainty = nothing,
        action_level          = _action_level(best.action),
        action_family         = _action_family(best.action),
        segment_summaries     = _segment_summaries(best.action, app_state),
        structure_summaries   = _structure_summaries(best.action),
        recommendation_scope  = regime.scope,
        target_profile_id     = regime.target_profile_id,
        detected_regime       = regime.regime_label,
    )

    return pkg, selection_reason, nothing
end

function _predicted_outcome_summary(
    treated_rollouts  :: Vector{RolloutResult},
    baseline_rollouts :: Vector{RolloutResult},
    treated_energies  :: Vector{Float64},
    baseline_energies :: Vector{Float64},
    treated_cvar      :: Float64,
    baseline_cvar     :: Float64,
)
    isempty(treated_rollouts) && return nothing
    isempty(baseline_rollouts) && return nothing

    treated = _rollout_signal_means(treated_rollouts)
    baseline = _rollout_signal_means(baseline_rollouts)
    isnothing(treated) && return nothing
    isnothing(baseline) && return nothing

    treated_cost = mean(treated_energies)
    baseline_cost = mean(baseline_energies)

    return (
        baseline_tir = baseline.tir,
        treated_tir = treated.tir,
        delta_tir = treated.tir - baseline.tir,
        baseline_pct_low = baseline.pct_low,
        treated_pct_low = treated.pct_low,
        delta_pct_low = treated.pct_low - baseline.pct_low,
        baseline_pct_high = baseline.pct_high,
        treated_pct_high = treated.pct_high,
        delta_pct_high = treated.pct_high - baseline.pct_high,
        baseline_bg_avg = baseline.bg_avg,
        treated_bg_avg = treated.bg_avg,
        delta_bg_avg = treated.bg_avg - baseline.bg_avg,
        baseline_cost_mean = baseline_cost,
        treated_cost_mean = treated_cost,
        delta_cost_mean = treated_cost - baseline_cost,
        baseline_cvar = baseline_cvar,
        treated_cvar = treated_cvar,
        delta_cvar = treated_cvar - baseline_cvar,
    )
end

function _predicted_uncertainty_summary(
    treated_rollouts  :: Vector{RolloutResult},
    baseline_rollouts :: Vector{RolloutResult},
    treated_energies  :: Vector{Float64},
    baseline_energies :: Vector{Float64},
)
    isempty(treated_rollouts) && return nothing
    isempty(baseline_rollouts) && return nothing

    treated = _rollout_signal_samples(treated_rollouts)
    baseline = _rollout_signal_samples(baseline_rollouts)
    isnothing(treated) && return nothing
    isnothing(baseline) && return nothing

    return (
        tir_std = _paired_delta_std(treated.tir, baseline.tir),
        pct_low_std = _paired_delta_std(treated.pct_low, baseline.pct_low),
        pct_high_std = _paired_delta_std(treated.pct_high, baseline.pct_high),
        bg_avg_std = _paired_delta_std(treated.bg_avg, baseline.bg_avg),
        cost_std = _paired_delta_std(treated_energies, baseline_energies),
    )
end

function _rollout_signal_means(rollouts :: Vector{RolloutResult})
    samples = _rollout_signal_samples(rollouts)
    isnothing(samples) && return nothing
    return (
        tir = mean(samples.tir),
        pct_low = mean(samples.pct_low),
        pct_high = mean(samples.pct_high),
        bg_avg = mean(samples.bg_avg),
    )
end

function _signal_float(
    signals::Dict{Symbol, Any},
    primary::Symbol,
    fallback::Symbol,
    default::Float64,
) :: Float64
    value = get(signals, primary, get(signals, fallback, default))
    value === nothing && (value = get(signals, fallback, default))
    value === nothing && return default
    return try
        Float64(value)
    catch
        default
    end
end

function _signal_float(
    signals::Dict{Symbol, Any},
    key::Symbol,
    default::Float64,
) :: Float64
    value = get(signals, key, default)
    value === nothing && return default
    return try
        Float64(value)
    catch
        default
    end
end

function _rollout_signal_samples(rollouts :: Vector{RolloutResult})
    tir = Float64[]
    pct_low = Float64[]
    pct_high = Float64[]
    bg_avg = Float64[]

    for rollout in rollouts
        isempty(rollout.phys_signals) && continue
        signals = rollout.phys_signals[end]
        push!(tir, _signal_float(signals, :tir_7d, :tir, 0.0))
        push!(pct_low, _signal_float(signals, :pct_low_7d, :pct_low, 0.0))
        push!(pct_high, _signal_float(signals, :pct_high_7d, :pct_high, 0.0))
        push!(bg_avg, _signal_float(signals, :bg_avg, 0.0))
    end

    isempty(tir) && return nothing
    return (
        tir = tir,
        pct_low = pct_low,
        pct_high = pct_high,
        bg_avg = bg_avg,
    )
end

function _shadow_exploration_due(
    current_day :: Int,
    config      :: ConfiguratorState
) :: Bool
    current_day <= 0 && return false
    bootstrap_start = max(4, config.φ_world.H_med)
    bootstrap_end = max(42, 6 * config.φ_world.H_med)
    bootstrap_start <= current_day <= bootstrap_end || return false
    return mod(current_day - bootstrap_start, 4) == 0
end

function _postgrad_probe_due(
    current_day :: Int,
    config      :: ConfiguratorState
) :: Bool
    current_day <= 21 && return false
    probe_start = max(22, config.φ_world.H_med + 15)
    current_day >= probe_start || return false
    return mod(current_day - probe_start, 3) == 0
end

function _pick_shadow_exploration_candidate(
    safe_results      :: Vector,
    baseline_energies :: Vector{Float64},
    config            :: ConfiguratorState,
    current_day       :: Int,
)
    non_null = filter(result -> !is_null(result.action), safe_results)
    isempty(non_null) && return nothing

    tolerated_regret = max(0.10, 0.25 * config.φ_act.δ_min_effect)
    scored = [
        (
            result = result,
            effect_size = compute_effect_size(
                result.energies,
                baseline_energies,
                config.φ_act.α_cvar,
            ),
            action_magnitude = magnitude(result.action),
        )
        for result in non_null
    ]

    near_neutral = filter(item -> item.effect_size >= -tolerated_regret, scored)
    pool = isempty(near_neutral) ? scored : near_neutral
    sort!(pool, by = item -> (item.action_magnitude, item.result.cvar, -item.effect_size))

    span = min(length(pool), 3)
    return pool[1 + mod(current_day - 1, span)].result
end

function _pick_postgrad_probe_candidate(
    safe_results      :: Vector,
    baseline_energies :: Vector{Float64},
    config            :: ConfiguratorState,
    current_day       :: Int,
)
    non_null = filter(result -> !is_null(result.action), safe_results)
    isempty(non_null) && return nothing

    probe_floor = max(0.05, 0.10 * config.φ_act.δ_min_effect)
    tolerated_regret = max(0.10, 0.25 * config.φ_act.δ_min_effect)
    scored = [
        (
            result = result,
            effect_size = compute_effect_size(
                result.energies,
                baseline_energies,
                config.φ_act.α_cvar,
            ),
            action_magnitude = magnitude(result.action),
        )
        for result in non_null
    ]

    mildly_positive = filter(item -> item.effect_size >= probe_floor, scored)
    near_neutral = filter(item -> item.effect_size >= -tolerated_regret, scored)
    pool = isempty(mildly_positive) ? near_neutral : mildly_positive
    isempty(pool) && return nothing

    sort!(pool, by = item -> (item.action_magnitude, item.result.cvar, -item.effect_size))
    span = min(length(pool), 2)
    return pool[1 + mod(current_day - 1, span)].result
end

# ─────────────────────────────────────────────────────────────────
# Select search strategy based on belief type and data availability
# Multiple dispatch on belief type routes naturally
# ─────────────────────────────────────────────────────────────────

function _select_strategy(
    belief :: GaussianBeliefState,
    config :: ConfiguratorState
) :: AbstractSearchStrategy
    GridSearch()
end

function _select_strategy(
    belief :: ParticleBeliefState,
    config :: ConfiguratorState
) :: AbstractSearchStrategy
    BeamSearch()   # particle filter → beam search (faster)
end

function _select_strategy(
    belief :: JEPABeliefState,
    config :: ConfiguratorState
) :: AbstractSearchStrategy
    # v2 — JEPA belief → gradient search if enough data, else beam
    OFFLINE_RL_MODEL.is_ready ? OfflineRLPolicy() : GradientSearch()
end

# ─────────────────────────────────────────────────────────────────
# Find burnout-safe alternative from survivors
# If best action has too high burnout risk, try next survivors
# ─────────────────────────────────────────────────────────────────

function _find_burnout_safe_alternative(
    survivors :: Vector,
    belief    :: AbstractBeliefState,
    twin      :: DigitalTwin,
    sim       :: AbstractSimulator,
    noise     :: RolloutNoise,
    config    :: ConfiguratorState
) :: Union{Nothing, NamedTuple}

    # try remaining survivors in order of CVaR
    for result in survivors[2:end]
        attribution = attribute_burnout(
            belief, result.action, twin, sim, noise, config;
            N = 50   # fewer pairs for alternatives — faster
        )

        if passes_burnout_gate(attribution, config)
            return (result=result, attribution=attribution)
        end
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Compute composite confidence score
# Combines κ, ρ, η, and effect size into one [0,1] number
# Used for UI display — not for any decision logic
# ─────────────────────────────────────────────────────────────────

function _compute_confidence(
    epistemic :: EpistemicState,
    result,
    config    :: ConfiguratorState
) :: Float64
    return _confidence_breakdown(epistemic, result, config, :primary).final_confidence
end

function _confidence_breakdown(
    epistemic :: EpistemicState,
    result,
    config    :: ConfiguratorState,
    selection_reason :: Symbol
)
    κ = clamp(epistemic.κ_familiarity, 0.0, 1.0)
    ρ = clamp(epistemic.ρ_concordance, 0.0, 1.0)
    η = clamp(epistemic.η_calibration, 0.0, 1.0)
    δ = min(1.0, result.cvar / (result.cvar + 0.1))
    effect_support = clamp(1.0 - δ, 0.0, 1.0)
    selection_penalty = selection_reason == :shadow_explore ? 0.65 :
                        selection_reason == :postgrad_probe ? 0.55 : 1.0
    final_confidence = clamp((κ * ρ * η * effect_support)^(1/4) * selection_penalty, 0.0, 1.0)
    return (
        familiarity = κ,
        concordance = ρ,
        calibration = η,
        effect_support = effect_support,
        selection_penalty = selection_penalty,
        final_confidence = final_confidence,
    )
end

function _paired_delta_std(
    treated  :: AbstractVector{<:Real},
    baseline :: AbstractVector{<:Real},
) :: Float64
    n = min(length(treated), length(baseline))
    n == 0 && return 0.0
    deltas = Float64[Float64(treated[i]) - Float64(baseline[i]) for i in 1:n]
    length(deltas) <= 1 && return 0.0
    return std(deltas)
end

function _action_level(action::AbstractAction) :: Int
    return action isa ScheduledAction ? action.level : 1
end

function _action_family(action::AbstractAction) :: Union{ActionFamily, Nothing}
    return action isa ScheduledAction ? action.family : parameter_adjustment
end

function _segment_label(segment::SegmentSurface) :: String
    return "$(segment.start_min)–$(segment.end_min) min"
end

function _delta_summary(label::String, value::Float64) :: String
    abs(value) < 1e-8 && return "$label unchanged"
    pct = round(value * 100; digits=1)
    sign = pct > 0 ? "+" : ""
    return "$label $(sign)$(pct)%"
end

function _segment_summaries(
    action    :: AbstractAction,
    app_state :: ConnectedAppState
)
    action isa ScheduledAction || return NamedTuple{(:segment_id, :label, :parameter_summaries), Tuple{String, String, Dict{String, String}}}[]

    segment_source = isempty(action.segments) ? app_state.current_segments : action.segments
    segment_lookup = Dict(segment.segment_id => segment for segment in segment_source)
    summaries = NamedTuple{(:segment_id, :label, :parameter_summaries), Tuple{String, String, Dict{String, String}}}[]
    for delta in action.segment_deltas
        isempty(delta.parameter_deltas) && continue
        all(abs(value) < 1e-8 for value in values(delta.parameter_deltas)) && continue
        segment = get(segment_lookup, delta.segment_id, nothing)
        label = isnothing(segment) ? delta.segment_id : _segment_label(segment)
        push!(summaries, (
            segment_id = delta.segment_id,
            label = label,
            parameter_summaries = Dict(
                String(key) => _delta_summary(String(key), value)
                for (key, value) in delta.parameter_deltas
            ),
        ))
    end
    return summaries
end

function _structure_summaries(action::AbstractAction) :: Vector{String}
    action isa ScheduledAction || return String[]

    summaries = String[]
    for edit in action.structural_edits
        if edit.edit_type == :split
            push!(summaries, "Split $(edit.target_segment_id) at $(edit.split_at_minute)")
        elseif edit.edit_type == :merge
            push!(summaries, "Merge $(edit.target_segment_id) with $(something(edit.neighbor_segment_id, "adjacent segment"))")
        elseif edit.edit_type == :remove
            push!(summaries, "Remove $(edit.target_segment_id)")
        elseif edit.edit_type == :add
            push!(summaries, "Add structure near $(edit.target_segment_id)")
        else
            push!(summaries, "Edit $(edit.target_segment_id) ($(edit.edit_type))")
        end
    end
    return summaries
end

export decide, ShadowScorecard, passes_graduation_gate,
       CandidateAction, ScheduledAction, search_actions, search_scheduled_actions,
       passes_epistemic_gate, passes_safety_gate,
       passes_effect_size_gate, passes_clinical_delta_gate, passes_burnout_gate,
       attribute_burnout, is_attribution_stable,
       summarize_attribution, check_safety, train_actor_cql!,
       GridSearch, BeamSearch, GradientSearch, OfflineRLPolicy

end # module Actor
