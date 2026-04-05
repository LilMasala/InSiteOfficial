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
using Statistics

using Main: AbstractDomainAdapter, UserPreferences,
            RegimeDetectionResult, ConnectedAppCapabilities, ConnectedAppState,
            MemoryBuffer, TwinPosterior, TwinPrior,
            ScheduledAction, SegmentSurface, SegmentDelta, StructureEdit,
            parameter_adjustment, structure_edit, BridgeDecodedAction
import Main: detect_regime, calibrate_posterior!, bridge_decode_action_path_result
using Main.Actor: CandidateAction

struct InSiteDomainAdapter <: AbstractDomainAdapter end

function _bridge_first_step(action_path) :: Union{Vector{Float64}, Nothing}
    action_path isa AbstractVector || return nothing
    isempty(action_path) && return nothing

    if first(action_path) isa AbstractVector
        return _bridge_first_step(first(action_path))
    end

    values = Float64[]
    for item in action_path
        item isa Number || return nothing
        push!(values, Float64(item))
    end
    return values
end

function _bridge_path_steps(action_path) :: Union{Vector{Vector{Float64}}, Nothing}
    action_path isa AbstractVector || return nothing
    isempty(action_path) && return nothing

    if first(action_path) isa AbstractVector
        steps = Vector{Vector{Float64}}()
        for raw_step in action_path
            step = _bridge_first_step(raw_step)
            isnothing(step) && return nothing
            push!(steps, step)
        end
        return steps
    end

    step = _bridge_first_step(action_path)
    isnothing(step) && return nothing
    return [step]
end

_bridge_positive(x::Float64) = max(0.0, tanh(x))

function _bridge_step_deltas(step::Vector{Float64}) :: Union{Dict{Symbol, Float64}, Nothing}
    length(step) >= 4 || return nothing

    hold_bias = _bridge_positive(step[1])
    basal_bias = tanh(step[2])
    correction_bias = tanh(step[3])
    meal_bias = tanh(step[4])
    support_bias = length(step) >= 5 ? _bridge_positive(step[5]) : 0.0
    stability_bias = length(step) >= 6 ? _bridge_positive(step[6]) : 0.0
    probe_bias = length(step) >= 7 ? tanh(step[7]) : 0.0
    trust_preservation = length(step) >= 8 ? _bridge_positive(step[8]) : 0.0

    conservatism = mean((hold_bias, support_bias, stability_bias, trust_preservation))
    damp = clamp(1.0 - 0.60 * conservatism, 0.15, 1.0)
    probe_boost = 1.0 + 0.15 * max(0.0, probe_bias)

    deltas = Dict{Symbol, Float64}(
        :isf_delta => clamp(0.10 * correction_bias * damp * probe_boost, -0.15, 0.15),
        :cr_delta => clamp(0.10 * meal_bias * damp, -0.15, 0.15),
        :basal_delta => clamp(0.08 * basal_bias * damp, -0.12, 0.12),
    )

    max_delta = maximum(abs, values(deltas))
    if hold_bias > 0.85 || max_delta < 1.0e-4
        return Dict(label => 0.0 for label in keys(deltas))
    end

    return deltas
end

function _bridge_schedule_segment_deltas(
    steps::Vector{Vector{Float64}},
    segments::Vector{SegmentSurface},
) :: Vector{SegmentDelta}
    isempty(steps) && return SegmentDelta[]
    isempty(segments) && return SegmentDelta[]

    broadcast_all = length(steps) == 1
    target_segments = broadcast_all ? segments : segments[1:min(length(steps), length(segments))]
    deltas = SegmentDelta[]
    for (idx, segment) in pairs(target_segments)
        step = broadcast_all ? steps[1] : steps[idx]
        step_deltas = _bridge_step_deltas(step)
        isnothing(step_deltas) && continue
        parameter_deltas = Dict{Symbol, Float64}(
            :isf => get(step_deltas, :isf_delta, 0.0),
            :cr => get(step_deltas, :cr_delta, 0.0),
            :basal => get(step_deltas, :basal_delta, 0.0),
        )
        all(abs(value) < 1.0e-8 for value in values(parameter_deltas)) && continue
        push!(deltas, SegmentDelta(segment_id=segment.segment_id, parameter_deltas=parameter_deltas))
    end
    return deltas
end

function _bridge_selector_index(selector::Float64, count::Int) :: Int
    count <= 1 && return 1
    scaled = clamp((selector + 1.0) / 2.0, 0.0, 1.0)
    return clamp(floor(Int, scaled * (count - 1)) + 1, 1, count)
end

function _bridge_split_minute(
    segment::SegmentSurface,
    capabilities::ConnectedAppCapabilities,
    position_bias::Float64,
) :: Union{Int, Nothing}
    min_dur = max(capabilities.min_segment_duration_min, 1)
    lower = segment.start_min + min_dur
    upper = segment.end_min - min_dur
    lower < upper || return nothing
    midpoint = (lower + upper) / 2
    radius = (upper - lower) / 2
    proposed = round(Int, midpoint + radius * clamp(position_bias, -1.0, 1.0))
    return clamp(proposed, lower, upper)
end

function _bridge_structure_delta_map(step::Vector{Float64}) :: Dict{Symbol, Float64}
    step_deltas = _bridge_step_deltas(step)
    isnothing(step_deltas) && return Dict{Symbol, Float64}()
    return Dict{Symbol, Float64}(
        :isf => get(step_deltas, :isf_delta, 0.0),
        :cr => get(step_deltas, :cr_delta, 0.0),
        :basal => get(step_deltas, :basal_delta, 0.0),
    )
end

function _bridge_structure_decode(
    steps::Vector{Vector{Float64}},
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Union{BridgeDecodedAction, Nothing}
    capabilities.level_2_enabled || return nothing
    app_state.allow_structural_recommendations || return nothing
    capabilities.supports_piecewise_schedule || return nothing
    isempty(app_state.current_segments) && return nothing

    control_step = steps[1]
    length(control_step) >= 11 || return nothing

    structure_gate = _bridge_positive(control_step[9])
    split_bias = tanh(control_step[10])
    merge_bias = tanh(control_step[11])
    structure_gate > 0.55 || return nothing
    max(abs(split_bias), abs(merge_bias)) > 0.35 || return nothing

    segments = sort(deepcopy(app_state.current_segments), by = seg -> seg.start_min)
    target_selector = length(control_step) >= 12 ? tanh(control_step[12]) : 0.0
    shape_selector = length(control_step) >= 13 ? tanh(control_step[13]) : 0.0
    child_selector = length(control_step) >= 14 ? tanh(control_step[14]) : 0.0
    delta_source_step = steps[min(length(steps), 2)]

    if split_bias >= merge_bias
        capabilities.max_segments_addable >= 1 || return nothing
        length(segments) < capabilities.max_segments || return nothing
        target_idx = _bridge_selector_index(target_selector, length(segments))
        target = segments[target_idx]
        split_at = _bridge_split_minute(target, capabilities, shape_selector)
        isnothing(split_at) && return nothing

        edit = StructureEdit(
            edit_type = :split,
            target_segment_id = target.segment_id,
            split_at_minute = split_at,
        )
        child_id = child_selector < 0 ? "$(target.segment_id)_a" : "$(target.segment_id)_b"
        parameter_deltas = _bridge_structure_delta_map(delta_source_step)
        segment_deltas = all(abs(value) < 1.0e-8 for value in values(parameter_deltas)) ?
            SegmentDelta[] :
            [SegmentDelta(segment_id=child_id, parameter_deltas=parameter_deltas)]
        action = ScheduledAction(2, structure_edit, segments, segment_deltas, [edit])
        metadata = Dict{String, Any}(
            "decoder" => "insite_structure_edit",
            "returned_action_kind" => "ScheduledAction",
            "used_schedule_surface" => true,
            "used_structural_recommendation" => true,
            "schedule_version" => app_state.schedule_version,
            "path_step_count" => length(steps),
            "path_steps_consumed" => min(length(steps), 2),
            "structure_edit_type" => "split",
            "target_segment_id" => target.segment_id,
            "split_at_minute" => split_at,
            "targeted_segment_ids" => isempty(segment_deltas) ? [target.segment_id] : [child_id],
            "supports_piecewise_schedule" => capabilities.supports_piecewise_schedule,
        )
        return BridgeDecodedAction(action, metadata)
    end

    length(segments) >= 2 || return nothing
    target_idx = _bridge_selector_index(target_selector, length(segments))
    neighbor_idx = if target_idx == 1
        2
    elseif target_idx == length(segments)
        length(segments) - 1
    else
        shape_selector >= 0 ? target_idx + 1 : target_idx - 1
    end
    target = segments[target_idx]
    neighbor = segments[neighbor_idx]
    first_segment, second_segment = target.start_min <= neighbor.start_min ? (target, neighbor) : (neighbor, target)
    first_segment.end_min == second_segment.start_min || return nothing

    edit = StructureEdit(
        edit_type = :merge,
        target_segment_id = first_segment.segment_id,
        neighbor_segment_id = second_segment.segment_id,
    )
    merged_id = "$(first_segment.segment_id)__$(second_segment.segment_id)"
    parameter_deltas = _bridge_structure_delta_map(delta_source_step)
    segment_deltas = all(abs(value) < 1.0e-8 for value in values(parameter_deltas)) ?
        SegmentDelta[] :
        [SegmentDelta(segment_id=merged_id, parameter_deltas=parameter_deltas)]
    action = ScheduledAction(2, structure_edit, segments, segment_deltas, [edit])
    metadata = Dict{String, Any}(
        "decoder" => "insite_structure_edit",
        "returned_action_kind" => "ScheduledAction",
        "used_schedule_surface" => true,
        "used_structural_recommendation" => true,
        "schedule_version" => app_state.schedule_version,
        "path_step_count" => length(steps),
        "path_steps_consumed" => min(length(steps), 2),
        "structure_edit_type" => "merge",
        "target_segment_id" => first_segment.segment_id,
        "neighbor_segment_id" => second_segment.segment_id,
        "targeted_segment_ids" => isempty(segment_deltas) ? [merged_id] : [merged_id],
        "supports_piecewise_schedule" => capabilities.supports_piecewise_schedule,
    )
    return BridgeDecodedAction(action, metadata)
end

function bridge_decode_action_path_result(
    ::InSiteDomainAdapter,
    action_path,
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Union{BridgeDecodedAction, Nothing}
    steps = _bridge_path_steps(action_path)
    isnothing(steps) && return nothing
    isempty(steps) && return nothing

    structured = _bridge_structure_decode(steps, capabilities, app_state)
    !isnothing(structured) && return structured

    if capabilities.level_1_enabled && !isempty(app_state.current_segments)
        segments = sort(deepcopy(app_state.current_segments), by = seg -> seg.start_min)
        segment_deltas = _bridge_schedule_segment_deltas(steps, segments)
        strategy = length(steps) == 1 ? "broadcast_all_segments" : "ordered_path_steps"
        action = ScheduledAction(1, parameter_adjustment, segments, segment_deltas, StructureEdit[])
        metadata = Dict{String, Any}(
            "decoder" => "insite_schedule_surface",
            "returned_action_kind" => "ScheduledAction",
            "used_schedule_surface" => true,
            "schedule_version" => app_state.schedule_version,
            "path_step_count" => length(steps),
            "path_steps_consumed" => isempty(segment_deltas) ? 0 : (length(steps) == 1 ? 1 : min(length(steps), length(segments))),
            "segment_decode_strategy" => strategy,
            "targeted_segment_ids" => [delta.segment_id for delta in segment_deltas],
            "supports_piecewise_schedule" => capabilities.supports_piecewise_schedule,
        )
        return BridgeDecodedAction(action, metadata)
    end

    step_deltas = _bridge_step_deltas(steps[1])
    isnothing(step_deltas) && return nothing
    action = CandidateAction(step_deltas)
    metadata = Dict{String, Any}(
        "decoder" => "insite_scalar_delta",
        "returned_action_kind" => "CandidateAction",
        "used_schedule_surface" => false,
        "path_step_count" => length(steps),
        "path_steps_consumed" => 1,
        "delta_map" => Dict(String(key) => value for (key, value) in step_deltas),
    )
    return BridgeDecodedAction(action, metadata)
end

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
