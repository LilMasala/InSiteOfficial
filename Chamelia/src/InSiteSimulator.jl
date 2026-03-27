"""
InSiteSimulator.jl
Concrete AbstractSimulator implementation for the InSite + t1d_sim bridge.

This keeps the Chamelia side self-contained in Julia while mirroring the
high-level physiology and sensor-noise structure from the Python simulator:
  - therapy multipliers drive daily glucose dynamics
  - context effectors perturb insulin sensitivity / endogenous glucose output
  - observations add Dexcom-like lag, drift, and missingness
"""

using Distributions
using Random
using Statistics

using Main: AbstractSimulator, AbstractAction, PatientState, PhysState, Observation,
            TwinPrior, RolloutNoise, register_physical_noise!,
            ScheduledAction, SegmentSurface, SegmentDelta, StructureEdit
using Main.Twin: register_physical_prior!

import Main.WorldModule: sim_step!, sim_observe, register_priors!, register_noise!,
                         action_dimensions, safety_thresholds, min_clinical_delta,
                         compute_frustration
import Main.Actor: check_safety
import Main: compute_physical_cost

struct InSiteSimulator <: AbstractSimulator
    persona::String
    seed::Int
end

InSiteSimulator(; persona::AbstractString="default", seed::Integer=42) =
    InSiteSimulator(String(persona), Int(seed))

InSiteSimulator(persona::AbstractString; seed::Integer=42) =
    InSiteSimulator(String(persona), Int(seed))

const _DEFAULT_BG_AVG = 135.0
const _DEFAULT_TIR = 0.65
const _DEFAULT_PCT_LOW = 0.02
const _DEFAULT_PCT_HIGH = 0.20
const _DEFAULT_BG_CV = 0.32

const _PERSONA_PRIOR_OVERRIDES = Dict{String, Dict{Symbol, Tuple{Float64, Float64}}}(
    "athlete" => Dict(
        :isf_multiplier => (1.22, 0.08),
        :stress_reactivity => (0.28, 0.10),
        :sleep_debt => (25.0, 12.0),
        :exercise_intensity => (0.75, 0.08)
    ),
    "sedentary" => Dict(
        :isf_multiplier => (0.83, 0.08),
        :cr_multiplier => (0.87, 0.08),
        :stress_reactivity => (0.58, 0.10),
        :sleep_debt => (80.0, 25.0),
        :exercise_intensity => (0.18, 0.08)
    ),
    "high_stress" => Dict(
        :isf_multiplier => (0.87, 0.08),
        :stress_acute => (0.55, 0.12),
        :stress_baseline => (0.45, 0.10),
        :stress_reactivity => (0.85, 0.08),
        :sleep_debt => (100.0, 30.0)
    ),
    "low_stress" => Dict(
        :stress_acute => (0.12, 0.05),
        :stress_baseline => (0.10, 0.05),
        :stress_reactivity => (0.20, 0.08),
        :sleep_debt => (30.0, 12.0)
    ),
    "insomniac" => Dict(
        :sleep_debt => (140.0, 35.0),
        :stress_reactivity => (0.75, 0.10),
        :stress_acute => (0.38, 0.10)
    ),
    "solid_sleeper" => Dict(
        :sleep_debt => (20.0, 10.0),
        :stress_reactivity => (0.45, 0.10),
        :stress_acute => (0.18, 0.06)
    )
)

function _state_value(
    vars::Dict{Symbol, Float64},
    key::Symbol,
    default::Float64
) :: Float64
    return Float64(get(vars, key, default))
end

function _clamp01(x::Float64) :: Float64
    return clamp(x, 0.0, 1.0)
end

function _phase_name(code::Float64) :: Symbol
    phase_idx = mod(round(Int, code), 4)
    phase_idx == 0 && return :follicular
    phase_idx == 1 && return :luteal
    phase_idx == 2 && return :menstrual
    return :ovulation
end

function _action_delta(action::AbstractAction, label::Symbol) :: Float64
    hasproperty(action, :deltas) || return 0.0
    deltas = getproperty(action, :deltas)
    deltas isa AbstractDict || return 0.0
    return Float64(get(deltas, label, 0.0))
end

function _segment_at_minute(
    segments::Vector{SegmentSurface},
    minute::Int
) :: Union{SegmentSurface, Nothing}
    wrapped = mod(minute, 1440)
    for segment in segments
        if segment.start_min <= wrapped < segment.end_min
            return segment
        end
    end
    return isempty(segments) ? nothing : segments[1]
end

function _apply_structure_edits(
    segments::Vector{SegmentSurface},
    edits::Vector{StructureEdit}
) :: Vector{SegmentSurface}
    current = deepcopy(segments)
    for edit in edits
        if edit.edit_type == :split
            idx = findfirst(seg -> seg.segment_id == edit.target_segment_id, current)
            isnothing(idx) && continue
            segment = current[idx]
            split_at = something(edit.split_at_minute, (segment.start_min + segment.end_min) ÷ 2)
            segment.start_min < split_at < segment.end_min || continue
            left = SegmentSurface(
                segment_id = "$(segment.segment_id)_a",
                start_min = segment.start_min,
                end_min = split_at,
                isf = segment.isf,
                cr = segment.cr,
                basal = segment.basal,
            )
            right = SegmentSurface(
                segment_id = "$(segment.segment_id)_b",
                start_min = split_at,
                end_min = segment.end_min,
                isf = segment.isf,
                cr = segment.cr,
                basal = segment.basal,
            )
            prefix = idx > 1 ? current[1:idx-1] : SegmentSurface[]
            suffix = idx < length(current) ? current[idx+1:end] : SegmentSurface[]
            current = vcat(prefix, [left, right], suffix)
        elseif edit.edit_type == :merge
            idx = findfirst(seg -> seg.segment_id == edit.target_segment_id, current)
            isnothing(idx) && continue
            neighbor_idx = isnothing(edit.neighbor_segment_id) ?
                (idx < length(current) ? idx + 1 : nothing) :
                findfirst(seg -> seg.segment_id == edit.neighbor_segment_id, current)
            isnothing(neighbor_idx) && continue
            a = current[idx]
            b = current[neighbor_idx]
            first, second = a.start_min <= b.start_min ? (a, b) : (b, a)
            first.end_min == second.start_min || continue
            merged = SegmentSurface(
                segment_id = "$(first.segment_id)__$(second.segment_id)",
                start_min = first.start_min,
                end_min = second.end_min,
                isf = (first.isf + second.isf) / 2,
                cr = (first.cr + second.cr) / 2,
                basal = (first.basal + second.basal) / 2,
            )
            current = sort(
                [seg for seg in current if seg.segment_id != first.segment_id && seg.segment_id != second.segment_id];
                by = seg -> seg.start_min
            )
            push!(current, merged)
            sort!(current, by = seg -> seg.start_min)
        end
    end
    return current
end

function _apply_segment_deltas(
    segments::Vector{SegmentSurface},
    deltas::Vector{SegmentDelta}
) :: Vector{SegmentSurface}
    delta_lookup = Dict(delta.segment_id => delta for delta in deltas)
    return [
        if isnothing(get(delta_lookup, segment.segment_id, nothing))
            segment
        else
            delta = delta_lookup[segment.segment_id]
            SegmentSurface(
                segment_id = segment.segment_id,
                start_min = segment.start_min,
                end_min = segment.end_min,
                isf = max(1e-6, segment.isf * (1.0 + delta.isf_delta)),
                cr = max(1e-6, segment.cr * (1.0 + delta.cr_delta)),
                basal = max(1e-6, segment.basal * (1.0 + delta.basal_delta)),
            )
        end
        for segment in segments
    ]
end

function _scheduled_segments(action::ScheduledAction) :: Vector{SegmentSurface}
    segments = deepcopy(action.segments)
    isempty(segments) && return SegmentSurface[]
    segments = _apply_structure_edits(segments, action.structural_edits)
    segments = _apply_segment_deltas(segments, action.segment_deltas)
    return sort(segments, by = seg -> seg.start_min)
end

function _normal(mean::Float64, std::Float64) :: Normal
    return Normal(mean, max(std, 1e-3))
end

function _base_prior(label::Symbol) :: Distribution
    label === :isf_multiplier && return _normal(1.0, 0.12)
    label === :cr_multiplier && return _normal(1.0, 0.12)
    label === :basal_multiplier && return _normal(1.0, 0.10)
    label === :bg_avg && return _normal(_DEFAULT_BG_AVG, 18.0)
    label === :tir_7d && return Beta(13, 7)
    label === :pct_low_7d && return Beta(2, 70)
    label === :pct_high_7d && return Beta(8, 26)
    label === :bg_cv && return _normal(_DEFAULT_BG_CV, 0.06)
    label === :prev_bg_avg && return _normal(_DEFAULT_BG_AVG, 18.0)
    label === :prev_tir_7d && return Beta(13, 7)
    label === :prev_pct_low_7d && return Beta(2, 70)
    label === :prev_pct_high_7d && return Beta(8, 26)
    label === :prev_bg_cv && return _normal(_DEFAULT_BG_CV, 0.06)
    label === :hours_since_exercise && return Uniform(0.0, 72.0)
    label === :hour_of_day && return Uniform(0.0, 23.0)
    label === :exercise_intensity && return Beta(3, 3)
    label === :sleep_debt && return _normal(55.0, 25.0)
    label === :stress_acute && return Beta(2, 8)
    label === :stress_baseline && return Beta(2, 10)
    label === :stress_reactivity && return Beta(3, 4)
    label === :cycle_sensitivity && return Beta(2, 5)
    label === :cycle_phase_code && return Uniform(0.0, 3.0)
    label === :cgm_lag_minutes && return Uniform(10.0, 20.0)
    label === :cgm_drift_bias && return Uniform(-3.0, 3.0)
    label === :cgm_missingness_rate && return Beta(2, 18)
    return _normal(0.0, 1.0)
end

function _persona_prior(sim::InSiteSimulator, label::Symbol) :: Distribution
    overrides = get(_PERSONA_PRIOR_OVERRIDES, sim.persona, nothing)
    if !isnothing(overrides) && haskey(overrides, label)
        μ, σ = overrides[label]
        return _normal(μ, σ)
    end
    return _base_prior(label)
end

function _register_physical_defaults!(sim::InSiteSimulator, prior::TwinPrior) :: Nothing
    for label in (
        :isf_multiplier, :cr_multiplier, :basal_multiplier,
        :bg_avg, :tir_7d, :pct_low_7d, :pct_high_7d, :bg_cv,
        :prev_bg_avg, :prev_tir_7d, :prev_pct_low_7d, :prev_pct_high_7d, :prev_bg_cv,
        :hours_since_exercise, :hour_of_day, :exercise_intensity, :sleep_debt,
        :stress_acute, :stress_baseline, :stress_reactivity,
        :cycle_sensitivity, :cycle_phase_code,
        :cgm_lag_minutes, :cgm_drift_bias, :cgm_missingness_rate
    )
        haskey(prior.physical_priors, label) && continue
        register_physical_prior!(prior, label, _persona_prior(sim, label))
    end
    return nothing
end

function _context_effectors(
    base_params::NamedTuple,
    vars::Dict{Symbol, Float64}
) :: NamedTuple
    k1 = base_params.k1
    k2 = base_params.k2
    egp0 = base_params.EGP0

    hours_since_exercise = _state_value(vars, :hours_since_exercise, 24.0)
    exercise_intensity = _clamp01(_state_value(vars, :exercise_intensity, 0.4))
    sleep_debt = max(0.0, _state_value(vars, :sleep_debt, 60.0))
    stress = _clamp01(_state_value(vars, :stress_acute, 0.2))
    stress_baseline = _clamp01(_state_value(vars, :stress_baseline, 0.15))
    stress_reactivity = _clamp01(_state_value(vars, :stress_reactivity, 0.45))
    cycle_sensitivity = _clamp01(_state_value(vars, :cycle_sensitivity, 0.3))
    phase = _phase_name(_state_value(vars, :cycle_phase_code, 0.0))

    if hours_since_exercise < 48.0
        decay = exp(-hours_since_exercise / 16.0)
        boost = 1.0 + 0.40 * exercise_intensity * decay
        k1 *= boost
        k2 *= boost
    end

    if phase == :luteal
        r = 1.0 - (0.03 + 0.15 * cycle_sensitivity)
        k1 *= r
        k2 *= r
    elseif phase == :menstrual
        # Gap 7: IS elevated during menstrual phase (Brown et al. 2015)
        k1 *= 1.0 + (0.03 + 0.08 * cycle_sensitivity)
        k2 *= 1.0 + (0.02 + 0.05 * cycle_sensitivity)
    elseif phase == :follicular
        k1 *= 1.0 + 0.02 * cycle_sensitivity
        k2 *= 1.0 + 0.02 * cycle_sensitivity
    elseif phase == :ovulation
        k1 *= 1.0 + 0.05 * cycle_sensitivity
        k2 *= 1.0 + 0.05 * cycle_sensitivity
    end

    sleep_ref_min = 510.0
    sleep_min_last_night = max(240.0, sleep_ref_min - sleep_debt)
    sleep_deficit_h = max(0.0, (sleep_ref_min - sleep_min_last_night) / 60.0)
    si_reduction = min(0.30, 0.035 * sleep_deficit_h) * (0.7 + 0.3 * stress_reactivity)
    k1 *= (1.0 - si_reduction)
    k2 *= (1.0 - 0.6 * si_reduction)
    egp0 *= (1.0 + min(0.28, 0.025 * sleep_deficit_h * (0.7 + 0.3 * stress_reactivity)))

    sleep_efficiency = clamp(0.92 - sleep_debt / 500.0, 0.55, 0.97)
    if sleep_efficiency < 0.82
        frag_penalty = 0.25 * max(0.0, (0.82 - sleep_efficiency) / 0.82)
        k1 *= (1.0 - 0.5 * frag_penalty)
        egp0 *= (1.0 + 0.3 * frag_penalty)
    end

    if stress > 0.1
        stress_egp_factor = if stress < 0.3
            0.12 * (stress - 0.1)
        elseif stress < 0.6
            0.024 + 0.35 * (stress - 0.3)
        else
            0.129 + 0.65 * (stress - 0.6)
        end
        effect = 1.0 + stress_egp_factor * stress_reactivity
        egp0 *= effect
        k1 *= 1.0 / (1.0 + 0.45 * (effect - 1.0))
    end

    if stress_baseline > 0.15
        chronic = min(0.20, 0.15 * (stress_baseline - 0.15) * stress_reactivity)
        egp0 *= (1.0 + chronic)
    end

    return (k1=k1, k2=k2, EGP0=egp0)
end

function _meal_load(hour_of_day::Int, vars::Dict{Symbol, Float64}) :: Float64
    stress = _clamp01(_state_value(vars, :stress_acute, 0.2))
    exercise_bonus = max(0.0, 1.0 - _state_value(vars, :hours_since_exercise, 24.0) / 48.0)
    cr_multiplier = _state_value(vars, :cr_multiplier, 1.0)
    meal_scale = clamp(1.0 + 0.25 * stress + 0.10 * exercise_bonus + 0.15 * (cr_multiplier - 1.0), 0.6, 1.5)
    if hour_of_day == 8
        return 40.0 * meal_scale
    elseif hour_of_day == 13
        return 55.0 * meal_scale
    elseif hour_of_day == 19
        return 65.0 * meal_scale
    end
    return 0.0
end

function _simulate_hour_cgm(
    base_params::NamedTuple,
    modified_params::NamedTuple,
    start_bg::Float64,
    meal_carbs::Float64,
    noise::Dict{Symbol, Float64},
    hour_of_day::Int = 0,
    twin_phys::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
) :: Vector{Float64}
    rng = MersenneTwister(abs(hash((base_params.k1, base_params.k2, base_params.EGP0,
                                    modified_params.k1, modified_params.k2,
                                    modified_params.EGP0, start_bg, meal_carbs))))
    bg = zeros(Float64, 12)
    bg[1] = clamp(start_bg + randn(rng) * 2.0, 45.0, 450.0)
    meal_effect = zeros(Float64, 12)

    if meal_carbs > 0.0
        for j in eachindex(meal_effect)
            gastric = exp(-(j - 1) * 0.22 * 0.55)
            elimination = exp(-(j - 1) * 0.22 * 1.35)
            meal_effect[j] = max(0.0, gastric - elimination)
        end
        meal_effect ./= max(sum(meal_effect), 1e-6)
        meal_effect .*= 9.0 * meal_carbs
    end

    sens = modified_params.k1 / max(1e-6, base_params.k1)
    egp = modified_params.EGP0 / max(1e-6, base_params.EGP0)
    bg_noise = get(noise, :bg_noise, 0.0)

    # Gap 5: Dawn phenomenon (GH surge 3-8 AM)
    hour_f = Float64(hour_of_day)
    if 3.0 <= hour_f <= 8.0
        dawn_sens = get(twin_phys, :dawn_sensitivity, 0.0)
        if dawn_sens > 0
            ramp = sin(π * (hour_f - 3.0) / 5.667)
            egp *= (1.0 + dawn_sens * 0.22 * ramp)
        end
    end

    for i in 2:12
        drift = (egp - 1.0) * 1.8 - (sens - 1.0) * 1.2
        bg[i] = clamp(
            bg[i - 1] + 0.12 * meal_effect[i] + drift + randn(rng) * 2.8 + 0.2 * bg_noise,
            45.0,
            450.0
        )
    end

    return bg
end

function _daily_metrics(bg::Vector{Float64}) :: NamedTuple
    μ = mean(bg)
    σ = std(bg)
    tir = count(x -> 70.0 <= x <= 180.0, bg) / length(bg)
    pct_low = count(x -> x < 70.0, bg) / length(bg)
    pct_high = count(x -> x > 180.0, bg) / length(bg)
    bg_cv = σ / max(μ, 1e-6)
    return (
        bg_avg = μ,
        tir = tir,
        pct_low = pct_low,
        pct_high = pct_high,
        bg_cv = bg_cv,
        bg_var = σ^2
    )
end

function sim_step!(
    sim::InSiteSimulator,
    state::PatientState,
    action::AbstractAction,
    noise::Dict{Symbol, Float64}
) :: PatientState
    vars = copy(state.phys.variables)

    current_isf = _state_value(vars, :isf_multiplier, 1.0)
    current_cr = _state_value(vars, :cr_multiplier, 1.0)
    current_basal = _state_value(vars, :basal_multiplier, 1.0)
    hour_of_day = mod(round(Int, _state_value(vars, :hour_of_day, 7.0)), 24)
    current_minute = hour_of_day * 60

    if action isa ScheduledAction && !isempty(action.segments)
        base_segment = _segment_at_minute(action.segments, current_minute)
        edited_segments = _scheduled_segments(action)
        edited_segment = _segment_at_minute(edited_segments, current_minute)
        if isnothing(base_segment) || isnothing(edited_segment)
            next_isf = current_isf
            next_cr = current_cr
            next_basal = current_basal
        else
            next_isf = clamp(current_isf * edited_segment.isf / max(base_segment.isf, 1e-6), 0.70, 1.35)
            next_cr = clamp(current_cr * edited_segment.cr / max(base_segment.cr, 1e-6), 0.70, 1.35)
            next_basal = clamp(current_basal * edited_segment.basal / max(base_segment.basal, 1e-6), 0.75, 1.25)
        end
    else
        next_isf = clamp(current_isf * (1.0 + _action_delta(action, :isf_delta)), 0.70, 1.35)
        next_cr = clamp(current_cr * (1.0 + _action_delta(action, :cr_delta)), 0.70, 1.35)
        next_basal = clamp(current_basal * (1.0 + _action_delta(action, :basal_delta)), 0.75, 1.25)
    end

    vars[:hours_since_exercise] = get(noise, :activity_noise, 0.0) > 0.55 ?
        max(0.0, 6.0 - 6.0 * get(noise, :activity_noise, 0.0)) :
        min(72.0, _state_value(vars, :hours_since_exercise, 24.0) + 24.0)
    vars[:exercise_intensity] = clamp(
        0.75 * _state_value(vars, :exercise_intensity, 0.4) +
        0.25 * abs(get(noise, :bg_noise, 0.0)) / 10.0,
        0.0,
        1.0
    )
    vars[:stress_acute] = _clamp01(
        0.70 * _state_value(vars, :stress_acute, 0.2) +
        0.30 * abs(get(noise, :bg_noise, 0.0)) / 10.0
    )
    vars[:stress_baseline] = _clamp01(
        0.92 * _state_value(vars, :stress_baseline, 0.15) +
        0.08 * vars[:stress_acute]
    )
    vars[:sleep_debt] = clamp(
        0.85 * _state_value(vars, :sleep_debt, 55.0) +
        15.0 * vars[:stress_acute] -
        10.0 * max(0.0, 1.0 - vars[:hours_since_exercise] / 48.0),
        0.0,
        240.0
    )
    vars[:cycle_phase_code] = mod(_state_value(vars, :cycle_phase_code, 0.0) + 1.0, 4.0)

    base_params = (
        k1 = current_isf / max(current_cr, 1e-6),
        k2 = current_isf / max(current_cr, 1e-6),
        EGP0 = 1.0 / max(current_basal, 1e-6)
    )
    vars[:isf_multiplier] = next_isf
    vars[:cr_multiplier] = next_cr
    vars[:basal_multiplier] = next_basal

    modified_params = _context_effectors((
        k1 = next_isf / max(next_cr, 1e-6),
        k2 = next_isf / max(next_cr, 1e-6),
        EGP0 = 1.0 / max(next_basal, 1e-6)
    ), vars)

    current_bg = _state_value(vars, :bg_avg, _DEFAULT_BG_AVG)
    hour_bg = _simulate_hour_cgm(base_params, modified_params, current_bg, _meal_load(hour_of_day, vars), noise, hour_of_day, vars)
    metrics = _daily_metrics(hour_bg)

    prev_bg_avg = _state_value(vars, :bg_avg, metrics.bg_avg)
    prev_tir = _state_value(vars, :tir_7d, _DEFAULT_TIR)
    prev_pct_low = _state_value(vars, :pct_low_7d, _DEFAULT_PCT_LOW)
    prev_pct_high = _state_value(vars, :pct_high_7d, _DEFAULT_PCT_HIGH)
    prev_bg_cv = _state_value(vars, :bg_cv, _DEFAULT_BG_CV)

    α = 1.0 / 168.0
    vars[:prev_bg_avg] = prev_bg_avg
    vars[:prev_tir_7d] = prev_tir
    vars[:prev_pct_low_7d] = prev_pct_low
    vars[:prev_pct_high_7d] = prev_pct_high
    vars[:prev_bg_cv] = prev_bg_cv

    vars[:bg_avg] = (1.0 - α) * prev_bg_avg + α * metrics.bg_avg
    vars[:tir_7d] = _clamp01((1.0 - α) * prev_tir + α * metrics.tir)
    vars[:pct_low_7d] = _clamp01((1.0 - α) * prev_pct_low + α * metrics.pct_low)
    vars[:pct_high_7d] = _clamp01((1.0 - α) * prev_pct_high + α * metrics.pct_high)
    vars[:bg_cv] = clamp((1.0 - α) * prev_bg_cv + α * metrics.bg_cv, 0.05, 1.0)
    vars[:cgm_lag_minutes] = clamp(get(noise, :cgm_lag, _state_value(vars, :cgm_lag_minutes, 15.0)), 10.0, 20.0)
    vars[:cgm_drift_bias] = clamp(
        0.85 * _state_value(vars, :cgm_drift_bias, 0.0) + 0.15 * get(noise, :cgm_drift, 0.0),
        -6.0,
        6.0
    )
    vars[:cgm_missingness_rate] = clamp(
        0.90 * _state_value(vars, :cgm_missingness_rate, 0.08) + 0.10 * max(0.0, get(noise, :cgm_missingness, 0.08)),
        0.0,
        0.35
    )
    vars[:hour_of_day] = mod(hour_of_day + 1, 24)

    return PatientState(PhysState(vars), state.psy)
end

function sim_observe(
    sim::InSiteSimulator,
    state::PatientState
) :: Observation
    _ = sim
    vars = state.phys.variables
    lag_minutes = clamp(_state_value(vars, :cgm_lag_minutes, 15.0), 10.0, 20.0)
    lag_frac = lag_minutes / 60.0
    drift_bias = _state_value(vars, :cgm_drift_bias, 0.0)
    missingness_rate = clamp(_state_value(vars, :cgm_missingness_rate, 0.08), 0.0, 1.0)

    current_bg = _state_value(vars, :bg_avg, _DEFAULT_BG_AVG)
    prev_bg = _state_value(vars, :prev_bg_avg, current_bg)
    observed_bg = clamp((1.0 - lag_frac) * current_bg + lag_frac * prev_bg + drift_bias + randn() * 5.0, 40.0, 450.0)

    signals = Dict{Symbol, Any}(
        :bg_avg => observed_bg,
        :tir_7d => _clamp01(_state_value(vars, :tir_7d, _DEFAULT_TIR)),
        :pct_low_7d => _clamp01(_state_value(vars, :pct_low_7d, _DEFAULT_PCT_LOW)),
        :pct_high_7d => _clamp01(_state_value(vars, :pct_high_7d, _DEFAULT_PCT_HIGH)),
        :bg_cv => clamp(_state_value(vars, :bg_cv, _DEFAULT_BG_CV), 0.05, 1.0),
        :bg_var => max(1.0, (clamp(_state_value(vars, :bg_cv, _DEFAULT_BG_CV), 0.05, 1.0) * observed_bg)^2),
        :sleep_debt => max(0.0, _state_value(vars, :sleep_debt, 55.0)),
        :stress_acute => _clamp01(_state_value(vars, :stress_acute, 0.2)),
        :hours_since_exercise => max(0.0, _state_value(vars, :hours_since_exercise, 24.0))
    )

    if rand() < missingness_rate
        signals[:stress_acute] = nothing
    end
    if rand() < 0.5 * missingness_rate
        signals[:hours_since_exercise] = nothing
    end
    if rand() < 0.35 * missingness_rate
        signals[:sleep_debt] = nothing
    end
    if rand() < 0.15 * missingness_rate
        signals[:bg_avg] = nothing
    end

    return Observation(timestamp=Float64(time()), signals=signals)
end

function register_priors!(sim::InSiteSimulator, prior::TwinPrior) :: Nothing
    _register_physical_defaults!(sim, prior)
    # Gap 5: dawn sensitivity — Beta(1.5, 3) with mean ~0.33; ~50% of T1D patients
    if !haskey(prior.physical_priors, :dawn_sensitivity)
        register_physical_prior!(prior, :dawn_sensitivity, Beta(1.5, 3))
    end
    return nothing
end

function register_noise!(::InSiteSimulator, noise::RolloutNoise) :: Nothing
    register_physical_noise!(noise, :bg_noise, Normal(0.0, 5.0))
    register_physical_noise!(noise, :cgm_lag, Uniform(10.0, 20.0))
    register_physical_noise!(noise, :cgm_drift, Uniform(-3.0, 3.0))
    register_physical_noise!(noise, :cgm_missingness, Beta(2, 18))
    register_physical_noise!(noise, :activity_noise, Beta(2, 5))
    return nothing
end

action_dimensions(::InSiteSimulator) = [:isf_delta, :cr_delta, :basal_delta]

"""
    min_clinical_delta(::InSiteSimulator) → 0.03

3% minimum relative change per parameter for InSite/T1D settings recommendations.

Rationale:
  - ISF of 50 mg/dL/U → must change by ≥1.5 mg/dL/U. Below this is within
    typical CGM noise and pump granularity.
  - CR of 10 g/U → must change by ≥0.3 g/U. Sub-threshold changes are not
    actionable on most pumps.
  - Basal of 1.0 U/hr → must change by ≥0.03 U/hr. Most pumps have 0.025–0.05
    U/hr minimum increment.

This does NOT replace the statistical effect size gate (δ_min_effect); both
gates must pass. This gate asks "is the action physically meaningful?"
The effect size gate asks "is the improvement statistically real?"
"""
min_clinical_delta(::InSiteSimulator) = 0.03

safety_thresholds(::InSiteSimulator) = Dict(
    :pct_low_max => 0.04,
    :pct_high_max => 0.25
)

function compute_frustration(
    ::InSiteSimulator,
    signals::Dict{Symbol, Any}
) :: Float64
    pct_low = Float64(get(signals, :pct_low_7d, get(signals, :pct_low, _DEFAULT_PCT_LOW)))
    tir = Float64(get(signals, :tir_7d, get(signals, :tir, _DEFAULT_TIR)))
    return clamp(pct_low + 0.5 * (1.0 - tir), 0.0, 1.0)
end

function compute_physical_cost(
    ::InSiteSimulator,
    signals::Dict{Symbol, Any},
    weights::Dict{Symbol, Float64}
) :: Float64
    pct_low = Float64(get(signals, :pct_low_7d, get(signals, :pct_low, _DEFAULT_PCT_LOW)))
    pct_high = Float64(get(signals, :pct_high_7d, get(signals, :pct_high, _DEFAULT_PCT_HIGH)))
    tir = Float64(get(signals, :tir_7d, get(signals, :tir, _DEFAULT_TIR)))
    bg_cv = Float64(get(signals, :bg_cv, _DEFAULT_BG_CV))

    w_low = get(weights, :w_low, 5.0)
    w_high = get(weights, :w_high, 1.0)
    w_var = get(weights, :w_var, 0.5)
    w_tir = get(weights, :w_tir, 1.0)

    return w_low * pct_low + w_high * pct_high + w_var * bg_cv - w_tir * tir
end

function check_safety(
    ::InSiteSimulator,
    rollout_signals::Vector{Dict{Symbol, Any}},
    thresholds::Dict{Symbol, Float64}
) :: Bool
    pct_low_max = get(thresholds, :pct_low_max, 0.04)
    pct_high_max = get(thresholds, :pct_high_max, 0.25)

    for signals in rollout_signals
        pct_low = Float64(get(signals, :pct_low_7d, get(signals, :pct_low, _DEFAULT_PCT_LOW)))
        pct_high = Float64(get(signals, :pct_high_7d, get(signals, :pct_high, _DEFAULT_PCT_HIGH)))
        if !(pct_low < pct_low_max && pct_high < pct_high_max)
            return false
        end
    end

    return true
end
