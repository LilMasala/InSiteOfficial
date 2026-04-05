if !isdefined(Main, :AbstractSimulator)
    include("types.jl")
end

if !isdefined(Main, :WorldModule)
    include("WorldModule/WorldModule.jl")
end

if !isdefined(Main, :Cost)
    include("Cost/Cost.jl")
end

if !isdefined(Main, :Actor)
    include("Actor/Actor.jl")
end

if !isdefined(Main, :Memory)
    include("Memory/Memory.jl")
end

if !isdefined(Main, :Perception)
    include("Perception/Perception.jl")
end

if !isdefined(Main, :Configurator)
    include("Configurator/Configurator.jl")
end

if !isdefined(Main, :Twin)
    include("Twin/Twin.jl")
end

if !isdefined(Main, :PythonBridge)
    include("PythonBridge.jl")
end

if !isdefined(Main, :InSiteSimulator)
    include("InSiteSimulator.jl")
end

if !isdefined(Main, :InSiteDomainAdapter)
    include("InSiteDomainAdapter.jl")
end

module Chamelia

using Serialization
using Statistics
using Distributions
using Flux
using BSON

using Main: AbstractAction, AbstractBeliefState, AbstractSimulator, AbstractDomainAdapter,
            DigitalTwin, MemoryBuffer, MemoryRecord,
            ConfiguratorState, RolloutNoise, UserPreferences, Observation,
            PsyState, GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            TwinPrior, TwinPosterior, NullAction, UserResponse,
            Accept, Reject, Partial,
            RecommendationPackage,
            ConnectedAppCapabilities, ConnectedAppState,
            KalmanBeliefEstimator, JEPABeliefEstimator, EpistemicState,
            SignalRegistry, register_signal!, initialize_noise,
            bridge_domain_name, bridge_encode_payload, bridge_domain_state,
            BridgeDecodedCandidate, BridgeProposalAdvisory,
            bridge_decode_candidate_proposals, bridge_action_summary

import Main: WorldModule, Cost, Actor, Memory, Perception, Configurator, Twin, calibrate_posterior!
import Main.PythonBridge

using Main.WorldModule
using Main.Cost
using Main.Actor
using Main.Memory
using Main.Perception
using Main.Configurator
using Main.Twin

mutable struct ChameliaSystem
    belief          :: AbstractBeliefState
    twin            :: DigitalTwin
    mem             :: MemoryBuffer
    config          :: ConfiguratorState
    sim             :: AbstractSimulator
    noise           :: RolloutNoise
    prefs           :: UserPreferences
    graduated       :: Bool
    current_day     :: Int
    last_obs_time   :: Float64
    last_decision_reason :: Symbol
    last_safety_diagnostics :: Any
    config_history  :: Vector{ConfiguratorState}
    meta_history    :: Vector{Any}
    perf_history    :: Vector{Float64}
    encoder         :: Union{HierarchicalJEPAEncoder, Nothing}
    predictor       :: Union{JEPAPredictor, Nothing}
    adapter         :: AbstractDomainAdapter   # domain adapter used at every step
end

const _JEPA_PARAMS = IdDict{ChameliaSystem, JEPAInferenceParams}()
const _SYSTEMS = Dict{Int64, ChameliaSystem}()
const _LAST_PACKAGES = Dict{Int64, RecommendationPackage}()
const _NEXT_SYSTEM_HANDLE = Ref{Int64}(1)
const _PYTHON_BRIDGES = IdDict{ChameliaSystem, PythonBridge.BridgeConfig}()
const _LAST_BRIDGE_DIAGNOSTICS = IdDict{ChameliaSystem, Any}()
const _LAST_BRIDGE_BUNDLES = IdDict{ChameliaSystem, Dict{String, Any}}()
const _PYTHON_BRIDGE_REQUEST_FNS = IdDict{ChameliaSystem, Function}()
const _PYTHON_BRIDGE_REPLAY_SYNC_CURSORS = IdDict{ChameliaSystem, Dict{String, Int}}()
const _LEGACY_JEPA_COMPAT = IdDict{ChameliaSystem, Bool}()

function _string_or_nothing(ptr::Cstring) :: Union{String, Nothing}
    ptr == C_NULL && return nothing
    return unsafe_string(ptr)
end

function _parse_signal_blob(blob::Union{String, Nothing}) :: Dict{Symbol, Any}
    isnothing(blob) && return Dict{Symbol, Any}()
    isempty(strip(blob)) && return Dict{Symbol, Any}()

    signals = Dict{Symbol, Any}()
    for entry in split(blob, ';')
        isempty(strip(entry)) && continue
        parts = split(entry, '='; limit=2)
        length(parts) == 2 || continue
        key = Symbol(strip(parts[1]))
        raw = strip(parts[2])
        parsed = tryparse(Float64, raw)
        signals[key] = isnothing(parsed) ? raw : parsed
    end

    return signals
end

function _register_system!(system::ChameliaSystem) :: Int64
    handle = _NEXT_SYSTEM_HANDLE[]
    _NEXT_SYSTEM_HANDLE[] += 1
    _SYSTEMS[handle] = system
    return handle
end

function _lookup_system(handle::Integer) :: Union{ChameliaSystem, Nothing}
    return get(_SYSTEMS, Int64(handle), nothing)
end

function _normalize_bridge_url(base_url::String) :: String
    normalized = strip(base_url)
    while !isempty(normalized) && endswith(normalized, "/")
        normalized = normalized[1:end-1]
    end
    return normalized
end

function _resolved_bridge_url(bridge_url::Union{String, Nothing}) :: Union{String, Nothing}
    if !isnothing(bridge_url)
        return _normalize_bridge_url(bridge_url)
    end
    env_url = strip(get(ENV, "CHAMELIA_PYTHON_BRIDGE_URL", ""))
    return isempty(env_url) ? nothing : _normalize_bridge_url(env_url)
end

function _validate_bridge_mode(mode::String) :: String
    normalized = strip(mode)
    normalized in ("v1.1", "v1.5", "v3") || throw(ArgumentError("unsupported bridge mode `$mode`"))
    return normalized
end

function set_legacy_jepa_compat!(
    system::ChameliaSystem;
    enabled::Bool=true,
) :: Nothing
    if enabled
        isnothing(system.encoder) && throw(ArgumentError("legacy JEPA compatibility requires loaded JEPA weights"))
        _LEGACY_JEPA_COMPAT[system] = true
    else
        pop!(_LEGACY_JEPA_COMPAT, system, nothing)
    end
    return nothing
end

function legacy_jepa_compat_enabled(system::ChameliaSystem) :: Bool
    return get(_LEGACY_JEPA_COMPAT, system, false)
end

function legacy_jepa_mode_label(system::ChameliaSystem) :: String
    isnothing(system.encoder) && return "disabled"
    return legacy_jepa_compat_enabled(system) ? "compatibility_enabled" : "compatibility_only"
end

function configure_python_bridge!(
    system::ChameliaSystem;
    base_url::String,
    mode::String="v3",
    session_id::String="default",
    model_version::String="unknown",
    timeout_s::Float64=5.0,
    rollout_horizon::Int=2,
) :: Nothing
    normalized_url = _normalize_bridge_url(base_url)
    isempty(normalized_url) && throw(ArgumentError("bridge URL cannot be empty"))
    normalized_mode = _validate_bridge_mode(mode)
    _PYTHON_BRIDGES[system] = PythonBridge.BridgeConfig(
        base_url=normalized_url,
        mode=normalized_mode,
        domain_name=bridge_domain_name(system.adapter),
        session_id=session_id,
        model_version=model_version,
        timeout_s=timeout_s,
        rollout_horizon=rollout_horizon,
    )
    return nothing
end

function _bridge_replay_sync_key(bridge::PythonBridge.BridgeConfig) :: String
    return string(bridge.domain_name, "::", bridge.mode, "::", bridge.model_version, "::", bridge.session_id)
end

function _record_bridge_replay_sync_status!(
    system::ChameliaSystem,
    status::Dict{String, Any},
) :: Dict{String, Any}
    diagnostics = get(_LAST_BRIDGE_DIAGNOSTICS, system, nothing)
    updated = diagnostics isa AbstractDict ?
        Dict{String, Any}(String(key) => deepcopy(value) for (key, value) in pairs(diagnostics)) :
        Dict{String, Any}()
    updated["replay_sync"] = deepcopy(status)
    _LAST_BRIDGE_DIAGNOSTICS[system] = updated
    return status
end

function sync_python_bridge_replay!(
    system::ChameliaSystem;
    full_resync::Bool=false,
) :: Union{Dict{String, Any}, Nothing}
    bridge = get(_PYTHON_BRIDGES, system, nothing)
    isnothing(bridge) && return nothing

    sync_key = _bridge_replay_sync_key(bridge)
    cursor_map = get!(() -> Dict{String, Int}(), _PYTHON_BRIDGE_REPLAY_SYNC_CURSORS, system)
    since_record_id = full_resync ? nothing : get(cursor_map, sync_key, nothing)
    examples = export_bridge_replay_examples(
        system;
        since_record_id=since_record_id,
        model_version=bridge.model_version,
    )

    if isempty(examples)
        status = Dict{String, Any}(
            "bridge_ok" => true,
            "domain_name" => bridge.domain_name,
            "mode" => bridge.mode,
            "session_id" => bridge.session_id,
            "model_version" => bridge.model_version,
            "full_resync" => full_resync,
            "since_record_id" => since_record_id,
            "exported_examples" => 0,
            "ingested" => 0,
            "duplicates" => 0,
            "skipped" => 0,
            "last_synced_record_id" => get(cursor_map, sync_key, nothing),
        )
        return _record_bridge_replay_sync_status!(system, status)
    end

    request_fn = get(_PYTHON_BRIDGE_REQUEST_FNS, system, nothing)
    result = PythonBridge.ingest_replay_examples(
        bridge,
        examples;
        request_fn=request_fn,
    )
    last_record_id = maximum(Int(get(example, "record_id", 0)) for example in examples)
    cursor_map[sync_key] = max(get(cursor_map, sync_key, 0), last_record_id)
    status = Dict{String, Any}(
        "bridge_ok" => true,
        "domain_name" => bridge.domain_name,
        "mode" => bridge.mode,
        "session_id" => bridge.session_id,
        "model_version" => bridge.model_version,
        "full_resync" => full_resync,
        "since_record_id" => since_record_id,
        "exported_examples" => length(examples),
        "ingested" => get(result, "ingested", 0),
        "duplicates" => get(result, "duplicates", 0),
        "skipped" => get(result, "skipped", 0),
        "last_synced_record_id" => cursor_map[sync_key],
    )
    return _record_bridge_replay_sync_status!(system, status)
end

function disable_python_bridge!(system::ChameliaSystem) :: Nothing
    pop!(_PYTHON_BRIDGES, system, nothing)
    pop!(_LAST_BRIDGE_DIAGNOSTICS, system, nothing)
    pop!(_LAST_BRIDGE_BUNDLES, system, nothing)
    pop!(_PYTHON_BRIDGE_REQUEST_FNS, system, nothing)
    pop!(_PYTHON_BRIDGE_REPLAY_SYNC_CURSORS, system, nothing)
    return nothing
end

function set_python_bridge_request_fn!(system::ChameliaSystem, request_fn::Function) :: Nothing
    _PYTHON_BRIDGE_REQUEST_FNS[system] = request_fn
    return nothing
end

function clear_python_bridge_request_fn!(system::ChameliaSystem) :: Nothing
    pop!(_PYTHON_BRIDGE_REQUEST_FNS, system, nothing)
    return nothing
end

function _performance_delta(mem::MemoryBuffer, rec_id::Int) :: Float64
    rec = Memory.get_record(mem, rec_id)
    (isnothing(rec) || isnothing(rec.realized_cost)) && return 0.0

    baseline = [
        something(r.realized_cost, 0.0)
        for r in mem.records
        if r.id != rec_id && !isnothing(r.realized_cost)
    ]

    isempty(baseline) && return -something(rec.realized_cost, 0.0)
    return mean(baseline) - something(rec.realized_cost, 0.0)
end

function _infer_encoder_counts(
    encoder::HierarchicalJEPAEncoder
) :: NamedTuple{(:n_subhourly, :n_ctx, :n_daily), NTuple{3, Int}}
    n_subhourly = size(encoder.hourly.input_proj.weight, 2)
    d_hourly = size(encoder.hourly.output_proj.weight, 1)
    n_ctx = max(0, size(encoder.daily.input_proj.weight, 2) - d_hourly)
    d_daily = size(encoder.daily.output_proj.weight, 1)
    n_daily = max(0, size(encoder.multiday.input_proj.weight, 2) - d_daily)
    return (n_subhourly=n_subhourly, n_ctx=n_ctx, n_daily=n_daily)
end

function _labels_for(
    obs::Observation,
    n::Int,
    prefix::String
) :: Vector{Symbol}
    n <= 0 && return Symbol[]
    base = sort!(collect(keys(obs.signals)); by=string)
    labels = Symbol[]

    for i in 1:n
        if i <= length(base)
            push!(labels, base[i])
        else
            push!(labels, Symbol(prefix * string(i)))
        end
    end

    return labels
end

function _ensure_jepa_params!(
    system::ChameliaSystem,
    obs::Observation
) :: JEPAInferenceParams
    haskey(_JEPA_PARAMS, system) && return _JEPA_PARAMS[system]
    isnothing(system.encoder) && error("JEPA weights are not available for this system")

    counts = _infer_encoder_counts(system.encoder)
    params = JEPAInferenceParams(
        system.encoder,
        _labels_for(obs, counts.n_subhourly, "subhourly_"),
        _labels_for(obs, counts.n_ctx, "ctx_"),
        _labels_for(obs, counts.n_daily, "daily_")
    )
    _JEPA_PARAMS[system] = params
    return params
end

function _load_optional_jepa(
    weights_dir::Union{String, Nothing}
) :: Tuple{Union{HierarchicalJEPAEncoder, Nothing}, Union{JEPAPredictor, Nothing}}
    # Legacy Julia JEPA assets are loadable for compatibility/reference work,
    # but they are not part of the default runtime path unless explicitly enabled.
    isnothing(weights_dir) && return nothing, nothing

    encoder_path = joinpath(weights_dir, "jepa_encoder.bson")
    predictor_path = joinpath(weights_dir, "jepa_predictor.bson")
    if !(isfile(encoder_path) && isfile(predictor_path))
        return nothing, nothing
    end

    encoder_state = BSON.load(encoder_path)
    predictor_state = BSON.load(predictor_path)

    encoder = encoder_state[:encoder]
    predictor = predictor_state[:predictor]

    Flux.loadmodel!(WorldModule.JEPA_PREDICTOR, predictor)
    return encoder, WorldModule.JEPA_PREDICTOR
end

function _build_twin_prior(prefs::UserPreferences) :: TwinPrior
    phys = Dict{Symbol, Distribution}()

    if !isempty(prefs.physical_priors)
        for (key, params) in prefs.physical_priors
            length(params) == 2 || continue
            μ = Float64(params[1])
            σ = max(Float64(params[2]), 0.02)
            phys[Symbol(key)] = Normal(μ, σ)
        end
    end

    return TwinPrior( 
        trust_growth_dist = Beta(2, 10),
        trust_decay_dist = Beta(5, 10),
        burnout_sensitivity_dist = Beta(2, 5),
        engagement_decay_dist = Beta(2, 8),
        physical_priors = phys,
        persona_label = isempty(prefs.physical_priors) ? prefs.persona : "questionnaire_derived"
    )
end

function initialize_patient(
    prefs   :: UserPreferences,
    sim     :: AbstractSimulator;
    adapter :: Union{AbstractDomainAdapter, Nothing} = nothing,
    weights_dir :: Union{String, Nothing} = nothing,
    legacy_jepa_compat :: Bool = false,
    bridge_url :: Union{String, Nothing} = nothing,
    bridge_mode :: String = "v3",
    bridge_session_id :: String = "default",
    bridge_model_version :: String = "unknown",
    bridge_timeout_s :: Float64 = 5.0,
) :: ChameliaSystem
    resolved_adapter = isnothing(adapter) ? Main.DefaultDomainAdapter() : adapter

    prior = _build_twin_prior(prefs)

    noise = initialize_noise()
    WorldModule.register_priors!(sim, prior)
    WorldModule.register_noise!(sim, noise)

    posterior = Twin.initialize_posterior(prior)

    # ── Cold-start calibration from self-reported glycemic targets ──────
    if !isempty(prefs.calibration_targets)
        calibrate_posterior!(resolved_adapter, posterior, prior, prefs.calibration_targets)
    end

    twin = DigitalTwin(prior, posterior, Float64(std(noise.trust_noise)))
    belief = Perception.initialize_belief(prior, KalmanBeliefEstimator())

    # Use domain adapter for physical cost weights when one is provided;
    # fall back to the InSite-domain backward-compat path otherwise.
    config = if !isnothing(adapter)
        Configurator.initialize_config(prefs, adapter)
    else
        Configurator.initialize_config(prefs)
    end

    encoder, predictor = _load_optional_jepa(weights_dir)
    legacy_jepa_compat && isnothing(encoder) && throw(ArgumentError("legacy JEPA compatibility was requested but no JEPA weights were loaded"))

    system = ChameliaSystem(
        belief,
        twin,
        MemoryBuffer(),
        config,
        sim,
        noise,
        prefs,
        false,
        0,
        0.0,
        :initialized,
        nothing,
        ConfiguratorState[],
        Any[],
        Float64[],
        encoder,
        predictor,
        resolved_adapter
    )
    legacy_jepa_compat && set_legacy_jepa_compat!(system; enabled=true)
    resolved_bridge_url = _resolved_bridge_url(bridge_url)
    if !isnothing(resolved_bridge_url)
        configure_python_bridge!(
            system;
            base_url=resolved_bridge_url,
            mode=bridge_mode,
            session_id=bridge_session_id,
            model_version=bridge_model_version,
            timeout_s=bridge_timeout_s,
        )
    end
    return system
end

function _maybe_run_python_bridge!(
    system::ChameliaSystem,
    obs::Observation,
) :: Nothing
    bridge = get(_PYTHON_BRIDGES, system, nothing)
    isnothing(bridge) && return nothing

    try
        encode_payload = bridge_encode_payload(system.adapter, obs)
        domain_state = bridge_domain_state(system.adapter, obs)
        request_fn = get(_PYTHON_BRIDGE_REQUEST_FNS, system, nothing)
        bundle = PythonBridge.run_pipeline(
            bridge,
            encode_payload,
            domain_state;
            request_fn=request_fn,
        )
        _LAST_BRIDGE_BUNDLES[system] = deepcopy(bundle)
        _LAST_BRIDGE_DIAGNOSTICS[system] = PythonBridge.summarize_bundle(bundle)
    catch err
        pop!(_LAST_BRIDGE_BUNDLES, system, nothing)
        _LAST_BRIDGE_DIAGNOSTICS[system] = Dict(
            "bridge_ok" => false,
            "mode" => bridge.mode,
            "domain_name" => bridge.domain_name,
            "session_id" => bridge.session_id,
            "model_version" => bridge.model_version,
            "error_type" => string(typeof(err)),
            "error_message" => sprint(showerror, err),
        )
    end

    return nothing
end

function _bridge_decoded_candidates(
    system::ChameliaSystem,
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Vector{BridgeDecodedCandidate}
    bundle = get(_LAST_BRIDGE_BUNDLES, system, nothing)
    proposal_bundle = bundle isa Dict ? get(bundle, "proposal_bundle", nothing) : nothing
    proposal_bundle isa AbstractDict || return BridgeDecodedCandidate[]
    proposal_payload = Dict{String, Any}(String(key) => value for (key, value) in pairs(proposal_bundle))
    return bridge_decode_candidate_proposals(system.adapter, proposal_payload, capabilities, app_state)
end

function _bridge_diagnostics_snapshot(
    system::ChameliaSystem;
    candidate_source_used::Union{Nothing, String}=nothing,
    decoded_candidate_count::Union{Nothing, Int}=nothing,
    selected_bridge_candidate_idx::Union{Nothing, Int}=nothing,
    selected_bridge_candidate_slot::Union{Nothing, Int}=nothing,
    python_advisory_available::Union{Nothing, Bool}=nothing,
    python_advisory_considered::Union{Nothing, Bool}=nothing,
    selected_matches_python_top_candidate::Union{Nothing, Bool}=nothing,
    selected_python_candidate_total::Union{Nothing, Float64}=nothing,
    selection_stage::Union{Nothing, String}=nothing,
    decision_reason::Union{Nothing, Symbol}=nothing,
    accepted_action::Union{Nothing, Bool}=nothing,
) :: Union{Dict{String, Any}, Nothing}
    diagnostics = get(_LAST_BRIDGE_DIAGNOSTICS, system, nothing)
    diagnostics isa AbstractDict || return nothing

    snapshot = Dict{String, Any}(String(key) => deepcopy(value) for (key, value) in pairs(diagnostics))
    !isnothing(candidate_source_used) && (snapshot["candidate_source_used"] = candidate_source_used)
    !isnothing(decoded_candidate_count) && (snapshot["decoded_candidate_count"] = decoded_candidate_count)
    snapshot["selected_bridge_candidate_idx"] = selected_bridge_candidate_idx
    snapshot["selected_bridge_candidate_slot"] = selected_bridge_candidate_slot
    !isnothing(python_advisory_available) && (snapshot["python_advisory_available"] = python_advisory_available)
    !isnothing(python_advisory_considered) && (snapshot["python_advisory_considered"] = python_advisory_considered)
    snapshot["selected_matches_python_top_candidate"] = selected_matches_python_top_candidate
    snapshot["selected_python_candidate_total"] = selected_python_candidate_total
    !isnothing(selection_stage) && (snapshot["selection_stage"] = selection_stage)
    !isnothing(decision_reason) && (snapshot["decision_reason"] = string(decision_reason))
    !isnothing(accepted_action) && (snapshot["accepted_action"] = accepted_action)
    return snapshot
end

function _bridge_vector_slot(values, slot::Int)
    values isa AbstractVector || return nothing
    1 <= slot <= length(values) || return nothing
    return deepcopy(values[slot])
end

function _bridge_float_or_nothing(value) :: Union{Float64, Nothing}
    value isa Number || return nothing
    return Float64(value)
end

function _bridge_candidate_advisories(
    bundle_snapshot::Union{Dict{String, Any}, Nothing},
    decoded_candidates::Vector{BridgeDecodedCandidate},
) :: Vector{BridgeProposalAdvisory}
    critic_scores = bundle_snapshot isa Dict ? get(bundle_snapshot, "critic_scores", Dict{String, Any}()) : Dict{String, Any}()
    totals = get(critic_scores, "candidate_total", nothing)
    ranked_slots = Int[]
    if totals isa AbstractVector
        numeric = Tuple{Int, Float64}[]
        for (slot, total) in pairs(totals)
            total isa Number || continue
            push!(numeric, (slot, Float64(total)))
        end
        sort!(numeric, by = item -> item[2])
        ranked_slots = [item[1] for item in numeric]
    end
    rank_lookup = Dict{Int, Int}(slot => rank for (rank, slot) in pairs(ranked_slots))

    advisories = BridgeProposalAdvisory[]
    for decoded in decoded_candidates
        slot = decoded.bridge_candidate_slot
        push!(advisories, BridgeProposalAdvisory(
            bridge_candidate_idx = decoded.bridge_candidate_idx,
            bridge_candidate_slot = slot,
            python_ic = _bridge_float_or_nothing(_bridge_vector_slot(get(critic_scores, "candidate_ic", nothing), slot)),
            python_tc = _bridge_float_or_nothing(_bridge_vector_slot(get(critic_scores, "candidate_tc", nothing), slot)),
            python_total = _bridge_float_or_nothing(_bridge_vector_slot(get(critic_scores, "candidate_total", nothing), slot)),
            python_rank = get(rank_lookup, slot, nothing),
        ))
    end
    return advisories
end

function _selected_decoded_candidate(
    decoded_candidates::Vector{BridgeDecodedCandidate},
    selected_action::Union{AbstractAction, Nothing},
) :: Union{BridgeDecodedCandidate, Nothing}
    isnothing(selected_action) && return nothing
    for decoded in decoded_candidates
        decoded.action == selected_action && return decoded
    end
    return nothing
end

function _bridge_decoded_candidate_details(
    bundle_snapshot::Union{Dict{String, Any}, Nothing},
    decoded_candidates::Vector{BridgeDecodedCandidate},
) :: Vector{Dict{String, Any}}
    proposal_bundle = bundle_snapshot isa Dict ? get(bundle_snapshot, "proposal_bundle", Dict{String, Any}()) : Dict{String, Any}()
    critic_scores = bundle_snapshot isa Dict ? get(bundle_snapshot, "critic_scores", Dict{String, Any}()) : Dict{String, Any}()

    details = Dict{String, Any}[]
    for decoded in decoded_candidates
        slot = decoded.bridge_candidate_slot
        push!(details, Dict{String, Any}(
            "bridge_candidate_idx" => decoded.bridge_candidate_idx,
            "bridge_candidate_slot" => slot,
            "decoded_action" => bridge_action_summary(decoded.action),
            "decode_metadata" => isnothing(decoded.decode_metadata) ? nothing : deepcopy(decoded.decode_metadata),
            "candidate_path" => _bridge_vector_slot(get(proposal_bundle, "candidate_paths", nothing), slot),
            "candidate_posture" => _bridge_vector_slot(get(proposal_bundle, "candidate_postures", nothing), slot),
            "candidate_reasoning_state" => _bridge_vector_slot(get(proposal_bundle, "candidate_reasoning_states", nothing), slot),
            "python_candidate_ic" => _bridge_vector_slot(get(critic_scores, "candidate_ic", nothing), slot),
            "python_candidate_tc" => _bridge_vector_slot(get(critic_scores, "candidate_tc", nothing), slot),
            "python_candidate_total" => _bridge_vector_slot(get(critic_scores, "candidate_total", nothing), slot),
        ))
    end
    return details
end

function _bridge_trace_and_diagnostics_snapshot(
    system::ChameliaSystem,
    decoded_candidates::Vector{BridgeDecodedCandidate},
    proposal_advisories::Vector{BridgeProposalAdvisory},
    selected_action::Union{AbstractAction, Nothing},
    reason::Symbol;
    selection_stage::String,
) :: Tuple{Union{Dict{String, Any}, Nothing}, Union{Dict{String, Any}, Nothing}}
    trace_snapshot = let bundle = get(_LAST_BRIDGE_BUNDLES, system, nothing)
        bundle isa Dict ? deepcopy(bundle) : nothing
    end

    decoded_details = _bridge_decoded_candidate_details(trace_snapshot, decoded_candidates)
    selected_decoded = _selected_decoded_candidate(decoded_candidates, selected_action)
    source = selection_stage == "actor_selection" ? (isempty(decoded_candidates) ? "legacy_fallback" : "python_bridge") : nothing
    advisory_available = any(!isnothing(advisory.python_total) || !isnothing(advisory.python_ic) || !isnothing(advisory.python_tc) for advisory in proposal_advisories)
    selected_advisory = if isnothing(selected_decoded)
        nothing
    else
        findfirst(advisory -> advisory.bridge_candidate_slot == selected_decoded.bridge_candidate_slot, proposal_advisories)
    end
    selected_advisory_total = isnothing(selected_advisory) ? nothing : proposal_advisories[selected_advisory].python_total
    selected_matches_python_top = if isnothing(selected_decoded)
        nothing
    else
        python_selected = get(get(_LAST_BRIDGE_DIAGNOSTICS, system, Dict{String, Any}()), "python_selected_candidate_idx", nothing)
        python_selected isa Integer ? Int(python_selected) == selected_decoded.bridge_candidate_idx : nothing
    end

    selection = Dict{String, Any}(
        "selection_stage" => selection_stage,
        "decision_reason" => string(reason),
        "accepted_action" => !isnothing(selected_action),
        "candidate_source_used" => source,
        "decoded_candidate_count" => length(decoded_candidates),
        "bridge_candidates_rejected" => selection_stage == "actor_selection" && !isempty(decoded_candidates) && isnothing(selected_action),
        "python_selected_candidate_idx" => get(get(_LAST_BRIDGE_DIAGNOSTICS, system, Dict{String, Any}()), "python_selected_candidate_idx", nothing),
        "python_advisory_available" => advisory_available,
        "python_advisory_considered" => selection_stage == "actor_selection" && advisory_available,
        "selected_matches_python_top_candidate" => selected_matches_python_top,
        "selected_python_candidate_total" => selected_advisory_total,
        "selected_bridge_candidate_idx" => isnothing(selected_decoded) ? nothing : selected_decoded.bridge_candidate_idx,
        "selected_bridge_candidate_slot" => isnothing(selected_decoded) ? nothing : selected_decoded.bridge_candidate_slot,
        "selected_action" => isnothing(selected_action) ? nothing : bridge_action_summary(selected_action),
    )

    if trace_snapshot isa Dict
        trace_snapshot["decoded_candidates"] = decoded_details
        if !isnothing(selected_decoded)
            selected_detail = findfirst(detail -> get(detail, "bridge_candidate_slot", nothing) == selected_decoded.bridge_candidate_slot, decoded_details)
            !isnothing(selected_detail) && (selection["selected_candidate"] = deepcopy(decoded_details[selected_detail]))
        end
        trace_snapshot["julia_selection"] = selection
    end

    diagnostics_snapshot = _bridge_diagnostics_snapshot(
        system;
        candidate_source_used=source,
        decoded_candidate_count=length(decoded_candidates),
        selected_bridge_candidate_idx=isnothing(selected_decoded) ? nothing : selected_decoded.bridge_candidate_idx,
        selected_bridge_candidate_slot=isnothing(selected_decoded) ? nothing : selected_decoded.bridge_candidate_slot,
        python_advisory_available=advisory_available,
        python_advisory_considered=selection_stage == "actor_selection" && advisory_available,
        selected_matches_python_top_candidate=selected_matches_python_top,
        selected_python_candidate_total=selected_advisory_total,
        selection_stage=selection_stage,
        decision_reason=reason,
        accepted_action=!isnothing(selected_action),
    )

    return trace_snapshot, diagnostics_snapshot
end

function observe!(
    system::ChameliaSystem,
    obs::Observation
) :: Nothing
    system.last_obs_time = obs.timestamp
    isempty(obs.signals) && return nothing

    # Old Julia JEPA belief updates are compatibility-only. Loading weights alone
    # does not activate this path in the default runtime.
    if !isnothing(system.encoder) && system.graduated && legacy_jepa_compat_enabled(system)
        if !(system.belief isa JEPABeliefState)
            system.belief = Perception.initialize_belief(system.twin.prior, JEPABeliefEstimator())
        end

        params = _ensure_jepa_params!(system, obs)
        system.belief = Perception.update_belief(
            system.belief,
            obs,
            NullAction(),
            system.twin,
            JEPABeliefEstimator(),
            system.mem,
            params,
            system.config
        )
        return nothing
    end

    if !(system.belief isa GaussianBeliefState)
        system.belief = Perception.initialize_belief(system.twin.prior, KalmanBeliefEstimator())
    end

    registry = SignalRegistry()
    for (label, value) in obs.signals
        dtype = isnothing(value) ? Float64 : typeof(value)
        register_signal!(registry, label, 1.0, dtype, false)
    end

    n_phys = length((system.belief::GaussianBeliefState).x̂_phys)
    params = Perception.initialize_kalman(system.twin, registry, n_phys)
    system.belief = Perception.update_belief(
        system.belief,
        obs,
        NullAction(),
        system.twin,
        KalmanBeliefEstimator(),
        system.mem,
        params,
        system.config
    )

    return nothing
end

function psy_from_belief(belief::GaussianBeliefState) :: PsyState
    return PsyState(
        trust = ScalarTrust(belief.x̂_trust),
        burden = ScalarBurden(belief.x̂_burden),
        engagement = ScalarEngagement(belief.x̂_engagement),
        burnout = ScalarBurnout(belief.x̂_burnout)
    )
end

function psy_from_belief(belief::ParticleBeliefState) :: PsyState
    isempty(belief.particles) && return PsyState(
        trust = ScalarTrust(0.0),
        burden = ScalarBurden(0.0),
        engagement = ScalarEngagement(0.0),
        burnout = ScalarBurnout(0.0)
    )

    weights = isempty(belief.weights) ? fill(1.0 / length(belief.particles), length(belief.particles)) : belief.weights
    wsum = sum(weights)
    normalized = wsum > 0 ? weights ./ wsum : fill(1.0 / length(weights), length(weights))

    weighted(field) = sum(normalized[i] * field(belief.particles[i]) for i in eachindex(belief.particles))

    return PsyState(
        trust = ScalarTrust(weighted(p -> p.psy.trust.value)),
        burden = ScalarBurden(weighted(p -> p.psy.burden.value)),
        engagement = ScalarEngagement(weighted(p -> p.psy.engagement.value)),
        burnout = ScalarBurnout(weighted(p -> p.psy.burnout.value))
    )
end

function psy_from_belief(belief::JEPABeliefState) :: PsyState
    summary = Cost.decode_latent_summary(belief)
    return PsyState(
        trust = ScalarTrust(summary.trust),
        burden = ScalarBurden(summary.burden),
        engagement = ScalarEngagement(summary.engagement),
        burnout = ScalarBurnout(summary.burnout)
    )
end

function psy_from_belief(belief::AbstractBeliefState) :: PsyState
    return PsyState(
        trust = ScalarTrust(0.0),
        burden = ScalarBurden(0.0),
        engagement = ScalarEngagement(0.0),
        burnout = ScalarBurnout(0.0)
    )
end

function step!(
    system::ChameliaSystem,
    obs::Observation;
    capabilities::ConnectedAppCapabilities=ConnectedAppCapabilities(),
    app_state::ConnectedAppState=ConnectedAppState(),
) :: Union{RecommendationPackage, Nothing}
    pkg_out = nothing
    system.current_day += 1
    observe!(system, obs)
    _maybe_run_python_bridge!(system, obs)

    try
        psy = psy_from_belief(system.belief)
        epistemic_for_adapt = Perception.compute_epistemic_state(
            system.belief,
            system.mem,
            NullAction(),
            system.config.φ_cost.thresholds
        )

        system.config = Configurator.adapt(
            system.config,
            system.belief,
            epistemic_for_adapt,
            system.mem,
            psy,
            system.prefs,
            system.current_day;
            graduated=system.graduated,
            last_decision_reason=system.last_decision_reason,
        )

        active_cfg_mode = Configurator.active_mode()

        meta = Configurator.compute_meta_state(
            system.belief,
            epistemic_for_adapt,
            system.mem,
            psy,
            system.current_day;
            graduated=system.graduated,
            last_decision_reason=system.last_decision_reason,
        )
        push!(system.config_history, deepcopy(system.config))
        push!(system.meta_history, meta)

        epistemic = Perception.compute_epistemic_state(
            system.belief,
            system.mem,
            NullAction(),
            system.config.φ_cost.thresholds
        )

        if !epistemic.feasible
            bridge_trace_snapshot, bridge_diagnostics_snapshot = _bridge_trace_and_diagnostics_snapshot(
                system,
                BridgeDecodedCandidate[],
                BridgeProposalAdvisory[],
                nothing,
                :epistemic_failed;
                selection_stage="epistemic_gate",
            )
            if bridge_diagnostics_snapshot isa Dict
                _LAST_BRIDGE_DIAGNOSTICS[system] = bridge_diagnostics_snapshot
            end
            system.last_decision_reason = :epistemic_failed
            Memory.store_hold!(
                system.mem,
                system.current_day,
                system.belief,
                epistemic,
                :epistemic_failed,
                system.config,
                psy;
                configurator_mode = active_cfg_mode,
                bridge_trace = bridge_trace_snapshot,
                bridge_diagnostics = bridge_diagnostics_snapshot,
            )
            return nothing
        end

        dims = WorldModule.action_dimensions(system.sim)
        thresholds = WorldModule.safety_thresholds(system.sim)
        decoded_candidates = _bridge_decoded_candidates(system, capabilities, app_state)
        proposal_advisories = _bridge_candidate_advisories(get(_LAST_BRIDGE_BUNDLES, system, nothing), decoded_candidates)
        pkg, reason, safety_diagnostics = Actor.decide(
            system.belief,
            system.twin,
            system.sim,
            system.noise,
            Memory.current_critic(system.mem),
            epistemic,
            system.config,
            dims,
            thresholds,
            system.current_day,
            capabilities,
            app_state,
            system.adapter,
            system.prefs,
            obs.signals,
            system.mem;
            proposal_actions = AbstractAction[decoded.action for decoded in decoded_candidates],
            proposal_advisories = proposal_advisories,
        )
        bridge_trace_snapshot, bridge_diagnostics_snapshot = _bridge_trace_and_diagnostics_snapshot(
            system,
            decoded_candidates,
            proposal_advisories,
            isnothing(pkg) ? nothing : pkg.action,
            reason;
            selection_stage="actor_selection",
        )
        if bridge_diagnostics_snapshot isa Dict
            _LAST_BRIDGE_DIAGNOSTICS[system] = bridge_diagnostics_snapshot
        end
        system.last_safety_diagnostics = safety_diagnostics

        if isnothing(pkg)
            system.last_decision_reason = reason
            Memory.store_hold!(
                system.mem,
                system.current_day,
                system.belief,
                epistemic,
                reason,
                system.config,
                psy;
                configurator_mode = active_cfg_mode,
                bridge_trace = bridge_trace_snapshot,
                bridge_diagnostics = bridge_diagnostics_snapshot,
            )
            return nothing
        end

        rec_id = Memory.store_record!(
            system.mem,
            system.current_day,
            system.belief,
            pkg.action,
            epistemic,
            system.config,
            psy;
            predicted_cvar    = pkg.cvar_value,
            configurator_mode = active_cfg_mode,
            bridge_trace = bridge_trace_snapshot,
            bridge_diagnostics = bridge_diagnostics_snapshot,
        )
        _ = rec_id
        system.last_decision_reason = reason

        if !system.graduated
            _refresh_graduation!(system)
            return nothing
        end

        pkg_out = pkg
        return pkg_out
    finally
        Memory.maybe_update_critic!(system.mem, system.current_day, system.config)
        Memory.maybe_update_twin_posterior!(system.twin, system.mem, system.current_day)
    end
end

function record_outcome!(
    system  :: ChameliaSystem,
    rec_id  :: Int,
    response::Union{UserResponse, Nothing},
    signals :: Dict{Symbol, Any},
    cost    :: Float64
) :: Nothing
    # Capture the current belief latent BEFORE anything mutates system state.
    # At this call site system.belief has already been updated by the observe/step
    # calls that happened between recommendation and outcome recording, so it IS
    # the z_{t+H} we need for JEPA predictor training.
    outcome_latent = _latent_μ_from_belief(system.belief)

    Memory.store_outcome!(system.mem, rec_id, response, signals, cost;
                          latent_μ_at_outcome = outcome_latent)
    _refresh_graduation!(system)
    Memory.maybe_update_critic!(system.mem, system.current_day, system.config)

    performance = _performance_delta(system.mem, rec_id)
    push!(system.perf_history, performance)

    epistemic = Perception.compute_epistemic_state(
        system.belief,
        system.mem,
        NullAction(),
        system.config.φ_cost.thresholds
    )

    meta = isempty(system.meta_history) ?
        Configurator.compute_meta_state(
            system.belief,
            epistemic,
            system.mem,
            psy_from_belief(system.belief),
            system.current_day;
            graduated=system.graduated,
            last_decision_reason=system.last_decision_reason,
        ) :
        (system.meta_history[end] isa Configurator.MetaState ?
            system.meta_history[end] :
            Configurator.compute_meta_state(
                system.belief,
                epistemic,
                system.mem,
                psy_from_belief(system.belief),
                system.current_day;
                graduated=system.graduated,
                last_decision_reason=system.last_decision_reason,
            ))

    Configurator.record_outcome!(system.config, meta, performance, system.prefs)

    n = min(length(system.meta_history), length(system.config_history), length(system.perf_history))
    if n > 0
        metas = Configurator.MetaState[
            system.meta_history[i]::Configurator.MetaState
            for i in (length(system.meta_history) - n + 1):length(system.meta_history)
        ]
        configs = system.config_history[(length(system.config_history) - n + 1):length(system.config_history)]
        perfs = system.perf_history[(length(system.perf_history) - n + 1):length(system.perf_history)]
        Configurator.maybe_train_cql!(metas, configs, perfs, system.prefs, system.current_day)
    end

    if legacy_jepa_compat_enabled(system) && !isnothing(system.encoder)
        Cost.train_decoder!(Cost.LATENT_DECODER, system.mem)
    end

    Actor.train_actor_cql!(system.mem)
    maybe_train_predictor!(system)
    return nothing
end

function _bridge_replay_example(
    system::ChameliaSystem,
    rec::MemoryRecord,
) :: Union{Dict{String, Any}, Nothing}
    trace = rec.bridge_trace
    outcome = rec.bridge_outcome
    trace isa AbstractDict || return nothing
    outcome isa AbstractDict || return nothing

    selection = get(trace, "julia_selection", nothing)
    selection isa AbstractDict || return nothing

    source = get(selection, "candidate_source_used", nothing)
    source == "python_bridge" || return nothing

    selected_slot = get(selection, "selected_bridge_candidate_slot", nothing)
    selected_slot isa Integer || return nothing

    encoded_state = get(trace, "encoded_state", nothing)
    configurator_output = get(trace, "configurator_output", nothing)
    proposal_bundle = get(trace, "proposal_bundle", nothing)
    critic_scores = get(trace, "critic_scores", nothing)
    retrieved_memory = get(trace, "retrieved_memory", nothing)

    encoded_state isa AbstractDict || return nothing
    configurator_output isa AbstractDict || return nothing
    proposal_bundle isa AbstractDict || return nothing
    critic_scores isa AbstractDict || return nothing

    z_t = get(encoded_state, "z_t", nothing)
    ctx_tokens = get(configurator_output, "ctx_tokens", nothing)
    selected_action_vec = _bridge_vector_slot(get(proposal_bundle, "candidate_actions", nothing), selected_slot)
    selected_path = _bridge_vector_slot(get(proposal_bundle, "candidate_paths", nothing), selected_slot)
    selected_posture = _bridge_vector_slot(get(proposal_bundle, "candidate_postures", nothing), selected_slot)
    selected_reasoning_state = _bridge_vector_slot(get(proposal_bundle, "candidate_reasoning_states", nothing), selected_slot)
    realized_cost = get(outcome, "realized_cost", nothing)
    outcome_key = get(outcome, "latent_μ_at_outcome", nothing)

    z_t isa AbstractVector || return nothing
    ctx_tokens isa AbstractVector || return nothing
    selected_action_vec isa AbstractVector || return nothing
    selected_path isa AbstractVector || return nothing
    realized_cost isa Real || return nothing
    outcome_key isa AbstractVector || return nothing

    retrieval_trace = Dict{String, Any}[]
    if retrieved_memory isa AbstractDict
        retrieved_keys = get(retrieved_memory, "retrieved_keys", nothing)
        retrieved_episode_summaries = get(retrieved_memory, "retrieved_episode_summaries", nothing)
        retrieval_base_quality_scores = get(retrieved_memory, "retrieval_base_quality_scores", nothing)
        if (
            retrieved_keys isa AbstractVector &&
            retrieved_episode_summaries isa AbstractVector &&
            retrieval_base_quality_scores isa AbstractVector
        )
            push!(retrieval_trace, Dict{String, Any}(
                "query_key" => deepcopy(z_t),
                "memory_keys" => deepcopy(retrieved_keys),
                "memory_summaries" => deepcopy(retrieved_episode_summaries),
                "base_quality_scores" => deepcopy(retrieval_base_quality_scores),
                "query_posture" => nothing,
                "memory_postures" => deepcopy(get(retrieved_memory, "retrieved_postures", nothing)),
                "base_scores" => deepcopy(get(retrieved_memory, "retrieval_base_scores", nothing)),
                "relevance_scores" => deepcopy(get(retrieved_memory, "retrieval_relevance_scores", nothing)),
                "relevance_weights" => deepcopy(get(retrieved_memory, "retrieval_relevance_weights", nothing)),
            ))
        end
    end

    return Dict{String, Any}(
        "bridge_version" => get(trace, "bridge_version", "v1"),
        "domain_name" => get(trace, "domain_name", bridge_domain_name(system.adapter)),
        "model_version" => get(trace, "model_version", nothing),
        "record_id" => rec.id,
        "day" => rec.day,
        "z_t" => deepcopy(z_t),
        "ctx_tokens" => deepcopy(ctx_tokens),
        "selected_candidate_idx" => get(selection, "selected_bridge_candidate_idx", nothing),
        "selected_candidate_slot" => selected_slot,
        "selected_action_vec" => deepcopy(selected_action_vec),
        "selected_path" => deepcopy(selected_path),
        "selected_posture" => deepcopy(selected_posture),
        "selected_reasoning_state" => deepcopy(selected_reasoning_state),
        "candidate_actions" => deepcopy(get(proposal_bundle, "candidate_actions", nothing)),
        "candidate_paths" => deepcopy(get(proposal_bundle, "candidate_paths", nothing)),
        "candidate_postures" => deepcopy(get(proposal_bundle, "candidate_postures", nothing)),
        "candidate_reasoning_states" => deepcopy(get(proposal_bundle, "candidate_reasoning_states", nothing)),
        "candidate_ic" => deepcopy(get(critic_scores, "candidate_ic", nothing)),
        "candidate_tc" => deepcopy(get(critic_scores, "candidate_tc", nothing)),
        "candidate_total" => deepcopy(get(critic_scores, "candidate_total", nothing)),
        "selected_candidate_ic" => deepcopy(_bridge_vector_slot(get(critic_scores, "candidate_ic", nothing), selected_slot)),
        "selected_candidate_tc" => deepcopy(_bridge_vector_slot(get(critic_scores, "candidate_tc", nothing), selected_slot)),
        "selected_candidate_total" => deepcopy(_bridge_vector_slot(get(critic_scores, "candidate_total", nothing), selected_slot)),
        "realized_ic" => Float64(realized_cost),
        "outcome_z_tH" => deepcopy(outcome_key),
        "retrieval_trace" => retrieval_trace,
        "source_patient_domain" => bridge_domain_name(system.adapter),
        "julia_selection" => deepcopy(Dict{String, Any}(String(key) => value for (key, value) in pairs(selection))),
        "selected_candidate" => deepcopy(get(selection, "selected_candidate", nothing)),
    )
end

function export_bridge_replay_examples(
    system::ChameliaSystem;
    limit::Union{Nothing, Int}=nothing,
    since_record_id::Union{Nothing, Int}=nothing,
    model_version::Union{Nothing, String}=nothing,
) :: Vector{Dict{String, Any}}
    exported = Dict{String, Any}[]
    for rec in system.mem.records
        !isnothing(since_record_id) && rec.id <= since_record_id && continue
        example = _bridge_replay_example(system, rec)
        isnothing(example) && continue
        if !isnothing(model_version)
            get(example, "model_version", nothing) == model_version || continue
        end
        push!(exported, example)
    end
    if limit === 0
        return Dict{String, Any}[]
    end
    if !isnothing(limit) && limit > 0 && length(exported) > limit
        return exported[(end - limit + 1):end]
    end
    return exported
end

# ─────────────────────────────────────────────────────────────────
# JEPA predictor action-conditioned training
#
# After enough shadow-period outcomes have accumulated in memory we
# fine-tune the predictor on real (z_t, a_eff, z_{t+H}) triples.
# This is SEPARATE from train_encoder! which only learns the latent
# space structure and correctly uses null actions for VICReg.
#
# Trigger: MIN_PREDICTOR_TRAINING_SAMPLES completed triples, then
# every PREDICTOR_TRAINING_INTERVAL new ones to amortise compute.
# ─────────────────────────────────────────────────────────────────

const _MIN_PREDICTOR_TRAINING_SAMPLES = 20
const _PREDICTOR_TRAINING_INTERVAL    = 5

# Extract μ from JEPA belief; nothing for Kalman / particle paths.
function _latent_μ_from_belief(::AbstractBeliefState) :: Union{Vector{Float32}, Nothing}
    return nothing
end
function _latent_μ_from_belief(belief::JEPABeliefState) :: Union{Vector{Float32}, Nothing}
    return Float32.(vec(belief.μ))
end

# Effective action features for a completed memory record.
# We scale by the patient's response so the predictor learns what
# actually happened, not just what was recommended:
#   Accept  → full action (patient implemented the change)
#   Partial → half-magnitude (partial implementation)
#   Reject  → zeros (null dynamics — patient made no change)
#   Hold / no response → zeros
function _effective_action_features(rec::MemoryRecord) :: Vector{Float32}
    a = WorldModule.action_to_features(rec.action)
    response = rec.user_response
    if isnothing(response) || response == Reject
        return zeros(Float32, length(a))
    elseif response == Partial
        return 0.5f0 .* a
    else  # Accept
        return a
    end
end

# Build a MemoryTransitionDataset from all completed records that
# have both latent snapshots populated.
function _build_latent_triples(mem::MemoryBuffer) :: Perception.MemoryTransitionDataset
    z_t_list     = Vector{Float32}[]
    a_feats_list = Vector{Float32}[]
    z_tH_list    = Vector{Float32}[]

    for rec in mem.records
        isnothing(rec.latent_μ_at_rec)     && continue
        isnothing(rec.latent_μ_at_outcome) && continue
        push!(z_t_list,     rec.latent_μ_at_rec)
        push!(a_feats_list, _effective_action_features(rec))
        push!(z_tH_list,    rec.latent_μ_at_outcome)
    end

    return Perception.MemoryTransitionDataset(z_t_list, a_feats_list, z_tH_list)
end

function maybe_train_predictor!(system::ChameliaSystem) :: Nothing
    # Only meaningful when a JEPA predictor is loaded
    legacy_jepa_compat_enabled(system) || return nothing
    isnothing(system.predictor) && return nothing

    # Count records with both latent snapshots
    n = count(
        r -> !isnothing(r.latent_μ_at_rec) && !isnothing(r.latent_μ_at_outcome),
        system.mem.records
    )

    n < _MIN_PREDICTOR_TRAINING_SAMPLES && return nothing

    # Retrain every _PREDICTOR_TRAINING_INTERVAL new triples, not every call
    n % _PREDICTOR_TRAINING_INTERVAL == 0 || return nothing

    dataset = _build_latent_triples(system.mem)
    Perception.train_predictor!(system.predictor, dataset)
    return nothing
end

function save_patient(system::ChameliaSystem, path::String) :: Nothing
    Serialization.serialize(path, system)
    return nothing
end

function load_patient(path::String) :: ChameliaSystem
    return Serialization.deserialize(path)
end

function _refresh_graduation!(system::ChameliaSystem) :: Nothing
    system.graduated && return nothing
    scorecard = Memory.compute_scorecard(system.mem)
    if Actor.passes_graduation_gate(scorecard)
        system.graduated = true
    end
    return nothing
end

function graduation_status(system::ChameliaSystem) :: NamedTuple
    sc = Memory.compute_scorecard(system.mem)
    bridge = get(_PYTHON_BRIDGES, system, nothing)
    belief_mode = system.belief isa JEPABeliefState ? :jepa :
                  system.belief isa GaussianBeliefState ? :kalman :
                  system.belief isa ParticleBeliefState ? :particle :
                  :unknown
    meta = if !isempty(system.meta_history) && system.meta_history[end] isa Configurator.MetaState
        system.meta_history[end]::Configurator.MetaState
    else
        last_epistemic = isempty(system.mem.records) ?
            EpistemicState(
                κ_familiarity=0.5,
                ρ_concordance=0.5,
                η_calibration=0.5,
                feasible=false,
            ) :
            system.mem.records[end].epistemic
        Configurator.compute_meta_state(
            system.belief,
            last_epistemic,
            system.mem,
            psy_from_belief(system.belief),
            system.current_day;
            graduated=system.graduated,
            last_decision_reason=system.last_decision_reason,
        )
    end
    return (
        graduated         = system.graduated || Actor.passes_graduation_gate(sc),
        n_days            = system.current_day,
        win_rate          = sc.win_rate,
        safety_violations = sc.safety_violations,
        consecutive_days  = sc.consecutive_days,
        belief_entropy    = meta.belief_entropy,
        κ_familiarity     = meta.κ_familiarity,
        ρ_concordance     = meta.ρ_concordance,
        η_calibration     = meta.η_calibration,
        trust_level       = meta.trust_level,
        burnout_level     = meta.burnout_level,
        no_surface_streak = meta.no_surface_streak,
        drift_detected    = meta.drift_detected,
        days_since_drift  = meta.days_since_drift,
        n_records         = meta.n_records,
        last_decision_reason = system.last_decision_reason,
        last_safety_diagnostics = system.last_safety_diagnostics,
        configurator_mode = Configurator.active_mode(),
        jepa_weights_loaded = !isnothing(system.encoder),
        jepa_active = system.belief isa JEPABeliefState,
        legacy_jepa_mode = legacy_jepa_mode_label(system),
        legacy_jepa_compat_enabled = legacy_jepa_compat_enabled(system),
        belief_mode = belief_mode,
        python_bridge_enabled = !isnothing(bridge),
        python_bridge_mode = isnothing(bridge) ? nothing : bridge.mode,
        python_bridge_model_version = isnothing(bridge) ? nothing : bridge.model_version,
        python_bridge_session_id = isnothing(bridge) ? nothing : bridge.session_id,
        last_bridge_diagnostics = get(_LAST_BRIDGE_DIAGNOSTICS, system, nothing),
        config = (
            Δ_max = system.config.φ_act.Δ_max,
            δ_min_effect = system.config.φ_act.δ_min_effect,
            α_cvar = system.config.φ_act.α_cvar,
            N_search = system.config.φ_act.N_search,
            N_roll = system.config.φ_world.N_roll,
            H_med = system.config.φ_world.H_med,
            H_burn = system.config.φ_cost.H_burn,
            ε_burn = system.config.φ_cost.ε_burn,
        ),
    )
end

Base.@ccallable function chamelia_initialize_patient(weights_dir::Cstring)::Clonglong
    try
        system = initialize_patient(
            UserPreferences(),
            Main.InSiteSimulator();
            adapter=Main.DefaultDomainAdapter(),
            weights_dir=_string_or_nothing(weights_dir)
        )
        return Clonglong(_register_system!(system))
    catch
        return Clonglong(0)
    end
end

Base.@ccallable function chamelia_observe(
    handle::Clonglong,
    timestamp::Cdouble,
    signals_blob::Cstring
)::Cint
    try
        system = _lookup_system(handle)
        isnothing(system) && return Cint(0)
        obs = Observation(
            timestamp=Float64(timestamp),
            signals=_parse_signal_blob(_string_or_nothing(signals_blob))
        )
        observe!(system, obs)
        return Cint(1)
    catch
        return Cint(0)
    end
end

Base.@ccallable function chamelia_step(
    handle::Clonglong,
    timestamp::Cdouble,
    signals_blob::Cstring
)::Clonglong
    try
        system = _lookup_system(handle)
        isnothing(system) && return Clonglong(0)
        obs = Observation(
            timestamp=Float64(timestamp),
            signals=_parse_signal_blob(_string_or_nothing(signals_blob))
        )
        pkg = step!(system, obs)
        if isnothing(pkg)
            delete!(_LAST_PACKAGES, Int64(handle))
            return Clonglong(0)
        end

        _LAST_PACKAGES[Int64(handle)] = pkg
        return Clonglong(system.mem.next_id - 1)
    catch
        return Clonglong(0)
    end
end

Base.@ccallable function chamelia_record_outcome(
    handle::Clonglong,
    rec_id::Clonglong,
    response_code::Cint,
    signals_blob::Cstring,
    cost::Cdouble
)::Cint
    try
        system = _lookup_system(handle)
        isnothing(system) && return Cint(0)
        response = response_code < 0 ? nothing : UserResponse(Int(response_code))
        signals = _parse_signal_blob(_string_or_nothing(signals_blob))
        record_outcome!(system, Int(rec_id), response, signals, Float64(cost))
        return Cint(1)
    catch
        return Cint(0)
    end
end

Base.@ccallable function chamelia_save_patient(
    handle::Clonglong,
    path::Cstring
)::Cint
    try
        system = _lookup_system(handle)
        path_str = _string_or_nothing(path)
        (isnothing(system) || isnothing(path_str)) && return Cint(0)
        save_patient(system, path_str)
        return Cint(1)
    catch
        return Cint(0)
    end
end

Base.@ccallable function chamelia_load_patient(path::Cstring)::Clonglong
    try
        path_str = _string_or_nothing(path)
        isnothing(path_str) && return Clonglong(0)
        system = load_patient(path_str)
        return Clonglong(_register_system!(system))
    catch
        return Clonglong(0)
    end
end

Base.@ccallable function chamelia_graduation_status(
    handle::Clonglong,
    graduated_out::Ptr{Cint},
    n_days_out::Ptr{Cint},
    win_rate_out::Ptr{Cdouble},
    safety_out::Ptr{Cint},
    consecutive_out::Ptr{Cint}
)::Cint
    try
        system = _lookup_system(handle)
        isnothing(system) && return Cint(0)
        status = graduation_status(system)
        graduated_out == C_NULL || unsafe_store!(graduated_out, status.graduated ? Cint(1) : Cint(0))
        n_days_out == C_NULL || unsafe_store!(n_days_out, Cint(status.n_days))
        win_rate_out == C_NULL || unsafe_store!(win_rate_out, Cdouble(status.win_rate))
        safety_out == C_NULL || unsafe_store!(safety_out, Cint(status.safety_violations))
        consecutive_out == C_NULL || unsafe_store!(consecutive_out, Cint(status.consecutive_days))
        return Cint(1)
    catch
        return Cint(0)
    end
end

Base.@ccallable function chamelia_free_patient(handle::Clonglong)::Cint
    deleted = pop!(_SYSTEMS, Int64(handle), nothing)
    pop!(_LAST_PACKAGES, Int64(handle), nothing)
    return isnothing(deleted) ? Cint(0) : Cint(1)
end

export ChameliaSystem,
       initialize_patient,
       observe!,
       step!,
       record_outcome!,
       set_legacy_jepa_compat!,
       legacy_jepa_compat_enabled,
       legacy_jepa_mode_label,
       sync_python_bridge_replay!,
       export_bridge_replay_examples,
       save_patient,
       load_patient,
       graduation_status

end # module Chamelia
