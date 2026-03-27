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

using Main: AbstractBeliefState, AbstractSimulator, AbstractDomainAdapter,
            DigitalTwin, MemoryBuffer,
            ConfiguratorState, RolloutNoise, UserPreferences, Observation,
            PsyState, GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            TwinPrior, TwinPosterior, NullAction, UserResponse, RecommendationPackage,
            ConnectedAppCapabilities, ConnectedAppState,
            KalmanBeliefEstimator, JEPABeliefEstimator, EpistemicState,
            SignalRegistry, register_signal!, initialize_noise

import Main: WorldModule, Cost, Actor, Memory, Perception, Configurator, Twin

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
    weights_dir :: Union{String, Nothing} = nothing
) :: ChameliaSystem
    prior = _build_twin_prior(prefs)

    noise = initialize_noise()
    WorldModule.register_priors!(sim, prior)
    WorldModule.register_noise!(sim, noise)

    posterior = Twin.initialize_posterior(prior)
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

    resolved_adapter = isnothing(adapter) ? Main.InSiteDomainAdapter() : adapter

    return ChameliaSystem(
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
end

function observe!(
    system::ChameliaSystem,
    obs::Observation
) :: Nothing
    system.last_obs_time = obs.timestamp
    isempty(obs.signals) && return nothing

    if !isnothing(system.encoder) && system.graduated
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
            system.last_decision_reason = :epistemic_failed
            Memory.store_hold!(
                system.mem,
                system.current_day,
                system.belief,
                epistemic,
                :epistemic_failed,
                system.config,
                psy;
                configurator_mode = active_cfg_mode
            )
            return nothing
        end

        dims = WorldModule.action_dimensions(system.sim)
        thresholds = WorldModule.safety_thresholds(system.sim)
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
            obs.signals,
            system.mem
        )
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
                configurator_mode = active_cfg_mode
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
            configurator_mode = active_cfg_mode
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
    Memory.store_outcome!(system.mem, rec_id, response, signals, cost)
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

    if !isnothing(system.encoder)
        Cost.train_decoder!(Cost.LATENT_DECODER, system.mem)
    end

    Actor.train_actor_cql!(system.mem)
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
        belief_mode = belief_mode,
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
            adapter=Main.InSiteDomainAdapter(),
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
       save_patient,
       load_patient,
       graduation_status

end # module Chamelia
