if !isdefined(Main, :AbstractSimulator)
    include(joinpath(@__DIR__, "src", "types.jl"))
end

for (name, relpath) in (
    (:WorldModule, joinpath("src", "WorldModule", "WorldModule.jl")),
    (:Cost, joinpath("src", "Cost", "Cost.jl")),
    (:Actor, joinpath("src", "Actor", "Actor.jl")),
    (:Memory, joinpath("src", "Memory", "Memory.jl")),
    (:Perception, joinpath("src", "Perception", "Perception.jl")),
    (:Configurator, joinpath("src", "Configurator", "Configurator.jl")),
    (:Twin, joinpath("src", "Twin", "Twin.jl"))
)
    if !isdefined(Main, name) || !(getfield(Main, name) isa Module)
        include(joinpath(@__DIR__, relpath))
    end
end

if !isdefined(Main, :Chamelia) || !(getfield(Main, :Chamelia) isa Module)
    include(joinpath(@__DIR__, "src", "Chamelia.jl"))
end

module ChameliaServer

using Base64
using Dates
using HTTP
using JSON3
using MbedTLS
using Random
using Serialization
using SHA
using UUIDs

using Main: Chamelia, Observation, UserPreferences, UserResponse, Reject, Partial, Accept

# InSiteDomainAdapter is loaded by Chamelia.jl; explicitly imported here so
# it is clear this server operates in InSite domain mode, not generic core mode.
using Main: InSiteDomainAdapter

abstract type AbstractStateBackend end

struct NotFoundError <: Exception
    message::String
end

Base.showerror(io::IO, err::NotFoundError) = print(io, err.message)

mutable struct InMemoryStateBackend <: AbstractStateBackend
    blobs::Dict{String, Vector{UInt8}}
end

InMemoryStateBackend() = InMemoryStateBackend(Dict{String, Vector{UInt8}}())

mutable struct FirebaseStorageBackend <: AbstractStateBackend
    access_token::Union{Nothing, String}
    expires_at::Float64
    bucket_hint::Union{Nothing, String}
end

FirebaseStorageBackend() = FirebaseStorageBackend(nothing, 0.0, nothing)

const PATIENTS = Dict{String, Chamelia.ChameliaSystem}()
const STATE_BACKEND = Ref{AbstractStateBackend}(FirebaseStorageBackend())

function _timestamp_utc() :: String
    return Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
end

function _nonempty_env(name::String) :: Union{Nothing, String}
    value = strip(get(ENV, name, ""))
    return isempty(value) ? nothing : value
end

function _project_id_for_logging() :: Union{Nothing, String}
    for name in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "FIREBASE_PROJECT_ID")
        value = _nonempty_env(name)
        !isnothing(value) && return value
    end
    return nothing
end

function _header_value(req::HTTP.Request, name::String) :: Union{Nothing, String}
    normalized = lowercase(name)
    for (header_name, header_value) in req.headers
        lowercase(String(header_name)) == normalized && return String(header_value)
    end
    return nothing
end

function _trace_id(req::HTTP.Request) :: Union{Nothing, String}
    trace_header = _header_value(req, "X-Cloud-Trace-Context")
    if !isnothing(trace_header)
        trace_id = split(trace_header, "/", limit=2)[1]
        return isempty(trace_id) ? nothing : trace_id
    end

    traceparent = _header_value(req, "traceparent")
    if !isnothing(traceparent)
        parts = split(traceparent, '-')
        length(parts) >= 2 || return nothing
        return isempty(parts[2]) ? nothing : parts[2]
    end

    return nothing
end

function _request_context(req::HTTP.Request) :: Dict{String, Any}
    method = String(req.method)
    path = HTTP.URIs.URI(String(req.target)).path
    trace_id = _trace_id(req)
    project_id = _project_id_for_logging()
    trace = (isnothing(trace_id) || isnothing(project_id)) ? nothing : "projects/$project_id/traces/$trace_id"

    return Dict(
        "request_id" => something(_header_value(req, "X-Request-Id"), trace_id, string(uuid4())),
        "method" => method,
        "path" => path,
        "service" => something(_nonempty_env("K_SERVICE"), "unknown"),
        "revision" => something(_nonempty_env("K_REVISION"), "unknown"),
        "trace_id" => trace_id,
        "trace" => trace
    )
end

function _merge_fields(dicts...) :: Dict{String, Any}
    merged = Dict{String, Any}()
    for dict in dicts
        for (key, value) in pairs(dict)
            merged[String(key)] = value
        end
    end
    return merged
end

function _normalize_fields(fields) :: Dict{String, Any}
    normalized = Dict{String, Any}()
    for (key, value) in pairs(fields)
        normalized[String(key)] = value
    end
    return normalized
end

function _require_config_value(config::AbstractDict, key::String) :: Any
    haskey(config, key) || error("service account missing $key")
    return config[key]
end

function _require_payload_value(payload::AbstractDict, key::String, message::String) :: Any
    haskey(payload, key) || error(message)
    return payload[key]
end

function _redact_identifier(value::AbstractString) :: String
    raw = strip(String(value))
    visible = min(length(raw), 4)
    suffix = last(raw, visible)
    return string(repeat("*", max(length(raw) - visible, 0)), suffix)
end

function _patient_fields(patient_id::Union{Nothing, AbstractString}) :: Dict{String, Any}
    fields = Dict{String, Any}()
    isnothing(patient_id) && return fields
    fields["patient_id_redacted"] = _redact_identifier(patient_id)
    fields["patient_id_length"] = length(patient_id)
    return fields
end

function _payload_summary(payload::Dict{String, Any}, body_bytes::Integer) :: Dict{String, Any}
    fields = Dict{String, Any}(
        "body_bytes" => Int(body_bytes),
        "body_keys" => sort!(String[String(key) for key in keys(payload)]),
        "auth_mode" => "patient_id_from_body"
    )

    raw_signals = get(payload, "signals", nothing)
    if raw_signals isa AbstractDict
        signal_keys = sort!(String[String(key) for key in keys(raw_signals)])
        sample_count = min(length(signal_keys), 12)
        fields["signals_count"] = length(signal_keys)
        fields["signal_keys_sample"] = signal_keys[1:sample_count]
    end

    raw_preferences = get(payload, "preferences", nothing)
    if raw_preferences isa AbstractDict
        fields["preference_keys"] = sort!(String[String(key) for key in keys(raw_preferences)])
        persona = get(raw_preferences, "persona", nothing)
        persona isa AbstractString && (fields["persona"] = String(persona))
    end

    patient_id = get(payload, "patient_id", nothing)
    patient_id isa AbstractString && merge!(fields, _patient_fields(strip(String(patient_id))))

    return fields
end

function _elapsed_ms(started_at::Float64) :: Float64
    return round((time() - started_at) * 1000; digits=3)
end

function _stacktrace_string(bt) :: String
    io = IOBuffer()
    for frame in stacktrace(bt)
        println(io, frame)
    end
    return String(take!(io))
end

function _emit_log(
    severity::String,
    message::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing,
    stage::Union{Nothing, String}=nothing,
    fields=Dict{String, Any}()
) :: Nothing
    record = Dict{String, Any}(
        "timestamp" => _timestamp_utc(),
        "severity" => severity,
        "message" => message,
        "component" => "chamelia_server"
    )

    if !isnothing(ctx)
        record["request_id"] = ctx["request_id"]
        record["http_method"] = ctx["method"]
        record["path"] = ctx["path"]
        record["service"] = ctx["service"]
        record["revision"] = ctx["revision"]
        !isnothing(ctx["trace_id"]) && (record["trace_id"] = ctx["trace_id"])
        !isnothing(ctx["trace"]) && (record["logging.googleapis.com/trace"] = ctx["trace"])
    end

    !isnothing(stage) && (record["stage"] = stage)

    for (key, value) in pairs(_normalize_fields(fields))
        value === nothing && continue
        record[String(key)] = value
    end

    io = severity in ("ERROR", "CRITICAL", "ALERT", "EMERGENCY") ? stderr : stdout
    println(io, JSON3.write(record))
    flush(io)
    return nothing
end

function _log_exception(
    err;
    bt=catch_backtrace(),
    ctx::Union{Nothing, Dict{String, Any}}=nothing,
    stage::Union{Nothing, String}=nothing,
    fields=Dict{String, Any}()
) :: Nothing
    error_fields = _normalize_fields(fields)
    error_fields["error_type"] = string(typeof(err))
    error_fields["error_message"] = sprint(showerror, err)
    error_fields["stacktrace"] = _stacktrace_string(bt)
    _emit_log("ERROR", "exception"; ctx=ctx, stage=stage, fields=error_fields)
    return nothing
end

function _with_stage_logging(
    f::Function,
    ctx::Union{Nothing, Dict{String, Any}},
    stage::String;
    fields=Dict{String, Any}()
)
    started_at = time()
    normalized_fields = _normalize_fields(fields)
    _emit_log("INFO", "stage_start"; ctx=ctx, stage=stage, fields=normalized_fields)
    try
        result = f()
        success_fields = copy(normalized_fields)
        success_fields["duration_ms"] = _elapsed_ms(started_at)
        _emit_log("INFO", "stage_complete"; ctx=ctx, stage=stage, fields=success_fields)
        return result
    catch err
        error_fields = copy(normalized_fields)
        error_fields["duration_ms"] = _elapsed_ms(started_at)
        _log_exception(err; bt=catch_backtrace(), ctx=ctx, stage=stage, fields=error_fields)
        rethrow()
    end
end

function set_state_backend!(backend::AbstractStateBackend) :: Nothing
    STATE_BACKEND[] = backend
    return nothing
end

function reset_patient_cache!() :: Nothing
    empty!(PATIENTS)
    return nothing
end

function _json_response(status::Integer, payload) :: HTTP.Response
    return HTTP.Response(
        status,
        ["Content-Type" => "application/json"],
        JSON3.write(payload)
    )
end

function _error_response(status::Integer, message::AbstractString) :: HTTP.Response
    return _json_response(status, Dict("ok" => false, "error" => String(message)))
end

function _json_body(req::HTTP.Request) :: Dict{String, Any}
    isempty(req.body) && return Dict{String, Any}()
    return JSON3.read(String(req.body), Dict{String, Any})
end

function _require_string(payload::Dict{String, Any}, key::String) :: String
    value = get(payload, key, nothing)
    value isa AbstractString || throw(ArgumentError("`$key` must be a string"))
    stripped = strip(String(value))
    isempty(stripped) && throw(ArgumentError("`$key` cannot be empty"))
    return stripped
end

function _optional_string(payload::Dict{String, Any}, key::String) :: Union{String, Nothing}
    value = get(payload, key, nothing)
    value === nothing && return nothing
    value isa AbstractString || throw(ArgumentError("`$key` must be a string"))
    stripped = strip(String(value))
    return isempty(stripped) ? nothing : stripped
end

function _as_float(value, key::String) :: Float64
    value isa Real || throw(ArgumentError("`$key` must be numeric"))
    return Float64(value)
end

function _as_bool(value, key::String) :: Bool
    value isa Bool || throw(ArgumentError("`$key` must be a boolean"))
    return Bool(value)
end

function _normalize_signal_value(value)
    if value isa Real
        return Float64(value)
    elseif value === nothing || value isa AbstractString || value isa Bool
        return value
    else
        throw(ArgumentError("signal values must be numbers, strings, booleans, or null"))
    end
end

function _signals(payload::Dict{String, Any}) :: Dict{Symbol, Any}
    raw = get(payload, "signals", Dict{String, Any}())
    raw isa AbstractDict || throw(ArgumentError("`signals` must be a JSON object"))
    return Dict(
        Symbol(String(key)) => _normalize_signal_value(value)
        for (key, value) in pairs(raw)
    )
end

function _preferences(payload::AbstractDict{String, <:Any}) :: UserPreferences
    defaults = UserPreferences()
    raw = get(payload, "preferences", Dict{String, Any}())
    raw isa AbstractDict || throw(ArgumentError("`preferences` must be a JSON object"))

    physical_priors = begin
        raw_phys = get(raw, "physical_priors", Dict{String, Any}())
        raw_phys isa AbstractDict || throw(ArgumentError("`preferences.physical_priors` must be a JSON object"))

        result = Dict{String, Vector{Float64}}()
        for (k, v) in pairs(raw_phys)
            v isa AbstractVector || continue
            length(v) == 2 || continue
            result[String(k)] = [Float64(v[1]), Float64(v[2])]
        end
        result
    end

    calibration_targets = begin
        raw_calib = get(raw, "calibration_targets", Dict{String, Any}())
        raw_calib isa AbstractDict || throw(ArgumentError("`preferences.calibration_targets` must be a JSON object"))
        result = Dict{String, Float64}()
        for key in ("recent_tir", "recent_pct_low", "recent_pct_high")
            v = get(raw_calib, key, nothing)
            isnothing(v) && continue
            result[key] = _as_float(v, "preferences.calibration_targets.$key")
        end
        result
    end

    minimum_action_delta_thresholds = begin
        raw_thresholds = get(raw, "minimum_action_delta_thresholds", Dict{String, Any}())
        raw_thresholds isa AbstractDict || throw(ArgumentError("`preferences.minimum_action_delta_thresholds` must be a JSON object"))
        result = Dict{String, Float64}()
        for (key, value) in raw_thresholds
            key isa AbstractString || throw(ArgumentError("`preferences.minimum_action_delta_thresholds` keys must be strings"))
            result[String(key)] = _as_float(value, "preferences.minimum_action_delta_thresholds.$key")
        end
        result
    end

    return UserPreferences(
        aggressiveness = _as_float(get(raw, "aggressiveness", defaults.aggressiveness), "preferences.aggressiveness"),
        hypoglycemia_fear = _as_float(get(raw, "hypoglycemia_fear", defaults.hypoglycemia_fear), "preferences.hypoglycemia_fear"),
        burden_sensitivity = _as_float(get(raw, "burden_sensitivity", defaults.burden_sensitivity), "preferences.burden_sensitivity"),
        persona = begin
            persona = get(raw, "persona", defaults.persona)
            persona isa AbstractString || throw(ArgumentError("`preferences.persona` must be a string"))
            String(persona)
        end,
        physical_priors = physical_priors,
        calibration_targets = calibration_targets,
        minimum_action_delta_thresholds = minimum_action_delta_thresholds
    )
end

function _segment_surface(raw, key::String) :: Main.SegmentSurface
    raw isa AbstractDict || throw(ArgumentError("`$key` must be an object"))
    segment_id = get(raw, "segment_id", nothing)
    if !(segment_id isa AbstractString) || isempty(strip(String(segment_id)))
        start_min = Int(round(_as_float(get(raw, "start_min", nothing), "$key.start_min")))
        end_min = Int(round(_as_float(get(raw, "end_min", nothing), "$key.end_min")))
        segment_id = "segment_$(start_min)_$(end_min)"
    end

    parameter_values = begin
        raw_parameters = get(raw, "parameter_values", nothing)
        if raw_parameters isa AbstractDict
            Dict{Symbol, Float64}(Symbol(String(k)) => _as_float(v, "$key.parameter_values.$k") for (k, v) in raw_parameters)
        else
            Dict{Symbol, Float64}(
                :isf => _as_float(get(raw, "isf", nothing), "$key.isf"),
                :cr => _as_float(get(raw, "cr", nothing), "$key.cr"),
                :basal => _as_float(get(raw, "basal", nothing), "$key.basal"),
            )
        end
    end

    return Main.SegmentSurface(
        segment_id = String(segment_id),
        start_min = Int(round(_as_float(get(raw, "start_min", nothing), "$key.start_min"))),
        end_min = Int(round(_as_float(get(raw, "end_min", nothing), "$key.end_min"))),
        parameter_values = parameter_values,
    )
end

function _connected_app_capabilities(payload) :: Main.ConnectedAppCapabilities
    payload isa AbstractDict || throw(ArgumentError("payload must be a JSON object"))
    defaults = Main.ConnectedAppCapabilities()
    raw = get(payload, "connected_app_capabilities", nothing)
    raw === nothing && return defaults
    raw isa AbstractDict || throw(ArgumentError("`connected_app_capabilities` must be a JSON object"))

    return Main.ConnectedAppCapabilities(
        app_id = begin
            value = get(raw, "app_id", defaults.app_id)
            value isa AbstractString || throw(ArgumentError("`connected_app_capabilities.app_id` must be a string"))
            String(value)
        end,
        supports_scalar_schedule = _as_bool(get(raw, "supports_scalar_schedule", defaults.supports_scalar_schedule), "connected_app_capabilities.supports_scalar_schedule"),
        supports_piecewise_schedule = _as_bool(get(raw, "supports_piecewise_schedule", defaults.supports_piecewise_schedule), "connected_app_capabilities.supports_piecewise_schedule"),
        supports_continuous_schedule = _as_bool(get(raw, "supports_continuous_schedule", defaults.supports_continuous_schedule), "connected_app_capabilities.supports_continuous_schedule"),
        max_segments = Int(round(_as_float(get(raw, "max_segments", defaults.max_segments), "connected_app_capabilities.max_segments"))),
        min_segment_duration_min = Int(round(_as_float(get(raw, "min_segment_duration_min", defaults.min_segment_duration_min), "connected_app_capabilities.min_segment_duration_min"))),
        max_segments_addable = Int(round(_as_float(get(raw, "max_segments_addable", defaults.max_segments_addable), "connected_app_capabilities.max_segments_addable"))),
        level_1_enabled = _as_bool(get(raw, "level_1_enabled", defaults.level_1_enabled), "connected_app_capabilities.level_1_enabled"),
        level_2_enabled = _as_bool(get(raw, "level_2_enabled", defaults.level_2_enabled), "connected_app_capabilities.level_2_enabled"),
        level_3_enabled = _as_bool(get(raw, "level_3_enabled", defaults.level_3_enabled), "connected_app_capabilities.level_3_enabled"),
        structural_change_requires_consent = _as_bool(get(raw, "structural_change_requires_consent", defaults.structural_change_requires_consent), "connected_app_capabilities.structural_change_requires_consent"),
    )
end

function _connected_app_state(payload) :: Main.ConnectedAppState
    payload isa AbstractDict || throw(ArgumentError("payload must be a JSON object"))
    defaults = Main.ConnectedAppState()
    raw = get(payload, "connected_app_state", nothing)
    raw === nothing && return defaults
    raw isa AbstractDict || throw(ArgumentError("`connected_app_state` must be a JSON object"))

    raw_segments = get(raw, "current_segments", Any[])
    raw_segments isa AbstractVector || throw(ArgumentError("`connected_app_state.current_segments` must be an array"))

    segments = Main.SegmentSurface[
        _segment_surface(entry, "connected_app_state.current_segments[$idx]")
        for (idx, entry) in enumerate(raw_segments)
    ]

    # Parse optional available_profiles array: [{id, name, segment_count}]
    raw_profiles = get(raw, "available_profiles", Any[])
    raw_profiles isa AbstractVector || throw(ArgumentError("`connected_app_state.available_profiles` must be an array"))
    available_profiles = map(enumerate(raw_profiles)) do (idx, entry)
        entry isa AbstractDict || throw(ArgumentError("`connected_app_state.available_profiles[$idx]` must be an object"))
        pid = get(entry, "id", nothing)
        pid isa AbstractString || throw(ArgumentError("`connected_app_state.available_profiles[$idx].id` must be a string"))
        pname = get(entry, "name", nothing)
        pname isa AbstractString || throw(ArgumentError("`connected_app_state.available_profiles[$idx].name` must be a string"))
        pcount_raw = get(entry, "segment_count", 1)
        Main.ProfileSummary((
            id           = String(pid),
            name         = String(pname),
            segment_count = Int(round(_as_float(pcount_raw, "connected_app_state.available_profiles[$idx].segment_count")))
        ))
    end

    # Parse optional active_profile_id
    raw_active_pid = get(raw, "active_profile_id", nothing)
    active_profile_id = if raw_active_pid === nothing
        nothing
    else
        raw_active_pid isa AbstractString || throw(ArgumentError("`connected_app_state.active_profile_id` must be a string"))
        String(raw_active_pid)
    end

    return Main.ConnectedAppState(
        schedule_version = begin
            value = get(raw, "schedule_version", defaults.schedule_version)
            value isa AbstractString || throw(ArgumentError("`connected_app_state.schedule_version` must be a string"))
            String(value)
        end,
        current_segments = segments,
        allow_structural_recommendations = _as_bool(get(raw, "allow_structural_recommendations", defaults.allow_structural_recommendations), "connected_app_state.allow_structural_recommendations"),
        allow_continuous_schedule = _as_bool(get(raw, "allow_continuous_schedule", defaults.allow_continuous_schedule), "connected_app_state.allow_continuous_schedule"),
        active_profile_id = active_profile_id,
        available_profiles = available_profiles,
    )
end

function _response_enum(payload::Dict{String, Any}) :: Union{UserResponse, Nothing}
    raw = get(payload, "response", get(payload, "response_code", nothing))
    raw === nothing && return nothing

    if raw isa Integer
        code = Int(raw)
        code in 0:2 || throw(ArgumentError("`response` code must be 0, 1, or 2"))
        return UserResponse(code)
    end

    raw isa AbstractString || throw(ArgumentError("`response` must be a string or integer"))
    label = lowercase(strip(String(raw)))
    label == "reject" && return Reject
    label == "partial" && return Partial
    label == "accept" && return Accept
    throw(ArgumentError("`response` must be one of Reject, Partial, Accept"))
end

function _serialize_system(
    system::Chamelia.ChameliaSystem;
    ctx::Union{Nothing, Dict{String, Any}}=nothing,
    patient_id::Union{Nothing, String}=nothing
) :: Vector{UInt8}
    return _with_stage_logging(ctx, "state.serialize"; fields=_patient_fields(patient_id)) do
        io = IOBuffer()
        Serialization.serialize(io, system)
        return take!(io)
    end
end

function _deserialize_system(
    bytes::Vector{UInt8};
    ctx::Union{Nothing, Dict{String, Any}}=nothing,
    patient_id::Union{Nothing, String}=nothing
) :: Chamelia.ChameliaSystem
    fields = _merge_fields(_patient_fields(patient_id), Dict("state_bytes" => length(bytes)))
    return _with_stage_logging(ctx, "state.deserialize"; fields=fields) do
        return Serialization.deserialize(IOBuffer(bytes))
    end
end

function _state_object_path(patient_id::String) :: String
    return "users/$patient_id/chamelia/state.jls"
end

function _base64url(data) :: String
    encoded = base64encode(data)
    encoded = replace(encoded, '+' => '-', '/' => '_')
    return replace(encoded, '=' => "")
end

function _escape_object_name(path::String) :: String
    safe(c::Char) = isascii(c) && (isletter(c) || isnumeric(c) || c in ['-', '_', '.', '~'])
    return HTTP.URIs.escapeuri(path, safe)
end

function _service_account_config(; ctx::Union{Nothing, Dict{String, Any}}=nothing) :: Dict{String, Any}
    return _with_stage_logging(ctx, "firebase.service_account_config") do
        credentials_path = get(ENV, "GOOGLE_APPLICATION_CREDENTIALS", "")
        isempty(strip(credentials_path)) && error("GOOGLE_APPLICATION_CREDENTIALS is not set")
        isfile(credentials_path) || error("service account file not found at $credentials_path")
        raw_config = read(credentials_path, String)
        parsed_config = JSON3.read(raw_config, Dict{String, Any})
        parsed_keys = sort!(collect(keys(parsed_config)))
        _emit_log(
            "INFO",
            "service_account_file_resolved";
            ctx=ctx,
            stage="firebase.service_account_config",
            fields=Dict(
                "credentials_path" => credentials_path,
                "credentials_file_exists" => true,
                "credentials_bytes" => sizeof(raw_config),
                "credentials_sha256_prefix" => bytes2hex(sha256(raw_config))[1:16],
                "service_account_keys" => parsed_keys,
                "service_account_has_client_email" => haskey(parsed_config, "client_email"),
                "service_account_has_private_key" => haskey(parsed_config, "private_key"),
                "service_account_type" => get(parsed_config, "type", nothing),
                "service_account_project_id" => get(parsed_config, "project_id", nothing)
            )
        )
        return parsed_config
    end
end

function _firebase_project_id(; ctx::Union{Nothing, Dict{String, Any}}=nothing) :: String
    return _with_stage_logging(ctx, "firebase.project_id") do
        project_id = strip(get(ENV, "FIREBASE_PROJECT_ID", ""))
        isempty(project_id) && error("FIREBASE_PROJECT_ID is not set")
        return project_id
    end
end

function _candidate_buckets(
    backend::FirebaseStorageBackend;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Vector{String}
    return _with_stage_logging(ctx, "firebase.candidate_buckets") do
        buckets = String[]
        if !isnothing(backend.bucket_hint)
            push!(buckets, backend.bucket_hint)
        end

        env_bucket = strip(get(ENV, "FIREBASE_STORAGE_BUCKET", ""))
        if !isempty(env_bucket)
            push!(buckets, env_bucket)
        end

        project_id = _firebase_project_id(ctx=ctx)
        push!(buckets, project_id * ".firebasestorage.app")
        push!(buckets, project_id * ".appspot.com")

        unique_buckets = unique(buckets)
        _emit_log(
            "INFO",
            "firebase_bucket_candidates";
            ctx=ctx,
            stage="firebase.candidate_buckets",
            fields=Dict("bucket_candidates" => unique_buckets)
        )
        return unique_buckets
    end
end

function _google_access_token!(
    backend::FirebaseStorageBackend;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: String
    return _with_stage_logging(ctx, "firebase.access_token") do
        now_ts = time()
        if !isnothing(backend.access_token) && now_ts < backend.expires_at
            _emit_log(
                "INFO",
                "firebase_access_token_cache_hit";
                ctx=ctx,
                stage="firebase.access_token",
                fields=Dict("expires_at_unix" => backend.expires_at)
            )
            return backend.access_token::String
        end

        _emit_log("INFO", "firebase_access_token_cache_miss"; ctx=ctx, stage="firebase.access_token")
        config = _service_account_config(ctx=ctx)
        client_email = String(_require_config_value(config, "client_email"))
        private_key = String(_require_config_value(config, "private_key"))

        issued_at = floor(Int, now_ts)
        header = Dict("alg" => "RS256", "typ" => "JWT")
        claims = Dict(
            "iss" => client_email,
            "scope" => "https://www.googleapis.com/auth/devstorage.read_write",
            "aud" => "https://oauth2.googleapis.com/token",
            "iat" => issued_at,
            "exp" => issued_at + 3600
        )

        signing_input = string(
            _base64url(JSON3.write(header)),
            ".",
            _base64url(JSON3.write(claims))
        )

        key = MbedTLS.PKContext()
        MbedTLS.parse_key!(key, private_key)
        digest = MbedTLS.digest(MbedTLS.MD_SHA256, signing_input)
        signature = MbedTLS.sign(key, MbedTLS.MD_SHA256, digest, Random.RandomDevice())
        assertion = string(signing_input, ".", _base64url(signature))

        form_body = string(
            "grant_type=",
            HTTP.URIs.escapeuri("urn:ietf:params:oauth:grant-type:jwt-bearer"),
            "&assertion=",
            HTTP.URIs.escapeuri(assertion)
        )

        response = HTTP.post(
            "https://oauth2.googleapis.com/token",
            ["Content-Type" => "application/x-www-form-urlencoded"],
            form_body
        )
        _emit_log(
            response.status == 200 ? "INFO" : "ERROR",
            "firebase_access_token_response";
            ctx=ctx,
            stage="firebase.access_token",
            fields=Dict("oauth_status" => response.status)
        )

        response.status == 200 || error("token exchange failed with status $(response.status): $(String(response.body))")
        payload = JSON3.read(String(response.body), Dict{String, Any})
        token = String(_require_payload_value(payload, "access_token", "token response missing access_token"))
        expires_in = Int(get(payload, "expires_in", 3600))

        backend.access_token = token
        backend.expires_at = now_ts + max(0, expires_in - 60)
        return token
    end
end

function _download_bytes(
    backend::InMemoryStateBackend,
    patient_id::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Union{Vector{UInt8}, Nothing}
    return get(backend.blobs, patient_id, nothing)
end

function _upload_bytes!(
    backend::InMemoryStateBackend,
    patient_id::String,
    bytes::Vector{UInt8};
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Nothing
    backend.blobs[patient_id] = copy(bytes)
    return nothing
end

function _download_bytes(
    backend::FirebaseStorageBackend,
    patient_id::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Union{Vector{UInt8}, Nothing}
    fields = _patient_fields(patient_id)
    return _with_stage_logging(ctx, "firebase.download"; fields=fields) do
        token = _google_access_token!(backend; ctx=ctx)
        object_name = _escape_object_name(_state_object_path(patient_id))

        for bucket in _candidate_buckets(backend; ctx=ctx)
            _emit_log(
                "INFO",
                "firebase_download_attempt";
                ctx=ctx,
                stage="firebase.download",
                fields=_merge_fields(fields, Dict("bucket" => bucket))
            )

            url = "https://storage.googleapis.com/storage/v1/b/$bucket/o/$object_name?alt=media"
            response = HTTP.get(
                url,
                ["Authorization" => "Bearer $token"];
                status_exception=false
            )

            _emit_log(
                response.status == 200 ? "INFO" : response.status == 404 ? "WARNING" : "ERROR",
                "firebase_download_response";
                ctx=ctx,
                stage="firebase.download",
                fields=_merge_fields(fields, Dict("bucket" => bucket, "storage_status" => response.status))
            )

            if response.status == 200
                backend.bucket_hint = bucket
                return Vector{UInt8}(response.body)
            elseif response.status == 404
                continue
            else
                error("Firebase download failed with status $(response.status): $(String(response.body))")
            end
        end

        _emit_log("WARNING", "firebase_state_not_found"; ctx=ctx, stage="firebase.download", fields=fields)
        return nothing
    end
end

function _upload_bytes!(
    backend::FirebaseStorageBackend,
    patient_id::String,
    bytes::Vector{UInt8};
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Nothing
    fields = _merge_fields(_patient_fields(patient_id), Dict("state_bytes" => length(bytes)))
    _with_stage_logging(ctx, "firebase.upload"; fields=fields) do
        token = _google_access_token!(backend; ctx=ctx)
        object_name = _escape_object_name(_state_object_path(patient_id))
        last_error = nothing

        for bucket in _candidate_buckets(backend; ctx=ctx)
            _emit_log(
                "INFO",
                "firebase_upload_attempt";
                ctx=ctx,
                stage="firebase.upload",
                fields=_merge_fields(fields, Dict("bucket" => bucket))
            )

            url = "https://storage.googleapis.com/upload/storage/v1/b/$bucket/o?uploadType=media&name=$object_name"
            response = HTTP.post(
                url,
                [
                    "Authorization" => "Bearer $token",
                    "Content-Type" => "application/octet-stream"
                ],
                bytes;
                status_exception=false
            )

            _emit_log(
                response.status in (200, 201) ? "INFO" : response.status in (400, 404) ? "WARNING" : "ERROR",
                "firebase_upload_response";
                ctx=ctx,
                stage="firebase.upload",
                fields=_merge_fields(fields, Dict("bucket" => bucket, "storage_status" => response.status))
            )

            if response.status in (200, 201)
                backend.bucket_hint = bucket
                return nothing
            elseif response.status in (400, 404)
                last_error = ErrorException("bucket $bucket rejected upload: $(String(response.body))")
                continue
            else
                error("Firebase upload failed with status $(response.status): $(String(response.body))")
            end
        end

        throw(something(last_error, ErrorException("no valid Firebase Storage bucket found")))
    end
    return nothing
end

function _load_from_backend(
    patient_id::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Union{Chamelia.ChameliaSystem, Nothing}
    fields = _patient_fields(patient_id)
    return _with_stage_logging(ctx, "state.load_from_backend"; fields=fields) do
        bytes = _download_bytes(STATE_BACKEND[], patient_id; ctx=ctx)
        bytes === nothing && return nothing
        return _deserialize_system(bytes; ctx=ctx, patient_id=patient_id)
    end
end

function _try_load_from_backend(
    patient_id::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing,
    route_stage::String="state.load_from_backend"
) :: Union{Chamelia.ChameliaSystem, Nothing}
    fields = _patient_fields(patient_id)
    try
        return _load_from_backend(patient_id; ctx=ctx)
    catch err
        _emit_log(
            "WARNING",
            "state_load_ignored";
            ctx=ctx,
            stage=route_stage,
            fields=_merge_fields(
                fields,
                Dict(
                    "error_type" => string(typeof(err)),
                    "error_message" => sprint(showerror, err)
                )
            )
        )
        return nothing
    end
end

function _try_load_from_backend_for_initialize(
    patient_id::String;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Union{Chamelia.ChameliaSystem, Nothing}
    return _try_load_from_backend(patient_id; ctx=ctx, route_stage="route.chamelia_initialize_patient")
end

function _persist_state!(
    patient_id::String,
    system::Chamelia.ChameliaSystem;
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Nothing
    fields = _patient_fields(patient_id)
    _with_stage_logging(ctx, "state.persist"; fields=fields) do
        bytes = _serialize_system(system; ctx=ctx, patient_id=patient_id)
        _upload_bytes!(STATE_BACKEND[], patient_id, bytes; ctx=ctx)
        return nothing
    end
    return nothing
end

function _lookup_patient(
    patient_id::String;
    autoload::Bool=true,
    ctx::Union{Nothing, Dict{String, Any}}=nothing
) :: Chamelia.ChameliaSystem
    fields = _merge_fields(_patient_fields(patient_id), Dict("autoload" => autoload))
    return _with_stage_logging(ctx, "patient.lookup"; fields=fields) do
        if haskey(PATIENTS, patient_id)
            _emit_log("INFO", "patient_cache_hit"; ctx=ctx, stage="patient.lookup", fields=fields)
            return PATIENTS[patient_id]
        end

        _emit_log("INFO", "patient_cache_miss"; ctx=ctx, stage="patient.lookup", fields=fields)
        if autoload
            loaded = _try_load_from_backend(patient_id; ctx=ctx, route_stage="patient.lookup")
            if !isnothing(loaded)
                PATIENTS[patient_id] = loaded
                _emit_log("INFO", "patient_autoloaded"; ctx=ctx, stage="patient.lookup", fields=fields)
                return loaded
            end
        end

        throw(NotFoundError("patient state not found for `$patient_id`"))
    end
end

function _status_payload(system::Chamelia.ChameliaSystem) :: Dict{String, Any}
    status = Chamelia.graduation_status(system)
    return Dict(
        "graduated" => status.graduated,
        "n_days" => status.n_days,
        "win_rate" => status.win_rate,
        "safety_violations" => status.safety_violations,
        "consecutive_days" => status.consecutive_days,
        "belief_entropy" => status.belief_entropy,
        "familiarity" => status.κ_familiarity,
        "concordance" => status.ρ_concordance,
        "calibration" => status.η_calibration,
        "trust_level" => status.trust_level,
        "burnout_level" => status.burnout_level,
        "no_surface_streak" => status.no_surface_streak,
        "drift_detected" => status.drift_detected,
        "days_since_drift" => status.days_since_drift,
        "n_records" => status.n_records,
        "last_decision_reason" => String(status.last_decision_reason),
        "last_safety_diagnostics" => _json_ready(status.last_safety_diagnostics),
        "configurator_mode" => String(status.configurator_mode),
        "jepa_weights_loaded" => status.jepa_weights_loaded,
        "jepa_active" => status.jepa_active,
        "belief_mode" => String(status.belief_mode),
        "config" => _json_ready(status.config),
    )
end

_json_ready(value::Nothing) = nothing
_json_ready(value::Bool) = value
_json_ready(value::Number) = value
_json_ready(value::String) = value
_json_ready(value::Symbol) = String(value)
_json_ready(value::AbstractVector) = [_json_ready(v) for v in value]
_json_ready(value::Dict) = Dict(String(k) => _json_ready(v) for (k, v) in value)
_json_ready(value::NamedTuple) = Dict(String(k) => _json_ready(v) for (k, v) in pairs(value))
_json_ready(value) = string(value)

function _serialize_action(action) :: Dict{String, Any}
    if action isa Main.NullAction
        return Dict("kind" => "null", "deltas" => Dict{String, Float64}())
    elseif action isa Main.Actor.CandidateAction
        return Dict(
            "kind" => "candidate",
            "deltas" => Dict(String(key) => value for (key, value) in action.deltas)
        )
    elseif action isa Main.ScheduledAction
        return Dict(
            "kind" => "scheduled",
            "level" => action.level,
            "family" => string(action.family),
            "segment_deltas" => [
                Dict(
                    "segment_id" => delta.segment_id,
                    "parameter_deltas" => Dict(String(key) => value for (key, value) in delta.parameter_deltas),
                )
                for delta in action.segment_deltas
            ],
            "structural_edits" => [
                Dict(
                    "edit_type" => String(edit.edit_type),
                    "target_segment_id" => edit.target_segment_id,
                    "split_at_minute" => edit.split_at_minute,
                    "neighbor_segment_id" => edit.neighbor_segment_id,
                )
                for edit in action.structural_edits
            ]
        )
    end

    return Dict("kind" => string(nameof(typeof(action))))
end

function _serialize_burnout(attr) :: Dict{String, Any}
    return Dict(
        "delta_hat" => attr.Δ_hat,
        "p_treated" => attr.P_treated,
        "p_baseline" => attr.P_baseline,
        "se_paired" => attr.se_paired,
        "ci_lower" => attr.ci_lower,
        "upper_ci" => attr.upper_ci,
        "n_pairs" => attr.n_pairs,
        "horizon" => attr.horizon,
        "horizon_sensitivity" => [
            Dict("horizon" => entry.H, "delta" => entry.Δ)
            for entry in attr.horizon_sensitivity
        ]
    )
end

function _serialize_recommendation(pkg) :: Dict{String, Any}
    return Dict(
        "action" => _serialize_action(pkg.action),
        "predicted_improvement" => pkg.predicted_improvement,
        "confidence" => pkg.confidence,
        "confidence_breakdown" => isnothing(pkg.confidence_breakdown) ? nothing : Dict(
            "familiarity" => pkg.confidence_breakdown.familiarity,
            "concordance" => pkg.confidence_breakdown.concordance,
            "calibration" => pkg.confidence_breakdown.calibration,
            "effect_support" => pkg.confidence_breakdown.effect_support,
            "selection_penalty" => pkg.confidence_breakdown.selection_penalty,
            "final_confidence" => pkg.confidence_breakdown.final_confidence,
        ),
        "alternatives" => [_serialize_action(action) for action in pkg.alternatives],
        "effect_size" => pkg.effect_size,
        "cvar_value" => pkg.cvar_value,
        "burnout_attribution" => isnothing(pkg.burnout_attribution) ? nothing : _serialize_burnout(pkg.burnout_attribution),
        "predicted_outcomes" => isnothing(pkg.predicted_outcomes) ? nothing : Dict(
            "baseline_tir" => pkg.predicted_outcomes.baseline_tir,
            "treated_tir" => pkg.predicted_outcomes.treated_tir,
            "delta_tir" => pkg.predicted_outcomes.delta_tir,
            "baseline_pct_low" => pkg.predicted_outcomes.baseline_pct_low,
            "treated_pct_low" => pkg.predicted_outcomes.treated_pct_low,
            "delta_pct_low" => pkg.predicted_outcomes.delta_pct_low,
            "baseline_pct_high" => pkg.predicted_outcomes.baseline_pct_high,
            "treated_pct_high" => pkg.predicted_outcomes.treated_pct_high,
            "delta_pct_high" => pkg.predicted_outcomes.delta_pct_high,
            "baseline_bg_avg" => pkg.predicted_outcomes.baseline_bg_avg,
            "treated_bg_avg" => pkg.predicted_outcomes.treated_bg_avg,
            "delta_bg_avg" => pkg.predicted_outcomes.delta_bg_avg,
            "baseline_cost_mean" => pkg.predicted_outcomes.baseline_cost_mean,
            "treated_cost_mean" => pkg.predicted_outcomes.treated_cost_mean,
            "delta_cost_mean" => pkg.predicted_outcomes.delta_cost_mean,
            "baseline_cvar" => pkg.predicted_outcomes.baseline_cvar,
            "treated_cvar" => pkg.predicted_outcomes.treated_cvar,
            "delta_cvar" => pkg.predicted_outcomes.delta_cvar,
        ),
        "predicted_uncertainty" => isnothing(pkg.predicted_uncertainty) ? nothing : Dict(
            "tir_std" => pkg.predicted_uncertainty.tir_std,
            "pct_low_std" => pkg.predicted_uncertainty.pct_low_std,
            "pct_high_std" => pkg.predicted_uncertainty.pct_high_std,
            "bg_avg_std" => pkg.predicted_uncertainty.bg_avg_std,
            "cost_std" => pkg.predicted_uncertainty.cost_std,
        ),
        "action_level" => pkg.action_level,
        "action_family" => isnothing(pkg.action_family) ? nothing : string(pkg.action_family),
        "segment_summaries" => [
            Dict(
                "segment_id" => summary.segment_id,
                "label" => summary.label,
                "parameter_summaries" => summary.parameter_summaries,
            )
            for summary in pkg.segment_summaries
        ],
        "structure_summaries" => pkg.structure_summaries,
        "recommendation_scope" => pkg.recommendation_scope,
        "target_profile_id" => pkg.target_profile_id,
        "detected_regime" => pkg.detected_regime,
    )
end

function _initialize_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_initialize_patient"; fields=Dict("route_name" => "chamelia_initialize_patient")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        _emit_log("INFO", "route_patient_resolved"; ctx=ctx, stage="route.chamelia_initialize_patient", fields=patient_fields)

        if haskey(PATIENTS, patient_id)
            _emit_log("INFO", "patient_already_initialized"; ctx=ctx, stage="route.chamelia_initialize_patient", fields=patient_fields)
            return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => _status_payload(PATIENTS[patient_id])))
        end

        system = _try_load_from_backend_for_initialize(patient_id; ctx=ctx)
        if isnothing(system)
            prefs = _with_stage_logging(ctx, "route.initialize.preferences"; fields=patient_fields) do
                return _preferences(payload)
            end

            weights_dir = _optional_string(payload, "weights_dir")
            _emit_log(
                "INFO",
                "initializing_new_patient";
                ctx=ctx,
                stage="route.initialize.create_system",
                fields=_merge_fields(
                    patient_fields,
                    Dict(
                        "persona" => prefs.persona,
                        "weights_dir_present" => !isnothing(weights_dir)
                    )
                )
            )

            system = _with_stage_logging(ctx, "route.initialize.create_system"; fields=patient_fields) do
                # Both InSiteSimulator and InSiteDomainAdapter are InSite-domain
                # components — explicitly named here so it is clear this server
                # is operating in InSite mode, not generic Chamelia core mode.
                return Chamelia.initialize_patient(
                    prefs,
                    Main.InSiteSimulator(prefs.persona);
                    adapter=InSiteDomainAdapter(),
                    weights_dir=weights_dir
                )
            end
            _persist_state!(patient_id, system; ctx=ctx)
        end

        PATIENTS[patient_id] = system
        status = _status_payload(system)
        _emit_log(
            "INFO",
            "route_response_ready";
            ctx=ctx,
            stage="route.chamelia_initialize_patient",
            fields=_merge_fields(patient_fields, Dict("status_n_days" => status["n_days"]))
        )
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => status))
    end
end

function _observe_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_observe"; fields=Dict("route_name" => "chamelia_observe")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _lookup_patient(patient_id; ctx=ctx)
        obs = _with_stage_logging(ctx, "route.observe.build_observation"; fields=patient_fields) do
            return Observation(
                timestamp=_as_float(get(payload, "timestamp", nothing), "timestamp"),
                signals=_signals(payload)
            )
        end

        _with_stage_logging(ctx, "route.observe.apply"; fields=patient_fields) do
            Chamelia.observe!(system, obs)
            return nothing
        end

        status = _status_payload(system)
        _emit_log(
            "INFO",
            "route_response_ready";
            ctx=ctx,
            stage="route.chamelia_observe",
            fields=_merge_fields(patient_fields, Dict("status_n_days" => status["n_days"]))
        )
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => status))
    end
end

function _step_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_step"; fields=Dict("route_name" => "chamelia_step")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _lookup_patient(patient_id; ctx=ctx)
        capabilities = _with_stage_logging(ctx, "route.step.connected_app_capabilities"; fields=patient_fields) do
            return _connected_app_capabilities(payload)
        end
        app_state = _with_stage_logging(ctx, "route.step.connected_app_state"; fields=patient_fields) do
            return _connected_app_state(payload)
        end
        obs = _with_stage_logging(ctx, "route.step.build_observation"; fields=patient_fields) do
            return Observation(
                timestamp=_as_float(get(payload, "timestamp", nothing), "timestamp"),
                signals=_signals(payload)
            )
        end

        recommendation = _with_stage_logging(ctx, "route.step.apply"; fields=patient_fields) do
            return Chamelia.step!(system, obs; capabilities=capabilities, app_state=app_state)
        end
        _persist_state!(patient_id, system; ctx=ctx)

        rec_id = system.mem.next_id > 1 ? system.mem.next_id - 1 : nothing
        status = _status_payload(system)
        _emit_log(
            "INFO",
            "route_response_ready";
            ctx=ctx,
            stage="route.chamelia_step",
            fields=_merge_fields(patient_fields, Dict("status_n_days" => status["n_days"], "recommendation_returned" => recommendation !== nothing))
        )
        return _json_response(200, Dict(
            "ok" => true,
            "patient_id" => patient_id,
            "rec_id" => rec_id,
            "recommendation" => recommendation === nothing ? nothing : _serialize_recommendation(recommendation),
            "status" => status
        ))
    end
end

function _record_outcome_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_record_outcome"; fields=Dict("route_name" => "chamelia_record_outcome")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _lookup_patient(patient_id; ctx=ctx)
        rec_id = Int(_as_float(get(payload, "rec_id", nothing), "rec_id"))
        cost = _as_float(get(payload, "cost", nothing), "cost")

        _with_stage_logging(ctx, "route.record_outcome.apply"; fields=_merge_fields(patient_fields, Dict("rec_id" => rec_id))) do
            Chamelia.record_outcome!(system, rec_id, _response_enum(payload), _signals(payload), cost)
            return nothing
        end
        _persist_state!(patient_id, system; ctx=ctx)

        status = _status_payload(system)
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => status))
    end
end

function _save_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_save_patient"; fields=Dict("route_name" => "chamelia_save_patient")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _lookup_patient(patient_id; ctx=ctx)
        _emit_log("INFO", "save_route_patient_loaded"; ctx=ctx, stage="route.chamelia_save_patient", fields=patient_fields)
        _persist_state!(patient_id, system; ctx=ctx)
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id))
    end
end

function _load_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_load_patient"; fields=Dict("route_name" => "chamelia_load_patient")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _load_from_backend(patient_id; ctx=ctx)
        isnothing(system) && throw(NotFoundError("patient state not found for `$patient_id`"))
        PATIENTS[patient_id] = system
        status = _status_payload(system)
        _emit_log(
            "INFO",
            "route_response_ready";
            ctx=ctx,
            stage="route.chamelia_load_patient",
            fields=_merge_fields(patient_fields, Dict("status_n_days" => status["n_days"]))
        )
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => status))
    end
end

function _graduation_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_graduation_status"; fields=Dict("route_name" => "chamelia_graduation_status")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        system = _lookup_patient(patient_id; ctx=ctx)
        status = _status_payload(system)
        _emit_log(
            "INFO",
            "route_response_ready";
            ctx=ctx,
            stage="route.chamelia_graduation_status",
            fields=_merge_fields(patient_fields, Dict("status_n_days" => status["n_days"]))
        )
        return _json_response(200, Dict("ok" => true, "patient_id" => patient_id, "status" => status))
    end
end

function _free_handler(payload::Dict{String, Any}, ctx::Dict{String, Any}) :: HTTP.Response
    return _with_stage_logging(ctx, "route.chamelia_free_patient"; fields=Dict("route_name" => "chamelia_free_patient")) do
        patient_id = _require_string(payload, "patient_id")
        patient_fields = _patient_fields(patient_id)
        deleted = pop!(PATIENTS, patient_id, nothing)
        _emit_log(
            "INFO",
            "patient_freed";
            ctx=ctx,
            stage="route.chamelia_free_patient",
            fields=_merge_fields(patient_fields, Dict("cache_entry_removed" => !isnothing(deleted)))
        )
        return _json_response(200, Dict("ok" => !isnothing(deleted), "patient_id" => patient_id))
    end
end

function handle_request(req::HTTP.Request) :: HTTP.Response
    ctx = _request_context(req)
    started_at = time()
    request_body_bytes = length(req.body)
    _emit_log(
        "INFO",
        "request_received";
        ctx=ctx,
        stage="request.start",
        fields=Dict("body_bytes" => request_body_bytes)
    )

    try
        method = String(req.method)
        path = HTTP.URIs.URI(String(req.target)).path

        if method == "GET" && path == "/health"
            response = _json_response(200, Dict("ok" => true, "status" => "ok"))
            _emit_log(
                "INFO",
                "request_complete";
                ctx=ctx,
                stage="request.complete",
                fields=Dict("status_code" => response.status, "duration_ms" => _elapsed_ms(started_at))
            )
            return response
        elseif method != "POST"
            response = _error_response(405, "method not allowed")
            _emit_log(
                "WARNING",
                "method_not_allowed";
                ctx=ctx,
                stage="request.route",
                fields=Dict("status_code" => response.status)
            )
            _emit_log(
                "INFO",
                "request_complete";
                ctx=ctx,
                stage="request.complete",
                fields=Dict("status_code" => response.status, "duration_ms" => _elapsed_ms(started_at))
            )
            return response
        end

        payload = _with_stage_logging(ctx, "request.parse_json"; fields=Dict("body_bytes" => request_body_bytes)) do
            return _json_body(req)
        end
        _emit_log(
            "INFO",
            "request_payload_parsed";
            ctx=ctx,
            stage="request.parse_json",
            fields=_payload_summary(payload, request_body_bytes)
        )

        if path == "/chamelia_initialize_patient"
            response = _initialize_handler(payload, ctx)
        elseif path == "/chamelia_observe"
            response = _observe_handler(payload, ctx)
        elseif path == "/chamelia_step"
            response = _step_handler(payload, ctx)
        elseif path == "/chamelia_record_outcome"
            response = _record_outcome_handler(payload, ctx)
        elseif path == "/chamelia_save_patient"
            response = _save_handler(payload, ctx)
        elseif path == "/chamelia_load_patient"
            response = _load_handler(payload, ctx)
        elseif path == "/chamelia_graduation_status"
            response = _graduation_handler(payload, ctx)
        elseif path == "/chamelia_free_patient"
            response = _free_handler(payload, ctx)
        else
            response = _error_response(404, "unknown route: $path")
            _emit_log(
                "WARNING",
                "unknown_route";
                ctx=ctx,
                stage="request.route",
                fields=Dict("status_code" => response.status)
            )
        end

        _emit_log(
            "INFO",
            "request_complete";
            ctx=ctx,
            stage="request.complete",
            fields=Dict("status_code" => response.status, "duration_ms" => _elapsed_ms(started_at))
        )
        return response
    catch err
        bt = catch_backtrace()
        status = 500
        message = sprint(showerror, err)
        if err isa NotFoundError
            status = 404
        elseif err isa ArgumentError
            status = 400
        end

        _log_exception(
            err;
            bt=bt,
            ctx=ctx,
            stage="request.error",
            fields=Dict("status_code" => status, "duration_ms" => _elapsed_ms(started_at))
        )

        response = _error_response(status, message)
        _emit_log(
            "INFO",
            "request_complete";
            ctx=ctx,
            stage="request.complete",
            fields=Dict("status_code" => response.status, "duration_ms" => _elapsed_ms(started_at))
        )
        return response
    end
end

function main() :: Nothing
    port = parse(Int, get(ENV, "PORT", "8080"))
    credentials_path = get(ENV, "GOOGLE_APPLICATION_CREDENTIALS", "")
    if get(ENV, "STORAGE_BACKEND", "") == "memory"
        set_state_backend!(InMemoryStateBackend())
    end
    _emit_log(
        "INFO",
        "server_starting";
        stage="startup",
        fields=Dict(
            "port" => port,
            "backend_type" => string(typeof(STATE_BACKEND[])),
            "service" => something(_nonempty_env("K_SERVICE"), "unknown"),
            "revision" => something(_nonempty_env("K_REVISION"), "unknown"),
            "firebase_project_id_configured" => !isnothing(_nonempty_env("FIREBASE_PROJECT_ID")),
            "firebase_storage_bucket_configured" => !isnothing(_nonempty_env("FIREBASE_STORAGE_BUCKET")),
            "credentials_path" => credentials_path,
            "credentials_file_exists" => !isempty(strip(credentials_path)) && isfile(credentials_path)
        )
    )
    HTTP.serve(handle_request, "0.0.0.0", port; verbose=false)
    return nothing
end

end # module ChameliaServer

if abspath(PROGRAM_FILE) == @__FILE__
    Main.ChameliaServer.main()
end
