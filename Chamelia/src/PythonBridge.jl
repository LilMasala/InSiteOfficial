module PythonBridge

using HTTP
using JSON3

export BridgeConfig, BridgeError, run_pipeline, summarize_bundle, ingest_replay_examples

Base.@kwdef struct BridgeConfig
    base_url::String
    bridge_version::String = "v1"
    mode::String = "v3"
    domain_name::String = "default"
    session_id::String = "default"
    model_version::String = "unknown"
    timeout_s::Float64 = 5.0
    rollout_horizon::Int = 2
end

struct BridgeError <: Exception
    endpoint::String
    status::Int
    message::String
end

Base.showerror(io::IO, err::BridgeError) = print(
    io,
    "bridge request to ",
    err.endpoint,
    " failed (status ",
    err.status,
    "): ",
    err.message,
)

function _http_readtimeout(timeout_s::Real) :: Int
    return max(1, ceil(Int, float(timeout_s)))
end

function _http_request_json(
    config::BridgeConfig,
    endpoint::String,
    payload::Dict{String, Any},
) :: Dict{String, Any}
    headers = ["Content-Type" => "application/json"]
    response = HTTP.request(
        "POST",
        string(config.base_url, endpoint),
        headers,
        JSON3.write(payload);
        readtimeout=_http_readtimeout(config.timeout_s),
    )

    body = isempty(response.body) ? Dict{String, Any}() : JSON3.read(String(response.body), Dict{String, Any})
    if response.status < 200 || response.status >= 300
        message = body isa Dict ? get(body, "error", String(response.body)) : String(response.body)
        throw(BridgeError(endpoint, response.status, string(message)))
    end
    return body isa Dict{String, Any} ? body : Dict{String, Any}(body)
end

function run_pipeline(
    config::BridgeConfig,
    encode_payload::Dict{String, Any},
    domain_state::Dict{String, Any}=Dict{String, Any}();
    request_fn::Union{Nothing, Function}=nothing,
) :: Dict{String, Any}
    base_payload = Dict{String, Any}(
        "bridge_version" => config.bridge_version,
        "mode" => config.mode,
        "domain_name" => config.domain_name,
        "session_id" => config.session_id,
        "model_version" => config.model_version,
    )
    requester = isnothing(request_fn) ?
        (endpoint, payload) -> _http_request_json(config, endpoint, payload) :
        request_fn

    encoded_state = requester("/encode", merge(copy(base_payload), encode_payload))
    retrieved_memory = requester(
        "/retrieve",
        merge(copy(base_payload), Dict{String, Any}("z_t" => get(encoded_state, "z_t", Any[]))),
    )
    configurator_output = requester(
        "/configure",
        merge(
            copy(base_payload),
            Dict{String, Any}(
                "encoded_state" => encoded_state,
                "retrieved_memory" => retrieved_memory,
            ),
        ),
    )
    proposal_bundle = requester(
        "/propose",
        merge(
            copy(base_payload),
            Dict{String, Any}(
                "encoded_state" => encoded_state,
                "configurator_output" => configurator_output,
                "retrieved_memory" => retrieved_memory,
            ),
        ),
    )
    rollout_bundle = requester(
        "/rollout",
        merge(
            copy(base_payload),
            Dict{String, Any}(
                "encoded_state" => encoded_state,
                "configurator_output" => configurator_output,
                "proposal_bundle" => proposal_bundle,
                "rollout_horizon" => config.rollout_horizon,
            ),
        ),
    )
    critic_scores = requester(
        "/critic",
        merge(
            copy(base_payload),
            Dict{String, Any}(
                "encoded_state" => encoded_state,
                "configurator_output" => configurator_output,
                "proposal_bundle" => proposal_bundle,
                "rollout_bundle" => rollout_bundle,
                "domain_state" => domain_state,
            ),
        ),
    )

    return Dict{String, Any}(
        "bridge_version" => config.bridge_version,
        "mode" => config.mode,
        "domain_name" => config.domain_name,
        "session_id" => config.session_id,
        "model_version" => config.model_version,
        "encoded_state" => encoded_state,
        "retrieved_memory" => retrieved_memory,
        "configurator_output" => configurator_output,
        "proposal_bundle" => proposal_bundle,
        "rollout_bundle" => rollout_bundle,
        "critic_scores" => critic_scores,
    )
end

function _argmin_index(values)::Union{Int, Nothing}
    values isa AbstractVector || return nothing
    isempty(values) && return nothing
    best_idx = 1
    best_value = Inf
    for (idx, value) in pairs(values)
        current = try
            Float64(value)
        catch
            continue
        end
        if current < best_value
            best_value = current
            best_idx = idx
        end
    end
    return best_idx - 1
end

function summarize_bundle(bundle::Dict{String, Any}) :: Dict{String, Any}
    proposal_bundle = get(bundle, "proposal_bundle", Dict{String, Any}())
    critic_scores = get(bundle, "critic_scores", Dict{String, Any}())
    candidate_paths = get(proposal_bundle, "candidate_paths", Any[])
    candidate_total = get(critic_scores, "candidate_total", Any[])

    num_candidates = candidate_paths isa AbstractVector ? length(candidate_paths) : 0
    path_length = (
        candidate_paths isa AbstractVector &&
        !isempty(candidate_paths) &&
        first(candidate_paths) isa AbstractVector
    ) ? length(first(candidate_paths)) : 0

    return Dict{String, Any}(
        "bridge_ok" => true,
        "mode" => get(bundle, "mode", "unknown"),
        "domain_name" => get(bundle, "domain_name", "unknown"),
        "session_id" => get(bundle, "session_id", "unknown"),
        "model_version" => get(bundle, "model_version", "unknown"),
        "num_candidates" => num_candidates,
        "path_length" => path_length,
        "python_selected_candidate_idx" => _argmin_index(candidate_total),
    )
end

function ingest_replay_examples(
    config::BridgeConfig,
    examples::Vector{Dict{String, Any}};
    request_fn::Union{Nothing, Function}=nothing,
) :: Dict{String, Any}
    base_payload = Dict{String, Any}(
        "bridge_version" => config.bridge_version,
        "mode" => config.mode,
        "domain_name" => config.domain_name,
        "session_id" => config.session_id,
        "model_version" => config.model_version,
        "examples" => examples,
    )
    requester = isnothing(request_fn) ?
        (endpoint, payload) -> _http_request_json(config, endpoint, payload) :
        request_fn
    return requester("/replay_ingest", base_payload)
end

end
