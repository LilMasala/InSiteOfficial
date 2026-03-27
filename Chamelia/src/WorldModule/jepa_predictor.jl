"""
jepa_predictor.jl
JEPA Predictor — v2.0 World Model
Predicts future latent belief states from current latent belief + action.
Runs in latent space — no explicit simulation needed at inference time.

Complements jepa_encoder.jl in Perception/:
  Encoder  (Perception): observation history → z_t
  Predictor (WorldModule): (z_t, action) → ẑ_{t+H}

Training objective: minimize prediction error in latent space
  L = ||ẑ_{t+H} - z_{t+H}||² (invariance term of VICReg)
"""

using Flux
using Statistics

# ─────────────────────────────────────────────────────────────────
# Action Embedding
# Maps AbstractAction → fixed-size vector for predictor input
# Domain-agnostic — simulator registers action dimensions
# ─────────────────────────────────────────────────────────────────

struct ActionEmbedder
    embedding :: Dense   # action features → d_action
end

Flux.@layer ActionEmbedder

function ActionEmbedder(n_action_features::Int, d_action::Int=16)
    ActionEmbedder(Dense(n_action_features, d_action, tanh))
end

function (emb::ActionEmbedder)(action_features::Vector{Float32}) :: Vector{Float32}
    return emb.embedding(action_features)
end

# ─────────────────────────────────────────────────────────────────
# JEPA Predictor
# (z_t, a_t) → ẑ_{t+H} for each horizon H
# Separate prediction head per horizon — H_short, H_med, H_long
# ─────────────────────────────────────────────────────────────────

struct JEPAPredictor
    action_embedder :: ActionEmbedder
    shared_layers   :: Chain          # shared trunk
    head_short      :: Dense          # H_short prediction
    head_med        :: Dense          # H_med prediction
    head_long       :: Dense          # H_long prediction
end

Flux.@layer JEPAPredictor

function JEPAPredictor(
    z_dim        :: Int = 64,
    d_action     :: Int = 16,
    d_hidden     :: Int = 128,
    n_action_features :: Int = 8
)
    d_input = z_dim + d_action   # concatenate latent + action

    JEPAPredictor(
        ActionEmbedder(n_action_features, d_action),
        Chain(
            Dense(d_input, d_hidden, gelu),
            Dense(d_hidden, d_hidden, gelu),
            Dense(d_hidden, z_dim)
        ),
        Dense(z_dim, z_dim),   # short horizon refinement
        Dense(z_dim, z_dim),   # medium horizon refinement
        Dense(z_dim, z_dim)    # long horizon refinement
    )
end

const JEPA_PREDICTOR = JEPAPredictor()

function (pred::JEPAPredictor)(
    z             :: AbstractArray,         # (z_dim, batch) current latent
    action_feats  :: AbstractArray,         # (n_action_features, batch)
    horizon       :: Symbol = :med          # :short, :med, or :long
) :: AbstractArray

    # embed action
    a_emb = pred.action_embedder.embedding(action_feats)

    # concatenate latent + action embedding
    x = vcat(z, a_emb)

    # shared trunk
    z_pred = pred.shared_layers(x)

    # horizon-specific head
    if horizon == :short
        return pred.head_short(z_pred)
    elseif horizon == :med
        return pred.head_med(z_pred)
    else
        return pred.head_long(z_pred)
    end
end

# ─────────────────────────────────────────────────────────────────
# Action feature extraction
# Convert AbstractAction → Float32 vector for the predictor
# Domain-agnostic — just uses magnitude and is_null
# Simulator plugin can override for richer action representations
# ─────────────────────────────────────────────────────────────────

function _delta_value(action::AbstractAction, keys::Symbol...) :: Float32
    hasproperty(action, :deltas) || return 0.0f0
    deltas = getproperty(action, :deltas)
    deltas isa AbstractDict || return 0.0f0
    for key in keys
        if haskey(deltas, key)
            return Float32(deltas[key])
        end
    end
    return 0.0f0
end

function _weighted_schedule_delta(
    action      :: AbstractAction,
    field_name  :: Symbol,
) :: Float32
    hasproperty(action, :segment_deltas) || return 0.0f0
    segment_deltas = getproperty(action, :segment_deltas)
    segment_deltas isa AbstractVector || return 0.0f0
    isempty(segment_deltas) && return 0.0f0

    segments = hasproperty(action, :segments) ? getproperty(action, :segments) : ()
    segment_lookup = Dict(
        getproperty(seg, :segment_id) => seg
        for seg in segments
        if hasproperty(seg, :segment_id)
    )
    weighted_total = 0.0
    total_weight = 0.0

    for delta in segment_deltas
        hasproperty(delta, :segment_id) || continue
        segment_id = getproperty(delta, :segment_id)
        segment = get(segment_lookup, segment_id, nothing)
        value = hasproperty(delta, field_name) ? Float64(getproperty(delta, field_name)) : 0.0
        weight = isnothing(segment) ? 1.0 : max(1.0, Float64(getproperty(segment, :end_min) - getproperty(segment, :start_min)))
        weighted_total += weight * value
        total_weight += weight
    end

    return total_weight <= 1e-8 ? 0.0f0 : Float32(weighted_total / total_weight)
end

function _action_level_feature(action::AbstractAction) :: Float32
    hasproperty(action, :level) || return 1.0f0 / 3.0f0
    return clamp(Float32(getproperty(action, :level)) / 3.0f0, 0.0f0, 1.0f0)
end

function _action_family_flags(action::AbstractAction) :: Tuple{Float32, Float32}
    hasproperty(action, :family) || return (1.0f0, 0.0f0)
    family = getproperty(action, :family)
    family_name = string(family)
    return (
        family_name == "parameter_adjustment" ? 1.0f0 : 0.0f0,
        family_name == "structure_edit" ? 1.0f0 : 0.0f0,
    )
end

function action_to_features(action::AbstractAction) :: Vector{Float32}
    family_parameter, family_structure = _action_family_flags(action)
    isf_feature = hasproperty(action, :segment_deltas) ?
        _weighted_schedule_delta(action, :isf_delta) :
        _delta_value(action, :isf_delta, :isf)
    cr_feature = hasproperty(action, :segment_deltas) ?
        _weighted_schedule_delta(action, :cr_delta) :
        _delta_value(action, :cr_delta, :cr)
    basal_feature = hasproperty(action, :segment_deltas) ?
        _weighted_schedule_delta(action, :basal_delta) :
        _delta_value(action, :basal_delta, :basal)
    Float32[
        is_null(action) ? 0.0f0 : 1.0f0,
        Float32(magnitude(action)),
        _action_level_feature(action),
        family_parameter,
        family_structure,
        isf_feature,
        cr_feature,
        basal_feature,
    ]
end

function _latent_entropy(log_σ::AbstractArray) :: Float32
    return 0.5f0 * Float32(sum(1.0f0 .+ log(2.0f0 * π) .+ 2.0f0 .* Float32.(log_σ)))
end

function _point_belief(
    z          :: AbstractVector,
    obs_log_lik :: Real
) :: JEPABeliefState
    point_log_σ = fill(-4.0f0, length(z))
    return JEPABeliefState(
        μ           = Float32.(z),
        log_σ       = point_log_σ,
        entropy     = _latent_entropy(point_log_σ),
        obs_log_lik = Float32(obs_log_lik)
    )
end

function _predict_latent_state(
    belief      :: JEPABeliefState,
    action      :: AbstractAction,
    predictor   :: JEPAPredictor,
    horizon     :: Symbol,
    uncertainty :: Float32
) :: JEPABeliefState
    z = reshape(Float32.(vec(belief.μ)), :, 1)
    a = reshape(action_to_features(action), :, 1)
    z_pred = predictor(z, a, horizon)[:, 1]
    new_log_σ = Float32.(vec(belief.log_σ)) .+ uncertainty

    return JEPABeliefState(
        μ           = Float32.(z_pred),
        log_σ       = new_log_σ,
        entropy     = _latent_entropy(new_log_σ),
        obs_log_lik = Float32(belief.obs_log_lik)
    )
end

# ─────────────────────────────────────────────────────────────────
# JEPA rollout — run H steps in latent space
# v2 equivalent of run_rollouts in rollout.jl
# Much faster than explicit simulation — just neural network forward passes
# ─────────────────────────────────────────────────────────────────

function jepa_rollout(
    belief    :: JEPABeliefState,
    action    :: AbstractAction,
    predictor :: JEPAPredictor,
    config    :: ConfiguratorState
) :: JEPABeliefState
    _ = config
    return _predict_latent_state(belief, action, predictor, :med, 0.10f0)
end

# ─────────────────────────────────────────────────────────────────
# Multi-horizon JEPA rollout
# Returns predictions at all three horizons simultaneously
# Used by Actor for comprehensive action evaluation
# ─────────────────────────────────────────────────────────────────

function jepa_rollout_all_horizons(
    belief    :: JEPABeliefState,
    action    :: AbstractAction,
    predictor :: JEPAPredictor
) :: NamedTuple{(:short, :med, :long), NTuple{3, JEPABeliefState}}
    return (
        short = _predict_latent_state(belief, action, predictor, :short, 0.05f0),
        med   = _predict_latent_state(belief, action, predictor, :med,   0.10f0),
        long  = _predict_latent_state(belief, action, predictor, :long,  0.20f0)
    )
end

function _sample_latent_state(belief::JEPABeliefState) :: Vector{Float32}
    μ = Float32.(vec(belief.μ))
    σ = exp.(Float32.(vec(belief.log_σ)))
    return μ .+ σ .* randn(Float32, length(μ))
end

function run_latent_rollouts(
    belief    :: JEPABeliefState,
    action    :: AbstractAction,
    predictor :: JEPAPredictor,
    config    :: ConfiguratorState
) :: Vector{LatentRolloutResult}
    N = config.φ_world.N_roll
    results = Vector{LatentRolloutResult}(undef, N)

    Threads.@threads for i in 1:N
        z0 = _sample_latent_state(belief)
        initial = _point_belief(z0, belief.obs_log_lik)
        horizons = jepa_rollout_all_horizons(initial, action, predictor)
        results[i] = LatentRolloutResult(
            action         = action,
            initial_belief = initial,
            short_belief   = horizons.short,
            med_belief     = horizons.med,
            long_belief    = horizons.long
        )
    end

    return results
end

function run_paired_latent_rollouts(
    belief      :: JEPABeliefState,
    action      :: AbstractAction,
    null_action :: AbstractAction,
    predictor   :: JEPAPredictor,
    config      :: ConfiguratorState;
    N           :: Int = config.φ_world.N_roll
) :: Tuple{Vector{LatentRolloutResult}, Vector{LatentRolloutResult}}
    treated  = Vector{LatentRolloutResult}(undef, N)
    baseline = Vector{LatentRolloutResult}(undef, N)

    Threads.@threads for i in 1:N
        z0 = _sample_latent_state(belief)
        initial = _point_belief(z0, belief.obs_log_lik)
        treated_horizons  = jepa_rollout_all_horizons(initial, action, predictor)
        baseline_horizons = jepa_rollout_all_horizons(initial, null_action, predictor)

        treated[i] = LatentRolloutResult(
            action         = action,
            initial_belief = initial,
            short_belief   = treated_horizons.short,
            med_belief     = treated_horizons.med,
            long_belief    = treated_horizons.long
        )

        baseline[i] = LatentRolloutResult(
            action         = null_action,
            initial_belief = initial,
            short_belief   = baseline_horizons.short,
            med_belief     = baseline_horizons.med,
            long_belief    = baseline_horizons.long
        )
    end

    return treated, baseline
end
