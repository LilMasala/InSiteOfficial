"""
latent.jl
Provisional latent cost bridge for the JEPA path.

The math document requires the cost function to remain explicit even when
Perception/World move to latent space. The learned decoder is not present
yet, so this bridge reserves a small set of latent dimensions as auditable
summary channels: trust, burnout, engagement, burden, and physical risk.
"""

Base.@kwdef struct LatentCostBridge
    trust_idx      :: Int = 1
    burnout_idx    :: Int = 2
    engagement_idx :: Int = 3
    burden_idx     :: Int = 4
    physical_range :: UnitRange{Int} = 5:8
end

const LATENT_COST_BRIDGE = LatentCostBridge()

mutable struct LatentCostDecoder
    model      :: Chain
    is_trained :: Bool
    n_trained  :: Int
end

const LATENT_DECODER = LatentCostDecoder(
    Chain(
        Dense(64, 32, relu),
        Dense(32, 5, sigmoid)
    ),
    false,
    0
)

_sigmoid(x::Real) = 1.0 / (1.0 + exp(-Float64(x)))

function _latent_component(
    z       :: AbstractVector,
    idx     :: Int,
    default :: Float64 = 0.0
) :: Float64
    idx <= length(z) ? Float64(z[idx]) : default
end

function decode_latent_summary(
    belief :: JEPABeliefState,
    bridge :: LatentCostBridge = LATENT_COST_BRIDGE
)
    z = Float32.(vec(belief.μ))
    physical_idxs = [i for i in bridge.physical_range if i <= length(z)]
    physical_risk = isempty(physical_idxs) ? 0.0 :
        mean(abs.(Float64.(z[physical_idxs])))

    return (
        trust         = _sigmoid(_latent_component(z, bridge.trust_idx)),
        burnout       = _sigmoid(_latent_component(z, bridge.burnout_idx)),
        engagement    = _sigmoid(_latent_component(z, bridge.engagement_idx)),
        burden        = log1p(exp(_latent_component(z, bridge.burden_idx))),
        physical_risk = physical_risk
    )
end

function _decoder_input(snapshot::AbstractVector) :: Vector{Float32}
    x = Float32.(vec(snapshot))
    if length(x) == 64
        return x
    elseif length(x) > 64
        return x[1:64]
    else
        return vcat(x, zeros(Float32, 64 - length(x)))
    end
end

function _decode_latent_summary(
    belief :: JEPABeliefState
)
    if LATENT_DECODER.is_trained
        decoded = Float32.(LATENT_DECODER.model(_decoder_input(belief.μ)))
        return (
            trust         = Float64(decoded[1]),
            burnout       = Float64(decoded[2]),
            engagement    = Float64(decoded[3]),
            burden        = 5.0 * Float64(decoded[4]),
            physical_risk = 10.0 * Float64(decoded[5])
        )
    end

    return decode_latent_summary(belief, LATENT_COST_BRIDGE)
end

function train_decoder!(
    decoder :: LatentCostDecoder,
    mem     :: MemoryBuffer
) :: Nothing
    training_records = filter(
        r -> !isnothing(r.latent_snapshot) && !isnothing(r.realized_cost),
        mem.records
    )

    length(training_records) < 20 && return nothing

    X = [_decoder_input(r.latent_snapshot) for r in training_records]
    Y = [Float32[
            r.trust_at_rec,
            r.burnout_at_rec,
            r.engagement_at_rec,
            clamp(r.burden_at_rec / 5.0, 0.0, 1.0),
            clamp(something(r.realized_cost, 0.0) / 10.0, 0.0, 1.0)
         ] for r in training_records]

    opt = Flux.setup(Adam(1e-3), decoder.model)

    for _ in 1:100
        for (x_i, y_i) in zip(X, Y)
            loss, grads = Flux.withgradient(decoder.model) do model
                ŷ = model(x_i)
                mean((ŷ .- y_i).^2)
            end
            Flux.update!(opt, decoder.model, grads[1])
        end
    end

    decoder.is_trained = true
    decoder.n_trained = length(training_records)
    return nothing
end

function compute_latent_intrinsic_cost(
    action        :: AbstractAction,
    current_state :: JEPABeliefState,
    next_state    :: JEPABeliefState,
    weights       :: CostWeights;
    bridge        :: LatentCostBridge = LATENT_COST_BRIDGE
) :: Float64
    current = LATENT_DECODER.is_trained ?
        _decode_latent_summary(current_state) :
        decode_latent_summary(current_state, bridge)
    next    = LATENT_DECODER.is_trained ?
        _decode_latent_summary(next_state) :
        decode_latent_summary(next_state, bridge)

    physical_scale = isempty(weights.physical) ? 1.0 : mean(values(weights.physical))
    C_physical = physical_scale * next.physical_risk
    C_burden   = compute_burden_cost(action, next.burden, weights)
    C_trust    = compute_trust_cost(current.trust, next.trust, weights)
    C_burnout  = compute_burnout_cost(
        next.burnout,
        next.burden,
        next.trust,
        next.engagement,
        0.0,
        weights
    )

    return C_physical + C_burden + C_trust + C_burnout
end

function compute_latent_rollout_energy(
    rollout :: LatentRolloutResult,
    config  :: ConfiguratorState
) :: Float64
    γ       = config.φ_cost.γ_discount
    weights = config.φ_cost.weights

    C_short = compute_latent_intrinsic_cost(
        rollout.action,
        rollout.initial_belief,
        rollout.short_belief,
        weights
    )
    C_med = compute_latent_intrinsic_cost(
        rollout.action,
        rollout.short_belief,
        rollout.med_belief,
        weights
    )
    C_long = compute_latent_intrinsic_cost(
        rollout.action,
        rollout.med_belief,
        rollout.long_belief,
        weights
    )

    return C_short +
           γ^(config.φ_world.H_short) * C_med +
           γ^(config.φ_world.H_med * 24) * C_long
end

function compute_energies(
    rollouts :: Vector{LatentRolloutResult},
    critic   :: AbstractCriticModel,
    config   :: ConfiguratorState
) :: Vector{Float64}
    _ = critic
    return [compute_latent_rollout_energy(r, config) for r in rollouts]
end
