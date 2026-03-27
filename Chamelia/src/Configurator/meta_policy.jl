"""
meta_policy.jl
Offline RL configurator — v2.0
Conservative Q-Learning (CQL) over configuration parameters.

Why CQL over standard Q-learning:
  - We only have offline data — no online exploration
  - Standard Q-learning overestimates Q for unseen (state, config) pairs
  - CQL adds conservatism penalty: penalizes high Q-values outside data support
  - This prevents the policy from exploiting configurations it has never tried

The Q-function:
  Q(meta_state, config) = expected downstream win_rate_delta

Training objective:
  L_CQL = E[(Q(s,a) - r)^2]                    (Bellman error)
        + alpha * E[log sum_a exp(Q(s,a)) - Q(s,a_data)]  (conservatism)

At inference: config = argmax_a Q(meta_state, a)
But a is continuous — use gradient ascent on Q with bounds from math.

Config bounds always enforced from system constraints — not from network.
Network decides WHERE in valid range, math decides WHAT the valid range is.
"""

using Flux
using Statistics
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Q-Network
# Maps (meta_state, config) → scalar Q-value
# Separate encoding of state and config, then combine
# ─────────────────────────────────────────────────────────────────

const CONFIG_DIM = 4   # Δ_max, N_roll_normalized, H_med_normalized, α_cvar

mutable struct QNetwork
    state_encoder  :: Chain    # meta_state → latent
    config_encoder :: Chain    # config → latent
    combiner       :: Chain    # (state_latent, config_latent) → Q
    n_trained      :: Int
    last_trained   :: Int
    is_ready       :: Bool
    min_samples    :: Int
end

function QNetwork(
    d_state  :: Int = 32,
    d_config :: Int = 16,
    d_hidden :: Int = 64
)
    QNetwork(
        # state encoder
        Chain(
            Dense(META_FEATURE_DIM, d_state, relu),
            Dense(d_state, d_state, relu)
        ),
        # config encoder
        Chain(
            Dense(CONFIG_DIM, d_config, relu),
            Dense(d_config, d_config, relu)
        ),
        # combiner
        Chain(
            Dense(d_state + d_config, d_hidden, relu),
            Dense(d_hidden, d_hidden, relu),
            Dense(d_hidden, 1)
        ),
        0, 0, false, 100
    )
end

Flux.@layer QNetwork

function (q::QNetwork)(state_feat::AbstractArray, config_feat::AbstractArray)
    s = q.state_encoder(state_feat)
    c = q.config_encoder(config_feat)
    return q.combiner(vcat(s, c))[1]
end

# Global Q-network
const Q_NET = QNetwork()

# ─────────────────────────────────────────────────────────────────
# Config encoding/decoding
# Normalize config parameters to [0,1] for the network
# Always use system-derived bounds — not hardcoded numbers
# ─────────────────────────────────────────────────────────────────

function encode_config(
    Δ_max   :: Float64,
    N_roll  :: Int,
    H_med   :: Int,
    α_cvar  :: Float64,
    meta    :: MetaState,
    prefs   :: UserPreferences
) :: Vector{Float32}

    # bounds from system constraints
    Δ_max_ceiling = compute_delta_max(prefs, meta.trust_level,
                                      meta.win_rate, meta.drift_detected,
                                      meta.n_records)
    Δ_max_floor   = 0.02
    N_min = Float64(min_n_roll(α_cvar))
    N_max = 150.0
    H_min = Float64(H_MED_MIN)
    H_max = meta.drift_detected ?
            Float64(max_h_med_during_drift(meta.days_since_drift)) : 14.0

    Float32[
        (Δ_max  - Δ_max_floor)  / (Δ_max_ceiling - Δ_max_floor + 1e-10),
        (Float64(N_roll) - N_min) / (N_max - N_min),
        (Float64(H_med)  - H_min) / (H_max - H_min + 1e-10),
        (α_cvar - 0.70)          / 0.25
    ]
end

function decode_config(
    encoded :: Vector{Float32},
    meta    :: MetaState,
    prefs   :: UserPreferences
) :: Tuple{Float64, Int, Int, Float64}

    Δ_max_ceiling = compute_delta_max(prefs, meta.trust_level,
                                      meta.win_rate, meta.drift_detected,
                                      meta.n_records)
    α_cvar_current = 0.80   # use current alpha for N_min
    N_min = Float64(min_n_roll(α_cvar_current))
    H_max = meta.drift_detected ?
            Float64(max_h_med_during_drift(meta.days_since_drift)) : 14.0

    Δ_max  = 0.02 + clamp(Float64(encoded[1]), 0.0, 1.0) *
             (Δ_max_ceiling - 0.02)
    N_roll = round(Int, N_min + clamp(Float64(encoded[2]), 0.0, 1.0) *
             (150.0 - N_min))
    H_med  = round(Int, Float64(H_MED_MIN) +
             clamp(Float64(encoded[3]), 0.0, 1.0) *
             (H_max - Float64(H_MED_MIN)))
    α_cvar = 0.70 + clamp(Float64(encoded[4]), 0.0, 1.0) * 0.25

    return Δ_max, N_roll, H_med, α_cvar
end

# ─────────────────────────────────────────────────────────────────
# CQL Training
# Trains Q-network on (meta_state, config, performance) triples
# from the experience buffer collected by the bandit phase.
#
# Loss = Bellman error + CQL conservatism penalty
#
# Bellman: (Q(s,a) - r)^2
# CQL:     alpha * (log sum_a exp(Q(s,a)) - Q(s, a_data))
#          This pushes down Q-values for actions not in the data.
# ─────────────────────────────────────────────────────────────────

function train_cql!(
    current_day  :: Int,
    meta_history :: Vector{MetaState},
    config_history :: Vector{ConfiguratorState},
    performance_history :: Vector{Float64},
    prefs        :: UserPreferences;
    n_epochs     :: Int = 200,
    lr           :: Float64 = 1e-3,
    α_cql        :: Float64 = 0.1,   # CQL conservatism strength
    n_random     :: Int = 10          # random configs per state for CQL term
) :: Nothing

    length(meta_history) < Q_NET.min_samples && return nothing

    opt = Flux.setup(Adam(lr), Q_NET)

    # build dataset
    states  = [Float32.(meta_to_features(m)) for m in meta_history]
    configs = [encode_config(
                    c.φ_act.Δ_max,
                    c.φ_world.N_roll,
                    c.φ_world.H_med,
                    c.φ_act.α_cvar,
                    m, prefs
               ) for (c, m) in zip(config_history, meta_history)]
    rewards = Float32.(performance_history)

    for epoch in 1:n_epochs
        total_loss = 0.0

        for (s, a, r) in zip(states, configs, rewards)

            loss, grads = Flux.withgradient(Q_NET) do q

                # Bellman error
                q_pred    = q(s, a)
                bellman   = (q_pred - r)^2

                # CQL conservatism — sample random configs and push their Q down
                # This ensures Q is only high for configs seen in training data
                random_configs = [Float32.(rand(CONFIG_DIM)) for _ in 1:n_random]
                q_random  = [q(s, rc) for rc in random_configs]
                log_sum   = log(sum(exp.(q_random)) + exp(q_pred))
                cql_penalty = α_cql * (log_sum - q_pred)

                bellman + cql_penalty
            end

            total_loss += loss
            Flux.update!(opt, Q_NET, grads[1])
        end

        if epoch % 50 == 0
            @info "CQL epoch $epoch, loss=$(round(total_loss/length(states), digits=4))"
        end
    end

    Q_NET.n_trained    = length(meta_history)
    Q_NET.last_trained = current_day
    Q_NET.is_ready     = true

    @info "CQL trained at day $current_day on $(length(meta_history)) experiences"
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Inference — find best config via gradient ascent on Q
# We maximize Q(meta_state, config) subject to system bounds
# Gradient ascent in config space — differentiable through Q-network
# ─────────────────────────────────────────────────────────────────

function adapt_cql(
    config :: ConfiguratorState,
    meta   :: MetaState,
    prefs  :: UserPreferences;
    n_steps     :: Int = 50,
    lr          :: Float64 = 0.05,
    n_restarts  :: Int = 5
) :: ConfiguratorState

    # fall back to bandit if not ready
    if !Q_NET.is_ready
        return adapt_bandit(config, meta, prefs)
    end

    state_feat = Float32.(meta_to_features(meta))
    lr32 = Float32(lr)

    best_config_enc = nothing
    best_q          = -Inf

    for _ in 1:n_restarts
        # random initialization in [0,1]^CONFIG_DIM
        config_enc = rand(Float32, CONFIG_DIM)

        for step in 1:n_steps
            # gradient of Q w.r.t. config encoding
            q_val, grads = Flux.withgradient(config_enc) do c
                -Q_NET(state_feat, c)   # negative because we maximize
            end

            if !isfinite(Float64(q_val)) || isnothing(grads[1]) ||
               any(!isfinite, Float64.(grads[1]))
                break
            end

            # gradient ascent step
            config_enc = clamp.(Float32.(config_enc .- lr32 .* grads[1]), 0.0f0, 1.0f0)
        end

        q_final = Float64(Q_NET(state_feat, config_enc))

        isfinite(q_final) || continue

        if q_final > best_q
            best_q          = q_final
            best_config_enc = Float32.(copy(config_enc))
        end
    end

    if isnothing(best_config_enc) || !isfinite(best_q)
        return BANDIT_CONFIG.is_ready ?
            adapt_bandit(config, meta, prefs) :
            adapt_rule_based(config, meta, prefs)
    end

    # decode back to actual config parameters
    Δ_max, N_roll, H_med, α_cvar = decode_config(best_config_enc, meta, prefs)

    # safety override — CQL cannot override safety constraints
    if meta.safety_violations > 0
        Δ_max  = 0.02
        N_roll = 100
        H_med  = H_MED_MIN
        α_cvar = 0.80
    end

    # get rule-based config for non-bandit parameters
    rule_config = adapt_rule_based(config, meta, prefs)

    ConfiguratorState(
        φ_perc  = rule_config.φ_perc,
        φ_world = WorldConfig(config.φ_world.H_short, H_med, N_roll),
        φ_cost  = rule_config.φ_cost,
        φ_act   = ActConfig(
            Δ_max,
            config.φ_act.δ_min_effect,
            α_cvar,
            config.φ_act.N_search
        ),
        last_update_day = meta.current_day
    )
end
