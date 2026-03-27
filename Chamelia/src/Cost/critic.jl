"""
critic.jl
Trainable Critic — V̂_ψ(s_{t+H})
Estimates residual cumulative cost BEYOND the explicit rollout horizon.

The World Module rolls out H days explicitly.
The Critic estimates what happens from day H to H_long.

Two implementations:
  ZeroCritic     — V̂ ≡ 0, pure finite-horizon MPC (POC start)
  RidgeCritic    — linear ridge regression on terminal state features
  (MLPCritic     — small neural network, future)

Training: minimize (V̂_ψ(s_τ) - R_τ)² over memory records
where R_τ = realized residual cost computed retrospectively from Memory.

Critical constraint: trained on REALIZED costs from Memory,
NOT on World Module predictions. Prevents self-reinforcing loops.
"""

using Statistics
using LinearAlgebra
using Flux


# ─────────────────────────────────────────────────────────────────
# Abstract Critic Interface
# ─────────────────────────────────────────────────────────────────

"""
critic.jl
Trainable Critic — V̂_ψ(s_{t+H})
Estimates residual cumulative cost BEYOND the explicit rollout horizon.

The World Module rolls out H days explicitly.
The Critic estimates what happens from day H to H_long.

Two implementations:
  ZeroCritic     — V̂ ≡ 0, pure finite-horizon MPC (POC start)
  RidgeCritic    — linear ridge regression on terminal state features
  (MLPCritic     — small neural network, future)

Training: minimize (V̂_ψ(s_τ) - R_τ)² over memory records
where R_τ = realized residual cost computed retrospectively from Memory.

Critical constraint: trained on REALIZED costs from Memory,
NOT on World Module predictions. Prevents self-reinforcing loops.
"""

using Statistics
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Abstract Critic interface
# ─────────────────────────────────────────────────────────────────

"""
    critic_value(critic, terminal_state) → Float64

Estimate residual cost from terminal rollout state onward.
Returns 0.0 for ZeroCritic (pure MPC mode).
"""
function critic_value(
    critic         :: AbstractCriticModel,
    terminal_state :: PatientState,
    terminal_psy   :: PsyState
) :: Float64
    error("$(typeof(critic)) must implement critic_value!")
end

function critic_value(
    critic         :: AbstractCriticModel,
    terminal_state :: PatientState,
    terminal_psy   :: PsyState,
    psy_trajectory :: Vector{PsyState}
) :: Float64
    _ = psy_trajectory
    return critic_value(critic, terminal_state, terminal_psy)
end

# ─────────────────────────────────────────────────────────────────
# ZeroCritic — V̂ ≡ 0
# Initial state — no Critic, pure finite-horizon MPC.
# Used for first 30 days until enough memory accumulates.
# ─────────────────────────────────────────────────────────────────

struct ZeroCritic <: AbstractCriticModel end

critic_value(::ZeroCritic, ::PatientState, ::PsyState) = 0.0

# ─────────────────────────────────────────────────────────────────
# Terminal State Features
# Hand-crafted features extracted from rollout terminal state.
# What the Critic learns from — designed to capture long-horizon signals.
#
# Features:
#   1. trust_terminal      — low trust → high future cost
#   2. burnout_terminal    — high burnout → high future cost
#   3. engagement_terminal — low engagement → high future cost
#   4. burden_terminal     — high burden → high future cost
#   5. trust_trend         — is trust improving or declining?
#   6. burnout_trend       — is burnout accelerating?
#   7. engagement_trend    — is engagement recovering?
# ─────────────────────────────────────────────────────────────────


function extract_terminal_features(
    terminal_psy   :: PsyState,
    psy_trajectory :: Vector{PsyState}
) :: Vector{Float64}

    τ = terminal_psy.trust.value
    B = terminal_psy.burnout.value
    ω = terminal_psy.engagement.value
    β = terminal_psy.burden.value

    # compute trends from trajectory if available
    if length(psy_trajectory) >= 3
        n = length(psy_trajectory)
        # trust trend — positive = improving
        τ_trend = psy_trajectory[n].trust.value - psy_trajectory[n-2].trust.value
        # burnout trend — positive = accelerating (bad)
        B_trend = psy_trajectory[n].burnout.value - psy_trajectory[n-2].burnout.value
        # engagement trend — positive = recovering (good)
        ω_trend = psy_trajectory[n].engagement.value - psy_trajectory[n-2].engagement.value
    else
        τ_trend = 0.0
        B_trend = 0.0
        ω_trend = 0.0
    end

    return Float64[τ, B, ω, β, τ_trend, B_trend, ω_trend]
end

# ─────────────────────────────────────────────────────────────────
# RidgeCritic — linear ridge regression on terminal features
# Bootstrapped after 30+ days of memory accumulation.
# Simple, interpretable, fast to train.
# ─────────────────────────────────────────────────────────────────

mutable struct RidgeCritic <: AbstractCriticModel
    weights        :: Vector{Float64}   # ψ — regression coefficients
    bias           :: Float64
    λ_ridge        :: Float64           # L2 regularization strength
    n_trained      :: Int               # how many records trained on
    last_trained_day :: Int
end

function RidgeCritic(λ_ridge::Float64 = 1.0)
    RidgeCritic(
        zeros(CRITIC_FEATURE_DIM),
        0.0,
        λ_ridge,
        0,
        0
    )
end

function critic_value(
    critic         :: RidgeCritic,
    terminal_state :: PatientState,
    terminal_psy   :: PsyState,
    psy_trajectory :: Vector{PsyState} = PsyState[]
) :: Float64
    features = extract_terminal_features(terminal_psy, psy_trajectory)
    return dot(critic.weights, features) + critic.bias
end

# ─────────────────────────────────────────────────────────────────
# Critic Training
# Trained on (terminal_features, realized_residual_cost) pairs from Memory.
# Uses ridge regression — closed form solution, no gradient descent needed.
#
# Loss: L(ψ) = Σ (V̂_ψ(s_τ) - R_τ)² + λ·||ψ||²
# Solution: ψ = (X'X + λI)^{-1} X'y
# ─────────────────────────────────────────────────────────────────

function train_critic!(
    critic  :: RidgeCritic,
    mem     :: MemoryBuffer,
    current_day :: Int
) :: Nothing

    # collect training pairs from memory
    # need records with realized critic targets
    training_records = filter(
        r -> !isnothing(r.critic_target) && !isnothing(r.realized_cost),
        mem.records
    )

    # need minimum data to train
    length(training_records) < 10 && return nothing

    n = length(training_records)

    # build feature matrix X and target vector y
    X = zeros(n, CRITIC_FEATURE_DIM)
    y = zeros(n)

    for (i, r) in enumerate(training_records)
        # reconstruct terminal psy state from memory snapshot
        terminal_psy = PsyState(
            trust      = ScalarTrust(r.trust_at_rec),
            burden     = ScalarBurden(r.burden_at_rec),
            engagement = ScalarEngagement(r.engagement_at_rec),
            burnout    = ScalarBurnout(r.burnout_at_rec)
        )
        X[i, :] = extract_terminal_features(terminal_psy, PsyState[])
        y[i]    = r.critic_target
    end

    # ridge regression closed form solution
    # ψ = (X'X + λI)^{-1} X'y
    λI = critic.λ_ridge * I(CRITIC_FEATURE_DIM)
    ψ  = (X' * X + λI) \ (X' * y)

    critic.weights        = ψ
    critic.bias           = mean(y) - dot(ψ, mean(X, dims=1)[:])
    critic.n_trained      = n
    critic.last_trained_day = current_day

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Compute realized residual cost R_τ from memory
# R_τ = Σ γ^k · C^int_{τ+k} for k = 1 to H_long - H_med
# Called when filling in retrospective memory fields
# ─────────────────────────────────────────────────────────────────

function compute_residual_cost(
    mem         :: MemoryBuffer,
    record_id   :: Int,
    current_day :: Int,
    γ           :: Float64 = 0.99,
    H_long      :: Int = 90,
    H_med       :: Int = 7
) :: Union{Float64, Nothing}

    # find the record
    rec_idx = findfirst(r -> r.id == record_id, mem.records)
    isnothing(rec_idx) && return nothing

    rec = mem.records[rec_idx]
    rec_day = rec.day

    # collect realized costs from records after this one
    future_records = filter(
        r -> r.day > rec_day &&
             r.day <= rec_day + (H_long - H_med) &&
             !isnothing(r.realized_cost),
        mem.records
    )

    isempty(future_records) && return nothing

    # discounted sum of future realized costs
    R = 0.0
    for r in future_records
        k = r.day - rec_day
        R += γ^k * r.realized_cost
    end

    return R
end


# ─────────────────────────────────────────────────────────────────
# MLPCritic — small neural network on terminal features
# More expressive than ridge — captures nonlinear interactions.
# e.g. "high burnout + low trust together are worse than sum of parts"
# Bootstrapped after 60+ days when enough data exists.
# ─────────────────────────────────────────────────────────────────

# mlp critic
mutable struct MLPCritic <: AbstractCriticModel
    model          :: Chain
    n_trained      :: Int
    last_trained_day :: Int
end


function MLPCritic(
    d_hidden :: Int = 32,
    dropout  :: Float64 = 0.1
)
    MLPCritic(
        Chain(
            Dense(CRITIC_FEATURE_DIM, d_hidden, relu),
            Dropout(dropout),
            Dense(d_hidden, d_hidden, relu),
            Dense(d_hidden, 1)         # scalar value estimate
        ),
        0,
        0
    )
end

Flux.@layer MLPCritic

function critic_value(
    critic         :: MLPCritic,
    terminal_state :: PatientState,
    terminal_psy   :: PsyState,
    psy_trajectory :: Vector{PsyState} = PsyState[]
) :: Float64
    features = Float32.(extract_terminal_features(terminal_psy, psy_trajectory))
    return Float64(critic.model(features)[1])
end

function train_critic!(
    critic      :: MLPCritic,
    mem         :: MemoryBuffer,
    current_day :: Int,
    n_epochs    :: Int = 50,
    lr          :: Float64 = 1e-3
) :: Nothing

    training_records = filter(
        r -> !isnothing(r.critic_target) && !isnothing(r.realized_cost),
        mem.records
    )

    length(training_records) < 20 && return nothing

    # build dataset
    X = [Float32.(extract_terminal_features(
            PsyState(
                trust      = ScalarTrust(r.trust_at_rec),
                burden     = ScalarBurden(r.burden_at_rec),
                engagement = ScalarEngagement(r.engagement_at_rec),
                burnout    = ScalarBurnout(r.burnout_at_rec)
            ),
            PsyState[]
         )) for r in training_records]

    y = Float32[r.critic_target for r in training_records]

    # simple SGD training loop
    opt = Flux.setup(Adam(lr), critic.model)

    for epoch in 1:n_epochs
        for (x_i, y_i) in zip(X, y)
            loss, grads = Flux.withgradient(critic.model) do m
                ŷ = m(x_i)[1]
                (ŷ - y_i)^2
            end
            Flux.update!(opt, critic.model, grads[1])
        end
    end

    critic.n_trained      = length(training_records)
    critic.last_trained_day = current_day

    return nothing
end
