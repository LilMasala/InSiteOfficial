"""
energy.jl
Total energy computation — E_t(a)
Combines explicit rollout cost with Critic terminal value estimate.

E_t(a) = C̄^int_{t:t+H}(a) + γ^H · V̂_ψ(s_{t+H})

Where:
  C̄^int = average discounted intrinsic cost across rollouts
  V̂_ψ   = Critic's estimate of residual cost beyond horizon H
  γ^H   = discount factor applied to terminal value
"""

using Statistics

# ─────────────────────────────────────────────────────────────────
# Compute total energy for one rollout
# E^(i)(a) = rollout cost + discounted critic value
# ─────────────────────────────────────────────────────────────────

function compute_rollout_energy(
    rollout :: RolloutResult,
    critic  :: AbstractCriticModel,
    γ       :: Float64,
    H       :: Int
) :: Float64

    # explicit rollout cost (already computed during rollout)
    rollout_cost = rollout.total_cost

    # critic terminal value estimate
    V̂ = critic_value(critic, rollout.terminal_state, rollout.terminal_psy,
                     rollout.psy_trajectory)

    # total energy
    return rollout_cost + γ^H * V̂
end

# ─────────────────────────────────────────────────────────────────
# Compute energies for all rollouts under a candidate action
# Returns Vector{Float64} — one energy per rollout
# ─────────────────────────────────────────────────────────────────

function compute_energies(
    rollouts :: Vector{RolloutResult},
    critic   :: AbstractCriticModel,
    config   :: ConfiguratorState
) :: Vector{Float64}

    γ = config.φ_cost.γ_discount
    H = config.φ_world.H_med * 24   # days → hours

    return [compute_rollout_energy(r, critic, γ, H) for r in rollouts]
end

# ─────────────────────────────────────────────────────────────────
# CVaR computation
# CVaR_α = mean of worst (1-α) fraction of energies
# α = 0.8 → average worst 20% of rollouts
# Focuses Actor on robust actions — best worst case
# ─────────────────────────────────────────────────────────────────

function compute_cvar(
    energies :: Vector{Float64},
    α        :: Float64 = 0.8
) :: Float64

    N = length(energies)
    sorted = sort(energies)   # ascending — worst are at the end

    # take worst (1-α) fraction
    n_tail = max(1, ceil(Int, (1.0 - α) * N))
    tail   = sorted[end - n_tail + 1 : end]

    return mean(tail)
end

# ─────────────────────────────────────────────────────────────────
# Effect size gate
# δ_eff = (CVaR(a⁰) - CVaR(a)) / std(energies(a))
# Only recommend if improvement exceeds noise by δ_min
# ─────────────────────────────────────────────────────────────────

function compute_effect_size(
    energies_action   :: Vector{Float64},   # energies under candidate action
    energies_baseline :: Vector{Float64},   # energies under null action
    α                 :: Float64 = 0.8
) :: Float64

    cvar_action   = compute_cvar(energies_action, α)
    cvar_baseline = compute_cvar(energies_baseline, α)

    # improvement over baseline
    improvement = cvar_baseline - cvar_action

    # normalize by rollout variance — is improvement real or just noise?
    σ_rollout = std(energies_action) + 1e-6

    return improvement / σ_rollout
end