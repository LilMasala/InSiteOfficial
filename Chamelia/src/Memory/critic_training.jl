"""
critic_training.jl
Critic training pipeline.
Computes realized residual costs R_tau from memory
and triggers critic updates when enough data exists.

Pipeline:
  1. For each completed record, compute R_tau retrospectively
  2. When enough (record, R_tau) pairs exist, train critic
  3. Critic bootstraps at day 30 (ridge), upgrades at day 60 (MLP)

Critical constraint: R_tau computed from REALIZED costs in memory,
NOT from World Module predictions. Prevents self-reinforcing loops.
"""

using Statistics
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# Compute realized residual cost R_tau for one record
# R_tau = sum_{k=1}^{H_long - H_med} gamma^k * C_int_{tau+k}
# Uses whatever realized future is available in memory
# ─────────────────────────────────────────────────────────────────

function compute_critic_target!(
    mem         :: MemoryBuffer,
    rec_id      :: Int,
    current_day :: Int,
    γ           :: Float64 = 0.99,
    H_long      :: Int = 90,
    H_med       :: Int = 7
) :: Nothing

    idx = findfirst(r -> r.id == rec_id, mem.records)
    isnothing(idx) && return nothing

    rec = mem.records[idx]
    isnothing(rec.realized_cost) && return nothing

    # collect future realized costs beyond H_med
    future = filter(
        r -> r.day > rec.day + H_med &&
             r.day <= rec.day + H_long &&
             !isnothing(r.realized_cost),
        mem.records
    )

    isempty(future) && return nothing

    # discounted sum
    R = 0.0
    for r in future
        k = r.day - rec.day
        R += γ^k * r.realized_cost
    end

    rec.critic_target = R
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Update all critic targets in memory
# Called periodically as new outcomes arrive
# ─────────────────────────────────────────────────────────────────

function update_all_critic_targets!(
    mem         :: MemoryBuffer,
    current_day :: Int,
    config      :: ConfiguratorState
) :: Nothing

    γ     = config.φ_cost.γ_discount
    H_med = config.φ_world.H_med

    for rec in mem.records
        isnothing(rec.realized_cost) && continue
        isnothing(rec.critic_target) || continue  # already computed

        compute_critic_target!(mem, rec.id, current_day, γ, 90, H_med)
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Maybe update critic
# Checks if enough data exists and triggers training if so.
# Bootstrap schedule:
#   Day 30+: train RidgeCritic if >= 10 records with targets
#   Day 60+: upgrade to MLPCritic if >= 20 records with targets
# ─────────────────────────────────────────────────────────────────

function maybe_update_critic!(
    mem         :: MemoryBuffer,
    current_day :: Int,
    config      :: ConfiguratorState
) :: Nothing

    # first update all targets
    update_all_critic_targets!(mem, current_day, config)

    # count records with critic targets
    n_with_targets = count(
        r -> !isnothing(r.critic_target),
        mem.records
    )

    # bootstrap RidgeCritic at day 30
    if current_day >= 30 && n_with_targets >= 10
        if isnothing(mem.critic) || mem.critic isa ZeroCritic
            @info "Bootstrapping RidgeCritic at day $current_day with $n_with_targets records"
            mem.critic = RidgeCritic()
        end
        mem.critic isa RidgeCritic && train_critic!(mem.critic, mem, current_day)
    end

    # upgrade to MLPCritic at day 60
    if current_day >= 60 && n_with_targets >= 20
        if isnothing(mem.critic) || mem.critic isa RidgeCritic
            @info "Upgrading to MLPCritic at day $current_day with $n_with_targets records"
            mem.critic = MLPCritic()
        end
        mem.critic isa MLPCritic && train_critic!(mem.critic, mem, current_day)
    end

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Current critic accessor
# Returns whatever critic is currently active
# ─────────────────────────────────────────────────────────────────

function current_critic(mem::MemoryBuffer) :: AbstractCriticModel
    return isnothing(mem.critic) ? ZeroCritic() : mem.critic
end
