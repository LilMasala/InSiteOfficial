"""
posterior.jl
TwinPosterior update logic.
Updated from data — starts at prior, drifts toward patient's true characteristics.
"""


using Statistics

# -------------------------------------------------------------------
# Initialize posterior from prior — day 0, no patient data yet.
# Start at the mean of each prior distribution.
# -------------------------------------------------------------------

function initialize_posterior(prior::TwinPrior) :: TwinPosterior
    TwinPosterior(
        trust_growth_rate    = mean(prior.trust_growth_dist),
        trust_decay_rate     = mean(prior.trust_decay_dist),
        burnout_sensitivity  = mean(prior.burnout_sensitivity_dist),
        engagement_decay     = mean(prior.engagement_decay_dist),
        physical             = Dict(k => mean(v) for (k,v) in prior.physical_priors),
        last_updated_day     = 0,
        n_observations       = 0
    )
end

# -------------------------------------------------------------------
# Update posterior from recent memory records.
# Called by the system after enough data has accumulated.
# Three separate update cadences per the formulation:
#   - Psychological params: daily
#   - Physical params: weekly (need stable windows)
#   - n_observations: every call
# -------------------------------------------------------------------

function update_posterior!(
    posterior :: TwinPosterior,
    prior     :: TwinPrior,
    records   :: Vector{MemoryRecord},
    current_day :: Int,
    min_accepts_for_trust_update :: Int = 5,   # configurable
    min_obs_for_burnout_update   :: Int = 10,  # configurable
    min_days_for_physical_update :: Int = 7    # configurable
)
    isempty(records) && return

    posterior.n_observations += length(records)
    posterior.last_updated_day = current_day

    # update psychological params from trust/burnout/engagement trajectories
    _update_psychological!(posterior, prior, records,min_accepts_for_trust_update, min_obs_for_burnout_update)

    # update physical params — simulator-specific, only if registered
    _update_physical!(posterior, prior, records, min_days_for_physical_update)
end

# -------------------------------------------------------------------
# Psychological update — trust, burnout, engagement dynamics
# Uses realized acceptance/rejection patterns from memory
# -------------------------------------------------------------------

function _update_psychological!(
    posterior :: TwinPosterior,
    prior     :: TwinPrior,
    records   :: Vector{MemoryRecord},
    min_accepts_for_trust_update :: Int = 5,   # configurable
    min_obs_for_burnout_update   :: Int = 10
)
    # only use records that have realized outcomes
    completed = filter(r -> !isnothing(r.user_response), records)
    isempty(completed) && return

    # estimate trust growth rate from acceptance patterns
    # more accepts after good outcomes → higher trust growth
    accepts = filter(r -> r.user_response == Accept, completed)
    if length(accepts) >= min_accepts_for_trust_update
        # simple MLE: average trust delta after acceptance
        trust_deltas = [r.trust_at_rec for r in accepts]
        raw_estimate = mean(trust_deltas)

        # regularize toward prior mean — pull estimate back if few observations
        n = length(accepts)
        prior_mean = mean(prior.trust_growth_dist)
        posterior.trust_growth_rate = _regularize(raw_estimate, prior_mean, n)
    end

    # estimate burnout sensitivity from burnout trajectory
    burnout_vals = [r.burnout_at_rec for r in completed]
    if length(burnout_vals) >= min_obs_for_burnout_update
        raw_estimate = mean(burnout_vals)
        prior_mean = mean(prior.burnout_sensitivity_dist)
        posterior.burnout_sensitivity = _regularize(raw_estimate, prior_mean, length(burnout_vals))
    end
end

# -------------------------------------------------------------------
# Physical update — simulator-specific parameters
# Only updates keys that the simulator has registered in physical_priors
# -------------------------------------------------------------------

function _update_physical!(
    posterior :: TwinPosterior,
    prior     :: TwinPrior,
    records   :: Vector{MemoryRecord},
    min_days_for_physical_update :: Int = 7    # configurable
)
    for (label, prior_dist) in prior.physical_priors
        # collect realized signal values for this physical variable
        vals = Float64[]
        for r in records
            if !isnothing(r.realized_signals) && haskey(r.realized_signals, label)
                push!(vals, r.realized_signals[label])
            end
        end

        length(vals) < min_days_for_physical_update && continue  # need at least a week of data

        raw_estimate = mean(vals)
        prior_mean = mean(prior_dist)
        posterior.physical[label] = _regularize(raw_estimate, prior_mean, length(vals))
    end
end

# -------------------------------------------------------------------
# Regularization toward prior
# Pulls estimate back toward prior mean when n is small.
# As n grows the estimate is trusted more.
# weight = n / (n + regularization_strength)
# -------------------------------------------------------------------

function _regularize(
    estimate :: Float64,
    prior_mean :: Float64,
    n :: Int,
    strength :: Float64 = 10.0  # higher = more conservative, trust prior longer
) :: Float64
    weight = n / (n + strength)
    return weight * estimate + (1 - weight) * prior_mean
end
