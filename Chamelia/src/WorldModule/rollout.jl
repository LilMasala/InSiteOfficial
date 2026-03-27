"""
rollout.jl
Stochastic rollout engine.
Runs N forward simulations from current belief to produce
a distribution of future patient states under a candidate action.

This is the core of the World Module:
  belief b_t + action a → N trajectories → distribution of futures

Used by:
  - Actor: evaluate candidate actions via CVaR
  - Burnout: paired counterfactual rollouts
  - World Module: short/medium horizon previews
"""

using Statistics

function _signal_float(signals::Dict{Symbol, Any}, key::Symbol, default::Float64) :: Float64
    value = get(signals, key, default)
    return value === nothing ? default : Float64(value)
end

# ─────────────────────────────────────────────────────────────────
# RolloutResult
# Summary of one complete forward simulation
# Add to types.jl
# ─────────────────────────────────────────────────────────────────

# (already defined in types.jl — add this if not there)
# struct RolloutResult
#   terminal_state :: PatientState
#   terminal_psy   :: PsyState
#   total_cost     :: Float64
#   psy_trajectory :: Vector{PsyState}
#   phys_signals   :: Vector{Dict{Symbol,Any}}
# end

# ─────────────────────────────────────────────────────────────────
# Sample initial state from belief
# Routes to correct sampling method per belief type
# ─────────────────────────────────────────────────────────────────

function sample_initial_state(belief::GaussianBeliefState) :: PatientState
    # sample from Gaussian belief
    phys_vars = Dict{Symbol, Float64}(
        label => rand(Normal(μ, sqrt(σ²)))
        for (label, μ) in belief.x̂_phys
        for (l2, σ²) in belief.Σ_phys
        if label == l2
    )

    psy = PsyState(
        trust      = ScalarTrust(clamp(rand(Normal(belief.x̂_trust, belief.σ_trust)), 0.0, 1.0)),
        burden     = ScalarBurden(max(0.0, rand(Normal(belief.x̂_burden, belief.σ_burden)))),
        engagement = ScalarEngagement(clamp(rand(Normal(belief.x̂_engagement, belief.σ_engagement)), 0.0, 1.0)),
        burnout    = ScalarBurnout(clamp(rand(Normal(belief.x̂_burnout, belief.σ_burnout)), 0.0, 1.0))
    )

    PatientState(PhysState(phys_vars), psy)
end

function sample_initial_state(belief::ParticleBeliefState) :: PatientState
    # sample one particle weighted by particle weights
    cumulative = cumsum(belief.weights)
    u = rand()
    idx = findfirst(c -> c >= u, cumulative)
    return belief.particles[idx]
end

function sample_initial_state(belief::JEPABeliefState) :: PatientState
    # JEPA belief is in latent space — return empty physical state
    # World Module uses JEPA predictor directly for v2 rollouts
    # this is only called for v1.1 rollouts
    error("JEPABeliefState cannot be sampled directly — use jepa_predictor.jl")
end

# ─────────────────────────────────────────────────────────────────
# Single rollout — one forward simulation
# Steps through H timesteps using simulator + behavioral dynamics
# ─────────────────────────────────────────────────────────────────

function single_rollout(
    initial_state :: PatientState,
    action        :: AbstractAction,
    twin          :: DigitalTwin,
    sim           :: AbstractSimulator,
    noise         :: RolloutNoise,
    weights       :: CostWeights,
    H             :: Int,              # horizon in steps
    γ             :: Float64 = 0.99    # discount factor
) :: RolloutResult

    state = initial_state
    total_cost = 0.0
    psy_trajectory = PsyState[]
    phys_signals   = Dict{Symbol, Any}[]

    for k in 1:H
        # sample noise for this step
        ξ = sample_noise(noise)

        # simulator advances physical state
        state = sim_step!(sim, state, action, ξ)

        # generate observation from new state
        obs = sim_observe(sim, state)

        # compute domain-specific frustration
        frustration = compute_frustration(sim, obs.signals)

        # extract sleep debt and stress from signals
        sleep_debt = _signal_float(obs.signals, :sleep_debt, 0.0)
        stress     = _signal_float(obs.signals, :stress_acute, 0.0)

        # update psychological state
        # during rollouts we assume patient accepts (conservative estimate)
        τ_prev = state.psy.trust.value
        new_psy = update_psy_state(
            state.psy,
            action,
            Accept,          # assume acceptance for rollout
            0.0,             # outcome quality unknown mid-rollout
            frustration,
            sleep_debt,
            stress,
            twin.posterior,
            ξ,
            0.95
        )

        state = PatientState(state.phys, new_psy)
        total_cost += γ^(k - 1) * compute_intrinsic_cost(
            action,
            new_psy,
            obs.signals,
            τ_prev,
            frustration,
            weights,
            sim
        )

        # record trajectory
        push!(psy_trajectory, new_psy)
        push!(phys_signals, obs.signals)
    end

    RolloutResult(
        action         = action,
        initial_psy    = initial_state.psy,
        terminal_state = state,
        terminal_psy   = state.psy,
        total_cost     = total_cost,   # filled in by Cost module
        psy_trajectory = psy_trajectory,
        phys_signals   = phys_signals
    )
end

# ─────────────────────────────────────────────────────────────────
# Run N rollouts in parallel
# Returns Vector{RolloutResult} — one per rollout
# This is the main entry point for Actor and Burnout modules
# ─────────────────────────────────────────────────────────────────

function run_rollouts(
    belief  :: AbstractBeliefState,
    action  :: AbstractAction,
    twin    :: DigitalTwin,
    sim     :: AbstractSimulator,
    noise   :: RolloutNoise,
    config  :: ConfiguratorState
) :: Vector{RolloutResult}

    N = config.φ_world.N_roll
    H = config.φ_world.H_med * 24   # days → hours
    γ = config.φ_cost.γ_discount

    # run N rollouts
    # Threads.@threads for parallel execution on CPU
    # replace with CUDA/Metal kernel for GPU
    results = Vector{RolloutResult}(undef, N)

    Threads.@threads for i in 1:N
        x0 = sample_initial_state(belief)
        results[i] = single_rollout(
            x0,
            action,
            twin,
            sim,
            noise,
            config.φ_cost.weights,
            H,
            γ
        )
    end

    return results
end

# ─────────────────────────────────────────────────────────────────
# Paired counterfactual rollouts for burnout attribution
# Shares initial state AND noise sequence between treated/baseline
# This is what makes the CI tight — common noise cancels out
# ─────────────────────────────────────────────────────────────────

function run_paired_rollouts(
    belief      :: AbstractBeliefState,
    action      :: AbstractAction,     # active policy π
    null_action :: AbstractAction,     # null policy π⁰
    twin        :: DigitalTwin,
    sim         :: AbstractSimulator,
    noise       :: RolloutNoise,
    H           :: Int,                # H_burn in days
    N           :: Int = 100           # number of pairs
) :: Tuple{Vector{RolloutResult}, Vector{RolloutResult}}

    treated  = Vector{RolloutResult}(undef, N)
    baseline = Vector{RolloutResult}(undef, N)

    Threads.@threads for i in 1:N
        # shared initial state — same starting point
        x0 = sample_initial_state(belief)

        # shared noise sequence — same exogenous randomness
        # this is what makes it a paired test
        noise_seq = [sample_noise(noise) for _ in 1:H*24]

        # treated path — active policy
        treated[i]  = _rollout_with_noise_seq(x0, action,      twin, sim, noise_seq, H*24)

        # baseline path — null action always
        baseline[i] = _rollout_with_noise_seq(x0, null_action, twin, sim, noise_seq, H*24)
    end

    return treated, baseline
end

# ─────────────────────────────────────────────────────────────────
# Rollout with pre-sampled noise sequence
# Used for paired counterfactual rollouts
# ─────────────────────────────────────────────────────────────────

function _rollout_with_noise_seq(
    initial_state :: PatientState,
    action        :: AbstractAction,
    twin          :: DigitalTwin,
    sim           :: AbstractSimulator,
    noise_seq     :: Vector{Dict{Symbol, Float64}},
    H             :: Int
) :: RolloutResult

    state = initial_state
    psy_trajectory = PsyState[]
    phys_signals   = Dict{Symbol, Any}[]

    for k in 1:H
        ξ = noise_seq[k]

        state      = sim_step!(sim, state, action, ξ)
        obs        = sim_observe(sim, state)
        frustration = compute_frustration(sim, obs.signals)
        sleep_debt = _signal_float(obs.signals, :sleep_debt, 0.0)
        stress     = _signal_float(obs.signals, :stress_acute, 0.0)

        new_psy = update_psy_state(
            state.psy, action, Accept, 0.0,
            frustration, sleep_debt, stress,
            twin.posterior, ξ
        )

        state = PatientState(state.phys, new_psy)
        push!(psy_trajectory, new_psy)
        push!(phys_signals, obs.signals)
    end

    RolloutResult(
        action         = action,
        initial_psy    = initial_state.psy,
        terminal_state = state,
        terminal_psy   = state.psy,
        total_cost     = 0.0,
        psy_trajectory = psy_trajectory,
        phys_signals   = phys_signals
    )
end
