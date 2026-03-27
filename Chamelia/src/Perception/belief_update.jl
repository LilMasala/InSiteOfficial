"""
belief_update.jl
Belief state estimator implementations.
All three implement the same predict/update interface.
Multiple dispatch routes to the correct math automatically.

v1.1  → KalmanBeliefEstimator  (Gaussian approximation)
v1.5  → ParticleBeliefEstimator (non-Gaussian, multimodal)
v2.0  → JEPABeliefEstimator    (learned latent representation)
"""

using Distributions
using LinearAlgebra
using Statistics

# ═══════════════════════════════════════════════════════════════════
# KALMAN BELIEF ESTIMATOR — v1.1
# Exact when dynamics are linear and noise is Gaussian.
# Approximate but tractable for patient state.
#
# State vector x̂ = [phys variables..., trust, burnout, engagement, burden]
# Belief = (x̂, Σ) — mean and covariance matrix
#
# Prediction step:
#   x̂_{t|t-1} = F · x̂_{t-1}
#   Σ_{t|t-1} = F · Σ_{t-1} · F' + Q
#
# Update step:
#   K = Σ_{t|t-1} · H' · (H · Σ_{t|t-1} · H' + R)^{-1}
#   x̂_t = x̂_{t|t-1} + K · (o_t - H · x̂_{t|t-1})
#   Σ_t = (I - K·H) · Σ_{t|t-1}
# ═══════════════════════════════════════════════════════════════════

mutable struct KalmanEstimatorParams
    F :: Matrix{Float64}    # state transition matrix
    Q :: Matrix{Float64}    # process noise covariance
    H :: Matrix{Float64}    # observation matrix
    R :: Matrix{Float64}    # observation noise covariance
end

# -------------------------------------------------------------------
# Initialize Kalman params from twin
# F, Q are psychology-driven (from posterior)
# H, R are signal-driven (from signal registry)
# -------------------------------------------------------------------

function initialize_kalman(
    twin     :: DigitalTwin,
    registry :: SignalRegistry,
    n_phys   :: Int    # number of physical state variables
) :: KalmanEstimatorParams

    n_psy = 4          # trust, burnout, engagement, burden — always 4
    n     = n_phys + n_psy

    # F — state transition
    # diagonal: each variable mostly evolves independently
    # psy variables decay toward baseline at rates from posterior
    F = Matrix{Float64}(I, n, n)

    # psychological decay rates from twin posterior
    psy_start = n_phys + 1
    F[psy_start,   psy_start]   = 1.0 - twin.posterior.engagement_decay   # trust
    F[psy_start+1, psy_start+1] = 1.0 - twin.posterior.burnout_sensitivity # burnout
    F[psy_start+2, psy_start+2] = 1.0 - twin.posterior.engagement_decay   # engagement
    F[psy_start+3, psy_start+3] = 0.95                                     # burden decay γ_β

    # Q — process noise
    # diagonal: independent noise per variable
    Q = Diagonal(fill(twin.rollout_noise_std^2, n)) |> Matrix

    # H — observation matrix
    # maps state variables to registered signals
    # identity for now — each signal directly observes one state variable
    n_obs = length(registry.signals)
    H = zeros(n_obs, n)
    for (i, (label, meta)) in enumerate(registry.signals)
        # simulator plugin should set this up properly
        # default: first n_obs state variables are directly observed
        if i <= n
            H[i, i] = 1.0
        end
    end

    # R — observation noise
    # diagonal: independent noise per signal
    # required signals get lower noise (more trusted)
    R = Diagonal([
        meta.required ? 0.1 : 1.0
        for (_, meta) in registry.signals
    ]) |> Matrix

    KalmanEstimatorParams(F, Q, H, R)
end

# -------------------------------------------------------------------
# Predict step — time update
# Move belief forward one timestep using dynamics
# -------------------------------------------------------------------

function predict_belief(
    estimator :: KalmanBeliefEstimator,
    belief    :: GaussianBeliefState,
    action    :: AbstractAction,
    twin      :: DigitalTwin,
    params    :: KalmanEstimatorParams
) :: GaussianBeliefState

    # assemble state vector from belief
    x̂ = _belief_to_vector(belief)
    Σ = _belief_to_matrix(belief)

    # prediction step
    x̂_pred = params.F * x̂
    Σ_pred  = params.F * Σ * params.F' + params.Q

    # action effect — shift mean based on action magnitude
    # larger actions = more uncertainty in prediction
    if !is_null(action)
        action_uncertainty = magnitude(action) * 0.1
        Σ_pred += Diagonal(fill(action_uncertainty^2, size(Σ_pred, 1))) |> Matrix
    end

    return _vector_to_belief(x̂_pred, Σ_pred, belief)
end

# -------------------------------------------------------------------
# Update step — measurement update
# Incorporate new observation into belief
# -------------------------------------------------------------------

function update_belief_step(
    estimator   :: KalmanBeliefEstimator,
    belief      :: GaussianBeliefState,
    observation :: Observation,
    params      :: KalmanEstimatorParams
) :: GaussianBeliefState

    x̂ = _belief_to_vector(belief)
    Σ = _belief_to_matrix(belief)

    # extract observation vector from signals
    z = _observation_to_vector(observation, params)

    # Kalman gain
    S = params.H * Σ * params.H' + params.R        # innovation covariance
    K = Σ * params.H' * inv(S)                      # Kalman gain

    # update mean and covariance
    innovation = z - params.H * x̂                  # how surprising was this observation?
    x̂_new = x̂ + K * innovation
    Σ_new  = (I - K * params.H) * Σ

    # observation log likelihood — used for anomaly detection
    obs_ll = _obs_log_likelihood(innovation, S)

    return _vector_to_belief(x̂_new, Σ_new, belief, obs_ll)
end

# -------------------------------------------------------------------
# Helper — assemble flat state vector from GaussianBeliefState
# order: [phys variables..., trust, burnout, engagement, burden]
# -------------------------------------------------------------------

function _belief_to_vector(belief::GaussianBeliefState) :: Vector{Float64}
    phys_vals = collect(values(belief.x̂_phys))
    psy_vals  = [belief.x̂_trust, belief.x̂_burnout, belief.x̂_engagement, belief.x̂_burden]
    return vcat(phys_vals, psy_vals)
end

function _belief_to_matrix(belief::GaussianBeliefState) :: Matrix{Float64}
    phys_vars = collect(values(belief.Σ_phys))
    psy_vars  = [belief.σ_trust^2, belief.σ_burnout^2, belief.σ_engagement^2, belief.σ_burden^2]
    return Diagonal(vcat(phys_vars, psy_vars)) |> Matrix
end

# -------------------------------------------------------------------
# Helper — extract observation vector from signals dict
# Only uses signals that have corresponding rows in H
# -------------------------------------------------------------------

function _observation_to_vector(
    obs    :: Observation,
    params :: KalmanEstimatorParams
) :: Vector{Float64}
    n_obs = size(params.H, 1)
    z = zeros(n_obs)
    for (i, (label, _)) in enumerate(obs.signals)
        if i <= n_obs && !isnothing(obs.signals[label])
            val = obs.signals[label]
            if val isa Float64
                z[i] = val
            end
        end
    end
    return z
end

# -------------------------------------------------------------------
# Helper — reconstruct GaussianBeliefState from updated vector
# -------------------------------------------------------------------

function _vector_to_belief(
    x̂     :: Vector{Float64},
    Σ     :: Matrix{Float64},
    prev  :: GaussianBeliefState,
    obs_ll :: Float64 = prev.obs_log_lik
) :: GaussianBeliefState

    n_phys = length(prev.x̂_phys)
    phys_keys = collect(keys(prev.x̂_phys))

    x̂_phys = Dict(phys_keys[i] => x̂[i] for i in 1:n_phys)
    Σ_phys  = Dict(phys_keys[i] => Σ[i,i] for i in 1:n_phys)

    psy_start = n_phys + 1

    entropy = _gaussian_entropy(Σ)

    GaussianBeliefState(
        x̂_phys        = x̂_phys,
        Σ_phys        = Σ_phys,
        x̂_trust       = x̂[psy_start],
        σ_trust        = sqrt(Σ[psy_start, psy_start]),
        x̂_burnout      = x̂[psy_start+1],
        σ_burnout      = sqrt(Σ[psy_start+1, psy_start+1]),
        x̂_engagement   = x̂[psy_start+2],
        σ_engagement   = sqrt(Σ[psy_start+2, psy_start+2]),
        x̂_burden       = x̂[psy_start+3],
        σ_burden        = sqrt(Σ[psy_start+3, psy_start+3]),
        entropy        = entropy,
        obs_log_lik    = obs_ll
    )
end

# -------------------------------------------------------------------
# Helper — Gaussian entropy H(b) = 0.5 * log((2πe)^n * det(Σ))
# High entropy = diffuse belief = system is uncertain
# -------------------------------------------------------------------

function _gaussian_entropy(Σ::Matrix{Float64}) :: Float64
    n = size(Σ, 1)
    sign, logdet_val = logabsdet(Σ)
    return 0.5 * (n * log(2π * ℯ) + logdet_val)
end

# -------------------------------------------------------------------
# Helper — observation log likelihood
# log p(o_t | b_{t-1}) — low value means anomaly
# innovation ~ N(0, S) under correct model
# -------------------------------------------------------------------

function _obs_log_likelihood(
    innovation :: Vector{Float64},
    S          :: Matrix{Float64}
) :: Float64
    d = MvNormal(zeros(length(innovation)), S)
    return logpdf(d, innovation)
end



# ═══════════════════════════════════════════════════════════════════
# PARTICLE BELIEF ESTIMATOR — v1.5
# No Gaussian assumption. Tracks N particles — each a complete
# hypothesis about the patient's true state.
#
# Three steps every timestep:
#   1. Propagate — move each particle through dynamics + noise
#   2. Weight    — score each particle by observation likelihood
#   3. Resample  — duplicate good particles, drop bad ones
#
# Handles multimodal beliefs — critical for psychological state.
# GPU parallelism opportunity: all N particles are independent.
# ═══════════════════════════════════════════════════════════════════

# -------------------------------------------------------------------
# Initialize N particles from the twin prior
# Day 0 — wide spread, honest about uncertainty
# Each particle is one complete hypothesis about patient state
# -------------------------------------------------------------------

function initialize_particles(
    prior   :: TwinPrior,
    N       :: Int = 100    # number of particles — configurable
) :: ParticleBeliefState

    particles = Vector{PatientState}(undef, N)

    for i in 1:N
        # sample psychological state from prior distributions
        psy = PsyState(
            trust      = ScalarTrust(rand(prior.trust_growth_dist)),
            burnout    = ScalarBurnout(rand(prior.burnout_sensitivity_dist)),
            engagement = ScalarEngagement(rand(prior.engagement_decay_dist)),
            burden     = ScalarBurden(0.0)   # no burden at day 0
        )

        # sample physical state from simulator-registered priors
        phys_vars = Dict{Symbol, Float64}(
            label => rand(dist)
            for (label, dist) in prior.physical_priors
        )

        particles[i] = PatientState(
            PhysState(phys_vars),
            psy
        )
    end

    # uniform weights at initialization — all hypotheses equally likely
    weights = fill(1.0 / N, N)

    ParticleBeliefState(
        particles   = particles,
        weights     = weights,
        entropy     = log(N),   # maximum entropy — complete uncertainty
        obs_log_lik = 0.0
    )
end

# -------------------------------------------------------------------
# Predict step — propagate each particle through dynamics + noise
# Each particle lives its own slightly different life
# This is where the simulator plugin runs per-particle
# -------------------------------------------------------------------

function predict_belief(
    estimator :: ParticleBeliefEstimator,
    belief    :: ParticleBeliefState,
    action    :: AbstractAction,
    twin      :: DigitalTwin,
    noise     :: RolloutNoise
) :: ParticleBeliefState

    N = length(belief.particles)
    new_particles = Vector{PatientState}(undef, N)

    for i in 1:N
        p = belief.particles[i]

        # sample noise for this particle
        ξ = sample_noise(noise)

        # propagate psychological state
        new_psy = _propagate_psy(p.psy, action, twin.posterior, ξ)

        # propagate physical state
        # simulator plugin handles this via AbstractSimulator interface
        # for now: apply action effect + noise to each physical variable
        new_phys_vars = Dict{Symbol, Float64}()
        for (label, val) in p.phys.variables
            noise_val = get(ξ, label, 0.0)
            new_phys_vars[label] = val + noise_val
        end

        new_particles[i] = PatientState(
            PhysState(new_phys_vars),
            new_psy
        )
    end

    # weights unchanged after prediction — only observation updates them
    ParticleBeliefState(
        particles   = new_particles,
        weights     = belief.weights,
        entropy     = belief.entropy,
        obs_log_lik = belief.obs_log_lik
    )
end

# -------------------------------------------------------------------
# Update step — weight + resample
# Score each particle by how well it explains new observation
# Then resample to concentrate on high-weight particles
# -------------------------------------------------------------------

function update_belief_step(
    estimator   :: ParticleBeliefEstimator,
    belief      :: ParticleBeliefState,
    observation :: Observation,
    noise       :: RolloutNoise
) :: ParticleBeliefState

    N = length(belief.particles)

    # ── Step 1: Weight ────────────────────────────────────────────
    # score each particle by observation likelihood
    # p(o_t | x^(i)_t) — how likely is this observation if particle i is true?
    log_weights = Vector{Float64}(undef, N)

    for i in 1:N
        log_weights[i] = _observation_log_likelihood(
            belief.particles[i],
            observation
        )
    end

    # normalize in log space for numerical stability
    log_weights .-= maximum(log_weights)
    weights = exp.(log_weights)
    weights ./= sum(weights)

    # observation log likelihood — average across particles
    obs_ll = mean(log_weights)

    # ── Step 2: Resample ──────────────────────────────────────────
    # systematic resampling — more stable than naive multinomial
    new_particles = _systematic_resample(belief.particles, weights, N)

    # ── Step 3: Add jitter — prevent particle degeneracy ─────────
    # small noise after resampling keeps particles from collapsing
    # to identical copies — allows exploration of nearby states
    new_particles = _add_jitter(new_particles, noise)

    # ── Entropy estimate from weights ────────────────────────────
    # H ≈ -Σ w_i log(w_i)
    entropy = -sum(w * log(w + 1e-10) for w in weights)

    ParticleBeliefState(
        particles   = new_particles,
        weights     = fill(1.0/N, N),   # uniform after resampling
        entropy     = entropy,
        obs_log_lik = obs_ll
    )
end

# -------------------------------------------------------------------
# Helper — propagate psychological state through one timestep
# Trust, burnout, engagement, burden dynamics
# -------------------------------------------------------------------

function _propagate_psy(
    psy     :: PsyState,
    action  :: AbstractAction,
    post    :: TwinPosterior,
    ξ       :: Dict{Symbol, Float64}
) :: PsyState

    # current values
    τ = psy.trust.value
    B = psy.burnout.value
    ω = psy.engagement.value
    β = psy.burden.value

    # burden accumulation — decays at γ_β, increases with recommendations
    β_new = 0.95 * β + (is_null(action) ? 0.0 : magnitude(action))

    # trust dynamics — erodes with large actions, builds slowly over time
    τ_new = clamp(
        τ - post.trust_decay_rate * magnitude(action) + get(ξ, :trust, 0.0),
        0.0, 1.0
    )

    # burnout dynamics — increases with burden and distrust
    ΔB_policy = post.burnout_sensitivity * β_new * (1.0 - τ_new)
    B_new = clamp(
        B + ΔB_policy + get(ξ, :burnout, 0.0),
        0.0, 1.0
    )

    # engagement — decays with burnout, recovers slowly
    ω_new = clamp(
        ω - post.engagement_decay * B_new + get(ξ, :engagement, 0.0),
        0.0, 1.0
    )

    PsyState(
        trust      = ScalarTrust(τ_new),
        burnout    = ScalarBurnout(B_new),
        engagement = ScalarEngagement(ω_new),
        burden     = ScalarBurden(β_new)
    )
end

# -------------------------------------------------------------------
# Helper — observation likelihood for one particle
# How likely is this observation if THIS particle is the true state?
# Higher = this particle is a good hypothesis
# -------------------------------------------------------------------

function _observation_log_likelihood(
    particle    :: PatientState,
    observation :: Observation
) :: Float64

    log_lik = 0.0

    for (label, value) in observation.signals
        isnothing(value) && continue
        value isa Float64 || continue

        # check if this signal corresponds to a physical variable
        if haskey(particle.phys.variables, label)
            true_val = particle.phys.variables[label]
            # assume observation noise ~ Normal(true_val, 1.0)
            # simulator plugin should override this with proper noise model
            log_lik += logpdf(Normal(true_val, 1.0), value)
        end

        # psychological signals
        if label == :mood_valence
            log_lik += logpdf(Normal(particle.psy.trust.value, 0.2), value)
        elseif label == :engagement_signal
            log_lik += logpdf(Normal(particle.psy.engagement.value, 0.15), value)
        end
    end

    return log_lik
end

# -------------------------------------------------------------------
# Helper — systematic resampling
# More numerically stable than naive multinomial resampling
# Guarantees exactly N particles with low variance
# -------------------------------------------------------------------

function _systematic_resample(
    particles :: Vector{PatientState},
    weights   :: Vector{Float64},
    N         :: Int
) :: Vector{PatientState}

    cumulative = cumsum(weights)
    new_particles = Vector{PatientState}(undef, N)

    step = 1.0 / N
    u = rand() * step   # single random draw — systematic

    j = 1
    for i in 1:N
        while u > cumulative[j]
            j += 1
        end
        new_particles[i] = particles[j]
        u += step
    end

    return new_particles
end

# -------------------------------------------------------------------
# Helper — add jitter after resampling
# Prevents particle degeneracy — keeps belief from collapsing
# Small noise so particles can explore nearby states
# -------------------------------------------------------------------

function _add_jitter(
    particles :: Vector{PatientState},
    noise     :: RolloutNoise
) :: Vector{PatientState}

    return map(particles) do p
        ξ = sample_noise(noise)

        # jitter psychological state
        new_psy = PsyState(
            trust      = ScalarTrust(clamp(p.psy.trust.value + 0.1*get(ξ,:trust,0.0), 0.0, 1.0)),
            burnout    = ScalarBurnout(clamp(p.psy.burnout.value + 0.1*get(ξ,:burnout,0.0), 0.0, 1.0)),
            engagement = ScalarEngagement(clamp(p.psy.engagement.value + 0.1*get(ξ,:engagement,0.0), 0.0, 1.0)),
            burden     = ScalarBurden(max(0.0, p.psy.burden.value + 0.1*get(ξ,:burden,0.0)))
        )

        # jitter physical state
        new_phys_vars = Dict{Symbol, Float64}(
            label => val + 0.1 * get(ξ, label, 0.0)
            for (label, val) in p.phys.variables
        )

        PatientState(PhysState(new_phys_vars), new_psy)
    end
end

# -------------------------------------------------------------------
# JEPA BELIEF ESTIMATOR — v2.0
# Latent observation-history encoder. The current implementation uses
# an explicit observation window object so the encoder can remain
# coherent without introducing a larger streaming data subsystem.
# -------------------------------------------------------------------

Base.@kwdef mutable struct JEPAInferenceParams
    encoder          :: HierarchicalJEPAEncoder
    subhourly        :: Array{Float32, 5}
    ctx              :: Array{Float32, 4}
    daily            :: Array{Float32, 3}
    subhourly_labels :: Vector{Symbol}
    ctx_labels       :: Vector{Symbol}
    daily_labels     :: Vector{Symbol}
end

function JEPAInferenceParams(
    encoder          :: HierarchicalJEPAEncoder,
    subhourly_labels :: Vector{Symbol},
    ctx_labels       :: Vector{Symbol},
    daily_labels     :: Vector{Symbol};
    n_days           :: Int = 14
) :: JEPAInferenceParams
    JEPAInferenceParams(
        encoder          = encoder,
        subhourly        = zeros(Float32, length(subhourly_labels), 12, 24, n_days, 1),
        ctx              = zeros(Float32, length(ctx_labels), 24, n_days, 1),
        daily            = zeros(Float32, length(daily_labels), n_days, 1),
        subhourly_labels = subhourly_labels,
        ctx_labels       = ctx_labels,
        daily_labels     = daily_labels
    )
end

function _signal_value(
    observation :: Observation,
    label       :: Symbol
) :: Float32
    value = get(observation.signals, label, 0.0)
    return value isa Number ? Float32(value) : 0.0f0
end

function _update_jepa_window!(
    params       :: JEPAInferenceParams,
    observation  :: Observation
) :: Nothing
    latest_day  = size(params.daily, 2)
    latest_hour = size(params.ctx, 2)

    for (i, label) in enumerate(params.subhourly_labels)
        params.subhourly[i, :, latest_hour, latest_day, 1] .= _signal_value(observation, label)
    end

    for (i, label) in enumerate(params.ctx_labels)
        params.ctx[i, latest_hour, latest_day, 1] = _signal_value(observation, label)
    end

    for (i, label) in enumerate(params.daily_labels)
        params.daily[i, latest_day, 1] = _signal_value(observation, label)
    end

    return nothing
end

function predict_belief(
    estimator :: JEPABeliefEstimator,
    belief    :: JEPABeliefState,
    action    :: AbstractAction,
    twin      :: DigitalTwin,
    params    :: JEPAInferenceParams
) :: JEPABeliefState
    _ = estimator
    _ = twin
    _ = params

    uncertainty_bump = Float32(0.05 * magnitude(action))
    new_log_σ = Float32.(belief.log_σ) .+ uncertainty_bump

    return JEPABeliefState(
        μ           = Float32.(belief.μ),
        log_σ       = new_log_σ,
        entropy     = _jepa_entropy(new_log_σ),
        obs_log_lik = belief.obs_log_lik
    )
end

function update_belief_step(
    estimator   :: JEPABeliefEstimator,
    belief      :: JEPABeliefState,
    observation :: Observation,
    params      :: JEPAInferenceParams
) :: JEPABeliefState
    _ = estimator

    _update_jepa_window!(params, observation)
    encoded = encode_observation_window(
        params.encoder,
        params.subhourly,
        params.ctx,
        params.daily
    )

    μ_prev    = Float32.(vec(belief.μ))
    μ_encoded = Float32.(vec(encoded.μ))
    log_prev  = Float32.(vec(belief.log_σ))
    log_enc   = Float32.(vec(encoded.log_σ))

    μ_new     = 0.5f0 .* (μ_prev .+ μ_encoded)
    log_σ_new = max.(log_prev, log_enc)
    obs_ll    = Float32(-mean((μ_encoded .- μ_prev).^2))

    return JEPABeliefState(
        μ           = μ_new,
        log_σ       = log_σ_new,
        entropy     = _jepa_entropy(log_σ_new),
        obs_log_lik = obs_ll
    )
end
