"""
Perception.jl
Perception module — belief state estimation.

Maintains and updates b_t ∈ 𝒫(𝒳) over the true latent patient state
from noisy multi-modal observations.

Exports:
  - update_belief   : main entry point, routes to correct estimator
  - compute_epistemic_state : κ, ρ, η and feasibility flag
  - detect_anomaly  : observation log likelihood anomaly detection
  - widen_belief    : expand uncertainty after anomaly
  - initialize_belief : create initial belief from twin prior
"""

include("../types.jl")

module Perception

using Distributions
using LinearAlgebra
using Statistics
using Flux

# load shared types from parent
using Main: AbstractSimulator, AbstractAction, AbstractBeliefState,
            AbstractBeliefEstimator, KalmanBeliefEstimator,
            ParticleBeliefEstimator, JEPABeliefEstimator,
            PatientState, PhysState, PsyState, TwinPrior, TwinPosterior,
            DigitalTwin, RolloutNoise, Observation, SignalRegistry,
            GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            EpistemicState, EpistemicThresholds, MemoryBuffer, MemoryRecord,
            ConfiguratorState, AnomalyResult, UserResponse,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            NullAction, Accept, Reject, Partial,
            sample_noise, is_null, magnitude

# load submodules
include("jepa_encoder.jl")
include("jepa_training.jl")
include("belief_update.jl")
include("epistemic.jl")
include("anomaly.jl")

# ─────────────────────────────────────────────────────────────────
# Initialize belief from twin prior
# Day 0 — maximum uncertainty, start from prior distributions
# Routes to correct initializer based on estimator type
# ─────────────────────────────────────────────────────────────────

function initialize_belief(
    prior     :: TwinPrior,
    estimator :: KalmanBeliefEstimator
) :: GaussianBeliefState

    # start at prior means with high uncertainty
    x̂_phys = Dict(label => mean(dist)
                   for (label, dist) in prior.physical_priors)

    Σ_phys  = Dict(label => var(dist)
                   for (label, dist) in prior.physical_priors)

    GaussianBeliefState(
        x̂_phys       = x̂_phys,
        Σ_phys       = Σ_phys,
        x̂_trust      = mean(prior.trust_growth_dist),
        σ_trust       = std(prior.trust_growth_dist),
        x̂_burnout     = mean(prior.burnout_sensitivity_dist),
        σ_burnout     = std(prior.burnout_sensitivity_dist),
        x̂_engagement  = mean(prior.engagement_decay_dist),
        σ_engagement  = std(prior.engagement_decay_dist),
        x̂_burden      = 0.0,
        σ_burden       = 0.1,
        entropy       = 0.0,   # computed after Σ is assembled
        obs_log_lik   = 0.0
    )
end

function initialize_belief(
    prior     :: TwinPrior,
    estimator :: ParticleBeliefEstimator;
    N         :: Int = 100
) :: ParticleBeliefState
    initialize_particles(prior, N)
end

function initialize_belief(
    prior     :: TwinPrior,
    estimator :: JEPABeliefEstimator
) :: JEPABeliefState
    # JEPA belief starts as high-entropy Gaussian in latent space
    # weights uninitialized until training begins
    z_dim = 64
    JEPABeliefState(
        μ           = zeros(Float32, z_dim),
        log_σ       = zeros(Float32, z_dim),   # σ = exp(0) = 1 — maximum uncertainty
        entropy     = 0.0f0,
        obs_log_lik = 0.0f0
    )
end

# ─────────────────────────────────────────────────────────────────
# Main entry point — update belief from new observation
# Routes to correct estimator via multiple dispatch
# Also runs anomaly detection and widens belief if needed
# ─────────────────────────────────────────────────────────────────

function update_belief(
    belief      :: AbstractBeliefState,
    observation :: Observation,
    action      :: AbstractAction,
    twin        :: DigitalTwin,
    estimator   :: AbstractBeliefEstimator,
    mem         :: MemoryBuffer,
    params,                                  # estimator-specific params
    config      :: ConfiguratorState
) :: AbstractBeliefState

    # 1. predict — move belief forward before seeing observation
    b_pred = predict_belief(estimator, belief, action, twin, params)

    # 2. update — incorporate new observation
    b_updated = update_belief_step(estimator, b_pred, observation, params)

    # 3. anomaly detection
    anomaly = detect_anomaly(
        b_updated.obs_log_lik,
        mem,
        observation,
        config.φ_perc.δ_anomaly
    )

    # 4. widen belief if anomaly detected
    b_final = widen_belief(b_updated, anomaly)

    return b_final
end

# ─────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────

export
    # initialization
    initialize_belief,

    # main update cycle
    update_belief,

    # estimator params
    initialize_kalman,
    initialize_particles,
    HierarchicalJEPAEncoder,
    JEPAInferenceParams,
    AbstractTrainingDataset,
    SQLiteTrainingDataset,
    SyntheticTrainingDataset,
    encode_observation_window,
    get_training_batch,
    n_samples,
    train_encoder!,
    save_jepa_weights,
    load_jepa_weights!,
    LAST_JEPA_TRAINING_LOSS,

    # epistemic
    compute_epistemic_state,

    # anomaly
    detect_anomaly,
    widen_belief,
    AnomalyResult

end # module Perception
