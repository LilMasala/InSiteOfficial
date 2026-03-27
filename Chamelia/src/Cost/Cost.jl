"""
Cost.jl
Cost module — normative cost and energy computation.

Two layers per the formulation:
  Layer 1: Intrinsic cost   — hardwired, auditable, no epistemic terms
  Layer 2: Critic           — learned terminal value estimator
  Layer 3: Energy           — combines both for Actor evaluation

Epistemic constraints live in Perception — NOT here.
This module only answers: "how bad is this state of affairs?"
"""

include("../types.jl")

module Cost

using Statistics
using LinearAlgebra
using Flux

using Main: AbstractSimulator, AbstractAction, AbstractBeliefState,
            PatientState, PhysState, PsyState,
            TwinPrior, TwinPosterior, DigitalTwin,
            RolloutNoise, RolloutResult, LatentRolloutResult, Observation,
            GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            EpistemicState, EpistemicThresholds,
            MemoryBuffer, MemoryRecord, ConfiguratorState,
            CostWeights, UserResponse,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            NullAction, Accept, Reject, Partial,
            AbstractCriticModel, CRITIC_FEATURE_DIM,
            compute_intrinsic_cost, compute_physical_cost,
            compute_burden_cost, compute_trust_cost,
            compute_burnout_cost, compute_burnout_hazard,
            sample_noise, is_null, magnitude
            
using Main.WorldModule: MockSimulator

include("intrinsic.jl")
include("latent.jl")
include("critic.jl")
include("energy.jl")

export
    # intrinsic cost
    compute_intrinsic_cost,
    compute_physical_cost,
    compute_burden_cost,
    compute_trust_cost,
    compute_burnout_cost,
    compute_burnout_hazard,
    decode_latent_summary,
    LatentCostDecoder,
    LATENT_DECODER,
    train_decoder!,
    compute_latent_intrinsic_cost,
    compute_latent_rollout_energy,

    # critic
    ZeroCritic,
    RidgeCritic,
    MLPCritic,
    critic_value,
    train_critic!,
    compute_residual_cost,
    extract_terminal_features,

    # energy
    compute_rollout_energy,
    compute_energies,
    compute_cvar,
    compute_effect_size

end # module Cost
