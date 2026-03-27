include("../types.jl")

module WorldModule

using Distributions
using Statistics
using Flux

# import all types from Main (defined by types.jl above)
using Main: AbstractSimulator, AbstractAction, AbstractBeliefState,
            PatientState, PhysState, PsyState, TwinPrior, TwinPosterior,
            DigitalTwin, RolloutNoise, RolloutResult, LatentRolloutResult, Observation,
            GaussianBeliefState, ParticleBeliefState, JEPABeliefState,
            ConfiguratorState, CostWeights, UserResponse, ScalarTrust, ScalarBurnout,
            ScalarEngagement, ScalarBurden, NullAction,
            Accept, Reject, Partial,
            sample_noise, is_null, magnitude, compute_intrinsic_cost

include("simulator.jl")
include("behavioral.jl")
include("rollout.jl")
include("jepa_predictor.jl")

export
    sim_step!,
    sim_observe,
    register_priors!,
    register_noise!,
    action_dimensions,
    safety_thresholds,
    min_clinical_delta,
    compute_frustration,
    MockSimulator,
    run_rollouts,
    run_paired_rollouts,
    update_psy_state,
    update_trust,
    update_burnout,
    update_engagement,
    update_burden,
    JEPAPredictor,
    JEPA_PREDICTOR,
    jepa_rollout,
    jepa_rollout_all_horizons,
    run_latent_rollouts,
    run_paired_latent_rollouts,
    action_to_features

end # module WorldModule
