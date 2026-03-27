"""
Memory.jl
Memory module — structured experience buffer.
Stores, scores, and learns from every recommendation and hold.
"""

include("../types.jl")

module Memory

using Statistics
using LinearAlgebra
using Flux

using Main: AbstractSimulator, AbstractAction, AbstractBeliefState,
            AbstractCriticModel, PatientState, PsyState,
            TwinPrior, TwinPosterior, DigitalTwin,
            MemoryBuffer, MemoryRecord, ConfiguratorState,
            EpistemicState, UserResponse, NullAction, JEPABeliefState,
            ScalarTrust, ScalarBurnout, ScalarEngagement, ScalarBurden,
            CRITIC_FEATURE_DIM, sample_noise

import Main: is_null, magnitude

using Main.Actor: ShadowScorecard
using Main.Cost: RidgeCritic, MLPCritic, train_critic!,
                 extract_terminal_features, ZeroCritic

include("buffer.jl")
include("critic_training.jl")
include("scorecard.jl")

export
    store_record!,
    store_outcome!,
    store_hold!,
    get_record,
    completed_records,
    recent_records,
    records_in_window,
    update_all_critic_targets!,
    maybe_update_critic!,
    current_critic,
    compute_scorecard,
    update_all_scores!,
    score_record!,
    maybe_update_twin_posterior!

end # module Memory
