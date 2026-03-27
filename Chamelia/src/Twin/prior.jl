"""
prior.jl
TwinPrior initialization from persona.
Fixed at patient creation — never updated.
"""

#given a persona label, initialize a TwinPrior with appropriate distributions

using Distributions

# -------------------------------------------------------------------
# Initialize a TwinPrior from distributions provided by the simulator.
# The simulator is responsible for deriving the persona and its distributions.
# -------------------------------------------------------------------
function initialize_prior(
    persona_label            :: String,
    trust_growth_dist        :: Distribution,
    trust_decay_dist         :: Distribution,
    burnout_sensitivity_dist :: Distribution,
    engagement_decay_dist    :: Distribution,
    physical_priors          :: Dict{Symbol, Distribution} = Dict{Symbol, Distribution}()
) :: TwinPrior

    TwinPrior(
        trust_growth_dist        = trust_growth_dist,
        trust_decay_dist         = trust_decay_dist,
        burnout_sensitivity_dist = burnout_sensitivity_dist,
        engagement_decay_dist    = engagement_decay_dist,
        physical_priors          = physical_priors,
        persona_label            = persona_label
    )
end

# -------------------------------------------------------------------
# Allow simulator plugins to register physical priors after creation
# e.g. t1d_sim calls register_physical_prior!(prior, :isf_multiplier, Normal(1.0, 0.2))
# -------------------------------------------------------------------

function register_physical_prior!( ### !() Modifies in place
    prior :: TwinPrior,
    label :: Symbol,
    dist  :: Distribution
)
    prior.physical_priors[label] = dist
end


