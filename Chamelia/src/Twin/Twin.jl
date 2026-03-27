""" 
Twin.jl
Digital Twin Module -- Model of the patient 
𝒯_i(t) = (θ_prior, θ_post(t), x_t, p(ξ))
"""

include("../types.jl")

module Twin

using Main: TwinPrior, TwinPosterior, RolloutNoise, DigitalTwin,
            MemoryRecord, Accept, initialize_noise,
            register_physical_noise!, sample_noise

include("prior.jl")
include("posterior.jl")
include("noise.jl")

end
