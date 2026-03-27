"""
noise.jl
Rollout noise model p(ξ).
Captures irreducible randomness in the patient's life.
Aleatoric uncertainty — cannot be reduced by more data.
"""

using Main: RolloutNoise, initialize_noise, register_physical_noise!, sample_noise
