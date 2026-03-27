"""
simulator.jl
AbstractSimulator interface — the plugin contract.
Any simulator (t1d_sim, mock, neural ODE) must implement these functions.
Chamelia's math never touches simulator internals — only this interface.
"""

import Main: compute_physical_cost

# ─────────────────────────────────────────────────────────────────
# Required interface — every simulator must implement these
# Julia will throw a clear error if a simulator forgets one
# ─────────────────────────────────────────────────────────────────

"""
    sim_step!(sim, state, action, noise) → PatientState

Advance patient state one timestep given action and noise.
This is the core dynamics function — f(x_t, a_t, ξ_t) → x_{t+1}
"""
function sim_step!(
    sim    :: AbstractSimulator,
    state  :: PatientState,
    action :: AbstractAction,
    noise  :: Dict{Symbol, Float64}
) :: PatientState
    error("$(typeof(sim)) must implement sim_step!")
end

"""
    sim_observe(sim, state) → Observation

Generate a noisy observation from the true state.
Models the observation noise ν_t — what sensors actually read.
"""
function sim_observe(
    sim   :: AbstractSimulator,
    state :: PatientState
) :: Observation
    error("$(typeof(sim)) must implement sim_observe!")
end

"""
    register_priors!(sim, prior) → nothing

Register simulator-specific physical priors into TwinPrior.
Called once at patient initialization.
e.g. t1d_sim registers :isf_multiplier => Normal(1.0, 0.2)
"""
function register_priors!(
    sim   :: AbstractSimulator,
    prior :: TwinPrior
) :: Nothing
    error("$(typeof(sim)) must implement register_priors!")
end

"""
    register_noise!(sim, noise) → nothing

Register simulator-specific physical noise into RolloutNoise.
Called once at patient initialization.
e.g. t1d_sim registers :bg_noise => Normal(0, 5.0)
"""
function register_noise!(
    sim   :: AbstractSimulator,
    noise :: RolloutNoise
) :: Nothing
    error("$(typeof(sim)) must implement register_noise!")
end

function action_dimensions(sim::AbstractSimulator) :: Vector{Symbol}
    error("$(typeof(sim)) must implement action_dimensions!")
end

function safety_thresholds(sim::AbstractSimulator) :: Dict{Symbol, Float64}
    error("$(typeof(sim)) must implement safety_thresholds!")
end

"""
    min_clinical_delta(sim) → Float64

Minimum relative parameter change (per dimension) considered clinically
meaningful for this domain. Recommendations whose largest per-dimension
delta is below this floor are suppressed — they are too small to represent
a real therapy change (pump granularity, clinical noise, user acceptance).

Domain-specific: InSite/T1D returns 0.03 (3%); future AID domain would
return a tighter value appropriate for real-time control.

Chamelia core uses this value without knowing the domain semantics.
"""
function min_clinical_delta(sim::AbstractSimulator) :: Float64
    return 0.01   # conservative default — domains should override
end

# ─────────────────────────────────────────────────────────────────
# Mock simulator — for testing without a real simulator
# Implements the interface with trivial dynamics
# Replace with t1d_sim plugin for real use
# ─────────────────────────────────────────────────────────────────

struct MockSimulator <: AbstractSimulator
    noise_std :: Float64
end

MockSimulator() = MockSimulator(0.1)

function sim_step!(
    sim    :: MockSimulator,
    state  :: PatientState,
    action :: AbstractAction,
    noise  :: Dict{Symbol, Float64}
) :: PatientState

    # trivial dynamics — physical variables drift randomly
    new_phys_vars = Dict{Symbol, Float64}(
        label => val + get(noise, label, 0.0)
        for (label, val) in state.phys.variables
    )

    # psychological state unchanged by mock simulator
    # behavioral.jl handles psy dynamics separately
    PatientState(PhysState(new_phys_vars), state.psy)
end

function sim_observe(
    sim   :: MockSimulator,
    state :: PatientState
) :: Observation
    # noisy observation of physical state
    signals = Dict{Symbol, Any}(
        label => val + randn() * sim.noise_std
        for (label, val) in state.phys.variables
    )
    Observation(timestamp=Float64(time()), signals=signals)
end

function register_priors!(sim::MockSimulator, prior::TwinPrior) :: Nothing
    # mock simulator registers no physical priors
    return nothing
end

# MockSimulator uses the default min_clinical_delta(::AbstractSimulator) = 0.01

function register_noise!(sim::MockSimulator, noise::RolloutNoise) :: Nothing
    # mock simulator registers no physical noise
    return nothing
end

action_dimensions(::MockSimulator) = [:dim1, :dim2]
safety_thresholds(::MockSimulator) = Dict{Symbol, Float64}()

# ─────────────────────────────────────────────────────────────────
# Physical Cost — domain-specific
# Simulator plugin implements this for its domain.
# InSite: glycemic cost (hypoglycemia, hyperglycemia, variability)
# Cardiac: symptom burden, arrhythmia risk, etc.
# ─────────────────────────────────────────────────────────────────

"""
    compute_physical_cost(sim, signals, weights) → Float64

Domain-specific physical outcome cost ∈ [0, ∞).
Must be implemented by simulator plugin.
Higher = worse physical outcomes.
"""

function compute_physical_cost(
    sim     :: MockSimulator,
    signals :: Dict{Symbol, Any},
    weights :: Dict{Symbol, Float64}
) :: Float64
    return 0.0
end
