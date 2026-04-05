"""
types.jl
Signal Registry
Register signals before use
"""

#### imports 
using Distributions
using LinearAlgebra
using Statistics

if !isdefined(@__MODULE__, :AbstractSimulator)

# ─────────────────────────────────────────────────────────────────
# constants 
# ─────────────────────────────────────────────────────────────────
const CRITIC_FEATURE_DIM = 7
const META_FEATURE_DIM   = 12   # MetaState fields excluding current_day

# -------------------------------------------------------------------
# Abstract types (swappable interfaces)
# -------------------------------------------------------------------

abstract type AbstractSimulator end
abstract type AbstractBeliefEstimator end

struct KalmanBeliefEstimator <: AbstractBeliefEstimator end
struct ParticleBeliefEstimator <: AbstractBeliefEstimator end  
struct JEPABeliefEstimator <: AbstractBeliefEstimator end

abstract type AbstractActor end
abstract type AbstractCritic end
abstract type AbstractConfigurator end 

# Math required concepts -- implmenetation is swappable
abstract type AbstractTrustModel end
abstract type AbstractBurnoutModel end
abstract type AbstractEngagementModel end
abstract type AbstractBurdenModel end


#distribution?
abstract type AbstractBeliefState end

# -------------------------------------------------------------------
# Domain Adapter
# Separates InSite-domain semantics from Chamelia's generic core.
#
# Chamelia core handles: belief/twin/memory structure, actor/configurator/
# world-model separation, the generic counterfactual decision loop, and
# hard-constraint interfaces (safety, epistemic, burnout).
#
# Domain adapters own: physical state variable definitions, psychological/
# burden variable conventions, safety envelope defaults, intrinsic-cost
# feature names and weights, action-family definitions, and control horizons.
#
# A future non-InSite domain (e.g., cardiovascular rehab, AID) supplies its
# own AbstractDomainAdapter without touching Chamelia core.
# -------------------------------------------------------------------

abstract type AbstractDomainAdapter end

struct DefaultDomainAdapter <: AbstractDomainAdapter end

"""
    default_physical_weights(adapter, prefs) :: Dict{Symbol, Float64}

Return domain-specific physical cost weight names and their initial values
given the user's preference profile.
This must NOT be hardcoded in Chamelia core — it belongs to the adapter.
"""
function default_physical_weights(adapter :: AbstractDomainAdapter, prefs) :: Dict{Symbol, Float64}
    error("$(typeof(adapter)) must implement default_physical_weights")
end

"""
    domain_name(adapter) :: String

Human-readable identifier for logging and diagnostics.
"""
function domain_name(adapter :: AbstractDomainAdapter) :: String
    error("$(typeof(adapter)) must implement domain_name")
end

function default_physical_weights(:: DefaultDomainAdapter, prefs) :: Dict{Symbol, Float64}
    _ = prefs
    return Dict{Symbol, Float64}()
end

function domain_name(:: DefaultDomainAdapter) :: String
    return "default"
end

#Actions
abstract type AbstractAction end

struct BridgeDecodedCandidate
    bridge_candidate_idx :: Int
    bridge_candidate_slot :: Int
    action :: AbstractAction
    decode_metadata :: Union{Dict{String, Any}, Nothing}
end

struct BridgeDecodedAction
    action :: AbstractAction
    metadata :: Union{Dict{String, Any}, Nothing}
end

Base.@kwdef struct BridgeProposalAdvisory
    bridge_candidate_idx :: Int
    bridge_candidate_slot :: Int
    python_ic :: Union{Float64, Nothing} = nothing
    python_tc :: Union{Float64, Nothing} = nothing
    python_total :: Union{Float64, Nothing} = nothing
    python_rank :: Union{Int, Nothing} = nothing
end

#abstract critic interfaces
abstract type AbstractCriticModel end

abstract type AbstractConfiguratorModel end

struct RuleBasedConfigurator    <: AbstractConfiguratorModel end
struct LearnedRuleConfigurator  <: AbstractConfiguratorModel end
struct MetaPolicyConfigurator   <: AbstractConfiguratorModel end

#actor searching
abstract type AbstractSearchStrategy end

struct GridSearch      <: AbstractSearchStrategy end
struct BeamSearch      <: AbstractSearchStrategy end  
struct GradientSearch  <: AbstractSearchStrategy end
struct OfflineRLPolicy <: AbstractSearchStrategy end


# -------------------------------------------------------------------
#Signal Registry
# -------------------------------------------------------------------

struct SignalMeta
    resolution_hours :: Float64
    dtype :: Type
    required :: Bool
end


mutable struct SignalRegistry
    signals :: Dict{Symbol, SignalMeta}
end

SignalRegistry() = SignalRegistry(Dict{Symbol, SignalMeta}())

function register_signal!(registry:: SignalRegistry, label:: Symbol, resolution_hours::Float64, dtype::Type,required::Bool=false)
    registry.signals[label] = SignalMeta(resolution_hours,dtype,required)
end

# -------------------------------------------------------------------
# section 2A: Core latent concepts 
# hardwired into chamelias cost/actor/perception math 
# their definitions can evolve 
# -------------------------------------------------------------------

struct ScalarTrust <: AbstractTrustModel
    value :: Float64
end 

struct ScalarBurnout <: AbstractBurnoutModel
    value :: Float64
end 

struct ScalarEngagement <: AbstractEngagementModel
    value :: Float64
end 

struct ScalarBurden <: AbstractBurdenModel
    value :: Float64
end 


Base.@kwdef struct BurnoutAttribution
    Δ_hat                :: Float64
    P_treated            :: Float64
    P_baseline           :: Float64
    se_paired            :: Float64
    ci_lower             :: Float64
    upper_ci             :: Float64
    n_pairs              :: Int
    horizon              :: Int
    horizon_sensitivity  :: Vector{NamedTuple{(:H, :Δ), Tuple{Int, Float64}}} =
        NamedTuple{(:H, :Δ), Tuple{Int, Float64}}[]
end

# -------------------------------------------------------------------
# section 2A: True Latent State
# This is never directly observed. Inferred by perception, generated by simulator
# Split into math-required psychological state + simulator-specific physical state
# -------------------------------------------------------------------

Base.@kwdef struct PsyState
    trust :: AbstractTrustModel
    burden :: AbstractBurdenModel
    engagement :: AbstractEngagementModel
    burnout :: AbstractBurnoutModel
end 

Base.@kwdef struct PhysState
    variables :: Dict{Symbol, Float64}
end 

Base.@kwdef struct PatientState
    phys :: PhysState
    psy :: PsyState
end 


# -------------------------------------------------------------------
#section 2A
#Observation (o_t)
#timestamp + whatever avail signals 
# -------------------------------------------------------------------

Base.@kwdef struct Observation
    timestamp :: Float64
    signals :: Dict{Symbol, Any}
end

function _bridge_json_value(value::Nothing)
    return nothing
end

function _bridge_json_value(value::Bool)
    return value
end

function _bridge_json_value(value::Number)
    return value
end

function _bridge_json_value(value::AbstractString)
    return String(value)
end

function _bridge_json_value(value::Symbol)
    return String(value)
end

function _bridge_json_value(value::AbstractVector)
    return [_bridge_json_value(item) for item in value]
end

function _bridge_json_value(value::Dict)
    return Dict(String(key) => _bridge_json_value(item) for (key, item) in value)
end

function _bridge_json_value(value)
    return string(value)
end

"""
    bridge_domain_name(adapter) :: String

Return the domain name carried over the Julia ↔ Python bridge.
The default bridge name matches the adapter's domain name.
"""
function bridge_domain_name(adapter::AbstractDomainAdapter) :: String
    return domain_name(adapter)
end

"""
    bridge_domain_state(adapter, obs) :: Dict{String,Any}

Build the plugin-owned opaque domain-state payload sent to the Python bridge.
Chamelia core does not interpret the returned fields.
"""
function bridge_domain_state(
    adapter::AbstractDomainAdapter,
    obs::Observation,
) :: Dict{String, Any}
    _ = adapter
    return Dict{String, Any}(
        "timestamp" => obs.timestamp,
        "signals" => Dict(String(key) => _bridge_json_value(value) for (key, value) in obs.signals),
    )
end

"""
    bridge_encode_payload(adapter, obs) :: Dict{String,Any}

Build the encode payload for the Python bridge.
Adapters own how raw observations are surfaced to the bridge. The default path
ships a plugin observation payload and lets the Python-side plugin tokenize it.
"""
function bridge_encode_payload(
    adapter::AbstractDomainAdapter,
    obs::Observation,
) :: Dict{String, Any}
    return Dict{String, Any}(
        "input_kind" => "plugin_observation",
        "observation" => bridge_domain_state(adapter, obs),
    )
end

# -------------------------------------------------------------------
#section 2A -- Actions
#what the system proposes. Domain-specific implmenetation
#Chamelia's math only needs magnitudes and nullpness
# -------------------------------------------------------------------

function is_null(a::AbstractAction) :: Bool
    error("$(typeof(a)) must implement is_null")
end

function magnitude(a::AbstractAction) :: Float64
    error("$(typeof(a)) must implement magnitude")
end

struct NullAction <: AbstractAction end

is_null(::NullAction) = true
magnitude(::NullAction) = 0.0

@enum UserResponse Reject Partial Accept

const ActionLevel = Int

@enum ActionFamily begin
    parameter_adjustment
    structure_edit
    continuous_schedule
end

Base.@kwdef struct ConnectedAppCapabilities
    app_id::String = "unknown"
    supports_scalar_schedule::Bool = true
    supports_piecewise_schedule::Bool = false
    supports_continuous_schedule::Bool = false
    max_segments::Int = 1
    min_segment_duration_min::Int = 0
    max_segments_addable::Int = 0
    level_1_enabled::Bool = true
    level_2_enabled::Bool = false
    level_3_enabled::Bool = false
    structural_change_requires_consent::Bool = true
end

Base.@kwdef struct SegmentSurface
    segment_id::String
    start_min::Int
    end_min::Int
    parameter_values::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
end

const ProfileSummary = NamedTuple{(:id, :name, :segment_count), Tuple{String, String, Int}}

Base.@kwdef struct ConnectedAppState
    schedule_version::String = ""
    current_segments::Vector{SegmentSurface} = SegmentSurface[]
    allow_structural_recommendations::Bool = false
    allow_continuous_schedule::Bool = false
    # Profile context — which profile is currently active, and what other profiles exist.
    # Chamelia uses these to eventually target patch-current vs patch-existing vs create-new.
    active_profile_id::Union{String, Nothing} = nothing
    available_profiles::Vector{ProfileSummary} = ProfileSummary[]
end

Base.@kwdef struct SegmentDelta
    segment_id::String
    parameter_deltas::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
end

Base.@kwdef struct StructureEdit
    edit_type::Symbol
    target_segment_id::String
    split_at_minute::Union{Int, Nothing} = nothing
    neighbor_segment_id::Union{String, Nothing} = nothing
end

struct ScheduledAction <: AbstractAction
    level::ActionLevel
    family::ActionFamily
    segments::Vector{SegmentSurface}
    segment_deltas::Vector{SegmentDelta}
    structural_edits::Vector{StructureEdit}
end

ScheduledAction(
    level::ActionLevel,
    family::ActionFamily
) = ScheduledAction(level, family, SegmentSurface[], SegmentDelta[], StructureEdit[])

ScheduledAction(
    level::ActionLevel,
    family::ActionFamily,
    segment_deltas::Vector{SegmentDelta},
    structural_edits::Vector{StructureEdit}
) = ScheduledAction(level, family, SegmentSurface[], segment_deltas, structural_edits)

function is_null(a::ScheduledAction) :: Bool
    no_segment_change = all(delta ->
        all(abs(value) < 1e-8 for value in values(delta.parameter_deltas)),
        a.segment_deltas
    )
    return no_segment_change && isempty(a.structural_edits)
end

function magnitude(a::ScheduledAction) :: Float64
    n_components = 0
    total = 0.0
    for delta in a.segment_deltas
        total += sum(abs(value) for value in values(delta.parameter_deltas))
        n_components += length(delta.parameter_deltas)
    end
    total += length(a.structural_edits)
    n_components += length(a.structural_edits)
    return n_components == 0 ? 0.0 : total / n_components
end

"""
    bridge_decode_action_path(adapter, action_path, capabilities, app_state) -> Union{AbstractAction, Nothing}

Plugin-owned decoding of a bridge candidate path into a Julia action.
Core Chamelia does not interpret the path semantics itself.
The default implementation returns `nothing`, which tells Julia to ignore the
bridge path for action-selection purposes and fall back to legacy search.
"""
function bridge_decode_action_path(
    adapter::AbstractDomainAdapter,
    action_path,
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Union{AbstractAction, Nothing}
    _ = adapter
    _ = action_path
    _ = capabilities
    _ = app_state
    return nothing
end

"""
    bridge_decode_action_path_result(adapter, action_path, capabilities, app_state)

Plugin-owned bridge decode that can attach structured metadata explaining how a
bridge path became a Julia-native action. The default implementation preserves
the legacy `bridge_decode_action_path(...)` hook and wraps its result.
"""
function bridge_decode_action_path_result(
    adapter::AbstractDomainAdapter,
    action_path,
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Union{BridgeDecodedAction, Nothing}
    action = bridge_decode_action_path(adapter, action_path, capabilities, app_state)
    isnothing(action) && return nothing
    return BridgeDecodedAction(action, Dict{String, Any}(
        "decoder" => "default_bridge_decoder",
        "returned_action_kind" => string(nameof(typeof(action))),
    ))
end

"""
    bridge_decode_candidate_proposals(adapter, proposal_bundle, capabilities, app_state)

Decode the non-baseline bridge candidates into Julia actions together with
their original proposal indices. Candidate slot 1 is reserved for the explicit
bridge baseline and is skipped because Julia owns the null-baseline comparison.
"""
function bridge_decode_candidate_proposals(
    adapter::AbstractDomainAdapter,
    proposal_bundle::Dict{String, Any},
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Vector{BridgeDecodedCandidate}
    candidate_paths = get(proposal_bundle, "candidate_paths", nothing)
    candidate_paths isa AbstractVector || return BridgeDecodedCandidate[]

    decoded = BridgeDecodedCandidate[]
    for (idx, action_path) in pairs(candidate_paths)
        idx == 1 && continue
        decoded_action = bridge_decode_action_path_result(adapter, action_path, capabilities, app_state)
        isnothing(decoded_action) && continue
        push!(decoded, BridgeDecodedCandidate(idx - 1, idx, decoded_action.action, decoded_action.metadata))
    end
    return decoded
end

"""
    bridge_decode_candidate_actions(adapter, proposal_bundle, capabilities, app_state)

Decode the non-baseline bridge candidates into Julia actions.
Candidate slot 1 is reserved for the explicit bridge baseline and is skipped
because Julia owns the null-baseline comparison.
"""
function bridge_decode_candidate_actions(
    adapter::AbstractDomainAdapter,
    proposal_bundle::Dict{String, Any},
    capabilities::ConnectedAppCapabilities,
    app_state::ConnectedAppState,
) :: Vector{AbstractAction}
    return AbstractAction[decoded.action for decoded in bridge_decode_candidate_proposals(adapter, proposal_bundle, capabilities, app_state)]
end

function bridge_action_summary(action::AbstractAction) :: Dict{String, Any}
    return Dict("kind" => string(nameof(typeof(action))))
end

function bridge_action_summary(::NullAction) :: Dict{String, Any}
    return Dict("kind" => "null", "deltas" => Dict{String, Float64}())
end

function bridge_action_summary(action::ScheduledAction) :: Dict{String, Any}
    return Dict(
        "kind" => "scheduled",
        "level" => action.level,
        "family" => string(action.family),
        "segment_deltas" => [
            Dict(
                "segment_id" => delta.segment_id,
                "parameter_deltas" => Dict(String(key) => value for (key, value) in delta.parameter_deltas),
            )
            for delta in action.segment_deltas
        ],
        "structural_edits" => [
            Dict(
                "edit_type" => String(edit.edit_type),
                "target_segment_id" => edit.target_segment_id,
                "split_at_minute" => edit.split_at_minute,
                "neighbor_segment_id" => edit.neighbor_segment_id,
            )
            for edit in action.structural_edits
        ]
    )
end


# -------------------------------------------------------------------
# Section 2B — Normative Cost Types
# Hardwired. No abstract types. Must be auditable by clinicians.
# -------------------------------------------------------------------


Base.@kwdef struct CostWeights
    # Always present — Chamelia's math requires these
    c_burden  :: Float64
    c_trust   :: Float64
    c_burnout :: Float64
    γ_β       :: Float64    # burden decay

    # Domain-specific physical outcome weights — registered by data source
    physical  :: Dict{Symbol, Float64}   # e.g :w_low => 5.0, :w_high => 1.0
end

# -------------------------------------------------------------------
# Explicit intrinsic cost helpers.
# These are shared by WorldModule rollouts and Cost energy evaluation.
# -------------------------------------------------------------------

function compute_physical_cost(
    sim     :: AbstractSimulator,
    signals :: Dict{Symbol, Any},
    weights :: Dict{Symbol, Float64}
) :: Float64
    error("$(typeof(sim)) must implement compute_physical_cost!")
end

function compute_burden_cost(
    action  :: AbstractAction,
    β       :: Float64,
    weights :: CostWeights
) :: Float64
    rec_cost = is_null(action) ? 0.0 : 1.0
    mag_cost = magnitude(action)
    return weights.c_burden * (rec_cost + mag_cost + β)
end

function compute_trust_cost(
    τ_prev    :: Float64,
    τ_current :: Float64,
    weights   :: CostWeights
) :: Float64
    return weights.c_trust * max(0.0, τ_prev - τ_current)
end

function compute_burnout_hazard(
    B           :: Float64,
    β           :: Float64,
    τ           :: Float64,
    ω           :: Float64,
    frustration :: Float64
) :: Float64
    λ0 = 0.01
    log_hazard =
        2.0 * B +
        0.5 * β +
        1.0 * (1.0 - τ) +
        1.0 * (1.0 - ω) +
        1.0 * frustration
    return λ0 * exp(log_hazard)
end

function compute_burnout_cost(
    B           :: Float64,
    β           :: Float64,
    τ           :: Float64,
    ω           :: Float64,
    frustration :: Float64,
    weights     :: CostWeights
) :: Float64
    λ_B = compute_burnout_hazard(B, β, τ, ω, frustration)
    return weights.c_burnout * (B + λ_B)
end

function compute_intrinsic_cost(
    action        :: AbstractAction,
    psy           :: PsyState,
    signals       :: Dict{Symbol, Any},
    τ_prev        :: Float64,
    frustration   :: Float64,
    weights       :: CostWeights,
    sim           :: AbstractSimulator
) :: Float64
    τ = psy.trust.value
    B = psy.burnout.value
    ω = psy.engagement.value
    β = psy.burden.value

    C_physical = compute_physical_cost(sim, signals, weights.physical)
    C_burden   = compute_burden_cost(action, β, weights)
    C_trust    = compute_trust_cost(τ_prev, τ, weights)
    C_burnout  = compute_burnout_cost(B, β, τ, ω, frustration, weights)

    return C_physical + C_burden + C_trust + C_burnout
end

# -------------------------------------------------------------------
# section 2C: Belief State
# Central Mathematical Object, Distribution over Patient State
# POC approximation: factored Gaussian over phys and psy substates.
# All downstream modules operate on this — never on the true state.
# -------------------------------------------------------------------


Base.@kwdef struct GaussianBeliefState <: AbstractBeliefState
    x̂_phys      :: Dict{Symbol, Float64}
    Σ_phys       :: Dict{Symbol, Float64}
    x̂_trust      :: Float64
    σ_trust       :: Float64
    x̂_burnout     :: Float64
    σ_burnout     :: Float64
    x̂_engagement  :: Float64
    σ_engagement  :: Float64
    x̂_burden      :: Float64
    σ_burden       :: Float64
    entropy       :: Float64
    obs_log_lik   :: Float64
end

Base.@kwdef struct ParticleBeliefState <: AbstractBeliefState
    particles   :: Vector{PatientState}
    weights     :: Vector{Float64}
    entropy     :: Float64
    obs_log_lik :: Float64
end

Base.@kwdef struct JEPABeliefState <: AbstractBeliefState
    μ           :: AbstractArray
    log_σ       :: AbstractArray
    entropy     :: Float32
    obs_log_lik :: Float32
end

# -------------------------------------------------------------------
# Section 2C — Epistemic State
# What the system doesn't know well.
# Hardwired as constraints — never traded off against cost.
# -------------------------------------------------------------------

Base.@kwdef struct EpistemicThresholds
    κ_min :: Float64   # GP familiarity minimum (default 0.6)
    ρ_min :: Float64   # ensemble concordance minimum (default 0.5)
    η_min :: Float64   # calibration quality minimum (default 0.7)
end

Base.@kwdef struct EpistemicState
    κ_familiarity  :: Float64   # ∈ [0,1]
    ρ_concordance  :: Float64   # ∈ [0,1]
    η_calibration  :: Float64   # ∈ [0,1]
    feasible       :: Bool      # F_t = 𝟙[all ≥ thresholds]
end


# -------------------------------------------------------------------
# Rollout Noise — p(ξ)
# Defined here because both Perception and WorldModule need it
# -------------------------------------------------------------------

mutable struct RolloutNoise
    trust_noise      :: Distribution
    burnout_noise    :: Distribution
    engagement_noise :: Distribution
    physical_noise   :: Dict{Symbol, Distribution}
end

function initialize_noise(
    trust_noise_std      :: Float64 = 0.02,
    burnout_noise_std    :: Float64 = 0.02,
    engagement_noise_std :: Float64 = 0.02
) :: RolloutNoise
    RolloutNoise(
        Normal(0.0, trust_noise_std),
        Normal(0.0, burnout_noise_std),
        Normal(0.0, engagement_noise_std),
        Dict{Symbol, Distribution}()
    )
end

function register_physical_noise!(
    noise :: RolloutNoise,
    label :: Symbol,
    dist  :: Distribution
)
    noise.physical_noise[label] = dist
end

function sample_noise(noise::RolloutNoise) :: Dict{Symbol, Float64}
    ξ = Dict{Symbol, Float64}(
        :trust      => rand(noise.trust_noise),
        :burnout    => rand(noise.burnout_noise),
        :engagement => rand(noise.engagement_noise)
    )

    for (label, dist) in noise.physical_noise
        ξ[label] = rand(dist)
    end

    return ξ
end

Base.@kwdef struct RolloutResult
  action         :: AbstractAction
  initial_psy    :: PsyState
  terminal_state :: PatientState
  terminal_psy   :: PsyState
  total_cost     :: Float64
  psy_trajectory :: Vector{PsyState}
  phys_signals   :: Vector{Dict{Symbol,Any}}
end

Base.@kwdef struct LatentRolloutResult
    action         :: AbstractAction
    initial_belief :: JEPABeliefState
    short_belief   :: JEPABeliefState
    med_belief     :: JEPABeliefState
    long_belief    :: JEPABeliefState
end


# -------------------------------------------------------------------
# Twin Types
# 𝒯_i(t) = (θ_prior, θ_post(t), x_t, p(ξ))
# θ_prior — fixed at patient creation, never updated
# uses Distributions.jl so parameters have actual prior distributions
# -------------------------------------------------------------------

Base.@kwdef mutable struct TwinPrior
    trust_growth_dist :: Distribution
    trust_decay_dist :: Distribution
    burnout_sensitivity_dist :: Distribution
    engagement_decay_dist :: Distribution

    #domain specific: physical priors

    physical_priors :: Dict{Symbol, Distribution}

    persona_label :: String
end 

# θ_post(t) — updated from data, starts at prior

Base.@kwdef mutable struct TwinPosterior
    #math required psychological estimates
    trust_growth_rate :: Float64
    trust_decay_rate :: Float64
    burnout_sensitivity :: Float64
    engagement_decay :: Float64

    physical :: Dict{Symbol, Float64}

    #when did the last update happen??? 
    last_updated_day :: Int
    n_observations :: Int
end


#The Twin
struct DigitalTwin
    prior :: TwinPrior 
    posterior :: TwinPosterior
    rollout_noise_std :: Float64
end 

# -------------------------------------------------------------------
# Configurator Types — φ_t
# Meta-policy output that parameterizes all other modules.
# Rule-based in v1.1 POC, learnable later.
# -------------------------------------------------------------------

Base.@kwdef mutable struct PercConfig
    signal_mask :: Dict{Symbol, Bool} #which signals are active
    H_perc :: Int
    δ_anomaly     :: Float64              # anomaly sensitivity threshold
end

Base.@kwdef mutable struct WorldConfig
    H_short :: Int #short rollout hours
    H_med :: Int #medium rollout days
    N_roll :: Int #number of stochastic rollouts, number of future simulations after an action
end 


Base.@kwdef mutable struct CostConfig 
    weights :: CostWeights
    thresholds :: EpistemicThresholds
    H_burn :: Int #burnout attribution horizon (14-60 days)
    ε_burn     :: Float64   # max tolerable policy-attributable burnout risk
    γ_discount :: Float64   # MDP discount factor
end 

Base.@kwdef mutable struct ActConfig
    Δ_max        :: Float64             # max relative action deviation
    δ_min_effect :: Float64             # minimum effect size
    α_cvar       :: Float64             # CVaR tail probability, given n futures, how do you pick the best outcomes
                                        # if 50 rollouts, alpha cvar of 0.8 means discard the best 80% of rollouts
                                        #focus on only the actions that cause the best worst case (hypoglycemias?)
    N_search     :: Int                 # search budget
end

Base.@kwdef mutable struct ConfiguratorState
    φ_perc  :: PercConfig
    φ_world :: WorldConfig
    φ_cost  :: CostConfig
    φ_act   :: ActConfig
    last_update_day :: Int
end

Base.@kwdef struct UserPreferences
    aggressiveness     :: Float64 = 0.5
    hypoglycemia_fear  :: Float64 = 0.7
    burden_sensitivity :: Float64 = 0.5
    persona            :: String  = "default"
    physical_priors    :: Dict{String, Vector{Float64}} = Dict{String, Vector{Float64}}()
    calibration_targets :: Dict{String, Float64} = Dict{String, Float64}()
    minimum_action_delta_thresholds :: Dict{String, Float64} = Dict{String, Float64}()
end

# -------------------------------------------------------------------
# Memory Types — m_τ
# One experience record. Three phases of information.
# -------------------------------------------------------------------


Base.@kwdef mutable struct MemoryRecord
    id :: Int
    day :: Int

    # reccomendation time (immutable once written)
    belief_entropy :: Float64 #how uncertain were we wehn we made this rec
    action  :: AbstractAction #what did we actually reccomend 
    epistemic  :: EpistemicState #were k, p, n all above threshold
    config_snapshot :: ConfiguratorState #what were the configruator settings at this point

    # outcome time (filled in delta days later)
    user_response :: Union{UserResponse,Nothing} #did they accept, reject, or partially accept??
    realized_signals :: Union{Dict{Symbol, Any}, Nothing} #what did the sensors actually show afterwards
    realized_cost :: Union{Float64, Nothing} #how bad were the actual outcomes 
    predicted_cvar :: Union{Float64, Nothing} = nothing

    # retrospective 
    critic_target :: Union{Float64, Nothing} #realized residual cost used to train the critic
    shadow_delta_score :: Union{Float64, Nothing} #did this reccomendation 

    # State snapshots at reccomendation time
    trust_at_rec :: Float64 #just these numbers at the reccomendation time
    burnout_at_rec :: Float64 # 
    engagement_at_rec :: Float64
    burden_at_rec :: Float64
    latent_snapshot :: Union{Vector{Float32}, Nothing} = nothing
    latent_μ_at_rec :: Union{Vector{Float32}, Nothing} = nothing
    latent_log_σ_at_rec :: Union{Vector{Float32}, Nothing} = nothing
    configurator_mode   :: Symbol = :rules

    # Python bridge artifacts (transport-stable, per-patient durable truth).
    # These are persisted in Julia-owned state so bridge restarts do not erase
    # the decision-time reasoning trace for a patient.
    bridge_trace :: Union{Dict{String, Any}, Nothing} = nothing
    bridge_diagnostics :: Union{Dict{String, Any}, Nothing} = nothing
    bridge_outcome :: Union{Dict{String, Any}, Nothing} = nothing

    # Outcome-time latent snapshot (filled in by record_outcome!).
    # Together with latent_μ_at_rec and action, this forms the (z_t, a_eff, z_{t+H})
    # triple used to train the JEPA predictor on real shadow-period transitions.
    # Nothing when the belief is non-JEPA or before the outcome is recorded.
    latent_μ_at_outcome :: Union{Vector{Float32}, Nothing} = nothing
end

mutable struct MemoryBuffer
    records :: Vector{MemoryRecord}
    H_mem :: Int
    next_id :: Int
    critic :: Union{AbstractCriticModel, Nothing}
end


MemoryBuffer() = MemoryBuffer(MemoryRecord[], 60, 1, nothing)
MemoryBuffer(records::Vector{MemoryRecord}, H_mem::Int, next_id::Int) =
    MemoryBuffer(records, H_mem, next_id, nothing)


# -------------------------------------------------------------------
# RegimeDetectionResult
# Output of domain adapter's regime detection step.
# Defined here (after ConnectedAppState and MemoryBuffer) so the
# detect_regime default can reference all three types.
#
# Chamelia core uses scope/target_profile_id directly when building
# RecommendationPackage — it never interprets the regime_label string.
# -------------------------------------------------------------------

struct RegimeDetectionResult
    regime_label      :: Union{String, Nothing}   # e.g. "weekend", "menstrual_phase"; nil = no regime
    scope             :: String                    # "patch_current" | "patch_existing" | "create_new"
    target_profile_id :: Union{String, Nothing}   # profile id when scope ≠ "patch_current"
end

"""
    detect_regime(adapter, signals, app_state, memory) → RegimeDetectionResult

Detect whether the current observation context corresponds to a distinct
recurring regime that warrants a profile-targeted recommendation.

Domain adapter owns this entirely — Chamelia core never interprets the
regime label or the signal names used to detect it.

Default: no regime detected, patch_current scope.
"""
function detect_regime(
    adapter   :: AbstractDomainAdapter,
    signals   :: Dict{Symbol, Any},
    app_state :: ConnectedAppState,
    memory    :: MemoryBuffer
) :: RegimeDetectionResult
    return RegimeDetectionResult(nothing, "patch_current", nothing)
end

"""
    calibrate_posterior!(adapter, posterior, prior, targets) :: Nothing

Domain adapter hook called at patient initialization when the user has supplied
self-reported glycemic calibration targets (e.g. recent TIR / %low / %high).
The default is a no-op — adapters override to narrow the physical posterior
using importance sampling against a domain-specific outcome heuristic.

`targets` keys are defined by the domain adapter (e.g. "recent_tir", "recent_pct_low",
"recent_pct_high" for InSite/T1D). Chamelia core passes them blindly.
"""
function calibrate_posterior!(
    adapter   :: AbstractDomainAdapter,
    posterior :: TwinPosterior,
    prior     :: TwinPrior,
    targets   :: Dict{String, Float64}
) :: Nothing
    return nothing
end

"""
    minimum_action_delta_threshold(adapter, prefs, dimension) :: Float64

Return the user-specific minimum worthwhile change threshold for an opaque
action-delta label. Chamelia core never interprets the label's domain meaning.
"""
function minimum_action_delta_threshold(
    adapter   :: AbstractDomainAdapter,
    prefs     :: UserPreferences,
    dimension :: Symbol
) :: Float64
    return max(0.0, get(prefs.minimum_action_delta_thresholds, String(dimension), 0.0))
end


# -------------------------------------------------------------------
# Output Types
# What Chamelia returns to the outside world.
# -------------------------------------------------------------------

Base.@kwdef struct RecommendationPackage
    action                :: AbstractAction
    predicted_improvement :: Float64         # CVaR(a^0) - CVaR(a*)
    confidence            :: Float64         # composite of κ, ρ, η, δ_eff
    confidence_breakdown  :: Union{
        NamedTuple{
            (:familiarity, :concordance, :calibration, :effect_support, :selection_penalty, :final_confidence),
            NTuple{6, Float64}
        },
        Nothing
    } = nothing
    alternatives          :: Vector{AbstractAction}  # top-2 other candidates
    effect_size           :: Float64         # δ_eff
    cvar_value            :: Float64         # CVaR of chosen action
    burnout_attribution   :: Union{BurnoutAttribution, Nothing}
    predicted_outcomes    :: Union{
        NamedTuple{
            (:baseline_tir, :treated_tir, :delta_tir,
             :baseline_pct_low, :treated_pct_low, :delta_pct_low,
             :baseline_pct_high, :treated_pct_high, :delta_pct_high,
             :baseline_bg_avg, :treated_bg_avg, :delta_bg_avg,
             :baseline_cost_mean, :treated_cost_mean, :delta_cost_mean,
             :baseline_cvar, :treated_cvar, :delta_cvar),
            NTuple{18, Float64}
        },
        Nothing
    } = nothing
    predicted_uncertainty :: Union{
        NamedTuple{
            (:tir_std, :pct_low_std, :pct_high_std, :bg_avg_std, :cost_std),
            NTuple{5, Float64}
        },
        Nothing
    } = nothing
    action_level          :: Int = 1
    action_family         :: Union{ActionFamily, Nothing} = nothing
    segment_summaries     :: Vector{NamedTuple{(:segment_id, :label, :parameter_summaries), Tuple{String, String, Dict{String, String}}}} =
        NamedTuple{(:segment_id, :label, :parameter_summaries), Tuple{String, String, Dict{String, String}}}[]
    structure_summaries   :: Vector{String} = String[]
    # Profile targeting — which profile this recommendation applies to.
    # "patch_current"  : edit the currently active profile (default, existing behavior)
    # "patch_existing" : edit a different existing profile by id
    # "create_new"     : propose creating a new profile from a base profile
    recommendation_scope  :: String = "patch_current"
    target_profile_id     :: Union{String, Nothing} = nothing
    # Regime context — which recurring regime triggered a non-default scope (if any).
    # Chamelia core never interprets this string; it is passed through to the app.
    detected_regime       :: Union{String, Nothing} = nothing
end


# ─────────────────────────────────────────────────────────────────
# AnomalyResult
# Output of anomaly detection for one observation
# ─────────────────────────────────────────────────────────────────

struct AnomalyResult
    is_anomaly     :: Bool
    severity       :: Float64    # [0,1] — how anomalous?
    z_score        :: Float64    # how many SDs from expected?
    flagged_signals :: Vector{Symbol}  # which signals were anomalous?
end

end
