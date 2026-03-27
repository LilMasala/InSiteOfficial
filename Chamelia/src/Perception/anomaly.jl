"""
anomaly.jl
Anomaly detection from observation log likelihood.
Low log p(o_t | b_{t-1}) → observation is surprising → flag anomaly.

Anomaly detection is a direct consequence of the belief-state framing:
  "is o_t consistent with the current belief?"
This is a likelihood test against b_t — not a separate heuristic.

Anomaly flags feed into:
  - Configurator: tighten constraints, increase rollout count
  - Perception: increase uncertainty (widen Σ)
  - Memory: flag record for review
"""

using Statistics


# ─────────────────────────────────────────────────────────────────
# Detect anomaly from observation log likelihood
# Compares current log likelihood against recent history.
# z_score > threshold → anomaly
# ─────────────────────────────────────────────────────────────────

function detect_anomaly(
    obs_log_lik   :: Real,
    mem           :: MemoryBuffer,
    observation   :: Observation,
    threshold     :: Float64 = 2.5   # configurable via Configurator δ_anomaly
) :: AnomalyResult

    # need history to compute z-score
    if length(mem.records) < 5
        return AnomalyResult(false, 0.0, 0.0, Symbol[])
    end

    # recent log likelihood history
    recent = last(mem.records, min(30, length(mem.records)))
    recent_lls = [r.epistemic.κ_familiarity for r in recent]   # proxy for now
    # TODO: store obs_log_lik directly in MemoryRecord

    mean_ll = mean(recent_lls)
    std_ll  = std(recent_lls) + 1e-6

    # z-score — how surprising is this observation?
    # note: log likelihood is negative — more negative = more surprising
    # so we flip the sign for the z-score
    z = -(obs_log_lik - mean_ll) / std_ll

    is_anomaly = z > threshold

    # severity — normalized to [0,1]
    severity = clamp(z / (threshold * 2.0), 0.0, 1.0)

    # identify which signals contributed to the anomaly
    flagged = _identify_anomalous_signals(observation, mem)

    AnomalyResult(is_anomaly, severity, z, flagged)
end

# ─────────────────────────────────────────────────────────────────
# Identify which specific signals are anomalous
# Checks each signal against its recent history
# ─────────────────────────────────────────────────────────────────

function _identify_anomalous_signals(
    observation :: Observation,
    mem         :: MemoryBuffer,
    threshold   :: Float64 = 2.5
) :: Vector{Symbol}

    flagged = Symbol[]
    isempty(mem.records) && return flagged

    recent = last(mem.records, min(30, length(mem.records)))

    for (label, value) in observation.signals
        isnothing(value) && continue
        value isa Float64 || continue

        # collect recent values for this signal
        recent_vals = Float64[]
        for r in recent
            if !isnothing(r.realized_signals) &&
               haskey(r.realized_signals, label) &&
               r.realized_signals[label] isa Float64
                push!(recent_vals, r.realized_signals[label])
            end
        end

        length(recent_vals) < 5 && continue

        # z-score of current value vs recent history
        μ = mean(recent_vals)
        σ = std(recent_vals) + 1e-6
        z = abs(value - μ) / σ

        z > threshold && push!(flagged, label)
    end

    return flagged
end

# ─────────────────────────────────────────────────────────────────
# widen_belief! for GaussianBeliefState
# Scale up variances by anomaly severity
# ─────────────────────────────────────────────────────────────────

function widen_belief(
    belief  :: GaussianBeliefState,
    anomaly :: AnomalyResult,
    factor  :: Float64 = 2.0
) :: GaussianBeliefState

    !anomaly.is_anomaly && return belief
    scale = 1.0 + (factor - 1.0) * anomaly.severity

    GaussianBeliefState(
        x̂_phys       = belief.x̂_phys,
        Σ_phys       = Dict(k => v * scale for (k,v) in belief.Σ_phys),
        x̂_trust      = belief.x̂_trust,
        σ_trust       = belief.σ_trust * sqrt(scale),
        x̂_burnout     = belief.x̂_burnout,
        σ_burnout     = belief.σ_burnout * sqrt(scale),
        x̂_engagement  = belief.x̂_engagement,
        σ_engagement  = belief.σ_engagement * sqrt(scale),
        x̂_burden      = belief.x̂_burden,
        σ_burden       = belief.σ_burden * sqrt(scale),
        entropy       = belief.entropy + log(scale),
        obs_log_lik   = belief.obs_log_lik
    )
end

# ─────────────────────────────────────────────────────────────────
# widen_belief! for ParticleBeliefState
# Add extra jitter to particles proportional to anomaly severity
# More anomalous = more jitter = wider particle spread
# ─────────────────────────────────────────────────────────────────

function widen_belief(
    belief  :: ParticleBeliefState,
    anomaly :: AnomalyResult,
    factor  :: Float64 = 2.0
) :: ParticleBeliefState

    !anomaly.is_anomaly && return belief

    # jitter scale proportional to anomaly severity
    jitter_scale = (factor - 1.0) * anomaly.severity

    new_particles = map(belief.particles) do p

        # widen psychological state
        new_psy = PsyState(
            trust      = ScalarTrust(clamp(
                p.psy.trust.value + randn() * jitter_scale * 0.1,
                0.0, 1.0)),
            burnout    = ScalarBurnout(clamp(
                p.psy.burnout.value + randn() * jitter_scale * 0.1,
                0.0, 1.0)),
            engagement = ScalarEngagement(clamp(
                p.psy.engagement.value + randn() * jitter_scale * 0.1,
                0.0, 1.0)),
            burden     = ScalarBurden(max(0.0,
                p.psy.burden.value + randn() * jitter_scale * 0.1))
        )

        # widen physical state
        new_phys_vars = Dict{Symbol, Float64}(
            label => val + randn() * jitter_scale * 0.1
            for (label, val) in p.phys.variables
        )

        PatientState(PhysState(new_phys_vars), new_psy)
    end

    # entropy increases with wider spread
    new_entropy = belief.entropy + log(1.0 + jitter_scale)

    ParticleBeliefState(
        particles   = new_particles,
        weights     = belief.weights,
        entropy     = new_entropy,
        obs_log_lik = belief.obs_log_lik
    )
end

# ─────────────────────────────────────────────────────────────────
# widen_belief! for JEPABeliefState
# Increase log variance of latent belief proportional to anomaly
# More anomalous = wider latent distribution = more uncertain
# ─────────────────────────────────────────────────────────────────

function widen_belief(
    belief  :: JEPABeliefState,
    anomaly :: AnomalyResult,
    factor  :: Float32 = 2.0f0
) :: JEPABeliefState

    !anomaly.is_anomaly && return belief

    scale = Float32(1.0 + (factor - 1.0) * anomaly.severity)

    # increase log variance — adding log(scale) to log_σ
    # is equivalent to multiplying σ by scale
    new_log_σ = belief.log_σ .+ log(scale)

    JEPABeliefState(
        μ           = belief.μ,
        log_σ       = new_log_σ,
        entropy     = _jepa_entropy(new_log_σ),
        obs_log_lik = belief.obs_log_lik
    )
end
