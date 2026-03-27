"""
search.jl
Action search strategies for the Actor.
All implement the same interface via multiple dispatch.

v1.1  → GridSearch      (exhaustive grid, discrete levels)
v1.5  → BeamSearch      (keeps top K at each dimension, scales better)
v2.0  → GradientSearch  (continuous optimization through JEPA predictor)
v2.0+ → OfflineRLPolicy (learned policy from fork-of-forks dataset)

The Configurator decides which strategy to use based on:
  - which estimator is active (Kalman/Particle/JEPA)
  - how much data exists in Memory
  - current belief entropy and trust level
"""

using Statistics
using LinearAlgebra

# ─────────────────────────────────────────────────────────────────
# CandidateAction
# Concrete action type for grid/beam/gradient search.
# Stores relative changes per dimension.
# e.g. :isf => +0.10 means +10% ISF from current value
# ─────────────────────────────────────────────────────────────────

struct CandidateAction <: AbstractAction
    deltas :: Dict{Symbol, Float64}
end

function is_null(a::CandidateAction) :: Bool
    return all(v -> abs(v) < 1e-8, values(a.deltas))
end

function magnitude(a::CandidateAction) :: Float64
    isempty(a.deltas) && return 0.0
    return sum(abs(v) for v in values(a.deltas)) / length(a.deltas)
end

# ─────────────────────────────────────────────────────────────────
# Shared interface
# Every search strategy implements this function.
# Returns sorted (action, energies, cvar) tuples — best first.
# ─────────────────────────────────────────────────────────────────

function search_actions(
    strategy   :: AbstractSearchStrategy,
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState,
    dimensions :: Vector{Symbol}
)
    error("$(typeof(strategy)) must implement search_actions!")
end

function search_scheduled_actions(
    strategy     :: AbstractSearchStrategy,
    belief       :: AbstractBeliefState,
    twin         :: DigitalTwin,
    sim          :: AbstractSimulator,
    noise        :: RolloutNoise,
    critic       :: AbstractCriticModel,
    config       :: ConfiguratorState,
    capabilities :: ConnectedAppCapabilities,
    app_state    :: ConnectedAppState;
    current_day  :: Int = 0,
)
    if isempty(app_state.current_segments) || !capabilities.level_1_enabled
        return search_actions(
            strategy,
            belief,
            twin,
            sim,
            noise,
            critic,
            config,
            action_dimensions(sim),
        )
    end

    candidates = _generate_level1_scheduled_candidates(config, app_state)
    if _level2_eligible(belief, capabilities, app_state, current_day)
        append!(candidates, generate_structure_edit_candidates(config, capabilities, app_state))
    end
    return _evaluate_scheduled_candidates(
        candidates,
        app_state,
        belief,
        twin,
        sim,
        noise,
        critic,
        config,
    )
end

function _generate_level1_scheduled_candidates(
    config    :: ConfiguratorState,
    app_state :: ConnectedAppState
) :: Vector{ScheduledAction}
    isempty(app_state.current_segments) && return ScheduledAction[]

    Δ_max = config.φ_act.Δ_max
    levels = [-Δ_max, -Δ_max / 2, Δ_max / 2, Δ_max]
    base_segments = deepcopy(app_state.current_segments)
    candidates = ScheduledAction[ScheduledAction(1, parameter_adjustment, base_segments, SegmentDelta[], StructureEdit[])]

    for segment in app_state.current_segments
        for level in levels
            push!(candidates, ScheduledAction(
                1,
                parameter_adjustment,
                base_segments,
                [SegmentDelta(segment_id=segment.segment_id, isf_delta=level)],
                StructureEdit[],
            ))
            push!(candidates, ScheduledAction(
                1,
                parameter_adjustment,
                base_segments,
                [SegmentDelta(segment_id=segment.segment_id, cr_delta=level)],
                StructureEdit[],
            ))
            push!(candidates, ScheduledAction(
                1,
                parameter_adjustment,
                base_segments,
                [SegmentDelta(segment_id=segment.segment_id, basal_delta=level)],
                StructureEdit[],
            ))
        end
    end

    for level in levels
        push!(candidates, ScheduledAction(
            1,
            parameter_adjustment,
            base_segments,
            [SegmentDelta(segment_id=segment.segment_id, isf_delta=level) for segment in app_state.current_segments],
            StructureEdit[],
        ))
        push!(candidates, ScheduledAction(
            1,
            parameter_adjustment,
            base_segments,
            [SegmentDelta(segment_id=segment.segment_id, cr_delta=level) for segment in app_state.current_segments],
            StructureEdit[],
        ))
        push!(candidates, ScheduledAction(
            1,
            parameter_adjustment,
            base_segments,
            [SegmentDelta(segment_id=segment.segment_id, basal_delta=level) for segment in app_state.current_segments],
            StructureEdit[],
        ))
    end

    if length(candidates) > config.φ_act.N_search
        return candidates[1:config.φ_act.N_search]
    end
    return candidates
end

function _current_trust(belief::GaussianBeliefState) :: Float64
    return belief.x̂_trust
end

function _current_trust(belief::ParticleBeliefState) :: Float64
    isempty(belief.particles) && return 0.0
    return mean(p.psy.trust.value for p in belief.particles)
end

function _current_trust(belief::JEPABeliefState) :: Float64
    return Float64(decode_latent_summary(belief).trust)
end

function _current_burnout(belief::GaussianBeliefState) :: Float64
    return belief.x̂_burnout
end

function _current_burnout(belief::ParticleBeliefState) :: Float64
    isempty(belief.particles) && return 1.0
    return mean(p.psy.burnout.value for p in belief.particles)
end

function _current_burnout(belief::JEPABeliefState) :: Float64
    return Float64(decode_latent_summary(belief).burnout)
end

function _level2_eligible(
    belief       :: AbstractBeliefState,
    capabilities :: ConnectedAppCapabilities,
    app_state    :: ConnectedAppState,
    current_day  :: Int,
) :: Bool
    capabilities.level_2_enabled || return false
    app_state.allow_structural_recommendations || return false
    capabilities.supports_piecewise_schedule || return false
    current_day >= 30 || return false
    _current_trust(belief) > 0.65 || return false
    _current_burnout(belief) < 0.40 || return false
    return !isempty(app_state.current_segments)
end

function _apply_structure_edits(
    segments :: Vector{SegmentSurface},
    edits    :: Vector{StructureEdit},
) :: Vector{SegmentSurface}
    current = deepcopy(segments)
    for edit in edits
        if edit.edit_type == :split
            idx = findfirst(seg -> seg.segment_id == edit.target_segment_id, current)
            isnothing(idx) && continue
            segment = current[idx]
            split_at = something(edit.split_at_minute, (segment.start_min + segment.end_min) ÷ 2)
            segment.start_min < split_at < segment.end_min || continue
            left = SegmentSurface(
                segment_id = "$(segment.segment_id)_a",
                start_min = segment.start_min,
                end_min = split_at,
                isf = segment.isf,
                cr = segment.cr,
                basal = segment.basal,
            )
            right = SegmentSurface(
                segment_id = "$(segment.segment_id)_b",
                start_min = split_at,
                end_min = segment.end_min,
                isf = segment.isf,
                cr = segment.cr,
                basal = segment.basal,
            )
            prefix = idx > 1 ? current[1:idx-1] : SegmentSurface[]
            suffix = idx < length(current) ? current[idx+1:end] : SegmentSurface[]
            current = vcat(prefix, [left, right], suffix)
        elseif edit.edit_type == :merge
            idx = findfirst(seg -> seg.segment_id == edit.target_segment_id, current)
            isnothing(idx) && continue
            neighbor_idx = isnothing(edit.neighbor_segment_id) ?
                (idx < length(current) ? idx + 1 : nothing) :
                findfirst(seg -> seg.segment_id == edit.neighbor_segment_id, current)
            isnothing(neighbor_idx) && continue
            a = current[idx]
            b = current[neighbor_idx]
            first, second = a.start_min <= b.start_min ? (a, b) : (b, a)
            first.end_min == second.start_min || continue
            merged = SegmentSurface(
                segment_id = "$(first.segment_id)__$(second.segment_id)",
                start_min = first.start_min,
                end_min = second.end_min,
                isf = (first.isf + second.isf) / 2,
                cr = (first.cr + second.cr) / 2,
                basal = (first.basal + second.basal) / 2,
            )
            current = sort(
                [seg for seg in current if seg.segment_id != first.segment_id && seg.segment_id != second.segment_id];
                by = seg -> seg.start_min
            )
            push!(current, merged)
            sort!(current, by = seg -> seg.start_min)
        end
    end
    return current
end

function _segments_valid(
    segments :: Vector{SegmentSurface},
    capabilities :: ConnectedAppCapabilities,
) :: Bool
    isempty(segments) && return false
    length(segments) <= capabilities.max_segments || return false
    segments[1].start_min == 0 || return false
    segments[end].end_min == 1440 || return false
    prev_end = 0
    for segment in sort(segments, by = seg -> seg.start_min)
        segment.start_min == prev_end || return false
        (segment.end_min - segment.start_min) >= capabilities.min_segment_duration_min || return false
        min(segment.isf, segment.cr, segment.basal) > 0 || return false
        prev_end = segment.end_min
    end
    return true
end

function generate_structure_edit_candidates(
    config        :: ConfiguratorState,
    capabilities  :: ConnectedAppCapabilities,
    app_state     :: ConnectedAppState,
) :: Vector{ScheduledAction}
    isempty(app_state.current_segments) && return ScheduledAction[]

    candidates = ScheduledAction[]
    Δ = config.φ_act.Δ_max / 2
    segments = sort(deepcopy(app_state.current_segments), by = seg -> seg.start_min)

    for segment in segments
        width = segment.end_min - segment.start_min
        width >= 2 * capabilities.min_segment_duration_min || continue
        split_at = (segment.start_min + segment.end_min) ÷ 2
        edit = StructureEdit(edit_type=:split, target_segment_id=segment.segment_id, split_at_minute=split_at)
        edited_segments = _apply_structure_edits(segments, [edit])
        _segments_valid(edited_segments, capabilities) || continue

        child_ids = [seg.segment_id for seg in edited_segments if startswith(seg.segment_id, "$(segment.segment_id)_")]
        for child_id in child_ids
            push!(candidates, ScheduledAction(
                2,
                structure_edit,
                edited_segments,
                [SegmentDelta(segment_id=child_id, basal_delta=Δ)],
                [edit],
            ))
            push!(candidates, ScheduledAction(
                2,
                structure_edit,
                edited_segments,
                [SegmentDelta(segment_id=child_id, isf_delta=Δ)],
                [edit],
            ))
            push!(candidates, ScheduledAction(
                2,
                structure_edit,
                edited_segments,
                [SegmentDelta(segment_id=child_id, cr_delta=-Δ)],
                [edit],
            ))
        end
    end

    for idx in 1:length(segments)-1
        a = segments[idx]
        b = segments[idx + 1]
        similar = abs(a.isf - b.isf) / max(a.isf, 1e-6) < 0.05 &&
            abs(a.cr - b.cr) / max(a.cr, 1e-6) < 0.05 &&
            abs(a.basal - b.basal) / max(a.basal, 1e-6) < 0.05
        similar || continue

        edit = StructureEdit(edit_type=:merge, target_segment_id=a.segment_id, neighbor_segment_id=b.segment_id)
        edited_segments = _apply_structure_edits(segments, [edit])
        _segments_valid(edited_segments, capabilities) || continue

        push!(candidates, ScheduledAction(
            2,
            structure_edit,
            edited_segments,
            SegmentDelta[],
            [edit],
        ))
    end

    if length(candidates) > config.φ_act.N_search
        return candidates[1:config.φ_act.N_search]
    end
    return candidates
end

function _evaluate_scheduled_candidates(
    candidates   :: Vector{ScheduledAction},
    app_state    :: ConnectedAppState,
    belief       :: AbstractBeliefState,
    twin         :: DigitalTwin,
    sim          :: AbstractSimulator,
    noise        :: RolloutNoise,
    critic       :: AbstractCriticModel,
    config       :: ConfiguratorState
)
    isempty(candidates) && return []

    results = Vector{Any}(undef, length(candidates))

    Threads.@threads for i in 1:length(candidates)
        rollouts = if belief isa JEPABeliefState
            run_latent_rollouts(belief, candidates[i], JEPA_PREDICTOR, config)
        else
            run_rollouts(belief, candidates[i], twin, sim, noise, config)
        end
        energies = compute_energies(rollouts, critic, config)
        cvar = compute_cvar(energies, config.φ_act.α_cvar)
        results[i] = (action=candidates[i], rollouts=rollouts, energies=energies, cvar=cvar)
    end

    sort!(results, by = r -> r.cvar)
    return results
end

# ─────────────────────────────────────────────────────────────────
# GRID SEARCH — v1.1
# Exhaustive search over all combinations of discrete action levels.
# Simple, interpretable, correct.
# Scales as O(levels^dimensions) — manageable for small action spaces.
# ─────────────────────────────────────────────────────────────────

function search_actions(
    ::GridSearch,
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState,
    dimensions :: Vector{Symbol}
)

    Δ_max  = config.φ_act.Δ_max
    levels = [-Δ_max, -Δ_max/2, 0.0, Δ_max/2, Δ_max]

    # generate all combinations
    candidates = _generate_grid(dimensions, levels)

    # limit to search budget
    if length(candidates) > config.φ_act.N_search
        candidates = candidates[1:config.φ_act.N_search]
    end

    return _evaluate_candidates(candidates, belief, twin, sim, noise, critic, config)
end

function _generate_grid(
    dimensions :: Vector{Symbol},
    levels     :: Vector{Float64}
) :: Vector{CandidateAction}

    candidates = CandidateAction[]

    # null action first
    push!(candidates, CandidateAction(Dict(d => 0.0 for d in dimensions)))

    function recurse!(current::Dict{Symbol,Float64}, dim_idx::Int)
        if dim_idx > length(dimensions)
            any(v -> abs(v) > 1e-8, values(current)) &&
                push!(candidates, CandidateAction(copy(current)))
            return
        end
        dim = dimensions[dim_idx]
        for level in levels
            current[dim] = level
            recurse!(current, dim_idx + 1)
        end
        delete!(current, dim)
    end

    recurse!(Dict{Symbol,Float64}(), 1)
    return candidates
end

# ─────────────────────────────────────────────────────────────────
# BEAM SEARCH — v1.5
# Keeps only top K candidates at each dimension expansion.
# Scales as O(K * levels * dimensions) — much better than grid.
# Trades off some optimality for computational tractability.
# ─────────────────────────────────────────────────────────────────

function search_actions(
    ::BeamSearch,
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState,
    dimensions :: Vector{Symbol};
    beam_width :: Int = 5
)

    Δ_max  = config.φ_act.Δ_max
    levels = [-Δ_max, -Δ_max/2, 0.0, Δ_max/2, Δ_max]

    # start with null action
    beam = [CandidateAction(Dict(d => 0.0 for d in dimensions))]

    for dim in dimensions
        # expand each beam candidate along this dimension
        expanded = CandidateAction[]
        for candidate in beam
            for level in levels
                new_deltas = copy(candidate.deltas)
                new_deltas[dim] = level
                push!(expanded, CandidateAction(new_deltas))
            end
        end

        # evaluate all expanded candidates
        evaluated = _evaluate_candidates(expanded, belief, twin, sim, noise, critic, config)

        # keep top beam_width by CVaR
        beam = [r.action for r in evaluated[1:min(beam_width, length(evaluated))]]
    end

    # final evaluation of beam survivors
    return _evaluate_candidates(beam, belief, twin, sim, noise, critic, config)
end

# ─────────────────────────────────────────────────────────────────
# GRADIENT SEARCH — v2.0
# Continuous optimization through the JEPA predictor.
# Requires differentiable world model — only valid with JEPABeliefState.
# Uses gradient descent on CVaR energy in latent action space.
# Falls back to grid search if belief is not JEPA.
# ─────────────────────────────────────────────────────────────────

function search_actions(
    ::GradientSearch,
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState,
    dimensions :: Vector{Symbol}
)

    # gradient search only valid with JEPA belief
    # fall back to grid search otherwise
    if !(belief isa JEPABeliefState)
        @warn "GradientSearch requires JEPABeliefState — falling back to GridSearch"
        return search_actions(GridSearch(), belief, twin, sim, noise,
                              critic, config, dimensions)
    end

    Δ_max   = config.φ_act.Δ_max
    n_steps = 50     # gradient steps
    lr      = 0.01   # step size
    n_restarts = 5   # random restarts to avoid local minima

    best_results = nothing
    best_cvar    = Inf

    for _ in 1:n_restarts
        # random initialization within bounds
        deltas = Dict(d => (rand() * 2 - 1) * Δ_max for d in dimensions)

        for step in 1:n_steps
            # numerical gradient — perturb each dimension
            grads = Dict{Symbol, Float64}()
            for dim in dimensions
                ε = 1e-3
                deltas_plus  = copy(deltas); deltas_plus[dim]  += ε
                deltas_minus = copy(deltas); deltas_minus[dim] -= ε

                a_plus  = CandidateAction(deltas_plus)
                a_minus = CandidateAction(deltas_minus)

                cvar_plus  = _quick_cvar(a_plus,  belief, twin, sim, noise, critic, config)
                cvar_minus = _quick_cvar(a_minus, belief, twin, sim, noise, critic, config)

                grads[dim] = (cvar_plus - cvar_minus) / (2ε)
            end

            # gradient step with projection onto bounds
            for dim in dimensions
                deltas[dim] = clamp(deltas[dim] - lr * grads[dim], -Δ_max, Δ_max)
            end
        end

        # evaluate final point
        action   = CandidateAction(deltas)
        rollouts = run_latent_rollouts(belief, action, JEPA_PREDICTOR, config)
        energies = compute_energies(rollouts, critic, config)
        cvar     = compute_cvar(energies, config.φ_act.α_cvar)

        if cvar < best_cvar
            best_cvar    = cvar
            best_results = [(action=action, rollouts=rollouts, energies=energies, cvar=cvar)]
        end
    end

    return best_results === nothing ? [] : best_results
end

function _quick_cvar(
    action  :: CandidateAction,
    belief  :: AbstractBeliefState,
    twin    :: DigitalTwin,
    sim     :: AbstractSimulator,
    noise   :: RolloutNoise,
    critic  :: AbstractCriticModel,
    config  :: ConfiguratorState
) :: Float64
    rollouts = run_rollouts(belief, action, twin, sim, noise, config)
    energies = compute_energies(rollouts, critic, config)
    return compute_cvar(energies, config.φ_act.α_cvar)
end

function _quick_cvar(
    action  :: CandidateAction,
    belief  :: JEPABeliefState,
    twin    :: DigitalTwin,
    sim     :: AbstractSimulator,
    noise   :: RolloutNoise,
    critic  :: AbstractCriticModel,
    config  :: ConfiguratorState
) :: Float64
    _ = twin
    _ = sim
    _ = noise
    rollouts = run_latent_rollouts(belief, action, JEPA_PREDICTOR, config)
    energies = compute_energies(rollouts, critic, config)
    return compute_cvar(energies, config.φ_act.α_cvar)
end

# ─────────────────────────────────────────────────────────────────
# OFFLINE RL POLICY — v2.0+
# Learned policy from fork-of-forks trajectory dataset.
# At inference: single forward pass through policy network.
# Conservative Q-Learning (CQL) prevents out-of-distribution actions.
# Requires training data — falls back to beam search until available.
# ─────────────────────────────────────────────────────────────────

mutable struct OfflineRLPolicyModel
    network       :: Union{Chain, Nothing}   # policy network (Flux)
    n_trained     :: Int
    is_ready      :: Bool                    # enough data to trust?
    min_samples   :: Int                     # minimum training samples
end

OfflineRLPolicyModel() = OfflineRLPolicyModel(nothing, 0, false, 1000)

# Global policy model — initialized once, updated from Memory
const OFFLINE_RL_MODEL = OfflineRLPolicyModel()

function search_actions(
    ::OfflineRLPolicy,
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState,
    dimensions :: Vector{Symbol}
)

    # fall back to beam search until policy is trained
    if !OFFLINE_RL_MODEL.is_ready
        @warn "OfflineRLPolicy not yet trained — falling back to BeamSearch"
        return search_actions(GradientSearch(), belief, twin, sim, noise,
                              critic, config, dimensions)
    end

    # policy forward pass — belief features → action deltas
    policy_input = belief isa JEPABeliefState ?
        _actor_policy_input(Float32.(vec(belief.μ)), action_to_features(NullAction())) :
        _extract_belief_features(belief)
    action_deltas = OFFLINE_RL_MODEL.network(policy_input)

    # clip to bounds
    Δ_max = config.φ_act.Δ_max
    deltas = Dict(
        dimensions[i] => clamp(Float64(action_deltas[i]), -Δ_max, Δ_max)
        for i in 1:min(length(dimensions), length(action_deltas))
    )

    action   = CandidateAction(deltas)
    rollouts = belief isa JEPABeliefState ?
        run_latent_rollouts(belief, action, JEPA_PREDICTOR, config) :
        run_rollouts(belief, action, twin, sim, noise, config)
    energies = compute_energies(rollouts, critic, config)
    cvar     = compute_cvar(energies, config.φ_act.α_cvar)

    return [(action=action, rollouts=rollouts, energies=energies, cvar=cvar)]
end

function _extract_belief_features(belief::GaussianBeliefState) :: Vector{Float32}
    Float32[
        belief.x̂_trust, belief.σ_trust,
        belief.x̂_burnout, belief.σ_burnout,
        belief.x̂_engagement, belief.σ_engagement,
        belief.x̂_burden, belief.σ_burden,
        belief.entropy
    ]
end

function _extract_belief_features(belief::ParticleBeliefState) :: Vector{Float32}
    # summarize particle distribution
    trusts = [p.psy.trust.value for p in belief.particles]
    burnouts = [p.psy.burnout.value for p in belief.particles]
    Float32[
        mean(trusts), std(trusts),
        mean(burnouts), std(burnouts),
        belief.entropy, 0.0, 0.0, 0.0, 0.0
    ]
end

function _extract_belief_features(belief::JEPABeliefState) :: Vector{Float32}
    # use latent mean directly
    Float32.(belief.μ[1:min(9, length(belief.μ))])
end

# ─────────────────────────────────────────────────────────────────
# Shared evaluation helper
# ─────────────────────────────────────────────────────────────────

function _evaluate_candidates(
    candidates :: Vector{CandidateAction},
    belief     :: AbstractBeliefState,
    twin       :: DigitalTwin,
    sim        :: AbstractSimulator,
    noise      :: RolloutNoise,
    critic     :: AbstractCriticModel,
    config     :: ConfiguratorState
)

    isempty(candidates) && return []

    results = Vector{Any}(undef, length(candidates))

    Threads.@threads for i in 1:length(candidates)
        rollouts = run_rollouts(belief, candidates[i], twin, sim, noise, config)
        energies = compute_energies(rollouts, critic, config)
        cvar     = compute_cvar(energies, config.φ_act.α_cvar)
        results[i] = (action=candidates[i], rollouts=rollouts, energies=energies, cvar=cvar)
    end

    sort!(results, by = r -> r.cvar)
    return results
end
