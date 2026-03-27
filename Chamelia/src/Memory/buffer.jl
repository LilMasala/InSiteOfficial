"""
buffer.jl
Memory buffer — structured experience store.
Handles record creation, outcome filling, and rolling window retention.

Each record has three phases:
  1. Recommendation time (immutable once written)
  2. Outcome time (filled in delta days later)
  3. Retrospective (updated progressively as future data arrives)
"""

using Statistics

function _latent_snapshot(
    belief :: AbstractBeliefState
) :: Tuple{Union{Vector{Float32}, Nothing}, Union{Vector{Float32}, Nothing}}
    return nothing, nothing
end

function _latent_snapshot(
    belief :: JEPABeliefState
) :: Tuple{Union{Vector{Float32}, Nothing}, Union{Vector{Float32}, Nothing}}
    return Float32.(vec(belief.μ)), Float32.(vec(belief.log_σ))
end

# ─────────────────────────────────────────────────────────────────
# Store new recommendation record
# Called immediately when recommendation is made or hold is logged
# ─────────────────────────────────────────────────────────────────

function store_record!(
    mem       :: MemoryBuffer,
    day       :: Int,
    belief    :: AbstractBeliefState,
    action    :: AbstractAction,
    epistemic :: EpistemicState,
    config    :: ConfiguratorState,
    psy       :: PsyState;
    latent_snapshot   :: Union{Vector{Float32}, Nothing} = nothing,
    predicted_cvar    :: Union{Float64, Nothing} = nothing,
    configurator_mode :: Symbol = :rules,
) :: Int   # returns record id

    id = mem.next_id
    mem.next_id += 1
    latent_μ_at_rec, latent_log_σ_at_rec = _latent_snapshot(belief)
    snapshot = isnothing(latent_snapshot) ? latent_μ_at_rec : copy(latent_snapshot)

    record = MemoryRecord(
        id              = id,
        day             = day,
        belief_entropy  = belief.entropy,
        action          = action,
        epistemic       = epistemic,
        config_snapshot = deepcopy(config),
        user_response   = nothing,
        realized_signals = nothing,
        realized_cost   = nothing,
        predicted_cvar  = predicted_cvar,
        critic_target   = nothing,
        shadow_delta_score = nothing,
        trust_at_rec    = psy.trust.value,
        burnout_at_rec  = psy.burnout.value,
        engagement_at_rec = psy.engagement.value,
        burden_at_rec   = psy.burden.value,
        latent_snapshot = snapshot,
        latent_μ_at_rec = latent_μ_at_rec,
        latent_log_σ_at_rec = latent_log_σ_at_rec,
        configurator_mode   = configurator_mode,
    )

    push!(mem.records, record)

    # prune old records outside retention window
    _prune_old_records!(mem, day)

    return id
end

# ─────────────────────────────────────────────────────────────────
# Fill in outcome fields
# Called delta days after recommendation when outcomes arrive
# ─────────────────────────────────────────────────────────────────

function store_outcome!(
    mem      :: MemoryBuffer,
    rec_id   :: Int,
    response :: Union{UserResponse, Nothing},
    signals  :: Dict{Symbol, Any},
    cost     :: Float64
) :: Nothing

    idx = findfirst(r -> r.id == rec_id, mem.records)
    isnothing(idx) && return nothing

    rec = mem.records[idx]
    rec.user_response    = response
    rec.realized_signals = signals
    rec.realized_cost    = cost

    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Store hold decision
# A hold is an active decision — logged and scored like a recommendation
# ─────────────────────────────────────────────────────────────────

function store_hold!(
    mem       :: MemoryBuffer,
    day       :: Int,
    belief    :: AbstractBeliefState,
    epistemic :: EpistemicState,
    reason    :: Symbol,
    config    :: ConfiguratorState,
    psy       :: PsyState;
    configurator_mode :: Symbol = :rules
) :: Int

    # null action represents hold
    null_action = NullAction()

    return store_record!(mem, day, belief, null_action, epistemic, config, psy;
                         configurator_mode = configurator_mode)
end

# ─────────────────────────────────────────────────────────────────
# Rolling window retention
# Drop records older than H_mem days
# ─────────────────────────────────────────────────────────────────

function _prune_old_records!(mem::MemoryBuffer, current_day::Int) :: Nothing
    cutoff = current_day - mem.H_mem
    filter!(r -> r.day >= cutoff, mem.records)
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────

function get_record(mem::MemoryBuffer, id::Int) :: Union{MemoryRecord, Nothing}
    idx = findfirst(r -> r.id == id, mem.records)
    isnothing(idx) ? nothing : mem.records[idx]
end

function completed_records(mem::MemoryBuffer) :: Vector{MemoryRecord}
    filter(r -> !isnothing(r.realized_cost), mem.records)
end

function recent_records(mem::MemoryBuffer, n::Int) :: Vector{MemoryRecord}
    last(mem.records, min(n, length(mem.records)))
end

function records_in_window(
    mem        :: MemoryBuffer,
    day_start  :: Int,
    day_end    :: Int
) :: Vector{MemoryRecord}
    filter(r -> day_start <= r.day <= day_end, mem.records)
end
