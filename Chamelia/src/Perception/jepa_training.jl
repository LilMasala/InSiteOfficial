"""
jepa_training.jl
Training interfaces for the latent JEPA encoder/predictor stack.

Chamelia stays data-agnostic here: datasets are responsible for returning
already-grouped tensors with the expected shapes, while the trainer only
knows how to optimize the encoder and predictor.
"""

using SQLite
using BSON

abstract type AbstractTrainingDataset end

function get_training_batch(dataset::AbstractTrainingDataset, batch_size::Int)
    error("$(typeof(dataset)) must implement get_training_batch!")
end

function n_samples(dataset::AbstractTrainingDataset) :: Int
    error("$(typeof(dataset)) must implement n_samples!")
end

struct SQLiteTrainingDataset <: AbstractTrainingDataset
    db_path        :: String
    window_days    :: Int
    n_subhourly    :: Int
    n_ctx          :: Int
    n_daily        :: Int
    subhourly_cols :: Vector{String}
    ctx_cols       :: Vector{String}
    daily_cols     :: Vector{String}
    table_name     :: String
    user_col       :: String
    time_col       :: String
end

struct SyntheticTrainingDataset <: AbstractTrainingDataset
    n_patients     :: Int
    n_days         :: Int
    window_days    :: Int
    n_subhourly    :: Int
    n_ctx          :: Int
    n_daily        :: Int
end

const LAST_JEPA_TRAINING_LOSS = Ref{Float64}(NaN)

_sql_ident(name::AbstractString) = "\"" * replace(String(name), "\"" => "\"\"") * "\""

function _row_value(row, col::String) :: Float32
    sym = Symbol(col)
    hasproperty(row, sym) || return 0.0f0
    value = getproperty(row, sym)
    (ismissing(value) || !(value isa Number)) && return 0.0f0
    return Float32(value)
end

# Returns the set of column names that actually exist in the target table.
# Columns requested by the dataset but absent from the DB silently return 0.
function _existing_cols(db_path::String, table_name::String) :: Set{String}
    db   = SQLite.DB(db_path)
    rows = map(NamedTuple, SQLite.DBInterface.execute(db, "PRAGMA table_info($('"' * table_name * '"'))"))
    SQLite.close(db)
    return Set{String}(String(r.name) for r in rows)
end

function _window_refs(dataset::SQLiteTrainingDataset)
    existing = _existing_cols(dataset.db_path, dataset.table_name)

    # Only SELECT columns that exist; missing signal columns will read as 0 via _row_value.
    wanted = unique(vcat(
        [dataset.user_col, dataset.time_col],
        dataset.subhourly_cols,
        dataset.ctx_cols,
        dataset.daily_cols
    ))
    cols = filter(c -> c in existing, wanted)

    sql = "SELECT " * join(_sql_ident.(cols), ", ") *
          " FROM " * _sql_ident(dataset.table_name) *
          " ORDER BY " * _sql_ident(dataset.user_col) * ", " * _sql_ident(dataset.time_col)

    db = SQLite.DB(dataset.db_path)
    rows = map(NamedTuple, SQLite.DBInterface.execute(db, sql))
    SQLite.close(db)

    grouped = Dict{Any, Vector{Any}}()
    for row in rows
        user_key = hasproperty(row, Symbol(dataset.user_col)) ?
            getproperty(row, Symbol(dataset.user_col)) :
            row[Symbol(dataset.user_col)]
        push!(get!(grouped, user_key, Any[]), row)
    end

    refs = Tuple{Vector{Any}, Int}[]
    for user_rows in values(grouped)
        n_days = length(user_rows) ÷ 24
        max_start = n_days - dataset.window_days
        for start_day in 1:max(0, max_start)
            push!(refs, (user_rows, start_day))
        end
    end

    return refs
end

function n_samples(dataset::SQLiteTrainingDataset) :: Int
    return length(_window_refs(dataset))
end

function n_samples(dataset::SyntheticTrainingDataset) :: Int
    return dataset.n_patients * max(0, dataset.n_days - dataset.window_days)
end

function _fill_window!(
    subhourly :: Array{Float32, 5},
    ctx       :: Array{Float32, 4},
    daily     :: Array{Float32, 3},
    dataset   :: SQLiteTrainingDataset,
    rows      :: Vector,
    start_day :: Int,
    batch_idx :: Int
) :: Nothing
    start_row = (start_day - 1) * 24 + 1
    window_rows = rows[start_row:start_row + dataset.window_days * 24 - 1]

    for day_idx in 1:dataset.window_days
        daily_accum = zeros(Float32, dataset.n_daily)
        for hour_idx in 1:24
            row = window_rows[(day_idx - 1) * 24 + hour_idx]
            for (i, col) in enumerate(dataset.subhourly_cols)
                subhourly[i, :, hour_idx, day_idx, batch_idx] .= _row_value(row, col)
            end
            for (i, col) in enumerate(dataset.ctx_cols)
                ctx[i, hour_idx, day_idx, batch_idx] = _row_value(row, col)
            end
            for (i, col) in enumerate(dataset.daily_cols)
                daily_accum[i] += _row_value(row, col)
            end
        end
        daily[:, day_idx, batch_idx] .= daily_accum ./ 24.0f0
    end

    return nothing
end

function get_training_batch(
    dataset    :: SQLiteTrainingDataset,
    batch_size :: Int
)
    refs = _window_refs(dataset)
    isempty(refs) && error("SQLiteTrainingDataset has no valid sliding windows")

    current_subhourly = zeros(Float32, dataset.n_subhourly, 12, 24, dataset.window_days, batch_size)
    current_ctx       = zeros(Float32, dataset.n_ctx, 24, dataset.window_days, batch_size)
    current_daily     = zeros(Float32, dataset.n_daily, dataset.window_days, batch_size)
    next_subhourly    = similar(current_subhourly)
    next_ctx          = similar(current_ctx)
    next_daily        = similar(current_daily)

    for batch_idx in 1:batch_size
        rows, start_day = refs[rand(1:length(refs))]
        _fill_window!(current_subhourly, current_ctx, current_daily, dataset, rows, start_day, batch_idx)
        _fill_window!(next_subhourly, next_ctx, next_daily, dataset, rows, start_day + 1, batch_idx)
    end

    return current_subhourly, current_ctx, current_daily,
           next_subhourly, next_ctx, next_daily
end

function get_training_batch(
    dataset    :: SyntheticTrainingDataset,
    batch_size :: Int
)
    current_subhourly = rand(Float32, dataset.n_subhourly, 12, 24, dataset.window_days, batch_size)
    current_ctx       = rand(Float32, dataset.n_ctx, 24, dataset.window_days, batch_size)
    current_daily     = rand(Float32, dataset.n_daily, dataset.window_days, batch_size)
    next_subhourly    = rand(Float32, dataset.n_subhourly, 12, 24, dataset.window_days, batch_size)
    next_ctx          = rand(Float32, dataset.n_ctx, 24, dataset.window_days, batch_size)
    next_daily        = rand(Float32, dataset.n_daily, dataset.window_days, batch_size)

    return current_subhourly, current_ctx, current_daily,
           next_subhourly, next_ctx, next_daily
end

function _encode_training_batch(
    encoder    :: HierarchicalJEPAEncoder,
    subhourly  :: AbstractArray,
    ctx        :: AbstractArray,
    daily      :: AbstractArray
) :: Tuple{Matrix{Float32}, Matrix{Float32}}
    n_days = size(subhourly, 4)
    batch  = size(subhourly, 5)

    # When there are no sub-hourly signals, skip the hourly encoder and use
    # zero summaries.  Calling Dense(0, d) through Zygote's backward pass on
    # zero-size arrays is unsupported and raises a DimensionMismatch.
    n_subhourly = size(subhourly, 1)
    d_hourly    = length(encoder.hourly.cls_token)

    hourly_by_day = if n_subhourly == 0
        [zeros(Float32, d_hourly, 24, batch) for _ in 1:n_days]
    else
        [
            begin
                hourly_tokens = [
                    reshape(
                        encoder.hourly(subhourly[:, :, hour_idx, day_idx, :]),
                        :,
                        1,
                        batch
                    )
                    for hour_idx in 1:24
                ]
                length(hourly_tokens) == 1 ? first(hourly_tokens) : cat(hourly_tokens..., dims=2)
            end
            for day_idx in 1:n_days
        ]
    end

    daily_tokens = [
        reshape(
            encoder.daily(hourly_by_day[day_idx], ctx[:, :, day_idx, :]),
            :,
            1,
            batch
        )
        for day_idx in 1:n_days
    ]
    daily_summaries = length(daily_tokens) == 1 ? first(daily_tokens) : cat(daily_tokens..., dims=2)

    μ, log_σ = encoder.multiday(daily_summaries, daily)
    return μ, log_σ
end

function train_encoder!(
    encoder,
    predictor,
    dataset    :: AbstractTrainingDataset;
    n_epochs   :: Int = 50,
    lr         :: Float64 = 1e-3,
    batch_size :: Int = 16
) :: Nothing
    models = (encoder = encoder, predictor = predictor)
    opt = Flux.setup(Adam(lr), models)

    for epoch in 1:n_epochs
        subhourly, ctx, daily,
        next_subhourly, next_ctx, next_daily = get_training_batch(dataset, batch_size)

        loss, grads = Flux.withgradient(models) do model_bundle
            z_current, _ = _encode_training_batch(
                model_bundle.encoder,
                subhourly,
                ctx,
                daily
            )
            z_next, _ = _encode_training_batch(
                model_bundle.encoder,
                next_subhourly,
                next_ctx,
                next_daily
            )

            null_action_features = zeros(Float32, 8, size(z_current, 2))
            z_next_pred = model_bundle.predictor(z_current, null_action_features, :med)

            vicreg_loss(z_current, z_next, z_next_pred)
        end

        LAST_JEPA_TRAINING_LOSS[] = Float64(loss)
        Flux.update!(opt, models, grads[1])

        if epoch % 10 == 0 || epoch == n_epochs
            @info "JEPA epoch $epoch, loss=$(round(Float64(loss), digits=4))"
        end
    end

    return nothing
end

function save_jepa_weights(encoder, predictor, dir::String) :: Nothing
    mkpath(dir)
    BSON.@save joinpath(dir, "jepa_encoder.bson") encoder
    BSON.@save joinpath(dir, "jepa_predictor.bson") predictor
    return nothing
end

function load_jepa_weights!(encoder, predictor, dir::String) :: Nothing
    encoder_state = BSON.load(joinpath(dir, "jepa_encoder.bson"))
    predictor_state = BSON.load(joinpath(dir, "jepa_predictor.bson"))
    Flux.loadmodel!(encoder, encoder_state[:encoder])
    Flux.loadmodel!(predictor, predictor_state[:predictor])
    return nothing
end

# ─────────────────────────────────────────────────────────────────
# MemoryTransitionDataset
#
# Training dataset for PHASE 2 predictor fine-tuning.
# Built from real shadow-period MemoryRecords by Chamelia.jl, which
# has access to both MemoryBuffer and WorldModule.action_to_features.
# Keeping that conversion outside this struct avoids a cross-module
# dependency between Perception and WorldModule.
#
# Each triple is causally grounded:
#   z_t     — latent belief μ at recommendation time
#   a_feats — effective action features (scaled by patient response;
#             see _effective_action_features in Chamelia.jl)
#   z_tH    — latent belief μ at outcome time, after the subsequent
#             observe/step calls have updated the encoder's belief
#
# Contrast with train_encoder! which uses null actions and VICReg to
# learn the latent space structure: this dataset teaches the predictor
# what ACTIONS do to that structure, using real patient behaviour.
# ─────────────────────────────────────────────────────────────────

struct MemoryTransitionDataset <: AbstractTrainingDataset
    z_t     :: Vector{Vector{Float32}}   # latent at recommendation time
    a_feats :: Vector{Vector{Float32}}   # effective action features
    z_tH    :: Vector{Vector{Float32}}   # latent at outcome time
end

function n_samples(dataset::MemoryTransitionDataset) :: Int
    return length(dataset.z_t)
end

# Returns three matrices of shape (dim, batch_size) sampled with replacement.
function get_training_batch(
    dataset    :: MemoryTransitionDataset,
    batch_size :: Int
) :: Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}}
    n = n_samples(dataset)
    n == 0 && error("MemoryTransitionDataset has no samples")

    z_dim = length(dataset.z_t[1])
    a_dim = length(dataset.a_feats[1])

    z_t_batch    = zeros(Float32, z_dim, batch_size)
    a_feats_batch = zeros(Float32, a_dim, batch_size)
    z_tH_batch   = zeros(Float32, z_dim, batch_size)

    for i in 1:batch_size
        idx = rand(1:n)
        z_t_batch[:, i]     = dataset.z_t[idx]
        a_feats_batch[:, i] = dataset.a_feats[idx]
        z_tH_batch[:, i]    = dataset.z_tH[idx]
    end

    return z_t_batch, a_feats_batch, z_tH_batch
end

# ─────────────────────────────────────────────────────────────────
# train_predictor!
#
# Fine-tunes the JEPA predictor on real action-conditioned latent
# transitions accumulated during the shadow period.
#
# Loss: MSE in latent space  L = ||ẑ_{t+H} - z_{t+H}||²
#
# Why MSE and not VICReg here:
#   train_encoder! uses VICReg because there is no supervision signal
#   — we need the variance/covariance terms to prevent collapse.
#   Here z_{t+H} is the GROUND TRUTH from the encoder, so this is
#   ordinary supervised regression.  Collapse cannot occur because
#   the encoder (not the predictor) defines the latent geometry.
#
# Horizon: :med — patient outcomes are recorded at medium horizon
#   (~7 days).  The short and long heads share the trunk so they
#   benefit indirectly from every update.
# ─────────────────────────────────────────────────────────────────

const LAST_PREDICTOR_TRAINING_LOSS = Ref{Float64}(NaN)

function train_predictor!(
    predictor,                          # JEPAPredictor — untyped to avoid WorldModule dependency
    dataset    :: MemoryTransitionDataset;
    n_epochs   :: Int    = 20,
    lr         :: Float64 = 1e-3,
    batch_size :: Int    = 16
) :: Nothing
    n_samples(dataset) == 0 && return nothing

    opt = Flux.setup(Adam(lr), predictor)

    for epoch in 1:n_epochs
        z_t_batch, a_feats_batch, z_tH_batch = get_training_batch(dataset, batch_size)

        loss, grads = Flux.withgradient(predictor) do pred
            z_pred = pred(z_t_batch, a_feats_batch, :med)
            mean((z_pred .- z_tH_batch).^2)
        end

        LAST_PREDICTOR_TRAINING_LOSS[] = Float64(loss)
        Flux.update!(opt, predictor, grads[1])

        if epoch % 10 == 0 || epoch == n_epochs
            @info "Predictor epoch $epoch, MSE=$(round(Float64(loss), digits=4))"
        end
    end

    return nothing
end
