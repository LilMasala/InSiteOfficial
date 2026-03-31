"""
train_local_jepa_weights.jl

Domain-agnostic Phase 1 JEPA encoder pretraining.

Two modes:

  Synthetic (default)
    Uses SyntheticTrainingDataset with random tensors.  Establishes a
    non-collapsed latent space but carries no real signal.  Use only as
    weight initialisation before real-data training.

    Dimensions are controlled by --n-ctx, --n-daily, --n-subhourly.

  Real data (--db-path + --ctx-cols + --daily-cols)
    Uses SQLiteTrainingDataset.  The caller supplies the column names
    appropriate for their domain; this script does not know or care what
    the columns represent.

Usage:
    # Synthetic (e.g. quick CI check)
    julia scripts/train_local_jepa_weights.jl --epochs 50

    # Real data — column lists come from the caller
    julia scripts/train_local_jepa_weights.jl \\
        --db-path /path/to/data.db \\
        --ctx-cols col_a,col_b,col_c \\
        --daily-cols col_x,col_y \\
        --epochs 50

Weights are saved to --out-dir (default: tmp/jepa_local_weights/).
"""

using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "WorldModule", "WorldModule.jl"))
include(joinpath(ROOT, "src", "Perception", "Perception.jl"))

# ─────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────

function parse_args(args::Vector{String})
    out_dir      = joinpath(ROOT, "tmp", "jepa_local_weights")
    n_epochs     = 50
    batch_size   = 16
    seed         = 42
    db_path      = nothing
    window_days  = 7
    n_subhourly  = 0
    n_ctx        = 37    # default matches InSite signal count
    n_daily      = 6
    ctx_cols     = nothing   # Vector{String} or nothing
    daily_cols   = nothing
    table_name   = "feature_frames"
    user_col     = "user_id"
    time_col     = "hour_utc"

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--out-dir"
            i += 1; out_dir = args[i]
        elseif arg == "--epochs"
            i += 1; n_epochs = parse(Int, args[i])
        elseif arg == "--batch-size"
            i += 1; batch_size = parse(Int, args[i])
        elseif arg == "--seed"
            i += 1; seed = parse(Int, args[i])
        elseif arg == "--db-path"
            i += 1; db_path = args[i]
        elseif arg == "--window-days"
            i += 1; window_days = parse(Int, args[i])
        elseif arg == "--n-subhourly"
            i += 1; n_subhourly = parse(Int, args[i])
        elseif arg == "--n-ctx"
            i += 1; n_ctx = parse(Int, args[i])
        elseif arg == "--n-daily"
            i += 1; n_daily = parse(Int, args[i])
        elseif arg == "--ctx-cols"
            i += 1; ctx_cols = split(args[i], ",")
        elseif arg == "--daily-cols"
            i += 1; daily_cols = split(args[i], ",")
        elseif arg == "--table"
            i += 1; table_name = args[i]
        elseif arg == "--user-col"
            i += 1; user_col = args[i]
        elseif arg == "--time-col"
            i += 1; time_col = args[i]
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    return (
        out_dir=out_dir, n_epochs=n_epochs, batch_size=batch_size, seed=seed,
        db_path=db_path, window_days=window_days,
        n_subhourly=n_subhourly, n_ctx=n_ctx, n_daily=n_daily,
        ctx_cols=ctx_cols, daily_cols=daily_cols,
        table_name=table_name, user_col=user_col, time_col=time_col,
    )
end

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

function main(args::Vector{String})
    cfg = parse_args(args)
    Random.seed!(cfg.seed)

    dataset, encoder = if cfg.db_path !== nothing
        isfile(cfg.db_path) || error("SQLite database not found: $(cfg.db_path)")

        # Column lists must be supplied when using a real database.
        ctx   = cfg.ctx_cols   !== nothing ? String.(cfg.ctx_cols)   : error("--ctx-cols required with --db-path")
        daily = cfg.daily_cols !== nothing ? String.(cfg.daily_cols) : error("--daily-cols required with --db-path")

        @info "Using SQLiteTrainingDataset" db=cfg.db_path n_ctx=length(ctx) n_daily=length(daily)
        ds = Perception.SQLiteTrainingDataset(
            cfg.db_path,
            cfg.window_days,
            cfg.n_subhourly,
            length(ctx),
            length(daily),
            String[],           # subhourly_cols — always empty for hourly-resolution data
            ctx,
            daily,
            cfg.table_name,
            cfg.user_col,
            cfg.time_col,
        )
        n = Perception.n_samples(ds)
        @info "Dataset has $n sliding windows"
        n > 0 || error("No valid training windows. Ensure at least $(cfg.window_days + 1) days of data per user.")
        enc = Perception.HierarchicalJEPAEncoder(cfg.n_subhourly, length(ctx), length(daily))
        ds, enc
    else
        @info "No --db-path given — using SyntheticTrainingDataset (weight init only)"
        ds = Perception.SyntheticTrainingDataset(64, 30, cfg.window_days, cfg.n_subhourly, cfg.n_ctx, cfg.n_daily)
        enc = Perception.HierarchicalJEPAEncoder(cfg.n_subhourly, cfg.n_ctx, cfg.n_daily)
        ds, enc
    end

    predictor = WorldModule.JEPAPredictor()

    @info "Training JEPA encoder" epochs=cfg.n_epochs batch_size=cfg.batch_size
    Perception.train_encoder!(encoder, predictor, dataset; n_epochs=cfg.n_epochs, batch_size=cfg.batch_size)

    mkpath(cfg.out_dir)
    Perception.save_jepa_weights(encoder, predictor, cfg.out_dir)

    println("Saved JEPA weights to: ", cfg.out_dir)
    println("Final training loss:   ", Perception.LAST_JEPA_TRAINING_LOSS[])
end

main(ARGS)
