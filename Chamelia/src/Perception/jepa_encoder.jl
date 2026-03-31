""" 
jepa_encoder.jl 
JEPA Encoder -- v2.0
Hierarchical transformer encoder: observation history → latent belief (μ_t, σ_t)

There are three levels: 
Level 1 -- within hour (sub hourly: raw cgm, HR)
Level 2-- within day (hourly: CTX feature)
level 3 -- across days (sleep, menstrual, site)

  [CLS] token → (μ_t, σ_t) latent belief mean and variance

Training: VICReg self-supervised loss (no labels needed)
Backend:  Flux.jl (CPU/Metal/CUDA via gpu() call)

"""


using Flux
using Statistics
using LinearAlgebra 

# ─────────────────────────────────────────────────────────────────
# Transformer Encoder Layer
#one layer = multiattention head + feed forward + layer norm 
#get stacked on N times per hierarchy level
# ─────────────────────────────────────────────────────────────────

struct TransformerEncoderLayer
    attention :: MultiHeadAttention
    feedforward :: Chain
    norm1 :: LayerNorm
    norm2 :: LayerNorm
    dropout :: Dropout
end

#now we need to make the function callable

function (layer:: TransformerEncoderLayer)(x)
    #prem-norm architecture
    #apparently this is more stable than the post norm architecture
    # x shape is expected to be: (d_model, seq_len, batch)

    #now we code th attention block 
    attn_out = layer.attention(layer.norm1(x))
    x = x + layer.dropout(attn_out[1])

    #feed forward block 
    ff_out = layer.feedforward(layer.norm2(x))
    x = x + layer.dropout(ff_out)

    return x
end

Flux.@layer TransformerEncoderLayer

function TransformerEncoderLayer(d_model :: Int, n_heads :: Int, d_ff :: Int, dropout :: Float64=0.1)
    TransformerEncoderLayer(
        MultiHeadAttention(d_model; nheads=n_heads, dropout_prob=dropout),
        Chain(
            Dense(d_model, d_ff,gelu),
            Dense(d_ff,d_model)
        ),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(dropout)
    )
end


# ─────────────────────────────────────────────────────────────────
# Hierarchy Level 1 — Within Hour
# Input:  sub-hourly signals (raw CGM, HR) — 12 tokens per hour
# Output: one hourly summary token per hour
# Small model — fast, low level features
# ─────────────────────────────────────────────────────────────────

struct HourlyEncoder
    input_proj  :: Dense              # project raw signals → d_model
    pos_embed   :: Embedding          # positional encoding (12 positions)
    layers      :: Vector{TransformerEncoderLayer}
    cls_token   :: Vector{Float32}    # learnable [CLS] token
    output_proj :: Dense              # → hourly summary dimension
end

Flux.@layer HourlyEncoder

function HourlyEncoder(
    n_signals   :: Int,      # number of sub-hourly signals
    d_model     :: Int = 32,
    n_heads     :: Int = 2,
    n_layers    :: Int = 1,
    d_ff        :: Int = 64,
    seq_len     :: Int = 12  # 12 × 5-min readings per hour
)
    HourlyEncoder(
        Dense(n_signals, d_model),
        Embedding(seq_len + 1, d_model),   # +1 for CLS position
        [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in 1:n_layers],
        randn(Float32, d_model),            # random init CLS token
        Dense(d_model, d_model)
    )
end

function (enc::HourlyEncoder)(x)
    # x shape: (n_signals, seq_len, batch)
    batch = size(x, 3)

    # project each token to d_model
    x = enc.input_proj(reshape(x, size(x,1), :))
    x = reshape(x, :, 12, batch)   # (d_model, seq_len, batch)

    # prepend CLS token
    cls = repeat(reshape(enc.cls_token, :, 1, 1), 1, 1, batch)
    x = cat(cls, x, dims=2)        # (d_model, seq_len+1, batch)

    # add positional embeddings
    positions = 1:(size(x,2))
    x = x .+ enc.pos_embed(positions)

    # run through transformer layers
    for layer in enc.layers
        x = layer(x)
    end

    # return CLS token as hourly summary
    summary = enc.output_proj(x[:, 1, :])  # (d_model, batch)
    return summary
end

# ─────────────────────────────────────────────────────────────────
# Hierarchy Level 2 — Within Day
# Input:  hourly tokens (from HourlyEncoder) + CTXBuilder features
# Output: one daily summary token
# ─────────────────────────────────────────────────────────────────

struct DailyEncoder
    input_proj  :: Dense
    pos_embed   :: Embedding          # 24 positions (hours)
    layers      :: Vector{TransformerEncoderLayer}
    cls_token   :: Vector{Float32}
    output_proj :: Dense
end

Flux.@layer DailyEncoder

function DailyEncoder(
    d_hourly    :: Int,      # dimension from HourlyEncoder output
    n_ctx_feats :: Int,      # number of CTXBuilder features per hour
    d_model     :: Int = 64,
    n_heads     :: Int = 4,
    n_layers    :: Int = 2,
    d_ff        :: Int = 128,
    seq_len     :: Int = 24  # 24 hours per day
)
    d_input = d_hourly + n_ctx_feats   # combine hourly summary + CTX features
    DailyEncoder(
        Dense(d_input, d_model),
        Embedding(seq_len + 1, d_model),
        [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in 1:n_layers],
        randn(Float32, d_model),
        Dense(d_model, d_model)
    )
end

function (enc::DailyEncoder)(hourly_summaries, ctx_features)
    # hourly_summaries: (d_hourly, 24, batch)
    # ctx_features:     (n_ctx_feats, 24, batch)
    batch = size(hourly_summaries, 3)

    # concatenate hourly summaries with CTX features
    x = cat(hourly_summaries, ctx_features, dims=1)  # (d_input, 24, batch)

    # project to d_model
    x = enc.input_proj(reshape(x, size(x,1), :))
    x = reshape(x, :, 24, batch)

    # prepend CLS token
    cls = repeat(reshape(enc.cls_token, :, 1, 1), 1, 1, batch)
    x = cat(cls, x, dims=2)

    # positional embeddings
    x = x .+ enc.pos_embed(1:size(x,2))

    # transformer layers
    for layer in enc.layers
        x = layer(x)
    end

    return enc.output_proj(x[:, 1, :])   # daily summary (d_model, batch)
end

# ─────────────────────────────────────────────────────────────────
# Hierarchy Level 3 — Across Days
# Input:  daily tokens (from DailyEncoder) + daily signals
#         (sleep, menstrual, site — naturally daily resolution)
# Output: z_t — latent belief vector
# ─────────────────────────────────────────────────────────────────

struct MultiDayEncoder
    input_proj  :: Dense
    pos_embed   :: Embedding
    layers      :: Vector{TransformerEncoderLayer}
    cls_token   :: Vector{Float32}
    μ_head      :: Dense              # mean of latent belief
    σ_head      :: Dense              # log variance of latent belief
end

Flux.@layer MultiDayEncoder

function MultiDayEncoder(
    d_daily       :: Int,
    n_daily_feats :: Int,
    d_model       :: Int = 128,
    n_heads       :: Int = 4,
    n_layers      :: Int = 2,
    d_ff          :: Int = 256,
    z_dim         :: Int = 64,   # 7th argument
    seq_len       :: Int = 14
)
    d_input = d_daily + n_daily_feats
    MultiDayEncoder(
        Dense(d_input, d_model),
        Embedding(seq_len + 1, d_model),
        [TransformerEncoderLayer(d_model, n_heads, d_ff) for _ in 1:n_layers],
        randn(Float32, d_model),
        Dense(d_model, z_dim),        # μ_t
        Dense(d_model, z_dim)         # log σ_t
    )
end

function (enc::MultiDayEncoder)(daily_summaries, daily_features)
    batch = size(daily_summaries, 3)

    x = cat(daily_summaries, daily_features, dims=1)
    x = enc.input_proj(reshape(x, size(x,1), :))
    x = reshape(x, :, size(daily_summaries,2), batch)

    cls = repeat(reshape(enc.cls_token, :, 1, 1), 1, 1, batch)
    x = cat(cls, x, dims=2)
    x = x .+ enc.pos_embed(1:size(x,2))

    for layer in enc.layers
        x = layer(x)
    end

    cls_out = x[:, 1, :]              # (d_model, batch)
    μ = enc.μ_head(cls_out)           # mean of latent belief
    log_σ = clamp.(enc.σ_head(cls_out), -4.0f0, 2.0f0)

    return μ, log_σ
end

# ─────────────────────────────────────────────────────────────────
# Hierarchical JEPA Encoder — full architecture
# Wires all three levels together.
# Input:  observation history (sub-hourly, hourly, daily signals)
# Output: JEPABeliefState (μ_t, σ_t) — latent belief
# ─────────────────────────────────────────────────────────────────

struct HierarchicalJEPAEncoder
    hourly    :: HourlyEncoder
    daily     :: DailyEncoder
    multiday  :: MultiDayEncoder
end

Flux.@layer HierarchicalJEPAEncoder

function HierarchicalJEPAEncoder(
    n_subhourly_signals :: Int,
    n_ctx_features      :: Int,
    n_daily_features    :: Int,   # ← semicolon was wrong here too, should be comma
    z_dim               :: Int = 64
)
    hourly   = HourlyEncoder(n_subhourly_signals)
    daily    = DailyEncoder(32, n_ctx_features)
    multiday = MultiDayEncoder(64, n_daily_features, 128, 4, 2, 256, z_dim)  # positional

    HierarchicalJEPAEncoder(hourly, daily, multiday)
end

function (enc::HierarchicalJEPAEncoder)(
    subhourly  :: AbstractArray,   # (n_subhourly, 12, 24, n_days, batch)
    ctx        :: AbstractArray,   # (n_ctx, 24, n_days, batch)
    daily      :: AbstractArray    # (n_daily, n_days, batch)
)
    n_days = size(subhourly, 4)
    batch  = size(subhourly, 5)

    # ── Level 1: encode each hour independently ───────────────────
    # When there are no sub-hourly signals, skip the HourlyEncoder and
    # use zero summaries — same as the training path in jepa_training.jl.
    d_hourly = length(enc.hourly.cls_token)
    hourly_summaries = if size(subhourly, 1) == 0
        zeros(Float32, d_hourly, 24, n_days, batch)
    else
        out = Array{Float32}(undef, d_hourly, 24, n_days, batch)
        for d in 1:n_days
            for h in 1:24
                x_hour = subhourly[:, :, h, d, :]   # (n_subhourly, 12, batch)
                out[:, h, d, :] = enc.hourly(x_hour)
            end
        end
        out
    end

    # ── Level 2: encode each day independently ────────────────────
    daily_summaries = Array{Float32}(undef, 64, n_days, batch)
    for d in 1:n_days
        h_sum = hourly_summaries[:, :, d, :]    # (32, 24, batch)
        ctx_d = ctx[:, :, d, :]                  # (n_ctx, 24, batch)
        daily_summaries[:, d, :] = enc.daily(h_sum, ctx_d)
    end

    # ── Level 3: encode across days → latent belief ───────────────
    # daily_summaries: (64, n_days, batch)
    # daily features:  (n_daily, n_days, batch)
    μ, log_σ = enc.multiday(daily_summaries, daily)
    μ_vec     = batch == 1 ? vec(μ[:, 1])     : vec(mean(μ, dims=2))
    log_σ_vec = batch == 1 ? vec(log_σ[:, 1]) : vec(mean(log_σ, dims=2))

    # return as JEPABeliefState
    return JEPABeliefState(
        μ           = Float32.(μ_vec),
        log_σ       = Float32.(log_σ_vec),
        entropy     = _jepa_entropy(log_σ_vec),
        obs_log_lik = 0.0f0   # updated separately by anomaly module
    )
end

encode_observation_window(
    enc        :: HierarchicalJEPAEncoder,
    subhourly  :: AbstractArray,
    ctx        :: AbstractArray,
    daily      :: AbstractArray
) = enc(subhourly, ctx, daily)

# ─────────────────────────────────────────────────────────────────
# Entropy of Gaussian latent belief
# H = 0.5 * sum(1 + log(2π) + log_σ²)
# Higher entropy = more uncertain latent belief
# ─────────────────────────────────────────────────────────────────

function _jepa_entropy(log_σ::AbstractArray)::Float32
    return 0.5f0 * sum(1.0f0 .+ log(2π) .+ 2.0f0 .* log_σ)
end


# ─────────────────────────────────────────────────────────────────
# VICReg Loss
# Prevents representational collapse during self-supervised training.
# Three terms — invariance + variance + covariance.
#
# L = λ·L_inv + μ·L_var + ν·L_cov
# Default weights from VICReg paper: λ=25, μ=25, ν=1
# ─────────────────────────────────────────────────────────────────

function vicreg_loss(
    z_a      :: AbstractMatrix,   # (z_dim, batch) — encoded current state
    z_b      :: AbstractMatrix,   # (z_dim, batch) — encoded next state
    z_b_pred :: AbstractMatrix;   # (z_dim, batch) — predicted next state
    λ        :: Float32 = 25.0f0, # invariance weight
    μ        :: Float32 = 25.0f0, # variance weight
    ν        :: Float32 = 1.0f0   # covariance weight
) :: Float32

    # ── Invariance loss ───────────────────────────────────────────
    # predicted latent should match actual next latent
    # MSE between prediction and reality
    L_inv = mean((z_b_pred .- z_b).^2)

    # ── Variance loss ─────────────────────────────────────────────
    # each dimension should have std ≥ 1 across the batch
    # if any dimension collapses to constant → std → 0 → penalty
    # hinge loss: max(0, 1 - std(dimension))
    std_a = sqrt.(var(z_a, dims=2) .+ 1f-4)   # (z_dim,) — std per dimension
    std_b = sqrt.(var(z_b, dims=2) .+ 1f-4)
    L_var = mean(relu.(1.0f0 .- std_a)) + mean(relu.(1.0f0 .- std_b))

    # ── Covariance loss ───────────────────────────────────────────
    # dimensions should be decorrelated — each captures different info
    # penalize off-diagonal entries of covariance matrix
    batch = size(z_a, 2)
    z_a_centered = z_a .- mean(z_a, dims=2)
    z_b_centered = z_b .- mean(z_b, dims=2)

    cov_a = (z_a_centered * z_a_centered') ./ (batch - 1)   # (z_dim, z_dim)
    cov_b = (z_b_centered * z_b_centered') ./ (batch - 1)

    # zero out diagonal — we only penalize off-diagonal correlations
    z_dim = size(z_a, 1)
    mask = 1.0f0 .- Matrix{Float32}(I, z_dim, z_dim)
    L_cov = sum((cov_a .* mask).^2) / z_dim +
            sum((cov_b .* mask).^2) / z_dim

    # ── Total VICReg loss ─────────────────────────────────────────
    return λ * L_inv + μ * L_var + ν * L_cov
end
