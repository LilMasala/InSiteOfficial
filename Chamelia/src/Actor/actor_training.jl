"""
actor_training.jl
Offline actor training over latent Memory records.

This augments the latent JEPA Actor path; it does not replace the online
latent search path. The learned policy is only marked ready after enough
Memory has accumulated.
"""

function _actor_policy_input(
    latent_snapshot :: AbstractVector,
    action_features :: AbstractVector
) :: Vector{Float32}
    latent = Float32.(vec(latent_snapshot))
    latent = length(latent) >= 64 ? latent[1:64] : vcat(latent, zeros(Float32, 64 - length(latent)))
    action = Float32.(vec(action_features))
    action = length(action) >= 8 ? action[1:8] : vcat(action, zeros(Float32, 8 - length(action)))
    return vcat(latent, action)
end

function train_actor_cql!(
    mem     :: MemoryBuffer;
    n_epochs::Int = 200,
    lr      :: Float64 = 1e-3,
    α_cql   :: Float64 = 0.1
) :: Nothing
    training_records = filter(
        r -> !isnothing(r.latent_snapshot) && !isnothing(r.realized_cost),
        mem.records
    )

    length(training_records) < 100 && return nothing

    mean_cost = mean(Float64(r.realized_cost) for r in training_records)
    std_cost  = std(Float64(r.realized_cost) for r in training_records) + 1e-6

    X = Vector{Vector{Float32}}(undef, length(training_records))
    Y = Vector{Vector{Float32}}(undef, length(training_records))
    rewards = Vector{Float32}(undef, length(training_records))

    for (i, rec) in enumerate(training_records)
        action_features = action_to_features(rec.action)
        X[i] = _actor_policy_input(rec.latent_snapshot, action_features)
        Y[i] = Float32.(action_features)
        rewards[i] = Float32((mean_cost - Float64(rec.realized_cost)) / std_cost)
    end

    if isnothing(OFFLINE_RL_MODEL.network)
        OFFLINE_RL_MODEL.network = Chain(
            Dense(64 + 8, 64, relu),
            Dense(64, 64, relu),
            Dense(64, 8)
        )
    end

    opt = Flux.setup(Adam(lr), OFFLINE_RL_MODEL.network)

    for _ in 1:n_epochs
        for (x_i, y_i, reward_i) in zip(X, Y, rewards)
            loss, grads = Flux.withgradient(OFFLINE_RL_MODEL.network) do model
                predicted_action = model(x_i)
                mse = mean((predicted_action .- y_i).^2)

                random_action = rand(Float32, 8)
                q_random = -mean((random_action .- y_i).^2)
                q_data = reward_i
                cql_penalty = α_cql * max(0.0f0, Float32(q_random - q_data))

                mse + cql_penalty
            end
            Flux.update!(opt, OFFLINE_RL_MODEL.network, grads[1])
        end
    end

    OFFLINE_RL_MODEL.n_trained = length(training_records)
    OFFLINE_RL_MODEL.is_ready = OFFLINE_RL_MODEL.n_trained >= OFFLINE_RL_MODEL.min_samples
    return nothing
end
