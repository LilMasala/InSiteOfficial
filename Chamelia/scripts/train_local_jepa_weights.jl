using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "WorldModule", "WorldModule.jl"))
include(joinpath(ROOT, "src", "Perception", "Perception.jl"))

function parse_args(args::Vector{String})
    out_dir = joinpath(ROOT, "tmp", "jepa_local_weights")
    n_epochs = 5
    batch_size = 8
    seed = 42

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--out-dir"
            i += 1
            out_dir = args[i]
        elseif arg == "--epochs"
            i += 1
            n_epochs = parse(Int, args[i])
        elseif arg == "--batch-size"
            i += 1
            batch_size = parse(Int, args[i])
        elseif arg == "--seed"
            i += 1
            seed = parse(Int, args[i])
        else
            error("unknown argument: $arg")
        end
        i += 1
    end

    return out_dir, n_epochs, batch_size, seed
end

function main(args::Vector{String})
    out_dir, n_epochs, batch_size, seed = parse_args(args)
    Random.seed!(seed)

    dataset = Perception.SyntheticTrainingDataset(32, 30, 7, 2, 12, 3)
    encoder = Perception.HierarchicalJEPAEncoder(2, 12, 3)
    predictor = WorldModule.JEPAPredictor()

    Perception.train_encoder!(encoder, predictor, dataset; n_epochs=n_epochs, batch_size=batch_size)
    mkpath(out_dir)
    Perception.save_jepa_weights(encoder, predictor, out_dir)

    println("Saved JEPA weights to: ", out_dir)
    println("Training loss: ", Perception.LAST_JEPA_TRAINING_LOSS[])
end

main(ARGS)
