using Test
using Distributions
using HTTP
using JSON3
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "WorldModule", "WorldModule.jl"))
include(joinpath(ROOT, "src", "Cost", "Cost.jl"))
include(joinpath(ROOT, "src", "Actor", "Actor.jl"))
include(joinpath(ROOT, "src", "Memory", "Memory.jl"))
include(joinpath(ROOT, "src", "Perception", "Perception.jl"))
include(joinpath(ROOT, "src", "Configurator", "Configurator.jl"))
include(joinpath(ROOT, "src", "Chamelia.jl"))
include(joinpath(ROOT, "server.jl"))

function make_test_prior()
    TwinPrior(
        trust_growth_dist = Beta(2, 5),
        trust_decay_dist = Beta(2, 8),
        burnout_sensitivity_dist = Beta(2, 5),
        engagement_decay_dist = Beta(2, 5),
        physical_priors = Dict{Symbol, Distribution}(),
        persona_label = "test"
    )
end

function make_test_twin()
    prior = make_test_prior()
    posterior = TwinPosterior(
        trust_growth_rate = 0.2,
        trust_decay_rate = 0.1,
        burnout_sensitivity = 0.1,
        engagement_decay = 0.05,
        physical = Dict{Symbol, Float64}(),
        last_updated_day = 0,
        n_observations = 0
    )

    return DigitalTwin(prior, posterior, 0.1)
end

function make_test_config()
    config = Configurator.initialize_config(UserPreferences())
    config.φ_world.N_roll = 8
    config.φ_act.N_search = 8
    return config
end

@testset "Module Load" begin
    @test isdefined(Main, :WorldModule)
    @test isdefined(Main, :Cost)
    @test isdefined(Main, :Actor)
    @test isdefined(Main, :Memory)
    @test isdefined(Main, :Perception)
    @test isdefined(Main, :Configurator)
    @test isdefined(Main, :Chamelia)
end

function server_request(method::String, path::String, payload=nothing)
    headers = payload === nothing ? Pair{String, String}[] : ["Content-Type" => "application/json"]
    body = payload === nothing ? UInt8[] : Vector{UInt8}(codeunits(JSON3.write(payload)))
    return Main.ChameliaServer.handle_request(HTTP.Request(method, path, headers, body))
end

function server_json(response::HTTP.Response)
    return JSON3.read(String(response.body), Dict{String, Any})
end

@testset "HTTP Server Smoke" begin
    Main.ChameliaServer.reset_patient_cache!()
    Main.ChameliaServer.set_state_backend!(Main.ChameliaServer.InMemoryStateBackend())

    health = server_request("GET", "/health")
    @test health.status == 200
    @test server_json(health)["status"] == "ok"

    patient_id = "server-smoke"
    init = server_request(
        "POST",
        "/chamelia_initialize_patient",
        Dict(
            "patient_id" => patient_id,
            "preferences" => Dict(
                "aggressiveness" => 0.4,
                "hypoglycemia_fear" => 0.8,
                "burden_sensitivity" => 0.5,
                "persona" => "test"
            )
        )
    )
    @test init.status == 200
    @test server_json(init)["status"]["n_days"] == 0

    observe = server_request(
        "POST",
        "/chamelia_observe",
        Dict(
            "patient_id" => patient_id,
            "timestamp" => 1.0,
            "signals" => Dict("bg_avg" => 110.0, "tir_7d" => 0.7)
        )
    )
    @test observe.status == 200

    step = server_request(
        "POST",
        "/chamelia_step",
        Dict(
            "patient_id" => patient_id,
            "timestamp" => 2.0,
            "signals" => Dict("bg_avg" => 112.0, "tir_7d" => 0.72, "pct_low_7d" => 0.01)
        )
    )
    step_payload = server_json(step)
    @test step.status == 200
    @test step_payload["status"]["n_days"] == 1
    @test !isnothing(step_payload["rec_id"])

    shadow_outcome = server_request(
        "POST",
        "/chamelia_record_outcome",
        Dict(
            "patient_id" => patient_id,
            "rec_id" => step_payload["rec_id"],
            "signals" => Dict("bg_avg" => 108.0, "tir_7d" => 0.76, "pct_low_7d" => 0.0),
            "cost" => 0.24
        )
    )
    @test shadow_outcome.status == 200

    free = server_request("POST", "/chamelia_free_patient", Dict("patient_id" => patient_id))
    @test free.status == 200
    @test server_json(free)["ok"] == true

    reinit = server_request("POST", "/chamelia_initialize_patient", Dict("patient_id" => patient_id))
    reinit_payload = server_json(reinit)
    @test reinit.status == 200
    @test reinit_payload["status"]["n_days"] == 1

    status = server_request("POST", "/chamelia_graduation_status", Dict("patient_id" => patient_id))
    @test status.status == 200
    status_payload = server_json(status)["status"]
    @test status_payload["n_days"] == 1
    @test haskey(status_payload, "configurator_mode")
    @test haskey(status_payload, "jepa_weights_loaded")
    @test haskey(status_payload, "jepa_active")
    @test haskey(status_payload, "belief_mode")
    @test haskey(status_payload, "familiarity")
    @test haskey(status_payload, "concordance")
    @test haskey(status_payload, "calibration")
    @test haskey(status_payload, "belief_entropy")

    free_again = server_request("POST", "/chamelia_free_patient", Dict("patient_id" => patient_id))
    @test free_again.status == 200

    load = server_request("POST", "/chamelia_load_patient", Dict("patient_id" => patient_id))
    @test load.status == 200
    @test server_json(load)["status"]["n_days"] == 1
end

@testset "Questionnaire Physical Priors" begin
    Main.ChameliaServer.reset_patient_cache!()
    Main.ChameliaServer.set_state_backend!(Main.ChameliaServer.InMemoryStateBackend())

    response = server_request(
        "POST",
        "/chamelia_initialize_patient",
        Dict(
            "patient_id" => "questionnaire-priors",
            "preferences" => Dict(
                "persona" => "questionnaire_derived",
                "physical_priors" => Dict(
                    "isf_multiplier" => [1.16, 0.10],
                    "sleep_regularity" => [0.72, 0.08]
                )
            )
        )
    )

    @test response.status == 200

    system = Main.ChameliaServer.PATIENTS["questionnaire-priors"]
    prior = system.twin.prior

    @test prior.persona_label == "questionnaire_derived"
    @test haskey(prior.physical_priors, :isf_multiplier)
    @test haskey(prior.physical_priors, :sleep_regularity)
    @test prior.physical_priors[:isf_multiplier] isa Normal
    @test mean(prior.physical_priors[:isf_multiplier]) ≈ 1.16 atol=1e-6
    @test std(prior.physical_priors[:isf_multiplier]) ≈ 0.10 atol=1e-6
    @test mean(prior.physical_priors[:sleep_regularity]) ≈ 0.72 atol=1e-6
end

@testset "Schedule-Aware Types And Server Parsing" begin
    caps = Main.ChameliaServer._connected_app_capabilities(Dict(
        "connected_app_capabilities" => Dict(
            "app_id" => "insite",
            "supports_scalar_schedule" => true,
            "supports_piecewise_schedule" => true,
            "max_segments" => 8,
            "min_segment_duration_min" => 120,
            "max_segments_addable" => 2,
            "level_1_enabled" => true,
            "level_2_enabled" => false,
            "level_3_enabled" => false,
            "structural_change_requires_consent" => true
        )
    ))
    @test caps.app_id == "insite"
    @test caps.supports_piecewise_schedule
    @test caps.max_segments == 8

    app_state = Main.ChameliaServer._connected_app_state(Dict(
        "connected_app_state" => Dict(
            "schedule_version" => "v1",
            "allow_structural_recommendations" => false,
            "current_segments" => [
                Dict("segment_id" => "overnight", "start_min" => 0, "end_min" => 360, "isf" => 45.0, "cr" => 12.0, "basal" => 0.8),
                Dict("segment_id" => "morning", "start_min" => 360, "end_min" => 720, "isf" => 42.0, "cr" => 10.0, "basal" => 0.9),
                Dict("segment_id" => "afternoon", "start_min" => 720, "end_min" => 1080, "isf" => 43.0, "cr" => 11.0, "basal" => 0.85),
                Dict("segment_id" => "evening", "start_min" => 1080, "end_min" => 1440, "isf" => 46.0, "cr" => 12.5, "basal" => 0.75)
            ]
        )
    ))
    @test app_state.schedule_version == "v1"
    @test length(app_state.current_segments) == 4
    @test app_state.current_segments[1].segment_id == "overnight"

    action = Main.ScheduledAction(
        1,
        Main.parameter_adjustment,
        [Main.SegmentDelta(segment_id="morning", parameter_deltas=Dict(:isf_delta => 0.05, :basal_delta => -0.03))],
        Main.StructureEdit[],
    )
    @test !Main.is_null(action)
    @test Main.magnitude(action) > 0.0

    serialized = Main.ChameliaServer._serialize_action(action)
    @test serialized["kind"] == "scheduled"
    @test serialized["level"] == 1
    @test serialized["family"] == "parameter_adjustment"
    @test length(serialized["segment_deltas"]) == 1

    pkg = Main.RecommendationPackage(
        action = action,
        predicted_improvement = 0.1,
        confidence = 0.8,
        confidence_breakdown = (
            familiarity = 0.84,
            concordance = 0.79,
            calibration = 0.75,
            effect_support = 0.88,
            selection_penalty = 1.0,
            final_confidence = 0.8,
        ),
        alternatives = AbstractAction[],
        effect_size = 0.05,
        cvar_value = -0.2,
        burnout_attribution = nothing,
        predicted_outcomes = (
            baseline_tir = 0.68,
            treated_tir = 0.74,
            delta_tir = 0.06,
            baseline_pct_low = 0.02,
            treated_pct_low = 0.018,
            delta_pct_low = -0.002,
            baseline_pct_high = 0.30,
            treated_pct_high = 0.24,
            delta_pct_high = -0.06,
            baseline_bg_avg = 168.0,
            treated_bg_avg = 151.0,
            delta_bg_avg = -17.0,
            baseline_cost_mean = 0.42,
            treated_cost_mean = 0.33,
            delta_cost_mean = -0.09,
            baseline_cvar = -0.1,
            treated_cvar = -0.2,
            delta_cvar = -0.1,
        ),
        predicted_uncertainty = (
            tir_std = 0.03,
            pct_low_std = 0.004,
            pct_high_std = 0.05,
            bg_avg_std = 11.0,
            cost_std = 0.08,
        ),
        action_level = 1,
        action_family = Main.parameter_adjustment,
        segment_summaries = [(
            segment_id = "morning",
            label = "360–720 min",
            parameter_summaries = Dict(
                "isf_delta" => "isf_delta +5.0%",
                "basal_delta" => "basal_delta -3.0%",
            ),
        )],
        structure_summaries = String[],
    )
    serialized_pkg = Main.ChameliaServer._serialize_recommendation(pkg)
    @test serialized_pkg["action_level"] == 1
    @test serialized_pkg["action_family"] == "parameter_adjustment"
    @test length(serialized_pkg["segment_summaries"]) == 1
    @test serialized_pkg["predicted_outcomes"]["delta_tir"] ≈ 0.06 atol=1e-6
    @test serialized_pkg["confidence_breakdown"]["familiarity"] ≈ 0.84 atol=1e-6
    @test serialized_pkg["predicted_uncertainty"]["tir_std"] ≈ 0.03 atol=1e-6

    level2_caps = Main.ConnectedAppCapabilities(
        app_id = "insite",
        supports_scalar_schedule = true,
        supports_piecewise_schedule = true,
        max_segments = 8,
        min_segment_duration_min = 120,
        max_segments_addable = 2,
        level_1_enabled = true,
        level_2_enabled = true,
        level_3_enabled = false,
        structural_change_requires_consent = true,
    )
    eligible_state = Main.ConnectedAppState(
        schedule_version = "v1",
        current_segments = app_state.current_segments,
        allow_structural_recommendations = true,
        allow_continuous_schedule = false,
    )
    ineligible_state = Main.ConnectedAppState(
        schedule_version = "v1",
        current_segments = app_state.current_segments,
        allow_structural_recommendations = false,
        allow_continuous_schedule = false,
    )
    belief = Main.GaussianBeliefState(
        x̂_phys = Dict{Symbol, Float64}(),
        Σ_phys = Dict{Symbol, Float64}(),
        x̂_trust = 0.8,
        σ_trust = 0.1,
        x̂_burnout = 0.1,
        σ_burnout = 0.1,
        x̂_engagement = 0.8,
        σ_engagement = 0.1,
        x̂_burden = 0.1,
        σ_burden = 0.1,
        entropy = 0.2,
        obs_log_lik = 0.0,
    )
    @test Main.Actor._level2_eligible(belief, level2_caps, ineligible_state, 40) == false
    @test Main.Actor._level2_eligible(belief, level2_caps, eligible_state, 40) == true
    @test !isempty(Main.Actor.generate_structure_edit_candidates(make_test_config(), level2_caps, eligible_state))
end

@testset "Schedule-Aware Server Smoke" begin
    Main.ChameliaServer.reset_patient_cache!()
    Main.ChameliaServer.set_state_backend!(Main.ChameliaServer.InMemoryStateBackend())

    patient_id = "schedule-aware-smoke"

    init = server_request(
        "POST",
        "/chamelia_initialize_patient",
        Dict(
            "patient_id" => patient_id,
            "preferences" => Dict(
                "aggressiveness" => 0.45,
                "hypoglycemia_fear" => 0.75,
                "burden_sensitivity" => 0.55,
                "persona" => "default"
            )
        )
    )
    @test init.status == 200

    capabilities = Dict(
        "app_id" => "insite",
        "supports_scalar_schedule" => true,
        "supports_piecewise_schedule" => true,
        "max_segments" => 8,
        "min_segment_duration_min" => 120,
        "max_segments_addable" => 2,
        "level_1_enabled" => true,
        "level_2_enabled" => true,
        "level_3_enabled" => false,
        "structural_change_requires_consent" => true
    )

    app_state = Dict(
        "schedule_version" => "v1",
        "allow_structural_recommendations" => true,
        "current_segments" => [
            Dict("segment_id" => "overnight", "start_min" => 0, "end_min" => 360, "isf" => 45.0, "cr" => 12.0, "basal" => 0.80),
            Dict("segment_id" => "morning", "start_min" => 360, "end_min" => 720, "isf" => 42.0, "cr" => 10.0, "basal" => 0.90),
            Dict("segment_id" => "afternoon", "start_min" => 720, "end_min" => 1080, "isf" => 43.0, "cr" => 11.0, "basal" => 0.85),
            Dict("segment_id" => "evening", "start_min" => 1080, "end_min" => 1440, "isf" => 46.0, "cr" => 12.5, "basal" => 0.75)
        ]
    )

    first_step = server_request(
        "POST",
        "/chamelia_step",
        Dict(
            "patient_id" => patient_id,
            "timestamp" => 1.0,
            "signals" => Dict("bg_avg" => 118.0, "tir_7d" => 0.71, "pct_low_7d" => 0.01),
            "connected_app_capabilities" => capabilities,
            "connected_app_state" => app_state
        )
    )
    first_payload = server_json(first_step)
    @test first_step.status == 200
    @test first_payload["status"]["n_days"] == 1
    @test haskey(first_payload, "recommendation")

    save = server_request("POST", "/chamelia_save_patient", Dict("patient_id" => patient_id))
    @test save.status == 200

    free = server_request("POST", "/chamelia_free_patient", Dict("patient_id" => patient_id))
    @test free.status == 200

    load = server_request("POST", "/chamelia_load_patient", Dict("patient_id" => patient_id))
    load_payload = server_json(load)
    @test load.status == 200
    @test load_payload["status"]["n_days"] == 1

    second_step = server_request(
        "POST",
        "/chamelia_step",
        Dict(
            "patient_id" => patient_id,
            "timestamp" => 2.0,
            "signals" => Dict("bg_avg" => 121.0, "tir_7d" => 0.73, "pct_low_7d" => 0.01),
            "connected_app_capabilities" => capabilities,
            "connected_app_state" => app_state
        )
    )
    second_payload = server_json(second_step)
    @test second_step.status == 200
    @test second_payload["status"]["n_days"] == 2
    @test haskey(second_payload, "recommendation")
end

@testset "InSite Simulator Plugin" begin
    Random.seed!(7)

    sim = Main.InSiteSimulator("athlete"; seed=7)
    prior = make_test_prior()
    Main.WorldModule.register_priors!(sim, prior)

    @test haskey(prior.physical_priors, :isf_multiplier)
    @test haskey(prior.physical_priors, :cr_multiplier)
    @test haskey(prior.physical_priors, :basal_multiplier)
    @test mean(prior.physical_priors[:isf_multiplier]) > 1.0

    noise = initialize_noise()
    Main.WorldModule.register_noise!(sim, noise)
    @test haskey(noise.physical_noise, :bg_noise)
    @test haskey(noise.physical_noise, :cgm_lag)

    phys = Dict{Symbol, Float64}(label => mean(dist) for (label, dist) in prior.physical_priors)
    state = PatientState(
        phys = PhysState(phys),
        psy = PsyState(
            trust = ScalarTrust(0.6),
            burden = ScalarBurden(0.2),
            engagement = ScalarEngagement(0.7),
            burnout = ScalarBurnout(0.1)
        )
    )
    action = Actor.CandidateAction(Dict(:isf_delta => 0.05, :cr_delta => -0.02, :basal_delta => 0.01))
    next_state = Main.WorldModule.sim_step!(sim, state, action, sample_noise(noise))
    obs = Main.WorldModule.sim_observe(sim, next_state)

    @test next_state.phys.variables[:isf_multiplier] > state.phys.variables[:isf_multiplier]
    @test Main.WorldModule.action_dimensions(sim) == [:isf_delta, :cr_delta, :basal_delta]
    @test Main.WorldModule.safety_thresholds(sim) == Dict(:pct_low_max => 0.04, :pct_high_max => 0.25)
    @test haskey(obs.signals, :bg_avg)
    @test haskey(obs.signals, :tir_7d)
    @test haskey(obs.signals, :pct_low_7d)

    frustration = Main.WorldModule.compute_frustration(
        sim,
        Dict{Symbol, Any}(:pct_low_7d => 0.03, :tir_7d => 0.70)
    )
    @test frustration ≈ 0.18 atol=1e-6

    cost = compute_physical_cost(
        sim,
        Dict{Symbol, Any}(
            :pct_low_7d => 0.03,
            :pct_high_7d => 0.20,
            :tir_7d => 0.70,
            :bg_cv => 0.32
        ),
        Dict(:w_low => 5.0, :w_high => 1.0, :w_var => 0.5, :w_tir => 1.0)
    )
    @test cost ≈ -0.19 atol=1e-6

    safe_rollout = [Dict{Symbol, Any}(:pct_low_7d => 0.02, :pct_high_7d => 0.15)]
    unsafe_rollout = [Dict{Symbol, Any}(:pct_low_7d => 0.05, :pct_high_7d => 0.15)]
    thresholds = Main.WorldModule.safety_thresholds(sim)
    @test Main.Actor.check_safety(sim, safe_rollout, thresholds)
    @test !Main.Actor.check_safety(sim, unsafe_rollout, thresholds)

    scheduled_action = Main.ScheduledAction(
        1,
        Main.parameter_adjustment,
        [
            Main.SegmentSurface(segment_id="overnight", start_min=0, end_min=720, parameter_values=Dict(:isf => 42.0, :cr => 10.0, :basal => 0.8)),
            Main.SegmentSurface(segment_id="day", start_min=720, end_min=1440, parameter_values=Dict(:isf => 48.0, :cr => 12.0, :basal => 0.7)),
        ],
        [Main.SegmentDelta(segment_id="day", parameter_deltas=Dict(:isf => 0.10, :basal => -0.05))],
        Main.StructureEdit[],
    )
    scheduled_state = Main.WorldModule.sim_step!(sim, state, scheduled_action, sample_noise(noise))
    @test scheduled_state.phys.variables[:isf_multiplier] != state.phys.variables[:isf_multiplier]

    Main.ChameliaServer.reset_patient_cache!()
    Main.ChameliaServer.set_state_backend!(Main.ChameliaServer.InMemoryStateBackend())
    server_request(
        "POST",
        "/chamelia_initialize_patient",
        Dict(
            "patient_id" => "insite-sim",
            "preferences" => Dict("persona" => "athlete")
        )
    )
    @test Main.ChameliaServer.PATIENTS["insite-sim"].sim isa Main.InSiteSimulator
end

@testset "Latent JEPA Path" begin
    config = make_test_config()
    twin = make_test_twin()
    encoder = Perception.HierarchicalJEPAEncoder(2, 2, 2, 64)
    params = Perception.JEPAInferenceParams(
        encoder,
        [:cgm, :hr],
        [:stress, :sleep],
        [:mood, :site]
    )
    mem = MemoryBuffer()
    obs = Observation(
        timestamp = 1.0,
        signals = Dict{Symbol, Any}(
            :cgm => 110.0,
            :hr => 65.0,
            :stress => 0.3,
            :sleep => 0.7,
            :mood => 0.2,
            :site => 1.0
        )
    )

    belief0 = Perception.initialize_belief(twin.prior, JEPABeliefEstimator())
    belief1 = Perception.update_belief(
        belief0,
        obs,
        NullAction(),
        twin,
        JEPABeliefEstimator(),
        mem,
        params,
        config
    )

    @test length(belief1.μ) == 64
    @test isfinite(Float64(belief1.entropy))

    action = Actor.CandidateAction(Dict(:isf => 0.05, :basal => -0.02))
    rollouts = WorldModule.run_latent_rollouts(
        belief1,
        action,
        WorldModule.JEPA_PREDICTOR,
        config
    )
    energies = Cost.compute_energies(rollouts, Cost.ZeroCritic(), config)
    results = Actor.search_actions(
        GradientSearch(),
        belief1,
        twin,
        WorldModule.MockSimulator(),
        initialize_noise(),
        Cost.ZeroCritic(),
        config,
        [:isf, :basal]
    )

    @test length(rollouts) == config.φ_world.N_roll
    @test all(isfinite, energies)
    @test !isempty(results)

    scheduled_action = Main.ScheduledAction(
        1,
        Main.parameter_adjustment,
        [
            Main.SegmentSurface(segment_id="overnight", start_min=0, end_min=720, parameter_values=Dict(:isf => 42.0, :cr => 10.0, :basal => 0.8)),
            Main.SegmentSurface(segment_id="day", start_min=720, end_min=1440, parameter_values=Dict(:isf => 48.0, :cr => 12.0, :basal => 0.7)),
        ],
        [Main.SegmentDelta(segment_id="day", parameter_deltas=Dict(:isf => 0.10, :basal => -0.05))],
        Main.StructureEdit[],
    )
    scheduled_features = Main.WorldModule.action_to_features(scheduled_action)
    scheduled_rollouts = Main.WorldModule.run_latent_rollouts(
        belief1,
        scheduled_action,
        Main.WorldModule.JEPA_PREDICTOR,
        config
    )
    scheduled_energies = Cost.compute_energies(scheduled_rollouts, Cost.ZeroCritic(), config)

    @test length(scheduled_features) == 8
    @test scheduled_features[4] == 1.0f0
    @test any(>(0.0f0), scheduled_features[6:8])
    @test any(<(0.0f0), scheduled_features[6:8])
    @test length(scheduled_rollouts) == config.φ_world.N_roll
    @test all(isfinite, scheduled_energies)

    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true
    )
    pkg, reason, _ = Actor.decide(
        belief1,
        twin,
        WorldModule.MockSimulator(),
        initialize_noise(),
        Cost.ZeroCritic(),
        epistemic,
        config,
        [:isf, :basal],
        Dict(:burnout => 0.98, :physical_risk => 10.0)
    )

    app_caps = Main.ConnectedAppCapabilities(
        app_id = "insite",
        supports_scalar_schedule = true,
        supports_piecewise_schedule = true,
        max_segments = 8,
        min_segment_duration_min = 120,
        max_segments_addable = 2,
        level_1_enabled = true,
        level_2_enabled = false,
        level_3_enabled = false,
        structural_change_requires_consent = true,
    )
    app_state = Main.ConnectedAppState(
        schedule_version = "latent-v1",
        current_segments = scheduled_action.segments,
        allow_structural_recommendations = false,
        allow_continuous_schedule = false,
    )
    scheduled_pkg, scheduled_reason, _ = Actor.decide(
        belief1,
        twin,
        WorldModule.MockSimulator(),
        initialize_noise(),
        Cost.ZeroCritic(),
        epistemic,
        config,
        [:isf, :basal],
        Dict(:burnout => 0.98, :physical_risk => 10.0),
        12,
        app_caps,
        app_state,
    )

    @test reason in (:recommended, :burnout_risk_exceeded, :no_survivors, :safety_violated, :effect_size_insufficient)
    @test (pkg === nothing) || isa(pkg, RecommendationPackage)
    @test scheduled_reason in (:recommended, :burnout_risk_exceeded, :no_survivors, :safety_violated, :effect_size_insufficient)
    @test (scheduled_pkg === nothing) || isa(scheduled_pkg, RecommendationPackage)
    @test isnothing(scheduled_pkg) || scheduled_pkg.action isa Main.ScheduledAction
end

@testset "Shadow Exploration Bootstrap" begin
    config = make_test_config()
    config.φ_act.δ_min_effect = 1.0e9

    belief = Main.GaussianBeliefState(
        x̂_phys = Dict(:bg => 100.0),
        Σ_phys = Dict(:bg => 4.0),
        x̂_trust = 0.7,
        σ_trust = 0.1,
        x̂_burnout = 0.1,
        σ_burnout = 0.05,
        x̂_engagement = 0.7,
        σ_engagement = 0.1,
        x̂_burden = 0.2,
        σ_burden = 0.05,
        entropy = 0.1,
        obs_log_lik = 0.0,
    )
    twin = make_test_twin()
    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true,
    )

    pkg, reason, _ = Actor.decide(
        belief,
        twin,
        WorldModule.MockSimulator(),
        initialize_noise(),
        Cost.ZeroCritic(),
        epistemic,
        config,
        [:isf, :cr, :basal],
        Dict(:burnout => 0.98, :physical_risk => 10.0),
        config.φ_world.H_med,
    )

    @test reason in (:shadow_explore, :burnout_risk_exceeded, :no_survivors)
    if reason == :shadow_explore
        @test pkg !== nothing
        @test !is_null(pkg.action)
        @test pkg.confidence < 1.0
    end
end

@testset "Memory Latent Snapshot" begin
    config = make_test_config()
    belief = JEPABeliefState(
        μ = zeros(Float32, 64),
        log_σ = zeros(Float32, 64),
        entropy = 0.0f0,
        obs_log_lik = 0.0f0
    )
    mem = MemoryBuffer()
    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true
    )
    psy = PsyState(
        trust = ScalarTrust(0.6),
        burden = ScalarBurden(0.2),
        engagement = ScalarEngagement(0.7),
        burnout = ScalarBurnout(0.1)
    )

    rec_id = Memory.store_record!(
        mem,
        1,
        belief,
        Actor.CandidateAction(Dict(:isf => 0.05)),
        epistemic,
        config,
        psy
    )
    rec = Memory.get_record(mem, rec_id)

    @test rec !== nothing
    @test rec.latent_snapshot !== nothing
    @test rec.latent_μ_at_rec !== nothing
    @test length(rec.latent_μ_at_rec) == 64
    @test Memory.current_critic(mem) isa Cost.ZeroCritic
end

@testset "Shadow Scorecard Math" begin
    mem = MemoryBuffer()
    config = make_test_config()
    belief = GaussianBeliefState(
        x̂_phys = Dict(:bg => 100.0),
        Σ_phys = Dict(:bg => 4.0),
        x̂_trust = 0.6,
        σ_trust = 0.1,
        x̂_burnout = 0.1,
        σ_burnout = 0.05,
        x̂_engagement = 0.7,
        σ_engagement = 0.1,
        x̂_burden = 0.2,
        σ_burden = 0.05,
        entropy = 0.1,
        obs_log_lik = 0.0,
    )
    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true,
    )
    psy = PsyState(
        trust = ScalarTrust(0.6),
        burden = ScalarBurden(0.2),
        engagement = ScalarEngagement(0.7),
        burnout = ScalarBurnout(0.1),
    )

    for day in 1:5
        rec_id = Memory.store_hold!(mem, day, belief, epistemic, :no_survivors, config, psy)
        Memory.store_outcome!(mem, rec_id, nothing, Dict{Symbol, Any}(), 0.20 + 0.005 * day)
    end

    hold_id = Memory.store_hold!(mem, 6, belief, epistemic, :no_survivors, config, psy)
    Memory.store_outcome!(mem, hold_id, nothing, Dict{Symbol, Any}(), 0.225)
    Memory.score_record!(mem, hold_id)
    hold_rec = Memory.get_record(mem, hold_id)
    @test hold_rec !== nothing
    @test hold_rec.shadow_delta_score > 0.0

    action_id = Memory.store_record!(
        mem,
        7,
        belief,
        Actor.CandidateAction(Dict(:isf => 0.05)),
        epistemic,
        config,
        psy;
        predicted_cvar = 0.18,
    )
    Memory.store_outcome!(mem, action_id, nothing, Dict{Symbol, Any}(), 0.26)
    Memory.score_record!(mem, action_id)
    action_rec = Memory.get_record(mem, action_id)
    @test action_rec !== nothing
    @test action_rec.shadow_delta_score ≈ 0.08 atol=1e-6
end

@testset "Relative Safety Gate" begin
    thresholds = Dict{Symbol, Float64}(
        :pct_low_max => 0.04,
        :pct_high_max => 0.25,
        :pct_low_hard_max => 0.10,
        :pct_high_hard_max => 0.40,
        :catastrophic_relief_tol => 0.05,
    )
    psy = PsyState(
        trust = ScalarTrust(0.6),
        burden = ScalarBurden(0.2),
        engagement = ScalarEngagement(0.7),
        burnout = ScalarBurnout(0.1),
    )
    state = PatientState(PhysState(Dict(:bg => 100.0)), psy)
    mkrollout(low, high) = RolloutResult(
        action = NullAction(),
        initial_psy = psy,
        terminal_state = state,
        terminal_psy = psy,
        total_cost = 0.0,
        psy_trajectory = PsyState[],
        phys_signals = [Dict{Symbol, Any}(:pct_low_7d => low, :pct_high_7d => high)],
    )

    treated_better = [mkrollout(0.05, 0.22), mkrollout(0.03, 0.20)]
    baseline_worse = [mkrollout(0.07, 0.28), mkrollout(0.06, 0.26)]
    @test Actor.passes_safety_gate(treated_better, baseline_worse, Main.InSiteSimulator(), thresholds) == true

    treated_worse = [mkrollout(0.08, 0.30), mkrollout(0.07, 0.29)]
    baseline_better = [mkrollout(0.04, 0.23), mkrollout(0.03, 0.21)]
    @test Actor.passes_safety_gate(treated_worse, baseline_better, Main.InSiteSimulator(), thresholds) == false

    treated_relief = [mkrollout(0.12, 0.42), mkrollout(0.11, 0.39)]
    baseline_catastrophic = [mkrollout(0.18, 0.50), mkrollout(0.16, 0.46)]
    relief_diag = Actor.safety_diagnostics(treated_relief, baseline_catastrophic, Main.InSiteSimulator(), thresholds)
    @test relief_diag.failure_mode == :passed
    @test relief_diag.passed == true

    treated_not_relieved = [mkrollout(0.179, 0.499), mkrollout(0.159, 0.459)]
    no_relief_diag = Actor.safety_diagnostics(treated_not_relieved, baseline_catastrophic, Main.InSiteSimulator(), thresholds)
    @test no_relief_diag.failure_mode == :catastrophic
    @test no_relief_diag.passed == false
end

@testset "Epistemic Calibration Neutrality" begin
    mem = MemoryBuffer()
    belief = GaussianBeliefState(
        x̂_phys = Dict(:bg => 100.0),
        Σ_phys = Dict(:bg => 4.0),
        x̂_trust = 0.6,
        σ_trust = 0.1,
        x̂_burnout = 0.1,
        σ_burnout = 0.05,
        x̂_engagement = 0.7,
        σ_engagement = 0.1,
        x̂_burden = 0.2,
        σ_burden = 0.05,
        entropy = 0.1,
        obs_log_lik = 0.0,
    )
    thresholds = EpistemicThresholds(κ_min = 0.60, ρ_min = 0.50, η_min = 0.70)

    for _ in 1:6
        push!(mem.records, MemoryRecord(
            id = mem.next_id,
            day = mem.next_id,
            belief_entropy = 0.2,
            action = NullAction(),
            epistemic = EpistemicState(κ_familiarity = 0.8, ρ_concordance = 0.8, η_calibration = 0.8, feasible = true),
            config_snapshot = make_test_config(),
            user_response = nothing,
            realized_signals = nothing,
            realized_cost = nothing,
            predicted_cvar = nothing,
            critic_target = nothing,
            shadow_delta_score = nothing,
            trust_at_rec = 0.6,
            burnout_at_rec = 0.1,
            engagement_at_rec = 0.7,
            burden_at_rec = 0.2,
            latent_snapshot = nothing,
            latent_μ_at_rec = nothing,
            latent_log_σ_at_rec = nothing,
        ))
        mem.next_id += 1
    end

    epistemic = Perception.compute_epistemic_state(
        belief,
        mem,
        NullAction(),
        thresholds,
    )

    @test epistemic.κ_familiarity >= thresholds.κ_min
    @test epistemic.ρ_concordance >= thresholds.ρ_min
    @test epistemic.η_calibration >= thresholds.η_min
    @test epistemic.feasible == true
end

@testset "JEPA Training Smoke" begin
    dataset = Perception.SyntheticTrainingDataset(3, 30, 7, 2, 12, 3)
    encoder = Perception.HierarchicalJEPAEncoder(2, 12, 3)
    predictor = WorldModule.JEPAPredictor()

    Perception.train_encoder!(encoder, predictor, dataset; n_epochs=3, batch_size=2)

    @test isfinite(Perception.LAST_JEPA_TRAINING_LOSS[])

    subhourly, ctx, daily,
    _, _, _ = Perception.get_training_batch(dataset, 1)
    belief = Perception.encode_observation_window(encoder, subhourly, ctx, daily)

    @test isa(belief, JEPABeliefState)
    @test length(belief.μ) == 64
    @test isfinite(Float64(belief.entropy))

    mktempdir() do dir
        Perception.save_jepa_weights(encoder, predictor, dir)
        @test isfile(joinpath(dir, "jepa_encoder.bson"))
        @test isfile(joinpath(dir, "jepa_predictor.bson"))

        encoder_loaded = Perception.HierarchicalJEPAEncoder(2, 12, 3)
        predictor_loaded = WorldModule.JEPAPredictor()
        Perception.load_jepa_weights!(encoder_loaded, predictor_loaded, dir)

        loaded_belief = Perception.encode_observation_window(
            encoder_loaded,
            subhourly,
            ctx,
            daily
        )
        @test isa(loaded_belief, JEPABeliefState)
        @test length(loaded_belief.μ) == 64
    end
end

@testset "Explicit Rollout Shared Core" begin
    config = make_test_config()
    twin = make_test_twin()
    belief = GaussianBeliefState(
        x̂_phys = Dict(:bg => 100.0),
        Σ_phys = Dict(:bg => 4.0),
        x̂_trust = 0.6,
        σ_trust = 0.1,
        x̂_burnout = 0.1,
        σ_burnout = 0.05,
        x̂_engagement = 0.7,
        σ_engagement = 0.1,
        x̂_burden = 0.2,
        σ_burden = 0.05,
        entropy = 0.2,
        obs_log_lik = 0.0
    )

    rollouts = WorldModule.run_rollouts(
        belief,
        Actor.CandidateAction(Dict(:isf => 0.05)),
        twin,
        WorldModule.MockSimulator(),
        initialize_noise(),
        config
    )

    @test !isempty(rollouts)
    @test rollouts[1].total_cost > 0.0
end

@testset "Configurator CQL Smoke" begin
    prefs = UserPreferences()
    config = make_test_config()
    meta = Configurator.MetaState(
        belief_entropy = 0.2,
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        win_rate = 0.6,
        safety_violations = 0,
        consecutive_days = 5,
        trust_level = 0.7,
        burnout_level = 0.1,
        drift_detected = false,
        days_since_drift = 30,
        n_records = 120,
        current_day = 120
    )

    metas = fill(meta, 100)
    configs = fill(config, 100)
    rewards = collect(range(-0.2, 0.2, length=100))

    getfield(Configurator, Symbol("train_cql!"))(
        120,
        metas,
        configs,
        rewards,
        prefs;
        n_epochs = 1,
        n_random = 2
    )
    adapted = Configurator.adapt_cql(config, meta, prefs; n_steps=2, n_restarts=1)

    @test Configurator.Q_NET.is_ready
    @test 0.70 <= adapted.φ_act.α_cvar <= 0.95
end

@testset "Configurator CQL Fallback On Nonfinite" begin
    prefs = UserPreferences()
    config = make_test_config()
    meta = Configurator.MetaState(
        belief_entropy = 0.2,
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        win_rate = 0.6,
        safety_violations = 0,
        consecutive_days = 5,
        trust_level = 0.7,
        burnout_level = 0.1,
        drift_detected = false,
        days_since_drift = 30,
        n_records = 120,
        current_day = 120
    )

    original_weight = copy(Configurator.Q_NET.state_encoder.layers[1].weight)
    original_ready = Configurator.Q_NET.is_ready
    try
        Configurator.Q_NET.is_ready = true
        Configurator.Q_NET.state_encoder.layers[1].weight .= NaN32
        adapted = Configurator.adapt_cql(config, meta, prefs; n_steps=2, n_restarts=1)
        @test adapted isa ConfiguratorState
        @test isfinite(adapted.φ_act.Δ_max)
    finally
        Configurator.Q_NET.state_encoder.layers[1].weight .= original_weight
        Configurator.Q_NET.is_ready = original_ready
    end
end

@testset "Postgrad No-Surface Adaptation" begin
    config = make_test_config()
    meta = Configurator.MetaState(
        belief_entropy = 0.2,
        κ_familiarity = 0.8,
        ρ_concordance = 0.8,
        η_calibration = 0.8,
        win_rate = 0.85,
        safety_violations = 0,
        consecutive_days = 14,
        trust_level = 0.7,
        burnout_level = 0.15,
        drift_detected = false,
        days_since_drift = 0,
        n_records = 30,
        current_day = 30,
        graduated = true,
        no_surface_streak = 8,
        last_decision_reason = :effect_size_insufficient,
    )
    adapted = Configurator._apply_postgrad_no_surface_adaptation(config, meta)
    @test adapted.φ_act.δ_min_effect < config.φ_act.δ_min_effect
    @test adapted.φ_act.N_search > config.φ_act.N_search
    @test adapted.φ_act.Δ_max >= config.φ_act.Δ_max

    not_graduated = Configurator.MetaState(
        belief_entropy = 0.2,
        κ_familiarity = 0.8,
        ρ_concordance = 0.8,
        η_calibration = 0.8,
        win_rate = 0.85,
        safety_violations = 0,
        consecutive_days = 14,
        trust_level = 0.7,
        burnout_level = 0.15,
        drift_detected = false,
        days_since_drift = 0,
        n_records = 30,
        current_day = 30,
        graduated = false,
        no_surface_streak = 8,
        last_decision_reason = :effect_size_insufficient,
    )
    unchanged = Configurator._apply_postgrad_no_surface_adaptation(config, not_graduated)
    @test unchanged.φ_act.δ_min_effect == config.φ_act.δ_min_effect
    @test unchanged.φ_act.N_search == config.φ_act.N_search
end

@testset "Latent Decoder Smoke" begin
    fake_mem = MemoryBuffer()
    config = make_test_config()
    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true
    )
    psy = PsyState(
        trust = ScalarTrust(0.6),
        burden = ScalarBurden(0.8),
        engagement = ScalarEngagement(0.7),
        burnout = ScalarBurnout(0.2)
    )

    for day in 1:25
        belief = JEPABeliefState(
            μ = rand(Float32, 64),
            log_σ = rand(Float32, 64),
            entropy = rand(Float32),
            obs_log_lik = 0.0f0
        )
        rec_id = Memory.store_record!(
            fake_mem,
            day,
            belief,
            Actor.CandidateAction(Dict(:isf => 0.05)),
            epistemic,
            config,
            psy;
            latent_snapshot = rand(Float32, 64)
        )
        Memory.store_outcome!(fake_mem, rec_id, Accept, Dict{Symbol, Any}(), rand())
    end

    Cost.LATENT_DECODER.is_trained = false
    Cost.LATENT_DECODER.n_trained = 0
    Cost.train_decoder!(Cost.LATENT_DECODER, fake_mem)

    @test Cost.LATENT_DECODER.is_trained == true
    @test Cost.LATENT_DECODER.n_trained >= 20

    current_belief = JEPABeliefState(
        μ = rand(Float32, 64),
        log_σ = rand(Float32, 64),
        entropy = rand(Float32),
        obs_log_lik = 0.0f0
    )
    next_belief = JEPABeliefState(
        μ = rand(Float32, 64),
        log_σ = rand(Float32, 64),
        entropy = rand(Float32),
        obs_log_lik = 0.0f0
    )
    decoded_cost = Cost.compute_latent_intrinsic_cost(
        Actor.CandidateAction(Dict(:isf => 0.05)),
        current_belief,
        next_belief,
        config.φ_cost.weights
    )

    @test decoded_cost isa Float64
end

@testset "Actor CQL Smoke" begin
    fake_mem = MemoryBuffer(MemoryRecord[], 365, 1)
    config = make_test_config()
    epistemic = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible = true
    )
    psy = PsyState(
        trust = ScalarTrust(0.6),
        burden = ScalarBurden(0.8),
        engagement = ScalarEngagement(0.7),
        burnout = ScalarBurnout(0.2)
    )

    for day in 1:110
        belief = JEPABeliefState(
            μ = rand(Float32, 64),
            log_σ = rand(Float32, 64),
            entropy = rand(Float32),
            obs_log_lik = 0.0f0
        )
        rec_id = Memory.store_record!(
            fake_mem,
            day,
            belief,
            Actor.CandidateAction(Dict(:isf => rand() * 0.1)),
            epistemic,
            config,
            psy;
            latent_snapshot = rand(Float32, 64)
        )
        Memory.store_outcome!(fake_mem, rec_id, Accept, Dict{Symbol, Any}(), rand())
    end

    Actor.OFFLINE_RL_MODEL.network = nothing
    Actor.OFFLINE_RL_MODEL.n_trained = 0
    Actor.OFFLINE_RL_MODEL.is_ready = false

    Actor.train_actor_cql!(fake_mem)

    @test Actor.OFFLINE_RL_MODEL.n_trained >= 100
    @test Actor.OFFLINE_RL_MODEL.is_ready == false
end

@testset "Full System Smoke" begin
    sim = WorldModule.MockSimulator()
    prefs = UserPreferences()
    system = Chamelia.initialize_patient(prefs, sim)
    obs = Observation(timestamp=1.0, signals=Dict{Symbol, Any}())

    for _ in 1:25
        Chamelia.step!(system, obs)
    end

    @test system.current_day == 25
    @test Chamelia.graduation_status(system).graduated == false
    @test system.mem.next_id > 1

    save_path = "/tmp/test_patient.jls"
    Chamelia.save_patient(system, save_path)
    loaded = Chamelia.load_patient(save_path)

    @test loaded.current_day == 25
end

# ─────────────────────────────────────────────────────────────────
# Week 4.5 Safety Integration Tests
# ─────────────────────────────────────────────────────────────────

@testset "Safety Gate: Catastrophic Block" begin
    psy = PsyState(
        trust      = ScalarTrust(0.6),
        burden     = ScalarBurden(0.2),
        engagement = ScalarEngagement(0.7),
        burnout    = ScalarBurnout(0.1),
    )
    state = PatientState(PhysState(Dict(:bg => 100.0)), psy)

    thresholds = Dict{Symbol, Float64}(
        :pct_low_max      => 0.04,
        :pct_low_hard_max => 0.08,
        :pct_high_max     => 0.25,
        :pct_high_hard_max => 0.40,
    )

    mkrollout(low, high) = RolloutResult(
        action         = NullAction(),
        initial_psy    = psy,
        terminal_state = state,
        terminal_psy   = psy,
        total_cost     = 0.0,
        psy_trajectory = PsyState[],
        phys_signals   = [Dict{Symbol, Any}(:pct_low_7d => low, :pct_high_7d => high)],
    )

    sim = WorldModule.MockSimulator()

    # pct_low > pct_low_hard_max (0.09 > 0.08) → catastrophic → blocked
    unsafe_rollouts = [mkrollout(0.09, 0.20)]
    @test Actor.passes_safety_gate(unsafe_rollouts, sim, thresholds) == false

    # pct_high > pct_high_hard_max (0.45 > 0.40) → catastrophic → blocked
    high_rollouts = [mkrollout(0.02, 0.45)]
    @test Actor.passes_safety_gate(high_rollouts, sim, thresholds) == false

    # both within hard limits → not catastrophic → passes
    safe_rollouts = [mkrollout(0.03, 0.20)]
    @test Actor.passes_safety_gate(safe_rollouts, sim, thresholds) == true
end

@testset "Burnout Gate" begin
    config = make_test_config()   # ε_burn = 0.05

    # upper_ci exactly at threshold → fails (strict <)
    at_threshold = BurnoutAttribution(
        Δ_hat = 0.04, P_treated = 0.10, P_baseline = 0.06,
        se_paired = 0.01, ci_lower = 0.02, upper_ci = 0.05,
        n_pairs = 20, horizon = 30,
    )
    @test Actor.passes_burnout_gate(at_threshold, config) == false

    # upper_ci above threshold → fails
    above = BurnoutAttribution(
        Δ_hat = 0.05, P_treated = 0.11, P_baseline = 0.06,
        se_paired = 0.01, ci_lower = 0.03, upper_ci = 0.07,
        n_pairs = 20, horizon = 30,
    )
    @test Actor.passes_burnout_gate(above, config) == false

    # upper_ci below threshold → passes
    below = BurnoutAttribution(
        Δ_hat = 0.01, P_treated = 0.07, P_baseline = 0.06,
        se_paired = 0.005, ci_lower = 0.0, upper_ci = 0.02,
        n_pairs = 20, horizon = 30,
    )
    @test Actor.passes_burnout_gate(below, config) == true
end

@testset "Epistemic Gate" begin
    # feasible=false → blocked regardless of individual values
    infeasible = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible      = false,
    )
    @test Actor.passes_epistemic_gate(infeasible) == false

    # feasible=true → passes
    feasible = EpistemicState(
        κ_familiarity = 0.9,
        ρ_concordance = 0.9,
        η_calibration = 0.9,
        feasible      = true,
    )
    @test Actor.passes_epistemic_gate(feasible) == true
end

@testset "Graduation Gate: All Four Criteria Required" begin
    good = Actor.ShadowScorecard(
        n_days            = 21,
        win_rate          = 0.60,
        safety_violations = 0,
        consecutive_days  = 7,
    )
    @test Actor.passes_graduation_gate(good) == true

    # Below minimum days
    @test Actor.passes_graduation_gate(Actor.ShadowScorecard(
        n_days=20, win_rate=0.60, safety_violations=0, consecutive_days=7)) == false

    # Below minimum win rate
    @test Actor.passes_graduation_gate(Actor.ShadowScorecard(
        n_days=21, win_rate=0.59, safety_violations=0, consecutive_days=7)) == false

    # Any safety violations
    @test Actor.passes_graduation_gate(Actor.ShadowScorecard(
        n_days=21, win_rate=0.60, safety_violations=1, consecutive_days=7)) == false

    # Below consecutive days
    @test Actor.passes_graduation_gate(Actor.ShadowScorecard(
        n_days=21, win_rate=0.60, safety_violations=0, consecutive_days=6)) == false
end

@testset "Configurator Safety Override After Violation" begin
    config = make_test_config()
    prefs  = UserPreferences()

    # MetaState with one safety violation → adapt_rule_based must clamp Δ_max = 0.02
    meta_with_violation = Configurator.MetaState(
        belief_entropy    = 0.3,
        κ_familiarity     = 0.8,
        ρ_concordance     = 0.8,
        η_calibration     = 0.8,
        win_rate          = 0.50,
        safety_violations = 1,
        consecutive_days  = 5,
        trust_level       = 0.6,
        burnout_level     = 0.2,
        drift_detected    = false,
        days_since_drift  = 0,
        n_records         = 15,
        current_day       = 20,
    )

    safe_config = Configurator.adapt_rule_based(config, meta_with_violation, prefs)
    @test safe_config.φ_act.Δ_max  ≈ 0.02
    @test safe_config.φ_world.N_roll == 100
    @test safe_config.φ_world.H_med  == 3   # H_MED_MIN

    # MetaState with zero violations → Δ_max not forced to floor
    meta_clean = Configurator.MetaState(
        belief_entropy    = 0.3,
        κ_familiarity     = 0.8,
        ρ_concordance     = 0.8,
        η_calibration     = 0.8,
        win_rate          = 0.65,
        safety_violations = 0,
        consecutive_days  = 14,
        trust_level       = 0.7,
        burnout_level     = 0.1,
        drift_detected    = false,
        days_since_drift  = 0,
        n_records         = 30,
        current_day       = 35,
    )

    clean_config = Configurator.adapt_rule_based(config, meta_clean, prefs)
    @test clean_config.φ_act.Δ_max > 0.02
end

@testset "Competence-Aware Model Selection" begin
    # ── Helpers ───────────────────────────────────────────────────
    function make_scored_record(mode::Symbol, win::Bool; use_jepa::Bool=false)
        # minimal MemoryRecord for win-rate computation
        MemoryRecord(
            id                  = 1,
            day                 = 1,
            belief_entropy      = 0.3,
            action              = NullAction(),
            epistemic           = EpistemicState(
                κ_familiarity=0.7, ρ_concordance=0.7,
                η_calibration=0.7, feasible=true
            ),
            config_snapshot     = make_test_config(),
            user_response       = nothing,
            realized_signals    = nothing,
            realized_cost       = 1.0,
            predicted_cvar      = nothing,
            critic_target       = nothing,
            shadow_delta_score  = win ? 1.0 : -1.0,
            trust_at_rec        = 0.7,
            burnout_at_rec      = 0.1,
            engagement_at_rec   = 0.8,
            burden_at_rec       = 0.2,
            latent_snapshot     = use_jepa ? Float32[0.1, 0.2] : nothing,
            latent_μ_at_rec     = use_jepa ? Float32[0.1, 0.2] : nothing,
            latent_log_σ_at_rec = use_jepa ? Float32[-3.0, -3.0] : nothing,
            configurator_mode   = mode,
        )
    end

    # Build a MemoryBuffer from a list of (mode, win) pairs
    function make_mem_with_modes(pairs::Vector{Tuple{Symbol,Bool}};
                                  jepa_pairs::Vector{Tuple{Symbol,Bool}}=Tuple{Symbol,Bool}[])
        mem = MemoryBuffer()
        for (mode, win) in pairs
            push!(mem.records, make_scored_record(mode, win))
        end
        for (mode, win) in jepa_pairs
            push!(mem.records, make_scored_record(mode, win; use_jepa=true))
        end
        return mem
    end

    # ── _mode_win_rates ───────────────────────────────────────────
    mem_wr = make_mem_with_modes([
        (:rules, true), (:rules, true), (:rules, false),  # rules: 2/3 ≈ 0.667
        (:cql, true),   (:cql, false),                    # cql:   1/2 = 0.500
    ])
    win_rates = Configurator._mode_win_rates(mem_wr)
    @test isapprox(win_rates[:rules], 2/3, atol=1e-8)
    @test isapprox(win_rates[:cql],   1/2, atol=1e-8)

    # ── _mode_sample_counts ───────────────────────────────────────
    counts = Configurator._mode_sample_counts(mem_wr)
    @test counts[:rules] == 3
    @test counts[:cql]   == 2

    # ── mode_competence_diagnostics: JEPA vs explicit ─────────────
    # 3 explicit wins out of 4, 2 JEPA wins out of 3
    mem_diag = make_mem_with_modes(
        [(:rules, true), (:rules, true), (:rules, true), (:rules, false)],
        jepa_pairs = [(:rules, true), (:rules, true), (:rules, false)]
    )
    diag = Configurator.mode_competence_diagnostics(mem_diag)
    @test !isnothing(diag.explicit_win_rate)
    @test !isnothing(diag.jepa_win_rate)
    @test isapprox(diag.explicit_win_rate, 3/4, atol=1e-8)
    @test isapprox(diag.jepa_win_rate,     2/3, atol=1e-8)
    # explicit (0.75) > JEPA (0.667) → jepa_preferred = false
    @test diag.jepa_preferred == false

    # ── Empty memory → no crash, neutral defaults ─────────────────
    empty_mem = MemoryBuffer()
    diag_empty = Configurator.mode_competence_diagnostics(empty_mem)
    @test isnothing(diag_empty.explicit_win_rate)
    @test isnothing(diag_empty.jepa_win_rate)
    @test diag_empty.jepa_preferred == false
    @test diag_empty.active_mode isa Symbol

    # ── store_record! propagates configurator_mode ────────────────
    mem_store = MemoryBuffer()
    config_s  = make_test_config()
    belief_s  = GaussianBeliefState(
        x̂_phys=Dict{Symbol,Float64}(), Σ_phys=Dict{Symbol,Float64}(),
        x̂_trust=0.7, σ_trust=0.05, x̂_burnout=0.1, σ_burnout=0.02,
        x̂_engagement=0.8, σ_engagement=0.05, x̂_burden=0.2, σ_burden=0.02,
        entropy=0.3, obs_log_lik=0.0
    )
    epistemic_s = EpistemicState(
        κ_familiarity=0.7, ρ_concordance=0.7, η_calibration=0.7, feasible=true
    )
    psy_s = PsyState(
        trust=ScalarTrust(0.7), burden=ScalarBurden(0.2),
        engagement=ScalarEngagement(0.8), burnout=ScalarBurnout(0.1)
    )
    Memory.store_record!(
        mem_store, 1, belief_s, NullAction(), epistemic_s, config_s, psy_s;
        configurator_mode = :cql
    )
    @test !isempty(mem_store.records)
    @test mem_store.records[1].configurator_mode == :cql
end

@testset "Preference-Aware Physical Cost Weights" begin
    # ── initialize_config (backward-compat form) ──────────────────
    # High hypoglycemia_fear → high w_low
    prefs_fearful = UserPreferences(aggressiveness=0.5, hypoglycemia_fear=1.0, burden_sensitivity=0.5)
    config_fearful = Configurator.initialize_config(prefs_fearful)
    @test config_fearful.φ_cost.weights.physical[:w_low]  ≈ 7.0   atol=1e-8   # 3 + 1.0*4 = 7
    @test config_fearful.φ_cost.weights.physical[:w_high] ≈ 1.25  atol=1e-8   # 1 + 0.5*0.5 = 1.25

    # Low hypoglycemia_fear + high aggressiveness → lower w_low, higher w_high
    prefs_aggr = UserPreferences(aggressiveness=1.0, hypoglycemia_fear=0.0, burden_sensitivity=0.5)
    config_aggr = Configurator.initialize_config(prefs_aggr)
    @test config_aggr.φ_cost.weights.physical[:w_low]  ≈ 3.0   atol=1e-8   # 3 + 0.0*4 = 3
    @test config_aggr.φ_cost.weights.physical[:w_high] ≈ 1.5   atol=1e-8   # 1 + 1.0*0.5 = 1.5

    # Conservative user: high hypo fear, low aggressiveness
    prefs_consv = UserPreferences(aggressiveness=0.0, hypoglycemia_fear=1.0, burden_sensitivity=0.5)
    config_consv = Configurator.initialize_config(prefs_consv)
    @test config_consv.φ_cost.weights.physical[:w_low]  ≈ 7.0   atol=1e-8
    @test config_consv.φ_cost.weights.physical[:w_high] ≈ 1.0   atol=1e-8

    # Tradeoff direction: aggressive user has lower w_low/w_high ratio than conservative
    ratio_aggr  = config_aggr.φ_cost.weights.physical[:w_low]  / config_aggr.φ_cost.weights.physical[:w_high]
    ratio_consv = config_consv.φ_cost.weights.physical[:w_low] / config_consv.φ_cost.weights.physical[:w_high]
    # aggressive + no hypo fear: 3.0/1.5 = 2.0
    # conservative + high hypo fear: 7.0/1.0 = 7.0
    @test ratio_aggr < ratio_consv   # aggressive user more willing to trade lows for high reduction

    # ── InSiteDomainAdapter: same scaling ─────────────────────────
    adapter = Main.InSiteDomainAdapter()
    weights_fearful = default_physical_weights(adapter, prefs_fearful)
    @test weights_fearful[:w_low]  ≈ 7.0   atol=1e-8
    @test weights_fearful[:w_high] ≈ 1.25  atol=1e-8

    weights_aggr = default_physical_weights(adapter, prefs_aggr)
    @test weights_aggr[:w_low]  ≈ 3.0   atol=1e-8
    @test weights_aggr[:w_high] ≈ 1.5   atol=1e-8

    # ── Safety invariant: w_high/w_low scaling does not affect safety thresholds ──
    # Safety thresholds come from the simulator, not from cost weights
    sim = Main.InSiteSimulator("default", 42)
    thresholds = WorldModule.safety_thresholds(sim)
    @test haskey(thresholds, :pct_low_max)
    @test thresholds[:pct_low_max] ≈ 0.04  atol=1e-8   # simulator hard limit unchanged by cost weights
end

@testset "Clinical Delta Gate" begin
    using Main.Actor: passes_clinical_delta_gate, CandidateAction, ScheduledAction
    using Main.WorldModule: min_clinical_delta
    using Main: UserPreferences

    sim     = Main.InSiteSimulator("default", 42)
    mock    = Main.WorldModule.MockSimulator()

    # ── InSite min_clinical_delta is 0.03 ──────────────────────────
    @test min_clinical_delta(sim) ≈ 0.03  atol=1e-8

    # ── CandidateAction: passes when at least one delta >= threshold ──
    big_action   = CandidateAction(Dict(:isf_delta => 0.05))   # 5% — passes
    small_action = CandidateAction(Dict(:isf_delta => 0.02))   # 2% — fails
    zero_action  = CandidateAction(Dict(:isf_delta => 0.0))    # 0% — fails
    multi_small  = CandidateAction(Dict(:isf_delta => 0.02, :cr_delta => 0.02))  # both small — fails
    multi_one_big = CandidateAction(Dict(:isf_delta => 0.02, :cr_delta => 0.05)) # one big — passes

    @test  passes_clinical_delta_gate(big_action,    sim)
    @test !passes_clinical_delta_gate(small_action,  sim)
    @test !passes_clinical_delta_gate(zero_action,   sim)
    @test !passes_clinical_delta_gate(multi_small,   sim)
    @test  passes_clinical_delta_gate(multi_one_big, sim)

    # ── ScheduledAction: passes when at least one segment delta >= threshold ──
    seg = Main.SegmentSurface(segment_id="s1", start_min=0, end_min=480,
                              parameter_values=Dict(:isf => 50.0, :cr => 10.0, :basal => 1.0))
    big_sched   = ScheduledAction(1, Main.parameter_adjustment, [seg],
                                  [Main.SegmentDelta(segment_id="s1", parameter_deltas=Dict(:isf_delta => 0.05))],
                                  Main.StructureEdit[])
    small_sched = ScheduledAction(1, Main.parameter_adjustment, [seg],
                                  [Main.SegmentDelta(segment_id="s1", parameter_deltas=Dict(:isf_delta => 0.01))],
                                  Main.StructureEdit[])
    empty_sched = ScheduledAction(1, Main.parameter_adjustment, [seg],
                                  Main.SegmentDelta[], Main.StructureEdit[])

    @test  passes_clinical_delta_gate(big_sched,   sim)
    @test !passes_clinical_delta_gate(small_sched, sim)
    @test !passes_clinical_delta_gate(empty_sched, sim)

    # ── NullAction never passes ─────────────────────────────────────
    @test !passes_clinical_delta_gate(NullAction(), sim)

    # ── MockSimulator uses conservative default 0.01 ───────────────
    @test min_clinical_delta(mock) ≈ 0.01  atol=1e-8
    @test passes_clinical_delta_gate(small_action, mock)   # 2% >= 1% default

    # ── User minimums can further suppress small-but-clinical changes ─────
    adapter = Main.InSiteDomainAdapter()
    prefs_isf = UserPreferences(minimum_action_delta_thresholds = Dict("isf_delta" => 0.06))
    @test !passes_clinical_delta_gate(big_action, sim, adapter, prefs_isf)
    @test passes_clinical_delta_gate(CandidateAction(Dict(:isf_delta => 0.07)), sim, adapter, prefs_isf)

    prefs_mixed = UserPreferences(
        minimum_action_delta_thresholds = Dict(
            "isf_delta" => 0.06,
            "cr_delta" => 0.04,
            "basal_delta" => 0.08
        )
    )
    @test passes_clinical_delta_gate(CandidateAction(Dict(:cr_delta => 0.05)), sim, adapter, prefs_mixed)
    @test !passes_clinical_delta_gate(CandidateAction(Dict(:basal_delta => 0.05)), sim, adapter, prefs_mixed)

    sched_pref = UserPreferences(minimum_action_delta_thresholds = Dict("basal_delta" => 0.07))
    sched_big_enough = ScheduledAction(1, Main.parameter_adjustment, [seg],
                                       [Main.SegmentDelta(segment_id="s1", parameter_deltas=Dict(:basal_delta => 0.08))],
                                       Main.StructureEdit[])
    sched_too_small = ScheduledAction(1, Main.parameter_adjustment, [seg],
                                      [Main.SegmentDelta(segment_id="s1", parameter_deltas=Dict(:basal_delta => 0.05))],
                                      Main.StructureEdit[])
    @test passes_clinical_delta_gate(sched_big_enough, sim, adapter, sched_pref)
    @test !passes_clinical_delta_gate(sched_too_small, sim, adapter, sched_pref)
end

@testset "Profile Context in ConnectedAppState" begin
    function parse_state(json_str)
        payload = JSON3.read(json_str, Dict{String, Any})
        ChameliaServer._connected_app_state(payload)
    end

    @testset "no app state → defaults" begin
        state = parse_state("{}")
        @test isnothing(state.active_profile_id)
        @test isempty(state.available_profiles)
    end

    @testset "active_profile_id round-trips" begin
        json = """{"connected_app_state":{
            "schedule_version":"v1",
            "current_segments":[],
            "allow_structural_recommendations":false,
            "allow_continuous_schedule":false,
            "active_profile_id":"profile-abc"
        }}"""
        state = parse_state(json)
        @test state.active_profile_id == "profile-abc"
        @test isempty(state.available_profiles)
    end

    @testset "available_profiles round-trips" begin
        json = """{"connected_app_state":{
            "schedule_version":"v1",
            "current_segments":[],
            "allow_structural_recommendations":false,
            "allow_continuous_schedule":false,
            "active_profile_id":"p1",
            "available_profiles":[
                {"id":"p1","name":"Weekday","segment_count":4},
                {"id":"p2","name":"Weekend","segment_count":3}
            ]
        }}"""
        state = parse_state(json)
        @test length(state.available_profiles) == 2
        @test state.available_profiles[1].id == "p1"
        @test state.available_profiles[1].name == "Weekday"
        @test state.available_profiles[1].segment_count == 4
        @test state.available_profiles[2].id == "p2"
        @test state.available_profiles[2].segment_count == 3
    end

    @testset "missing profile id throws" begin
        json = """{"connected_app_state":{
            "schedule_version":"v1","current_segments":[],
            "allow_structural_recommendations":false,"allow_continuous_schedule":false,
            "available_profiles":[{"name":"NoId","segment_count":2}]
        }}"""
        @test_throws ArgumentError parse_state(json)
    end

    @testset "missing profile name throws" begin
        json = """{"connected_app_state":{
            "schedule_version":"v1","current_segments":[],
            "allow_structural_recommendations":false,"allow_continuous_schedule":false,
            "available_profiles":[{"id":"p1","segment_count":2}]
        }}"""
        @test_throws ArgumentError parse_state(json)
    end
end

@testset "RecommendationPackage scope fields" begin
    @testset "defaults" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.0,
            confidence           = 0.5,
            alternatives         = AbstractAction[],
            effect_size          = 0.0,
            cvar_value           = 0.0,
            burnout_attribution  = nothing
        )
        @test pkg.recommendation_scope == "patch_current"
        @test isnothing(pkg.target_profile_id)
    end

    @testset "patch_existing scope" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.0,
            confidence           = 0.5,
            alternatives         = AbstractAction[],
            effect_size          = 0.0,
            cvar_value           = 0.0,
            burnout_attribution  = nothing,
            recommendation_scope = "patch_existing",
            target_profile_id    = "weekend-profile"
        )
        @test pkg.recommendation_scope == "patch_existing"
        @test pkg.target_profile_id == "weekend-profile"
    end

    @testset "serialization includes scope fields" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.1,
            confidence           = 0.8,
            alternatives         = AbstractAction[],
            effect_size          = 0.5,
            cvar_value           = 0.2,
            burnout_attribution  = nothing,
            recommendation_scope = "create_new",
            target_profile_id    = "base-profile"
        )
        json_dict = ChameliaServer._serialize_recommendation(pkg)
        @test get(json_dict, "recommendation_scope", nothing) == "create_new"
        @test get(json_dict, "target_profile_id", nothing) == "base-profile"
    end

    @testset "serialization of default scope" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.0,
            confidence           = 0.5,
            alternatives         = AbstractAction[],
            effect_size          = 0.0,
            cvar_value           = 0.0,
            burnout_attribution  = nothing
        )
        json_dict = ChameliaServer._serialize_recommendation(pkg)
        @test get(json_dict, "recommendation_scope", nothing) == "patch_current"
        @test isnothing(get(json_dict, "target_profile_id", nothing))
    end
end

@testset "InSite Regime Detection" begin
    adapter   = InSiteDomainAdapter()
    empty_mem = MemoryBuffer()

    # ── No regime — empty signals ──────────────────────────────────
    @testset "no signals → no regime, patch_current" begin
        result = detect_regime(adapter, Dict{Symbol,Any}(), ConnectedAppState(), empty_mem)
        @test isnothing(result.regime_label)
        @test result.scope == "patch_current"
        @test isnothing(result.target_profile_id)
    end

    # ── Menstrual phase (highest priority) ────────────────────────
    @testset "menstrual phase detected" begin
        sigs = Dict{Symbol,Any}(:cycle_phase_menstrual => 1.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "menstrual_phase"
        @test result.scope == "patch_current"   # no other profiles
    end

    # ── Luteal phase ──────────────────────────────────────────────
    @testset "luteal phase detected" begin
        sigs = Dict{Symbol,Any}(:cycle_phase_luteal => 1.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "luteal_phase"
    end

    # ── High activity day ─────────────────────────────────────────
    @testset "high_activity_day at threshold" begin
        sigs = Dict{Symbol,Any}(:exercise_mins => 60.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "high_activity_day"
    end

    @testset "below threshold not flagged" begin
        sigs = Dict{Symbol,Any}(:exercise_mins => 30.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test isnothing(result.regime_label)
    end

    # ── Weekend (day_of_week) ─────────────────────────────────────
    @testset "Saturday is weekend" begin
        sigs = Dict{Symbol,Any}(:day_of_week => 6.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "weekend"
    end

    @testset "Sunday is weekend" begin
        sigs = Dict{Symbol,Any}(:day_of_week => 0.0)
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "weekend"
    end

    @testset "weekday not flagged" begin
        for dow in [1.0, 2.0, 3.0, 4.0, 5.0]
            sigs = Dict{Symbol,Any}(:day_of_week => dow)
            result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
            @test isnothing(result.regime_label)
        end
    end

    # ── Menstrual wins over weekend (priority) ────────────────────
    @testset "menstrual phase beats weekend in priority" begin
        sigs = Dict{Symbol,Any}(
            :cycle_phase_menstrual => 1.0,
            :day_of_week => 6.0
        )
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test result.regime_label == "menstrual_phase"
    end

    # ── Scope logic with available_profiles ───────────────────────
    @testset "regime + matching profile → patch_existing" begin
        app = ConnectedAppState(
            schedule_version = "v1",
            current_segments = SegmentSurface[],
            allow_structural_recommendations = false,
            allow_continuous_schedule = false,
            active_profile_id = "p1",
            available_profiles = [
                (id="p1", name="Weekday", segment_count=4),
                (id="p2", name="Weekend",  segment_count=3),
            ]
        )
        sigs = Dict{Symbol,Any}(:day_of_week => 6.0)
        result = detect_regime(adapter, sigs, app, empty_mem)
        @test result.regime_label == "weekend"
        @test result.scope == "patch_existing"
        @test result.target_profile_id == "p2"
    end

    @testset "regime + no matching profile → create_new from active" begin
        app = ConnectedAppState(
            schedule_version = "v1",
            current_segments = SegmentSurface[],
            allow_structural_recommendations = false,
            allow_continuous_schedule = false,
            active_profile_id = "p1",
            available_profiles = [
                (id="p1", name="Default", segment_count=4),
            ]
        )
        sigs = Dict{Symbol,Any}(:day_of_week => 6.0)
        result = detect_regime(adapter, sigs, app, empty_mem)
        @test result.regime_label == "weekend"
        @test result.scope == "create_new"
        @test result.target_profile_id == "p1"   # base = active profile
    end

    @testset "regime + no other profiles → patch_current" begin
        app = ConnectedAppState()   # no available_profiles
        sigs = Dict{Symbol,Any}(:day_of_week => 6.0)
        result = detect_regime(adapter, sigs, app, empty_mem)
        @test result.regime_label == "weekend"
        @test result.scope == "patch_current"
    end

    # ── detected_regime serialized through package ────────────────
    @testset "detected_regime appears in serialized package" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.0,
            confidence           = 0.5,
            alternatives         = AbstractAction[],
            effect_size          = 0.0,
            cvar_value           = 0.0,
            burnout_attribution  = nothing,
            detected_regime      = "weekend"
        )
        json_dict = ChameliaServer._serialize_recommendation(pkg)
        @test get(json_dict, "detected_regime", nothing) == "weekend"
    end

    @testset "nil detected_regime serializes as nothing" begin
        pkg = RecommendationPackage(
            action               = NullAction(),
            predicted_improvement = 0.0,
            confidence           = 0.5,
            alternatives         = AbstractAction[],
            effect_size          = 0.0,
            cvar_value           = 0.0,
            burnout_attribution  = nothing
        )
        json_dict = ChameliaServer._serialize_recommendation(pkg)
        @test isnothing(get(json_dict, "detected_regime", nothing))
    end

    # ── InSiteDomainAdapter returns no regime for non-regime signals ─
    @testset "zero cycle signals → no regime" begin
        sigs = Dict{Symbol,Any}(
            :cycle_phase_menstrual => 0.0,
            :cycle_phase_luteal    => 0.0,
            :exercise_mins         => 10.0,
            :day_of_week           => 3.0   # Wednesday
        )
        result = detect_regime(adapter, sigs, ConnectedAppState(), empty_mem)
        @test isnothing(result.regime_label)
        @test result.scope == "patch_current"
    end
end

@testset "Cold-Start Calibration" begin
    Main.ChameliaServer.reset_patient_cache!()
    Main.ChameliaServer.set_state_backend!(Main.ChameliaServer.InMemoryStateBackend())

    adapter = Main.InSiteDomainAdapter()

    # ── 1. UserPreferences defaults to empty calibration_targets ──────────
    @testset "UserPreferences defaults" begin
        prefs = UserPreferences()
        @test isa(prefs.calibration_targets, Dict{String, Float64})
        @test isempty(prefs.calibration_targets)
    end

    # ── 2. calibration_targets round-trip ─────────────────────────────────
    @testset "calibration_targets round-trip" begin
        prefs = UserPreferences(calibration_targets = Dict("recent_tir" => 0.72, "recent_pct_low" => 0.04))
        @test prefs.calibration_targets["recent_tir"] ≈ 0.72 atol=1e-9
        @test prefs.calibration_targets["recent_pct_low"] ≈ 0.04 atol=1e-9
    end

    # Helper: build a fresh posterior from a default prior
    function _fresh_posterior()
        prior = TwinPrior(
            trust_growth_dist       = Beta(2, 5),
            trust_decay_dist        = Beta(2, 8),
            burnout_sensitivity_dist = Beta(2, 5),
            engagement_decay_dist   = Beta(2, 5),
            physical_priors         = Dict{Symbol, Distribution}(
                :isf_multiplier   => Normal(1.0, 0.12),
                :basal_multiplier => Normal(1.0, 0.10),
            ),
            persona_label = "test"
        )
        posterior = TwinPosterior(
            trust_growth_rate   = 0.2,
            trust_decay_rate    = 0.1,
            burnout_sensitivity = 0.1,
            engagement_decay    = 0.05,
            physical            = Dict{Symbol, Float64}(:isf_multiplier => 1.0, :basal_multiplier => 1.0),
            last_updated_day    = 0,
            n_observations      = 0,
        )
        return prior, posterior
    end

    # ── 3. No-op when targets are empty ───────────────────────────────────
    @testset "no-op when targets empty" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior, Dict{String, Float64}())
        @test posterior.physical[:isf_multiplier]   ≈ 1.0 atol=1e-9
        @test posterior.physical[:basal_multiplier] ≈ 1.0 atol=1e-9
    end

    # ── 4. High TIR → ISF estimate shifts up ──────────────────────────────
    @testset "high TIR shifts ISF up" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior, Dict("recent_tir" => 0.85))
        # tir_hat = 0.85 requires higher ISF → posterior should shift above 1.0
        @test posterior.physical[:isf_multiplier] > 1.0
    end

    # ── 5. Low TIR + high pct_high → ISF estimate shifts down ─────────────
    @testset "low TIR + high pct_high shifts ISF down" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior, Dict("recent_tir" => 0.25, "recent_pct_high" => 0.60))
        # Low TIR with lots of high → ISF must be lower → posterior should shift below 1.0
        @test posterior.physical[:isf_multiplier] < 1.0
    end

    # ── 6. Good control → bounded update near prior ────────────────────────
    @testset "good control stays near prior" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior,
            Dict("recent_tir" => 0.72, "recent_pct_low" => 0.04, "recent_pct_high" => 0.24))
        # Should update but stay within reasonable bounds
        @test 0.7 < posterior.physical[:isf_multiplier]   < 1.5
        @test 0.7 < posterior.physical[:basal_multiplier] < 1.5
    end

    # ── 7. Inconsistent targets → n_eff < 5 → no update ──────────────────
    @testset "inconsistent targets → no update" begin
        prior, posterior = _fresh_posterior()
        # pct_low + pct_high > 1 is impossible; all particles will get near-zero weight
        calibrate_posterior!(adapter, posterior, prior,
            Dict("recent_pct_low" => 0.5, "recent_pct_high" => 0.6))
        # Posterior should remain at prior mean (1.0) since targets are inconsistent
        @test posterior.physical[:isf_multiplier]   ≈ 1.0 atol=1e-6
        @test posterior.physical[:basal_multiplier] ≈ 1.0 atol=1e-6
    end

    # ── 8. Partial targets: only recent_tir → still updates ───────────────
    @testset "partial targets update" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior, Dict("recent_tir" => 0.80))
        # With high TIR target, ISF should shift; at least one physical param should change
        @test posterior.physical[:isf_multiplier] != 1.0 || posterior.physical[:basal_multiplier] != 1.0
    end

    # ── 9. Soft bound: calibrated posterior stays within prior support ─────
    @testset "posterior within prior support" begin
        prior, posterior = _fresh_posterior()
        calibrate_posterior!(adapter, posterior, prior,
            Dict("recent_tir" => 0.90, "recent_pct_low" => 0.02, "recent_pct_high" => 0.08))
        @test 0.5 <= posterior.physical[:isf_multiplier]   <= 1.8
        @test 0.5 <= posterior.physical[:basal_multiplier] <= 1.8
    end

    # ── 10. server.jl: calibration_targets parses from JSON ───────────────
    @testset "server parses calibration_targets" begin
        prefs = Main.ChameliaServer._preferences(Dict(
            "preferences" => Dict(
                "calibration_targets" => Dict(
                    "recent_tir" => 0.70,
                    "recent_pct_low" => 0.05,
                    "recent_pct_high" => 0.25
                )
            )
        ))
        @test prefs.calibration_targets["recent_tir"]      ≈ 0.70 atol=1e-9
        @test prefs.calibration_targets["recent_pct_low"]  ≈ 0.05 atol=1e-9
        @test prefs.calibration_targets["recent_pct_high"] ≈ 0.25 atol=1e-9
    end

    # ── 11. server.jl: missing keys are silently ignored ──────────────────
    @testset "server ignores missing calibration keys" begin
        prefs = Main.ChameliaServer._preferences(Dict(
            "preferences" => Dict(
                "calibration_targets" => Dict("recent_tir" => 0.65)
            )
        ))
        @test haskey(prefs.calibration_targets, "recent_tir")
        @test !haskey(prefs.calibration_targets, "recent_pct_low")
        @test !haskey(prefs.calibration_targets, "recent_pct_high")
    end

    # ── 12. server.jl: invalid value throws ───────────────────────────────
    @testset "server rejects non-numeric calibration_target value" begin
        @test_throws ArgumentError Main.ChameliaServer._preferences(Dict(
            "preferences" => Dict(
                "calibration_targets" => Dict("recent_tir" => "bad_value")
            )
        ))
    end

    # ── 13. Determinism: same targets → same posterior ────────────────────
    @testset "deterministic with same targets" begin
        targets = Dict("recent_tir" => 0.68, "recent_pct_low" => 0.06, "recent_pct_high" => 0.26)
        prior, post1 = _fresh_posterior()
        calibrate_posterior!(adapter, post1, prior, targets)
        _, post2 = _fresh_posterior()
        calibrate_posterior!(adapter, post2, prior, targets)
        @test post1.physical[:isf_multiplier]   ≈ post2.physical[:isf_multiplier]   atol=1e-12
        @test post1.physical[:basal_multiplier] ≈ post2.physical[:basal_multiplier] atol=1e-12
    end

    # ── 14. end-to-end: initialize_patient uses calibration_targets ────────
    # Must pass InSiteDomainAdapter explicitly — the default adapter is a
    # domain-agnostic no-op stub that cannot know about ISF/TIR relationships.
    @testset "initialize_patient applies calibration" begin
        prefs = UserPreferences(calibration_targets = Dict("recent_tir" => 0.85))
        sim   = Main.InSiteSimulator()
        system = Chamelia.initialize_patient(prefs, sim; adapter = Main.InSiteDomainAdapter())
        # With high TIR target, ISF should shift above 1.0
        @test system.twin.posterior.physical[:isf_multiplier] > 1.0
    end
end

# ─────────────────────────────────────────────────────────────────
# JEPA Predictor — Action-Conditioned Training
#
# Tests for the shadow-period predictor fine-tuning pipeline:
#   1. MemoryRecord.latent_μ_at_outcome is written by store_outcome!
#   2. record_outcome! on a JEPA system populates the outcome latent
#   3. MemoryTransitionDataset builds correct triples from memory
#   4. Effective action scaling (accept/partial/reject) is correct
#   5. train_predictor! runs without error and reduces MSE loss
# ─────────────────────────────────────────────────────────────────

@testset "JEPA Predictor Action-Conditioned Training" begin

    # ── helpers ──────────────────────────────────────────────────
    function _make_jepa_belief(z_dim::Int = 4)
        JEPABeliefState(
            μ           = randn(Float32, z_dim),
            log_σ       = fill(-1.0f0, z_dim),
            entropy     = 0.0f0,
            obs_log_lik = 0.0f0
        )
    end

    function _make_jepa_record(id::Int, belief::JEPABeliefState, action::AbstractAction)
        μ, log_σ = Float32.(vec(belief.μ)), Float32.(vec(belief.log_σ))
        MemoryRecord(
            id                  = id,
            day                 = id,
            belief_entropy      = 1.0,
            action              = action,
            epistemic           = EpistemicState(0.8, 0.8, 0.8, true),
            config_snapshot     = make_test_config(),
            user_response       = nothing,
            realized_signals    = nothing,
            realized_cost       = nothing,
            critic_target       = nothing,
            shadow_delta_score  = nothing,
            trust_at_rec        = 0.5,
            burnout_at_rec      = 0.1,
            engagement_at_rec   = 0.7,
            burden_at_rec       = 0.2,
            latent_μ_at_rec     = μ,
            latent_log_σ_at_rec = log_σ,
        )
    end

    z_dim = 4

    # ── 1. store_outcome! writes latent_μ_at_outcome ─────────────
    @testset "store_outcome! sets latent_μ_at_outcome" begin
        mem = MemoryBuffer()
        rec = _make_jepa_record(1, _make_jepa_belief(z_dim), NullAction())
        push!(mem.records, rec)

        outcome_μ = Float32[0.1, 0.2, 0.3, 0.4]
        Memory.store_outcome!(mem, 1, Accept, Dict{Symbol,Any}(), 0.5;
                              latent_μ_at_outcome = outcome_μ)

        stored = Memory.get_record(mem, 1)
        @test !isnothing(stored.latent_μ_at_outcome)
        @test stored.latent_μ_at_outcome ≈ outcome_μ
    end

    # ── 2. store_outcome! without latent leaves field nothing ─────
    @testset "store_outcome! without latent leaves field nothing" begin
        mem = MemoryBuffer()
        rec = _make_jepa_record(1, _make_jepa_belief(z_dim), NullAction())
        push!(mem.records, rec)

        Memory.store_outcome!(mem, 1, Reject, Dict{Symbol,Any}(), 0.0)
        stored = Memory.get_record(mem, 1)
        @test isnothing(stored.latent_μ_at_outcome)
    end

    # ── 3. record_outcome! populates latent on JEPA system ────────
    @testset "record_outcome! captures outcome latent from JEPA belief" begin
        sim    = Main.InSiteSimulator()
        system = Chamelia.initialize_patient(UserPreferences(), sim)

        # Manually set a JEPA belief so _latent_μ_from_belief fires
        outcome_μ = randn(Float32, 64)
        system.belief = JEPABeliefState(
            μ           = outcome_μ,
            log_σ       = fill(-1.0f0, 64),
            entropy     = 0.0f0,
            obs_log_lik = 0.0f0
        )

        # Seed a record so store_outcome! has something to fill in
        epistemic = Perception.compute_epistemic_state(
            system.belief, system.mem, NullAction(),
            system.config.φ_cost.thresholds
        )
        psy = Chamelia.psy_from_belief(system.belief)
        rec_id = Memory.store_record!(system.mem, 1, system.belief,
                                      NullAction(), epistemic, system.config, psy)

        Chamelia.record_outcome!(system, rec_id, Accept,
                                 Dict{Symbol,Any}(:tir => 0.7), 0.3)

        stored = Memory.get_record(system.mem, rec_id)
        @test !isnothing(stored.latent_μ_at_outcome)
        @test stored.latent_μ_at_outcome ≈ outcome_μ
    end

    # ── 4. MemoryTransitionDataset builds triples correctly ───────
    @testset "MemoryTransitionDataset builds from completed records" begin
        mem    = MemoryBuffer()
        action = Actor.CandidateAction(Dict(:x => 0.1))

        for i in 1:5
            rec = _make_jepa_record(i, _make_jepa_belief(z_dim), action)
            rec.latent_μ_at_outcome = randn(Float32, z_dim)
            rec.user_response       = Accept
            push!(mem.records, rec)
        end

        # Record without outcome latent — should be excluded
        incomplete = _make_jepa_record(6, _make_jepa_belief(z_dim), action)
        push!(mem.records, incomplete)

        dataset = Chamelia._build_latent_triples(mem)
        @test Perception.n_samples(dataset) == 5
        @test length(dataset.z_t[1])     == z_dim
        @test length(dataset.a_feats[1]) == 8   # action_to_features output dim
        @test length(dataset.z_tH[1])    == z_dim
    end

    # ── 5. Effective action feature scaling ───────────────────────
    @testset "effective action features scale by response" begin
        action  = Actor.CandidateAction(Dict(:x => 1.0))
        full_a  = WorldModule.action_to_features(action)

        make_rec = (response) -> begin
            rec = _make_jepa_record(1, _make_jepa_belief(z_dim), action)
            rec.user_response       = response
            rec.latent_μ_at_outcome = zeros(Float32, z_dim)
            mem = MemoryBuffer(); push!(mem.records, rec)
            Chamelia._build_latent_triples(mem).a_feats[1]
        end

        # Accept → full action features
        @test make_rec(Accept)  ≈ full_a

        # Partial → half magnitude
        @test make_rec(Partial) ≈ 0.5f0 .* full_a

        # Reject → zeros
        @test all(iszero, make_rec(Reject))
    end

    # ── 6. get_training_batch returns correct shapes ──────────────
    @testset "get_training_batch returns correct tensor shapes" begin
        n = 10; a_dim = 8; batch = 4
        dataset = Perception.MemoryTransitionDataset(
            [randn(Float32, z_dim) for _ in 1:n],
            [randn(Float32, a_dim) for _ in 1:n],
            [randn(Float32, z_dim) for _ in 1:n],
        )
        z_t_b, a_b, z_tH_b = Perception.get_training_batch(dataset, batch)
        @test size(z_t_b)  == (z_dim, batch)
        @test size(a_b)    == (a_dim, batch)
        @test size(z_tH_b) == (z_dim, batch)
    end

    # ── 7. train_predictor! runs and loss is finite ───────────────
    @testset "train_predictor! runs without error and loss is finite" begin
        n = 30; a_dim = 8; z_dim_pred = 64
        dataset = Perception.MemoryTransitionDataset(
            [randn(Float32, z_dim_pred) for _ in 1:n],
            [randn(Float32, a_dim)      for _ in 1:n],
            [randn(Float32, z_dim_pred) for _ in 1:n],
        )
        predictor = WorldModule.JEPAPredictor(z_dim_pred, 16, 128, a_dim)
        Perception.train_predictor!(predictor, dataset; n_epochs=5, batch_size=8)
        @test isfinite(Perception.LAST_PREDICTOR_TRAINING_LOSS[])
    end

    # ── 8. train_predictor! is a no-op on empty dataset ──────────
    @testset "train_predictor! is a no-op on empty dataset" begin
        dataset   = Perception.MemoryTransitionDataset([], [], [])
        predictor = WorldModule.JEPAPredictor()
        # Should return nothing without throwing
        @test isnothing(Perception.train_predictor!(predictor, dataset))
    end

end
