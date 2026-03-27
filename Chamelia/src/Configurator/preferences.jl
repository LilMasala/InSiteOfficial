"""
preferences.jl
User preferences → initial ConfiguratorState.
The starting point before any learning or adaptation.

UserPreferences encodes what the patient wants:
  - how aggressive should recommendations be?
  - how much do they fear hypoglycemia?
  - how sensitive are they to recommendation burden?

These seed the initial configuration but are overridden
by learned adaptation as data accumulates.

Domain adapter separation (Week 5.45):
  Chamelia core owns: aggressiveness → Δ_max, burden_sensitivity → c_burden.
  These are domain-agnostic behavioral/psychological parameters.

  Domain adapters own: the physical cost weight names and the mapping of
  domain-specific preferences (e.g. hypoglycemia_fear → w_low) into those
  weights. Pass an AbstractDomainAdapter to initialize_config to inject
  domain-specific physical weights without touching Chamelia core.
"""

# ─────────────────────────────────────────────────────────────────
# Default configurations
# Conservative starting points — earn trust before expanding
# ─────────────────────────────────────────────────────────────────

const DEFAULT_PERC_CONFIG = PercConfig(
    signal_mask = Dict{Symbol, Bool}(),   # all signals active by default
    H_perc      = 12,                     # 12 hour lookback
    δ_anomaly   = 2.5                     # z-score threshold for anomaly
)

const DEFAULT_WORLD_CONFIG = WorldConfig(
    H_short = 6,    # 6 hour short rollout
    H_med   = 7,    # 7 day medium rollout
    N_roll  = 50    # 50 stochastic rollouts
)

const DEFAULT_COST_CONFIG = CostConfig(
    weights    = CostWeights(
        c_burden  = 0.2,
        c_trust   = 2.0,
        c_burnout = 3.0,
        γ_β       = 0.95,
        physical  = Dict{Symbol, Float64}()
    ),
    thresholds = EpistemicThresholds(
        κ_min = 0.60,
        ρ_min = 0.50,
        η_min = 0.70
    ),
    H_burn     = 30,
    ε_burn     = 0.05,
    γ_discount = 0.99
)

const DEFAULT_ACT_CONFIG = ActConfig(
    Δ_max        = 0.05,   # 5% max change — very conservative start
    δ_min_effect = 0.50,
    α_cvar       = 0.80,
    N_search     = 27
)

# ─────────────────────────────────────────────────────────────────
# Initialize ConfiguratorState from UserPreferences + DomainAdapter
# ─────────────────────────────────────────────────────────────────

"""
    initialize_config(prefs, adapter) → ConfiguratorState

Preferred form. Physical cost weights come from the domain adapter,
keeping Chamelia core free of InSite-specific signal names.
"""
function initialize_config(
    prefs   :: UserPreferences,
    adapter :: AbstractDomainAdapter
) :: ConfiguratorState
    Δ_max    = clamp(0.03 + prefs.aggressiveness * 0.07, 0.03, 0.10)
    c_burden = 0.1 + prefs.burden_sensitivity * 0.3

    act_config = ActConfig(
        Δ_max        = Δ_max,
        δ_min_effect = DEFAULT_ACT_CONFIG.δ_min_effect,
        α_cvar       = DEFAULT_ACT_CONFIG.α_cvar,
        N_search     = DEFAULT_ACT_CONFIG.N_search
    )

    phys_weights = default_physical_weights(adapter, prefs)

    cost_config = CostConfig(
        weights = CostWeights(
            c_burden  = c_burden,
            c_trust   = DEFAULT_COST_CONFIG.weights.c_trust,
            c_burnout = DEFAULT_COST_CONFIG.weights.c_burnout,
            γ_β       = DEFAULT_COST_CONFIG.weights.γ_β,
            physical  = phys_weights
        ),
        thresholds = DEFAULT_COST_CONFIG.thresholds,
        H_burn     = DEFAULT_COST_CONFIG.H_burn,
        ε_burn     = DEFAULT_COST_CONFIG.ε_burn,
        γ_discount = DEFAULT_COST_CONFIG.γ_discount
    )

    ConfiguratorState(
        φ_perc          = DEFAULT_PERC_CONFIG,
        φ_world         = DEFAULT_WORLD_CONFIG,
        φ_cost          = cost_config,
        φ_act           = act_config,
        last_update_day = 0
    )
end

"""
    initialize_config(prefs) → ConfiguratorState

Backward-compatible form. Hardcodes InSite-domain physical cost weights
(w_low, w_high, w_tir, w_var) so existing call sites continue to work.

New domains: call initialize_config(prefs, YourDomainAdapter()) instead.
"""
function initialize_config(prefs :: UserPreferences) :: ConfiguratorState
    Δ_max    = clamp(0.03 + prefs.aggressiveness * 0.07, 0.03, 0.10)
    w_hypo   = 3.0 + prefs.hypoglycemia_fear * 4.0   # InSite/T1D: maps fear → w_low [3,7]
    # w_high scales with aggressiveness to express high-BG motivation — must match
    # InSiteDomainAdapter.default_physical_weights for consistency
    w_high   = 1.0 + prefs.aggressiveness * 0.5       # [1.0, 1.5]
    c_burden = 0.1 + prefs.burden_sensitivity * 0.3

    act_config = ActConfig(
        Δ_max        = Δ_max,
        δ_min_effect = DEFAULT_ACT_CONFIG.δ_min_effect,
        α_cvar       = DEFAULT_ACT_CONFIG.α_cvar,
        N_search     = DEFAULT_ACT_CONFIG.N_search
    )

    cost_config = CostConfig(
        weights = CostWeights(
            c_burden  = c_burden,
            c_trust   = DEFAULT_COST_CONFIG.weights.c_trust,
            c_burnout = DEFAULT_COST_CONFIG.weights.c_burnout,
            γ_β       = DEFAULT_COST_CONFIG.weights.γ_β,
            # InSite-domain signal names — owned by InSiteDomainAdapter going forward
            physical  = Dict{Symbol, Float64}(:w_low => w_hypo, :w_high => w_high,
                                               :w_tir => 1.0,   :w_var  => 0.5)
        ),
        thresholds = DEFAULT_COST_CONFIG.thresholds,
        H_burn     = DEFAULT_COST_CONFIG.H_burn,
        ε_burn     = DEFAULT_COST_CONFIG.ε_burn,
        γ_discount = DEFAULT_COST_CONFIG.γ_discount
    )

    ConfiguratorState(
        φ_perc          = DEFAULT_PERC_CONFIG,
        φ_world         = DEFAULT_WORLD_CONFIG,
        φ_cost          = cost_config,
        φ_act           = act_config,
        last_update_day = 0
    )
end