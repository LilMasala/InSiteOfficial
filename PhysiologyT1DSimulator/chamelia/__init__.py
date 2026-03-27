"""Chamelia — self-learning therapy recommender framework.

Nine-block architecture:
    Block 1: Data Layer (t1d_sim + iOS pipeline)
    Block 2: Simulation Layer (t1d_sim closed-loop)
    Block 3: Model Zoo (chamelia.models)
    Block 4: Confidence Module (chamelia.confidence)
    Block 5: Shadow Module (chamelia.shadow)
    Block 6: Meta-Controller (chamelia.meta_controller)
    Block 7: Optimization Engine (chamelia.optimizer)
    Block 8: User Agency (t1d_sim.agency)
    Block 9: Evaluation Layer (chamelia.evaluation)
"""
from chamelia.models.base import PredictionEnvelope, PredictorCard
from chamelia.confidence import ConfidenceModule, GateResult
from chamelia.shadow import ShadowModule, ShadowRecord, Scorecard, GraduationStatus
from chamelia.meta_controller import MetaController, ModelRegistryEntry
from chamelia.optimizer import (
    GridSearchOptimizer,
    TherapyAction,
    ObjectiveWeights,
    RecommendationPackage,
    RecommendationDecision,
)
from chamelia.evaluation import build_robustness_report, RobustnessReport
from chamelia.personality import (
    UserPersonality,
    RecommendationBudget,
    RecommendationFraming,
    sample_personality,
)
from chamelia.therapy_modes import (
    TherapyLevel,
    TherapyModeState,
    get_level_constraints,
    compute_personalization_weight,
)

__all__ = [
    # Block 3
    "PredictionEnvelope",
    "PredictorCard",
    # Block 4
    "ConfidenceModule",
    "GateResult",
    # Block 5
    "ShadowModule",
    "ShadowRecord",
    "Scorecard",
    "GraduationStatus",
    # Block 6
    "MetaController",
    "ModelRegistryEntry",
    # Block 7
    "GridSearchOptimizer",
    "TherapyAction",
    "ObjectiveWeights",
    "RecommendationPackage",
    "RecommendationDecision",
    # Block 9
    "build_robustness_report",
    "RobustnessReport",
    # Personality & Therapy Modes
    "UserPersonality",
    "RecommendationBudget",
    "RecommendationFraming",
    "sample_personality",
    "TherapyLevel",
    "TherapyModeState",
    "get_level_constraints",
    "compute_personalization_weight",
]
