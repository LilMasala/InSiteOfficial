"""Chamelia model zoo."""
from chamelia.models.base import PredictionEnvelope, PredictorCard
from chamelia.models.stub import StubRecommender
from chamelia.models.aggregate import AggregateOutcomePredictor
from chamelia.models.anomaly import AnomalyDetector
from chamelia.models.behavior import BehaviorResponseModel

# Torch-dependent models — import lazily so the package works without torch.
try:
    from chamelia.models.temporal import TemporalSequenceModel
    from chamelia.models.surrogate import BGDynamicsSurrogate
except ImportError:
    TemporalSequenceModel = None  # type: ignore[assignment,misc]
    BGDynamicsSurrogate = None    # type: ignore[assignment,misc]

__all__ = [
    "PredictionEnvelope",
    "PredictorCard",
    "StubRecommender",
    "AggregateOutcomePredictor",
    "AnomalyDetector",
    "TemporalSequenceModel",
    "BGDynamicsSurrogate",
    "BehaviorResponseModel",
]
