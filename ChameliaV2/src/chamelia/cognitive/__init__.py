"""Additive cognitive-architecture components for Chamelia."""

from src.chamelia.cognitive.clustering import (
    AttentionLoRAAdapter,
    DomainCluster,
    DomainIndex,
    DomainRoute,
    LoRAAdapterBank,
)
from src.chamelia.cognitive.lancedb_assessment import (
    BackendAssessment,
    assess_vector_backends,
)
from src.chamelia.cognitive.latent_action import (
    LatentActionEncoder,
    LatentSkillCandidate,
    estimate_target_delta,
)
from src.chamelia.cognitive.mamba_world_model import (
    MambaActionConditionedWorldModel,
    WorldModelBenchmark,
    benchmark_world_models,
)
from src.chamelia.cognitive.planning import (
    FrozenReasoningChain,
    HighLevelPlanner,
    MCTSSearch,
    Talker,
    ThinkerOutput,
)
from src.chamelia.cognitive.procedural import (
    ProceduralMemory,
    RetrievedSkill,
    SkillRecord,
)
from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    InformationOrderedBottleneck,
    IsotropicSkillCodec,
    VectorQuantizer,
)
from src.chamelia.cognitive.sleep import (
    ChoreographerEvaluator,
    DreamDecompiler,
    LILOAutoDoc,
    LOVEDecomposer,
    SleepCoordinator,
    SleepCycleReport,
    StitchCompressor,
)
from src.chamelia.cognitive.storage import CognitiveStorage, StoragePaths

__all__ = [
    "CognitiveStorage",
    "StoragePaths",
    "AttentionLoRAAdapter",
    "DomainCluster",
    "DomainIndex",
    "DomainRoute",
    "LoRAAdapterBank",
    "ProceduralMemory",
    "RetrievedSkill",
    "SkillRecord",
    "BackendAssessment",
    "assess_vector_backends",
    "LatentActionEncoder",
    "LatentSkillCandidate",
    "estimate_target_delta",
    "MambaActionConditionedWorldModel",
    "WorldModelBenchmark",
    "benchmark_world_models",
    "FrozenReasoningChain",
    "HighLevelPlanner",
    "MCTSSearch",
    "ThinkerOutput",
    "Talker",
    "VectorQuantizer",
    "InformationOrderedBottleneck",
    "ContrastiveSparseRepresentation",
    "IsotropicSkillCodec",
    "LOVEDecomposer",
    "DreamDecompiler",
    "ChoreographerEvaluator",
    "LILOAutoDoc",
    "StitchCompressor",
    "SleepCoordinator",
    "SleepCycleReport",
]
