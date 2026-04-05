"""Curriculum scaffolding for Chamelia V2.

Keep package imports lazy so stage-specific heavy data dependencies are not loaded
unless the caller actually asks for those domains.
"""

from __future__ import annotations

from importlib import import_module

from training.curriculum.cost_schedule import CostLevel, MaturingIntrinsicCost
from training.curriculum.graduation import GraduationManager
from training.curriculum.stage_runner import CurriculumStageRunner

__all__ = [
    "CollaborativeCurriculumDomain",
    "CostLevel",
    "CurriculumStageRunner",
    "GamesCurriculumDomain",
    "GraduationManager",
    "HealthCurriculumDomain",
    "LanguageCurriculumDomain",
    "MaturingIntrinsicCost",
    "PatternCurriculumDomain",
    "ReasoningCurriculumDomain",
]


def __getattr__(name: str):
    """Resolve curriculum-domain exports lazily."""
    module_map = {
        "LanguageCurriculumDomain": "training.curriculum.domains.stage0_language",
        "ReasoningCurriculumDomain": "training.curriculum.domains.stage1_reasoning",
        "PatternCurriculumDomain": "training.curriculum.domains.stage2_patterns",
        "GamesCurriculumDomain": "training.curriculum.domains.stage3_games",
        "CollaborativeCurriculumDomain": "training.curriculum.domains.stage4_collaborative",
        "HealthCurriculumDomain": "training.curriculum.domains.stage5_health",
    }
    module_name = module_map.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
