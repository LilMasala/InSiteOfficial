"""Curriculum domains.

Expose stage domain classes lazily so importing the package does not force every
public-data backend to initialize.
"""

from __future__ import annotations

from importlib import import_module

from training.curriculum.domains.base import BaseCurriculumDomain, CurriculumDomain, MaskingStrategy

__all__ = [
    "BaseCurriculumDomain",
    "CollaborativeCurriculumDomain",
    "CurriculumDomain",
    "GamesCurriculumDomain",
    "HealthCurriculumDomain",
    "LanguageCurriculumDomain",
    "MaskingStrategy",
    "PatternCurriculumDomain",
    "ReasoningCurriculumDomain",
]


def __getattr__(name: str):
    """Resolve domain exports lazily."""
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
