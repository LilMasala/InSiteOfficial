"""Curriculum domains."""

from training.curriculum.domains.base import BaseCurriculumDomain, CurriculumDomain, MaskingStrategy
from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain
from training.curriculum.domains.stage5_health import HealthCurriculumDomain

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
