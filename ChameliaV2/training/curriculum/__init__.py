"""Curriculum scaffolding for Chamelia V2."""

from training.curriculum.cost_schedule import CostLevel, MaturingIntrinsicCost
from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain
from training.curriculum.domains.stage5_health import HealthCurriculumDomain
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
