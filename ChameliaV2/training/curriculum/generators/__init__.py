"""Synthetic generators and environments for curriculum scaffolding."""

from training.curriculum.generators.chess_env import StockfishChessEnv
from training.curriculum.generators.collab_env import CollaborativeSelfPlayEnv
from training.curriculum.generators.gridworld_gen import HiddenRegimeGridworldGenerator
from training.curriculum.generators.health_sim import SyntheticPatientEnv
from training.curriculum.generators.logic_gen import BasicArithmeticGenerator, LogicProblemGenerator
from training.curriculum.generators.poker_env import PokerEnv
from training.curriculum.generators.sequence_gen import (
    ArithmeticSequenceGenerator,
    HiddenMarkovSequenceGenerator,
)

__all__ = [
    "ArithmeticSequenceGenerator",
    "BasicArithmeticGenerator",
    "CollaborativeSelfPlayEnv",
    "HiddenMarkovSequenceGenerator",
    "HiddenRegimeGridworldGenerator",
    "LogicProblemGenerator",
    "PokerEnv",
    "StockfishChessEnv",
    "SyntheticPatientEnv",
]
