"""Basic validation for the curriculum scaffold."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import LatentMemory
from src.chamelia.tokenizers import BoardTokenizer, StructuredStateTokenizer, TimeSeriesTokenizer
from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain
from training.curriculum.domains.stage5_health import HealthCurriculumDomain
from training.curriculum.generators.health_sim import SyntheticPatientEnv
from training.curriculum.graduation import GraduationManager
from training.curriculum.stage_runner import CurriculumStageRunner


class DummyHJEPA(torch.nn.Module):
    """Small HJEPA-compatible stub for curriculum runtime tests."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def _apply_fpn(self, features: torch.Tensor, is_prediction: bool = False) -> list[torch.Tensor]:
        _ = is_prediction
        return [features, features[:, ::2, :], features.mean(dim=1, keepdim=True)]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = mask
        cls = tokens.mean(dim=1, keepdim=True)
        target_features = torch.cat([cls, tokens], dim=1)
        return {
            "predictions": [tokens],
            "targets": [tokens],
            "mask_valid": torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool),
            "context_features": target_features,
            "target_features": target_features,
        }


def test_curriculum_domains_and_cost_progression() -> None:
    """Instantiate scaffold domains and validate loader, masking, and advancement."""
    domains = [
        LanguageCurriculumDomain(batch_size=2, seq_len=16),
        ReasoningCurriculumDomain(domain_variant="basic_arithmetic", batch_size=2, seq_len=16),
        PatternCurriculumDomain(domain_variant="basic_arithmetic_patterns", batch_size=2, seq_len=16),
        GamesCurriculumDomain(domain_variant="chess", batch_size=2, seq_len=16),
        CollaborativeCurriculumDomain(batch_size=2, seq_len=16),
        HealthCurriculumDomain(batch_size=2, seq_len=16),
    ]

    for domain in domains:
        loader = domain.get_data_loader(domain.cost.current_level, split="train")
        batch = next(iter(loader))
        assert batch.tokens is not None
        masked, mask = domain.get_masking_strategy(domain.cost.current_level).apply(
            batch.tokens, domain.cost.current_level
        )
        assert masked.shape == batch.tokens.shape
        assert mask.shape == batch.tokens.shape
        z = torch.randn(batch.tokens.shape[0], 8)
        action = torch.randn(batch.tokens.shape[0], domain.action_dim)
        domain_state = dict(batch.domain_state)
        domain_state["level"] = domain.cost.current_level
        domain_state["premises"] = torch.randn(batch.tokens.shape[0], 4, 8)
        cost_value = domain.cost(z, action, domain_state)
        assert cost_value.shape == (batch.tokens.shape[0],)
        domain.cost.episodes_at_current_level = max(
            domain.cost.episodes_at_current_level,
            domain.get_cost_schedule()[domain.cost.current_level].min_episodes_at_level,
        )
        advanced = domain.cost.maybe_advance(domain.run_advancement_probe(None, domain.cost.current_level))
        assert isinstance(advanced, bool)


def test_curriculum_runner_and_config(tmp_path: Path) -> None:
    """Load curriculum config and validate runner/graduation scaffolding."""
    config_path = Path("/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["checkpoint_dir"] = str(tmp_path)

    stages = [
        [LanguageCurriculumDomain(batch_size=2, seq_len=8)],
        [ReasoningCurriculumDomain(domain_variant="basic_arithmetic", batch_size=2, seq_len=8)],
    ]
    manager = GraduationManager(stages, config)
    runner = CurriculumStageRunner(
        model=torch.nn.Identity(),
        stages=stages,
        graduation_manager=manager,
        config=config,
        device="cpu",
    )

    report = manager.get_status_report()
    assert "stage 0" in report
    probe = manager.run_stage_probe(None, 0)
    assert "language" in probe

    runner.save_stage_checkpoint(0, {"event": "smoke", "metrics": probe})
    saved = list(tmp_path.glob("*.pt"))
    assert saved


def test_tokenizer_family_shapes() -> None:
    """Validate the additional board, time-series, and structured-state tokenizers."""
    board = BoardTokenizer(vocab_size=16, embed_dim=32, max_seq_len=80)
    board_out = board(torch.randint(0, 16, (2, 69)))
    assert board_out.tokens.shape == (2, 69, 32)

    timeseries = TimeSeriesTokenizer(num_features=4, embed_dim=32, max_seq_len=32)
    ts_out = timeseries(torch.randn(2, 10, 4))
    assert ts_out.tokens.shape == (2, 10, 32)

    structured = StructuredStateTokenizer(vocab_size=32, num_continuous=3, embed_dim=32, max_seq_len=16)
    structured_out = structured(
        {
            "categorical_tokens": torch.randint(0, 32, (2, 4)),
            "history_tokens": torch.randint(0, 32, (2, 3)),
            "continuous_values": torch.randn(2, 3),
        }
    )
    assert structured_out.tokens.shape == (2, 8, 32)


def test_arithmetic_runtime_domain_default_train_step() -> None:
    """Verify the arithmetic runtime domain can drive a real default Chamelia train step."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = ReasoningCurriculumDomain(domain_variant="basic_arithmetic", batch_size=2, seq_len=8)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    hjepa = DummyHJEPA(embed_dim=embed_dim)
    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        memory_read_k=4,
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
    )
    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
    )
    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        optimizer=optimizer,
    )

    batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train"))).to_device("cpu")
    step_batch = runner._to_step_batch(batch, domain)
    loss = runner._default_train_step(step_batch, domain)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_synthetic_patient_env_support_action() -> None:
    """Verify the health simulation scaffold reflects gentle vs aggressive care."""
    env = SyntheticPatientEnv()
    env.reset()
    state_aggressive, _, _, _ = env.step("aggressive_optimize", {})
    env.reset()
    state_support, _, _, _ = env.step("support", {})
    assert state_support["trust"] > state_aggressive["trust"]
