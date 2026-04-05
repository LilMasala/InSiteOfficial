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
from training.curriculum.probes.health_probe import HealthProbe
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


class ScriptedHealthModel(torch.nn.Module):
    """Small scripted model used to validate health-behavior probe metrics."""

    def __init__(
        self,
        runtime_domain,
        embed_dim: int = 32,
        num_ctx_tokens: int = 4,
        num_candidates: int = 4,
        path_length: int = 2,
        posture_dim: int = 6,
        stable_choice: str = "hold",
        fragile_choice: str = "support",
        memory_choice: str | None = None,
    ) -> None:
        super().__init__()
        self.domain = runtime_domain
        self.embed_dim = embed_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.num_candidates = num_candidates
        self.path_length = path_length
        self.posture_dim = posture_dim
        self.stable_choice = stable_choice
        self.fragile_choice = fragile_choice
        self.memory_choice = memory_choice or fragile_choice
        self.memory = LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu")

    def set_domain(self, domain) -> None:
        self.domain = domain

    def _action_template(self, label: str) -> torch.Tensor:
        action = torch.zeros(self.domain.get_action_dim(), dtype=torch.float32)
        action[("hold", "stabilize", "support", "aggressive_optimize").index(label)] = 1.0
        return action

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        domain_state: dict,
        actor_mode: str = "mode2",
        store_to_memory: bool = False,
        input_kind: str = "embedded_tokens",
    ) -> dict[str, torch.Tensor]:
        _ = tokens
        _ = mask
        _ = actor_mode
        _ = store_to_memory
        _ = input_kind
        patient_state = domain_state["patient_state"]
        batch_size = patient_state.shape[0]
        z = torch.zeros(batch_size, self.embed_dim, dtype=torch.float32)
        z[:, : patient_state.shape[1]] = patient_state.float()
        ctx_tokens = torch.zeros(batch_size, self.num_ctx_tokens, self.embed_dim, dtype=torch.float32)
        candidate_postures = torch.zeros(
            batch_size,
            self.num_candidates,
            self.posture_dim,
            dtype=torch.float32,
        )
        for idx in range(self.num_candidates):
            if idx < self.posture_dim:
                candidate_postures[:, idx, idx] = 1.0
        candidate_paths = torch.zeros(
            batch_size,
            self.num_candidates,
            self.path_length,
            self.domain.get_action_dim(),
            dtype=torch.float32,
        )
        candidate_paths[:, 1, :, :] = self._action_template("stabilize").view(1, 1, -1)
        candidate_paths[:, 2, :, :] = self._action_template("support").view(1, 1, -1)
        candidate_paths[:, 3, :, :] = self._action_template("aggressive_optimize").view(1, 1, -1)
        candidate_actions = candidate_paths[:, :, 0, :]

        trust = patient_state[:, 3]
        burden = patient_state[:, 5]
        bg = patient_state[:, 0]
        fragile_mask = (trust < 0.55) | (burden > 0.45) | (bg > 160.0)
        selected_idx = torch.zeros(batch_size, dtype=torch.long)
        fragile_label = self.memory_choice if len(self.memory.records) > 0 else self.fragile_choice
        stable_idx = ("hold", "stabilize", "support", "aggressive_optimize").index(self.stable_choice)
        fragile_idx = ("hold", "stabilize", "support", "aggressive_optimize").index(fragile_label)
        selected_idx[~fragile_mask] = stable_idx
        selected_idx[fragile_mask] = fragile_idx

        selected_path = candidate_paths[torch.arange(batch_size), selected_idx]
        selected_posture = candidate_postures[torch.arange(batch_size), selected_idx]
        action_vec = selected_path[:, 0, :]
        totals = torch.tensor(
            [[0.10, 0.18, 0.22, 0.35]],
            dtype=torch.float32,
        ).repeat(batch_size, 1)
        if fragile_mask.any():
            totals[fragile_mask] = torch.tensor(
                [0.40, 0.24, 0.18, 0.55],
                dtype=torch.float32,
            )
        return {
            "z": z,
            "ctx_tokens": ctx_tokens,
            "candidate_postures": candidate_postures,
            "candidate_paths": candidate_paths,
            "candidate_actions": candidate_actions,
            "selected_path": selected_path,
            "selected_posture": selected_posture,
            "selected_candidate_idx": selected_idx,
            "action_vec": action_vec,
            "candidate_costs": {
                "ic": totals,
                "tc": torch.zeros_like(totals),
                "total": totals,
            },
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


def test_curriculum_runner_exports_bridge_artifact_checkpoint(tmp_path: Path) -> None:
    """Stage checkpoints should emit a real bridge-loadable artifact when export metadata is provided."""
    config_path = Path("/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum.yaml")
    config = yaml.safe_load(config_path.read_text())
    config["checkpoint_dir"] = str(tmp_path)

    stages = [[LanguageCurriculumDomain(batch_size=2, seq_len=8)]]
    manager = GraduationManager(stages, config)
    model = torch.nn.Linear(4, 4)
    runner = CurriculumStageRunner(
        model=model,
        stages=stages,
        graduation_manager=manager,
        config=config,
        device="cpu",
        export_model_config={
            "embed_dim": 4,
            "configurator": {
                "num_ctx_tokens": 2,
                "num_heads": 1,
                "num_layers": 1,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "memory_read_k": 2,
            },
            "actor": {"num_heads": 1, "num_layers": 1, "mlp_ratio": 2.0, "dropout": 0.0},
            "cost": {
                "critic_num_heads": 1,
                "critic_num_layers": 1,
                "critic_mlp_ratio": 2.0,
                "critic_dropout": 0.0,
                "critic_horizon": 4,
            },
            "memory": {"max_episodes": 8, "retrieval_k": 2, "device": "cpu"},
        },
        export_backbone_mode="stub",
        export_model_version="bridge-export-test-v1",
    )

    runner.save_stage_checkpoint(0, {"event": "smoke", "metrics": {"language": {"score": 1.0}}})
    artifacts = list((tmp_path / "bridge_artifacts").glob("*.pth"))
    assert len(artifacts) == 1
    payload = torch.load(artifacts[0], map_location="cpu")
    assert payload["model_version"] == "bridge-export-test-v1"
    assert payload["bridge_backbone_mode"] == "stub"
    assert "model_state_dict" in payload
    assert payload["config"]["embed_dim"] == 4


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


def test_health_runtime_domain_produces_delayed_outcomes_for_memory_training() -> None:
    """Verify the health curriculum can fill delayed outcomes and train from memory."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=2, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(
                list(zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False))[0],
                list(zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False))[1],
            ),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        store_to_memory=True,
        critic_train_interval=1,
        world_model_train_interval=1,
    )

    batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train"))).to_device("cpu")
    step_batch = runner._to_step_batch(batch, domain)
    loss = runner._default_train_step(step_batch, domain)

    assert torch.isfinite(loss)
    assert model.memory.size > 0
    record = model.memory.records[0]
    assert record.ic_realized is not None
    assert record.outcome_key is not None
    assert record.candidate_actions is not None
    assert model.train_critic_from_memory() is not None
    assert model.train_world_model_from_memory() is not None


def test_path_baseline_loss_prefers_complex_path_only_when_it_beats_baseline() -> None:
    """Verify the runner can compute a path-vs-baseline ranking loss."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=2, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        path_baseline_loss_weight=1.0,
        path_baseline_margin=0.05,
    )

    batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train"))).to_device("cpu")
    step_batch = runner._to_step_batch(batch, domain)
    tokenized = runtime_domain.get_tokenizer()(step_batch.model_inputs.long())
    outputs = model(
        tokens=tokenized.tokens,
        mask=step_batch.input_mask,
        domain_state=step_batch.domain_state,
        actor_mode="mode2",
        store_to_memory=False,
        input_kind="embedded_tokens",
    )
    path_loss = runner._compute_path_baseline_loss(
        outputs=outputs,
        step_batch=step_batch,
        runtime_domain=runtime_domain,
    )

    assert path_loss is not None
    assert path_loss.dim() == 0
    assert torch.isfinite(path_loss)


def test_posture_diversity_loss_penalizes_candidate_posture_collapse() -> None:
    """Verify the runner exposes non-baseline candidate-posture diversity pressure."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=2, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        posture_diversity_loss_weight=1.0,
        posture_similarity_cap=0.5,
        posture_path_similarity_cap=0.5,
    )

    collapsed_outputs = {
        "candidate_postures": torch.zeros(1, 4, model.actor.posture_dim),
        "candidate_paths": torch.zeros(1, 4, 3, runtime_domain.get_action_dim()),
    }
    collapsed_outputs["candidate_paths"][:, 1:, :, :] = 1.0
    collapsed_loss = runner._compute_posture_diversity_loss(collapsed_outputs)

    diverse_outputs = {
        "candidate_postures": torch.zeros(1, 4, model.actor.posture_dim),
        "candidate_paths": torch.zeros(1, 4, 3, runtime_domain.get_action_dim()),
    }
    diverse_outputs["candidate_postures"][0, 1, 0] = 1.0
    diverse_outputs["candidate_postures"][0, 2, 1] = 1.0
    diverse_outputs["candidate_postures"][0, 3, 2] = 1.0
    diverse_outputs["candidate_paths"][0, 1, :, 0] = 1.0
    diverse_outputs["candidate_paths"][0, 2, :, 1] = 1.0
    diverse_outputs["candidate_paths"][0, 3, :, 2] = 1.0
    diverse_loss = runner._compute_posture_diversity_loss(diverse_outputs)

    assert collapsed_loss is not None
    assert diverse_loss is not None
    assert float(collapsed_loss.item()) > float(diverse_loss.item())


def test_posture_specialization_loss_prefers_realized_better_nonbaseline_postures() -> None:
    """Delayed-outcome posture specialization should favor non-baseline paths that truly help."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=1, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
            num_candidates=3,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        posture_specialization_loss_weight=1.0,
        posture_specialization_temperature=0.25,
    )

    action_dim = runtime_domain.get_action_dim()
    candidate_paths = torch.zeros(1, 3, 3, action_dim)
    candidate_paths[0, 1, :, 2] = 5.0
    candidate_paths[0, 2, :, 3] = 5.0
    domain_state = {
        "patient_state": torch.tensor([[165.0, -0.5, 0.2, 0.25, 0.75, 0.8]], dtype=torch.float32)
    }
    expanded_state = runner._expand_domain_state_for_candidates(domain_state, num_candidates=3)
    realized = runtime_domain.simulate_path_outcome(
        candidate_paths.reshape(3, 3, action_dim),
        expanded_state,
    )
    assert realized is not None
    realized_costs = realized["realized_intrinsic_cost"].reshape(1, 3)
    assert not torch.isclose(realized_costs[0, 1], realized_costs[0, 2])
    better_nonbaseline = 1 + realized_costs[:, 1:].argmin(dim=1)
    worse_nonbaseline = 1 + realized_costs[:, 1:].argmax(dim=1)

    aligned_total = torch.full((1, 3), 1.0)
    aligned_total[0, 0] = 0.5
    aligned_total[0, int(better_nonbaseline.item())] = 0.0
    aligned_total[0, int(worse_nonbaseline.item())] = 2.0
    misaligned_total = torch.full((1, 3), 1.0)
    misaligned_total[0, 0] = 0.5
    misaligned_total[0, int(better_nonbaseline.item())] = 2.0
    misaligned_total[0, int(worse_nonbaseline.item())] = 0.0

    step_batch = type(
        "StepBatch",
        (),
        {"domain_state": domain_state},
    )()
    aligned_loss = runner._compute_posture_specialization_loss(
        outputs={
            "candidate_postures": torch.zeros(1, 3, model.actor.posture_dim),
            "candidate_paths": candidate_paths,
            "candidate_costs": {"total": aligned_total},
        },
        step_batch=step_batch,
        runtime_domain=runtime_domain,
    )
    misaligned_loss = runner._compute_posture_specialization_loss(
        outputs={
            "candidate_postures": torch.zeros(1, 3, model.actor.posture_dim),
            "candidate_paths": candidate_paths,
            "candidate_costs": {"total": misaligned_total},
        },
        step_batch=step_batch,
        runtime_domain=runtime_domain,
    )

    assert aligned_loss is not None
    assert misaligned_loss is not None
    assert float(aligned_loss.item()) < float(misaligned_loss.item())


def test_retrieval_relevance_loss_prefers_memories_with_better_outcomes_and_posture_match() -> None:
    """Delayed outcomes should directly supervise the learned retrieval reranker."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=1, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
            num_candidates=3,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )
    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        retrieval_relevance_loss_weight=1.0,
        retrieval_relevance_temperature=0.25,
    )

    selected_posture = torch.tensor([[1.0] + [0.0] * (model.actor.posture_dim - 1)], dtype=torch.float32)
    retrieved_postures = torch.zeros(1, 2, model.actor.posture_dim)
    retrieved_postures[0, 0, 0] = 1.0
    retrieved_postures[0, 1, 1] = 1.0
    base_quality_scores = torch.tensor([[-0.1, -0.8]], dtype=torch.float32)
    aligned_outputs = {
        "retrieval_relevance_scores": torch.tensor([[2.0, -1.0]], dtype=torch.float32),
        "retrieved_postures": retrieved_postures,
        "selected_posture": selected_posture,
        "retrieval_base_quality_scores": base_quality_scores,
    }
    misaligned_outputs = {
        "retrieval_relevance_scores": torch.tensor([[-1.0, 2.0]], dtype=torch.float32),
        "retrieved_postures": retrieved_postures,
        "selected_posture": selected_posture,
        "retrieval_base_quality_scores": base_quality_scores,
    }

    aligned_loss = runner._compute_retrieval_relevance_loss(
        outputs=aligned_outputs,
        realized_ic=torch.tensor([0.9], dtype=torch.float32),
    )
    misaligned_loss = runner._compute_retrieval_relevance_loss(
        outputs=misaligned_outputs,
        realized_ic=torch.tensor([0.9], dtype=torch.float32),
    )

    assert aligned_loss is not None
    assert misaligned_loss is not None
    assert float(aligned_loss.item()) < float(misaligned_loss.item())


def test_default_train_step_runs_retrieval_replay_interval() -> None:
    """The runner should schedule retrieval replay alongside other memory updates."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = ReasoningCurriculumDomain(domain_variant="basic_arithmetic", batch_size=1, seq_len=8)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )

    calls = {"count": 0}

    def capture_replay(temperature: float = 0.25) -> torch.Tensor:
        calls["count"] += 1
        assert temperature == 0.25
        return torch.tensor(0.5, dtype=torch.float32)

    model.train_retrieval_from_memory = capture_replay  # type: ignore[method-assign]

    config = {"curriculum": {"start_stage": 0, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device="cpu",
        retrieval_train_interval=1,
        retrieval_replay_loss_weight=1.0,
    )

    batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train"))).to_device("cpu")
    step_batch = runner._to_step_batch(batch, domain)
    loss = runner._default_train_step(step_batch, domain)
    assert torch.isfinite(loss)
    assert calls["count"] == 1


def test_health_probe_scores_simple_and_supportive_behavior() -> None:
    """Health probe should measure simple-path preference and fragile supportive actions."""
    domain = HealthCurriculumDomain(batch_size=1, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None

    probe = HealthProbe(simplicity_margin=0.05)
    model = ScriptedHealthModel(
        runtime_domain=runtime_domain,
        stable_choice="hold",
        fragile_choice="support",
        memory_choice="support",
    )
    metrics = probe.evaluate(
        model,
        {
            "runtime_domain": runtime_domain,
            "cases": [
                {
                    "name": "stable",
                    "state": {
                        "bg": 112.0,
                        "mood": 0.2,
                        "engagement": 0.82,
                        "trust": 0.9,
                        "burnout": 0.08,
                        "burden": 0.1,
                    },
                    "fragile": False,
                    "memory_relevant": False,
                },
                {
                    "name": "fragile",
                    "state": {
                        "bg": 176.0,
                        "mood": -0.25,
                        "engagement": 0.45,
                        "trust": 0.4,
                        "burnout": 0.28,
                        "burden": 0.6,
                    },
                    "fragile": True,
                    "memory_relevant": False,
                },
            ],
        },
    )

    assert metrics["baseline_competitive_rate"] == 0.5
    assert metrics["simple_path_preference"] == 1.0
    assert metrics["fragile_supportive_rate"] == 1.0
    assert metrics["fragile_aggressive_avoidance"] == 1.0


def test_health_probe_measures_memory_plan_shift_without_leaking_memory() -> None:
    """Health probe should detect memory-induced plan shifts and restore probe memory state."""
    domain = HealthCurriculumDomain(batch_size=1, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None

    probe = HealthProbe(simplicity_margin=0.05)
    model = ScriptedHealthModel(
        runtime_domain=runtime_domain,
        stable_choice="hold",
        fragile_choice="aggressive_optimize",
        memory_choice="support",
    )
    metrics = probe.evaluate(
        model,
        {
            "runtime_domain": runtime_domain,
            "cases": [
                {
                    "name": "memory_fragile",
                    "state": {
                        "bg": 182.0,
                        "mood": -0.35,
                        "engagement": 0.38,
                        "trust": 0.32,
                        "burnout": 0.31,
                        "burden": 0.64,
                    },
                    "fragile": True,
                    "memory_relevant": True,
                }
            ],
        },
    )

    assert metrics["memory_plan_shift_rate"] == 1.0
    assert len(model.memory.records) == 0


def test_health_probe_runs_with_real_chamelia_model() -> None:
    """Health probe should produce finite metrics against the repaired Chamelia substrate."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    domain = HealthCurriculumDomain(batch_size=1, seq_len=12)
    runtime_domain = domain.build_runtime_domain(embed_dim=embed_dim)
    assert runtime_domain is not None

    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=runtime_domain.get_action_dim(),
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=16, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    )

    probe = HealthProbe(simplicity_margin=0.05)
    metrics = probe.evaluate(model, {"runtime_domain": runtime_domain})

    expected_keys = {
        "simple_path_preference",
        "baseline_competitive_rate",
        "fragile_supportive_rate",
        "fragile_aggressive_avoidance",
        "mean_realized_advantage_over_baseline",
        "memory_plan_shift_rate",
    }
    assert expected_keys.issubset(metrics.keys())
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
