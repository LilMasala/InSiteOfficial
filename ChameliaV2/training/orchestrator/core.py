"""Unified orchestration loop for Chamelia cognitive training."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import shutil
import time
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.cognitive.clustering import DomainIndex, LoRAAdapterBank
from src.chamelia.cognitive.latent_action import LatentActionEncoder
from src.chamelia.cognitive.mamba_world_model import MambaActionConditionedWorldModel
from src.chamelia.cognitive.planning import HighLevelPlanner, MCTSSearch
from src.chamelia.cognitive.procedural import ProceduralMemory
from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    InformationOrderedBottleneck,
    IsotropicSkillCodec,
)
from src.chamelia.cognitive.sleep import SleepCoordinator
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.hjepa_adapter import forward_hjepa
from src.chamelia.memory import LatentMemory
from src.chamelia.plugins import CartPoleDomain, Connect4Domain, InteractiveDomainAdapter
from src.chamelia.retrieval import (
    MemoryRelevanceScorer,
    ProceduralRelevanceScorer,
    compute_procedural_reranker_loss,
)
from src.chamelia.world_model import ActionConditionedWorldModel
from src.losses.combined import CombinedLoss
from src.losses.hjepa_loss import HJEPALoss
from src.models.hjepa import HJEPA


def _move_nested_to_device(value: Any, device: torch.device | str) -> Any:
    """Move arbitrarily nested tensor payloads to a device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    return value


def _clone_nested_cpu(value: Any) -> Any:
    """Detach nested tensor payloads onto CPU for replay/checkpoint storage."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _clone_nested_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested_cpu(item) for item in value)
    return value


def _collate_domain_state_batch(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of plugin domain-state payloads into a batched payload."""
    if not states:
        return {}
    keys = set().union(*(state.keys() for state in states))
    collated: dict[str, Any] = {}
    for key in keys:
        values = [state.get(key) for state in states]
        first = values[0]
        if torch.is_tensor(first):
            tensors = []
            for value in values:
                if not torch.is_tensor(value):
                    break
                tensor_value = value
                if tensor_value.dim() == 0:
                    tensor_value = tensor_value.unsqueeze(0)
                tensors.append(tensor_value)
            else:
                collated[key] = torch.cat(tensors, dim=0)
                continue
        collated[key] = values
    return collated


def _infer_encoder_embed_dim(encoder_type: str, configured_embed_dim: int) -> int:
    lowered = encoder_type.lower()
    if "vit_tiny" in lowered:
        return 192
    if "vit_small" in lowered:
        return 384
    if "vit_base" in lowered:
        return 768
    if "vit_large" in lowered:
        return 1024
    if "vit_huge" in lowered:
        return 1280
    return configured_embed_dim


def _build_zero_mask(tokens: torch.Tensor) -> torch.Tensor:
    return torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.float32, device=tokens.device)


def _build_random_mask(
    batch_size: int,
    seq_len: int,
    ratio: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    width = max(1, min(seq_len, int(round(seq_len * ratio))))
    for batch_idx in range(batch_size):
        start = 0 if seq_len == width else random.randint(0, seq_len - width)
        mask[batch_idx, start : start + width] = 1.0
    return mask


def _hierarchy_weights(num_hierarchies: int) -> list[float]:
    """Return a short, stable hierarchy-weight schedule."""
    base = [1.0, 0.7, 0.5, 0.35]
    if num_hierarchies <= len(base):
        return base[:num_hierarchies]
    tail = [base[-1]] * (num_hierarchies - len(base))
    return base + tail


@dataclass
class ReplayRecord:
    """Canonical environment-facing replay entry for unified training."""

    domain_id: str
    modality_family: str
    episode_id: int
    step_idx: int
    obs_raw: Any
    tokenizer_input: Any
    tokens: torch.Tensor
    mask: torch.Tensor
    domain_state: dict[str, Any]
    action: Any
    action_logits_or_vec: torch.Tensor
    legal_actions_mask: torch.Tensor | None
    reward: float
    cost: float
    done: bool
    next_obs_raw: Any
    next_tokens: torch.Tensor
    next_mask: torch.Tensor
    next_domain_state: dict[str, Any]
    latent_z: torch.Tensor | None
    next_latent_z: torch.Tensor | None
    ctx_tokens: torch.Tensor | None
    search_policy: torch.Tensor | None
    search_value: float | None
    selected_path: torch.Tensor | None
    selected_posture: torch.Tensor | None
    memory_hits: int = 0
    procedural_skill_ids: tuple[int, ...] = ()
    reasoning_trace_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "modality_family": self.modality_family,
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "obs_raw": _clone_nested_cpu(self.obs_raw),
            "tokenizer_input": _clone_nested_cpu(self.tokenizer_input),
            "tokens": self.tokens.detach().cpu(),
            "mask": self.mask.detach().cpu(),
            "domain_state": _clone_nested_cpu(self.domain_state),
            "action": _clone_nested_cpu(self.action),
            "action_logits_or_vec": self.action_logits_or_vec.detach().cpu(),
            "legal_actions_mask": None if self.legal_actions_mask is None else self.legal_actions_mask.detach().cpu(),
            "reward": float(self.reward),
            "cost": float(self.cost),
            "done": bool(self.done),
            "next_obs_raw": _clone_nested_cpu(self.next_obs_raw),
            "next_tokens": self.next_tokens.detach().cpu(),
            "next_mask": self.next_mask.detach().cpu(),
            "next_domain_state": _clone_nested_cpu(self.next_domain_state),
            "latent_z": None if self.latent_z is None else self.latent_z.detach().cpu(),
            "next_latent_z": None if self.next_latent_z is None else self.next_latent_z.detach().cpu(),
            "ctx_tokens": None if self.ctx_tokens is None else self.ctx_tokens.detach().cpu(),
            "search_policy": None if self.search_policy is None else self.search_policy.detach().cpu(),
            "search_value": self.search_value,
            "selected_path": None if self.selected_path is None else self.selected_path.detach().cpu(),
            "selected_posture": None if self.selected_posture is None else self.selected_posture.detach().cpu(),
            "memory_hits": int(self.memory_hits),
            "procedural_skill_ids": tuple(int(value) for value in self.procedural_skill_ids),
            "reasoning_trace_id": self.reasoning_trace_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ReplayRecord":
        return cls(
            domain_id=str(payload["domain_id"]),
            modality_family=str(payload["modality_family"]),
            episode_id=int(payload["episode_id"]),
            step_idx=int(payload["step_idx"]),
            obs_raw=payload["obs_raw"],
            tokenizer_input=payload["tokenizer_input"],
            tokens=payload["tokens"].float(),
            mask=payload["mask"].float(),
            domain_state=payload["domain_state"],
            action=payload["action"],
            action_logits_or_vec=payload["action_logits_or_vec"].float(),
            legal_actions_mask=payload["legal_actions_mask"],
            reward=float(payload["reward"]),
            cost=float(payload["cost"]),
            done=bool(payload["done"]),
            next_obs_raw=payload["next_obs_raw"],
            next_tokens=payload["next_tokens"].float(),
            next_mask=payload["next_mask"].float(),
            next_domain_state=payload["next_domain_state"],
            latent_z=None if payload["latent_z"] is None else payload["latent_z"].float(),
            next_latent_z=None if payload["next_latent_z"] is None else payload["next_latent_z"].float(),
            ctx_tokens=None if payload["ctx_tokens"] is None else payload["ctx_tokens"].float(),
            search_policy=None if payload["search_policy"] is None else payload["search_policy"].float(),
            search_value=payload["search_value"],
            selected_path=None if payload["selected_path"] is None else payload["selected_path"].float(),
            selected_posture=None if payload["selected_posture"] is None else payload["selected_posture"].float(),
            memory_hits=int(payload.get("memory_hits", 0)),
            procedural_skill_ids=tuple(int(value) for value in payload.get("procedural_skill_ids", ())),
            reasoning_trace_id=payload.get("reasoning_trace_id"),
        )


@dataclass(frozen=True)
class ReplayWindow:
    """Contiguous replay window sampled from one episode."""

    records: tuple[ReplayRecord, ...]

    @property
    def horizon(self) -> int:
        return len(self.records)


class TransitionReplayBuffer:
    """Simple replay buffer over canonical orchestrator records."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = int(capacity)
        self.records: list[ReplayRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: ReplayRecord) -> None:
        if len(self.records) >= self.capacity:
            self.records.pop(0)
        self.records.append(record)

    def sample(
        self,
        batch_size: int,
        *,
        domain_id: str | None = None,
        require_next_latent: bool = True,
    ) -> list[ReplayRecord]:
        candidates = [
            record
            for record in self.records
            if (domain_id is None or record.domain_id == domain_id)
            and (not require_next_latent or record.next_latent_z is not None)
        ]
        if not candidates:
            return []
        count = min(int(batch_size), len(candidates))
        return random.sample(candidates, count)

    def observation_pool(self, domain_id: str | None = None) -> list[Any]:
        return [
            record.obs_raw
            for record in self.records
            if domain_id is None or record.domain_id == domain_id
        ]

    def sample_windows(
        self,
        batch_size: int,
        *,
        horizon: int,
        domain_id: str | None = None,
    ) -> list[ReplayWindow]:
        if horizon < 1:
            return []
        by_episode: dict[tuple[str, int], list[ReplayRecord]] = defaultdict(list)
        for record in self.records:
            if domain_id is not None and record.domain_id != domain_id:
                continue
            if record.latent_z is None or record.next_latent_z is None:
                continue
            by_episode[(record.domain_id, record.episode_id)].append(record)
        candidates: list[ReplayWindow] = []
        for episode_records in by_episode.values():
            ordered = sorted(episode_records, key=lambda item: item.step_idx)
            for start_idx in range(0, len(ordered) - horizon + 1):
                window = ordered[start_idx : start_idx + horizon]
                contiguous = all(
                    window[offset].step_idx == window[0].step_idx + offset
                    for offset in range(horizon)
                )
                if not contiguous:
                    continue
                candidates.append(ReplayWindow(records=tuple(window)))
        if not candidates:
            return []
        count = min(int(batch_size), len(candidates))
        return random.sample(candidates, count)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "records": [record.to_payload() for record in self.records],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.records = [ReplayRecord.from_payload(payload) for payload in state["records"]]


@dataclass
class DomainPhaseConfig:
    """One subphase budget inside a domain run."""

    episodes: int
    use_memory: bool = False
    use_sleep: bool = False
    train_hjepa: bool = False
    hjepa_lr: float = 1.0e-5
    module_lr: float = 1.0e-4
    actor_loss_weight: float = 1.0
    critic_loss_weight: float = 1.0
    world_model_loss_weight: float = 1.0
    retrieval_loss_weight: float = 0.05
    representation_loss_weight: float = 0.2


@dataclass
class DomainRunConfig:
    """One domain block inside the orchestrator curriculum."""

    name: str
    family: str
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    bootstrap_random_episodes: int = 16
    bootstrap_pretrain_steps: int = 64
    bootstrap_batch_size: int = 16
    mask_ratio: float = 0.25
    max_episode_steps: int = 256
    optimizer_interval: int = 1
    sleep_interval_episodes: int = 250
    checkpoint_interval_episodes: int = 250
    evaluation_episodes: int = 16
    planner_backend: str = "mcts"
    world_model_backend: str = "mamba"
    baselines: tuple[str, ...] = ("random",)
    mcts_simulations: int = 16
    mcts_depth: int = 3
    mcts_rollout_horizon: int = 3
    primary_metric: str = "episode_reward_mean"
    primary_mode: str = "max"
    parity_threshold: float | None = None
    memory_gain_threshold: float | None = None
    phases: dict[str, DomainPhaseConfig] = field(default_factory=dict)


@dataclass
class OrchestratorConfig:
    """Top-level orchestrator configuration."""

    seed: int = 42
    device: str = "cpu"
    run_dir: str = "checkpoints/orchestrator"
    num_ctx_tokens: int = 8
    rollout_horizon: int = 3
    reasoning_steps: int = 2
    replay_capacity: int = 100_000
    family_backbones: dict[str, dict[str, Any]] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    procedural: dict[str, Any] = field(default_factory=dict)
    sleep: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    domains: list[DomainRunConfig] = field(default_factory=list)


def load_orchestrator_config(path: str | Path) -> OrchestratorConfig:
    """Load the orchestrator YAML config."""
    raw = yaml.safe_load(Path(path).read_text())
    if "embed_dim" in raw:
        raise ValueError(
            "Top-level orchestrator 'embed_dim' has been removed; set width in family_backbones only."
        )
    domains: list[DomainRunConfig] = []
    for entry in raw.get("domains", []):
        phases = {
            key: DomainPhaseConfig(**value)
            for key, value in entry.get("phases", {}).items()
        }
        domains.append(
            DomainRunConfig(
                name=str(entry["name"]),
                family=str(entry["family"]),
                adapter_kwargs=dict(entry.get("adapter_kwargs", {})),
                bootstrap_random_episodes=int(entry.get("bootstrap_random_episodes", 16)),
                bootstrap_pretrain_steps=int(entry.get("bootstrap_pretrain_steps", 64)),
                bootstrap_batch_size=int(entry.get("bootstrap_batch_size", 16)),
                mask_ratio=float(entry.get("mask_ratio", 0.25)),
                max_episode_steps=int(entry.get("max_episode_steps", 256)),
                optimizer_interval=int(entry.get("optimizer_interval", 1)),
                sleep_interval_episodes=int(entry.get("sleep_interval_episodes", 250)),
                checkpoint_interval_episodes=int(entry.get("checkpoint_interval_episodes", 250)),
                evaluation_episodes=int(entry.get("evaluation_episodes", 16)),
                planner_backend=str(entry.get("planner_backend", "mcts")),
                world_model_backend=str(entry.get("world_model_backend", "mamba")),
                baselines=tuple(entry.get("baselines", ("random",))),
                mcts_simulations=int(entry.get("mcts_simulations", 16)),
                mcts_depth=int(entry.get("mcts_depth", 3)),
                mcts_rollout_horizon=int(entry.get("mcts_rollout_horizon", 3)),
                primary_metric=str(entry.get("primary_metric", "episode_reward_mean")),
                primary_mode=str(entry.get("primary_mode", "max")),
                parity_threshold=entry.get("parity_threshold"),
                memory_gain_threshold=entry.get("memory_gain_threshold"),
                phases=phases,
            )
        )
    return OrchestratorConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cpu")),
        run_dir=str(raw.get("run_dir", "checkpoints/orchestrator")),
        num_ctx_tokens=int(raw.get("num_ctx_tokens", 8)),
        rollout_horizon=int(raw.get("rollout_horizon", 3)),
        reasoning_steps=int(raw.get("reasoning_steps", 2)),
        replay_capacity=int(raw.get("replay_capacity", 100_000)),
        family_backbones=dict(raw.get("family_backbones", {})),
        memory=dict(raw.get("memory", {})),
        procedural=dict(raw.get("procedural", {})),
        sleep=dict(raw.get("sleep", {})),
        logging=dict(raw.get("logging", {})),
        domains=domains,
    )


class BackboneRegistry:
    """Registry of modality-family HJEPA backbones."""

    def __init__(self, config: dict[str, dict[str, Any]], *, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.models: dict[str, HJEPA] = {}

    def get(self, family_name: str) -> HJEPA:
        if family_name in self.models:
            return self.models[family_name]
        family_cfg = dict(self.config.get(family_name, {}))
        configured_embed_dim = int(family_cfg.get("embed_dim", 128))
        encoder_type = str(family_cfg.get("encoder_type", "vit_tiny_patch16_224"))
        resolved_embed_dim = _infer_encoder_embed_dim(encoder_type, configured_embed_dim)
        model = HJEPA(
            encoder_type=encoder_type,
            img_size=int(family_cfg.get("img_size", 224)),
            embed_dim=resolved_embed_dim,
            predictor_depth=int(family_cfg.get("predictor_depth", 4)),
            predictor_num_heads=int(family_cfg.get("predictor_num_heads", 4)),
            predictor_mlp_ratio=float(family_cfg.get("predictor_mlp_ratio", 2.0)),
            num_hierarchies=int(family_cfg.get("num_hierarchies", 3)),
            pretrained=bool(family_cfg.get("pretrained", False)),
            drop_path_rate=float(family_cfg.get("drop_path_rate", 0.0)),
            use_fpn=bool(family_cfg.get("use_fpn", False)),
            fpn_feature_dim=(
                resolved_embed_dim if bool(family_cfg.get("use_fpn", False)) else None
            ),
            fpn_fusion_method=str(family_cfg.get("fpn_fusion_method", "add")),
            use_gradient_checkpointing=bool(family_cfg.get("use_gradient_checkpointing", False)),
            use_layerscale=bool(family_cfg.get("use_layerscale", False)),
            layerscale_init=float(family_cfg.get("layerscale_init", 1.0e-5)),
            use_flash_attention=bool(family_cfg.get("use_flash_attention", False)),
            sequence_mode=bool(family_cfg.get("sequence_mode", True)),
            use_vq=bool(family_cfg.get("use_vq", False)),
            vq_codebook_size=int(family_cfg.get("vq_codebook_size", 512)),
            vq_beta=float(family_cfg.get("vq_beta", 0.25)),
        ).to(self.device)
        self.models[family_name] = model
        return model

    def state_dict(self) -> dict[str, Any]:
        return {
            family_name: {
                "config": dict(self.config.get(family_name, {})),
                "state_dict": model.state_dict(),
            }
            for family_name, model in self.models.items()
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for family_name, payload in state.items():
            if family_name not in self.config:
                self.config[family_name] = dict(payload.get("config", {}))
            model = self.get(family_name)
            model.load_state_dict(payload["state_dict"])


class UnifiedTrainingOrchestrator:
    """Drive staged HJEPA + Chamelia training over interactive domains."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.replay = TransitionReplayBuffer(capacity=config.replay_capacity)
        self.backbones = BackboneRegistry(config.family_backbones, device=self.device)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self._loss_window: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self._transfer_loss_window: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=50))
        self._optimization_calls = 0
        self._optimizer_steps = 0
        self._transfer_steps = 0
        self.adapter_builders: dict[str, Any] = {
            "cartpole": CartPoleDomain,
            "connect4": Connect4Domain,
        }

    def _diagnostic_path(self, domain_name: str, phase_name: str) -> Path:
        directory = self.run_dir / domain_name / "diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{phase_name}.jsonl"

    def _append_diagnostic_event(
        self,
        *,
        domain_name: str,
        phase_name: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        event = {
            "kind": kind,
            "time": time.time(),
            **payload,
        }
        with self._diagnostic_path(domain_name, phase_name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _gradient_norm(self, model: Chamelia) -> float:
        total = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm = float(param.grad.detach().data.norm(2).item())
            total += grad_norm * grad_norm
        return total ** 0.5

    def _build_transfer_optimizer(self, model: Chamelia) -> torch.optim.Optimizer | None:
        params: list[torch.nn.Parameter] = []
        for module_name in (
            "iob_encoder",
            "csr_encoder",
            "skill_codec",
            "latent_action_encoder",
            "procedural_reranker",
        ):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            params.extend(param for param in module.parameters() if param.requires_grad)
        if not params:
            return None
        return torch.optim.AdamW(params, lr=1.0e-4, weight_decay=1.0e-4)

    def _planner_discount_weights(
        self,
        horizon: int,
        gamma: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        powers = torch.arange(horizon, device=device, dtype=dtype)
        weights = torch.pow(torch.full((horizon,), float(gamma), device=device, dtype=dtype), powers)
        return weights / weights.sum().clamp_min(1.0e-6)

    def _extract_planner_diagnostics(
        self,
        outputs: dict[str, Any],
        *,
        batch_idx: int = 0,
    ) -> dict[str, Any] | None:
        planner_diagnostics = outputs.get("planner_diagnostics") or []
        if (
            isinstance(planner_diagnostics, list)
            and batch_idx < len(planner_diagnostics)
            and isinstance(planner_diagnostics[batch_idx], dict)
        ):
            return planner_diagnostics[batch_idx]
        mcts_traces = outputs.get("mcts_traces") or []
        if (
            isinstance(mcts_traces, list)
            and batch_idx < len(mcts_traces)
            and isinstance(mcts_traces[batch_idx], dict)
        ):
            counterfactual = mcts_traces[batch_idx].get("counterfactual")
            if isinstance(counterfactual, dict):
                return counterfactual
        return None

    def _summarize_outputs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "planner_backend": str(outputs.get("planner_backend", "unknown")),
            "world_model_backend": str(outputs.get("world_model_backend", "unknown")),
        }
        selected_candidate_idx = outputs.get("selected_candidate_idx")
        if torch.is_tensor(selected_candidate_idx):
            summary["selected_candidate_idx"] = int(selected_candidate_idx[0].item())
        candidate_costs = outputs.get("candidate_costs")
        if isinstance(candidate_costs, dict):
            total = candidate_costs.get("total")
            ic = candidate_costs.get("ic")
            tc = candidate_costs.get("tc")
            if torch.is_tensor(total):
                summary["candidate_count"] = int(total.shape[1]) if total.dim() > 1 else int(total.shape[0])
                summary["candidate_total_min"] = float(total[0].min().item())
                summary["candidate_total_mean"] = float(total[0].mean().item())
                summary["candidate_total_max"] = float(total[0].max().item())
            if torch.is_tensor(ic):
                summary["candidate_ic_mean"] = float(ic[0].mean().item())
            if torch.is_tensor(tc):
                summary["candidate_tc_mean"] = float(tc[0].mean().item())
        retrieved_scores = outputs.get("retrieved_episode_scores")
        if torch.is_tensor(retrieved_scores):
            summary["memory_hits"] = int(retrieved_scores[0].numel())
            if retrieved_scores.numel() > 0:
                summary["memory_score_mean"] = float(retrieved_scores[0].mean().item())
        skill_traces = outputs.get("skill_traces") or []
        if skill_traces:
            summary["procedural_skill_count"] = int(len(skill_traces[0] or ()))
        candidate_postures = outputs.get("candidate_postures")
        if torch.is_tensor(candidate_postures):
            summary["candidate_posture_std"] = float(candidate_postures[0].std().item())
            summary["selected_posture_norm"] = float(outputs["selected_posture"][0].norm().item())
        action_vec = outputs.get("action_vec")
        if torch.is_tensor(action_vec):
            summary["selected_action_norm"] = float(action_vec[0].norm().item())
        z = outputs.get("z")
        if torch.is_tensor(z):
            summary["latent_norm"] = float(z[0].norm().item())
        ctx = outputs.get("ctx_tokens")
        if torch.is_tensor(ctx):
            summary["ctx_norm"] = float(ctx[0].norm().item())
        planner_debug = self._extract_planner_diagnostics(outputs, batch_idx=0)
        if planner_debug is not None:
            summary["planner_selection_reason"] = str(planner_debug.get("selection_reason", "unknown"))
            if planner_debug.get("best_actual_idx") is not None:
                summary["planner_best_actual_idx"] = int(planner_debug["best_actual_idx"])
            if planner_debug.get("selected_candidate_idx") is not None:
                summary["planner_selected_idx"] = int(planner_debug["selected_candidate_idx"])
            if planner_debug.get("selected_minus_baseline_predicted") is not None:
                summary["selected_vs_baseline_predicted"] = float(
                    planner_debug["selected_minus_baseline_predicted"]
                )
            if planner_debug.get("selected_minus_baseline_predicted_ic") is not None:
                summary["selected_vs_baseline_predicted_ic"] = float(
                    planner_debug["selected_minus_baseline_predicted_ic"]
                )
            if planner_debug.get("selected_minus_baseline_predicted_discounted_tc") is not None:
                summary["selected_vs_baseline_predicted_discounted_tc"] = float(
                    planner_debug["selected_minus_baseline_predicted_discounted_tc"]
                )
            if planner_debug.get("selected_minus_baseline_actual_cost") is not None:
                summary["selected_vs_baseline_actual_cost"] = float(
                    planner_debug["selected_minus_baseline_actual_cost"]
                )
            if planner_debug.get("selected_minus_baseline_actual_reward") is not None:
                summary["selected_vs_baseline_actual_reward"] = float(
                    planner_debug["selected_minus_baseline_actual_reward"]
                )
            if planner_debug.get("selected_terminal_state_mae") is not None:
                summary["selected_terminal_state_mae"] = float(
                    planner_debug["selected_terminal_state_mae"]
                )
            if planner_debug.get("required_predicted_improvement") is not None:
                summary["required_predicted_improvement"] = float(
                    planner_debug["required_predicted_improvement"]
                )
            if planner_debug.get("root_predicted_cost_std") is not None:
                summary["root_predicted_cost_std"] = float(
                    planner_debug["root_predicted_cost_std"]
                )
            if planner_debug.get("predicted_advantage_source") is not None:
                summary["predicted_advantage_source"] = str(
                    planner_debug["predicted_advantage_source"]
                )
            if planner_debug.get("harmful_pick_source") is not None:
                summary["harmful_pick_source"] = str(
                    planner_debug["harmful_pick_source"]
                )
            if (
                planner_debug.get("best_actual_idx") is not None
                and planner_debug.get("selected_candidate_idx") is not None
            ):
                summary["planner_counterfactual_miss"] = int(
                    int(planner_debug["best_actual_idx"])
                    != int(planner_debug["selected_candidate_idx"])
                )
        return summary

    def _summarize_metrics(
        self,
        metrics: dict[str, Any],
        *,
        primary_metric: str,
    ) -> str:
        pieces = []
        for ablation in ("full", "no_memory", "no_sleep"):
            value = float(metrics.get(ablation, {}).get(primary_metric, 0.0))
            pieces.append(f"{ablation}_{primary_metric}={value:.4f}")
        if "full_train_mode" in metrics:
            value = float(metrics.get("full_train_mode", {}).get(primary_metric, 0.0))
            pieces.append(f"full_train_mode_{primary_metric}={value:.4f}")
        for baseline, baseline_metrics in metrics.get("baselines", {}).items():
            value = float(baseline_metrics.get(primary_metric, 0.0))
            pieces.append(f"{baseline}_{primary_metric}={value:.4f}")
        return " ".join(pieces)

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for domain_cfg in self.config.domains:
            results[domain_cfg.name] = self._run_domain(domain_cfg)
        return results

    def _cleanup_model(self, model: Chamelia | None, adapter: InteractiveDomainAdapter | None = None) -> None:
        """Tear down background workers and storage handles deterministically."""
        if model is not None:
            if model.sleep_coordinator is not None:
                try:
                    model.sleep_coordinator.stop()
                except Exception:
                    pass
            if model.procedural_memory is not None:
                try:
                    model.procedural_memory.close()
                except Exception:
                    pass
            if model.domain_index is not None:
                try:
                    model.domain_index.close()
                except Exception:
                    pass
        if adapter is not None:
            close_fn = getattr(adapter, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        # Give Rust/tokio worker pools a beat to settle before interpreter exit.
        time.sleep(0.1)

    def _build_adapter(self, domain_cfg: DomainRunConfig) -> InteractiveDomainAdapter:
        if domain_cfg.name not in self.adapter_builders:
            raise KeyError(f"No orchestrator adapter registered for domain '{domain_cfg.name}'.")
        family_embed_dim = self.backbones.get(domain_cfg.family).embed_dim
        adapter = self.adapter_builders[domain_cfg.name](
            embed_dim=family_embed_dim,
            **domain_cfg.adapter_kwargs,
        )
        tokenizer = adapter.get_tokenizer()
        if isinstance(tokenizer, torch.nn.Module):
            tokenizer.to(self.device)
        return adapter

    def _build_representation_loss(self, family_name: str) -> torch.nn.Module:
        loss_type = str(self.config.logging.get("representation_loss", "combined")).lower()
        family_cfg = dict(self.config.family_backbones.get(family_name, {}))
        num_hierarchies = int(family_cfg.get("num_hierarchies", 3))
        weights = _hierarchy_weights(num_hierarchies)
        if loss_type == "hjepa":
            return HJEPALoss(
                loss_type="smoothl1",
                hierarchy_weights=weights,
                num_hierarchies=num_hierarchies,
                normalize_embeddings=True,
            ).to(self.device)
        return CombinedLoss(
            jepa_loss_type="smoothl1",
            jepa_hierarchy_weights=weights,
            num_hierarchies=num_hierarchies,
            normalize_embeddings=True,
            vicreg_weight=0.1,
        ).to(self.device)

    def _build_model(
        self,
        domain_cfg: DomainRunConfig,
        adapter: InteractiveDomainAdapter,
    ) -> Chamelia:
        hjepa = self.backbones.get(domain_cfg.family)
        embed_dim = int(hjepa.embed_dim)
        memory_cfg = self.config.memory
        iob = None
        if bool(memory_cfg.get("use_iob", True)):
            iob = InformationOrderedBottleneck(
                input_dim=embed_dim,
                bottleneck_dim=int(memory_cfg.get("iob_dim", embed_dim)),
            ).to(self.device)
        memory = LatentMemory(
            embed_dim=embed_dim,
            max_episodes=int(memory_cfg.get("max_episodes", 20_000)),
            retrieval_k=int(memory_cfg.get("retrieval_k", 8)),
            device=str(memory_cfg.get("device", "cpu")),
            iob_encoder=iob,
            iob_widths=tuple(memory_cfg.get("iob_widths", (32, 64, 128))),
        )
        procedural_cfg = self.config.procedural
        csr = None
        if bool(procedural_cfg.get("use_csr", True)):
            csr = ContrastiveSparseRepresentation(
                input_dim=embed_dim,
                output_dim=int(procedural_cfg.get("csr_dim", 1024)),
                active_dims=int(procedural_cfg.get("csr_active_dims", 64)),
            ).to(self.device)
        codec = None
        if bool(procedural_cfg.get("use_isotropic_storage", True)):
            codec = IsotropicSkillCodec(
                embed_dim=embed_dim,
                num_tokens=int(procedural_cfg.get("num_tokens", 32)),
                codebook_size=int(procedural_cfg.get("codebook_size", 128)),
            ).to(self.device)
        domain_root = self.run_dir / domain_cfg.name
        domain_root.mkdir(parents=True, exist_ok=True)
        procedural = ProceduralMemory(
            root=domain_root / "procedural",
            skill_dim=embed_dim,
            use_faiss=bool(procedural_cfg.get("use_faiss", False)),
            use_lancedb=bool(procedural_cfg.get("use_lancedb", True)),
            csr_encoder=csr,
            codec=codec,
        )
        configurator = Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=self.config.num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.1,
            memory_read_k=int(memory_cfg.get("retrieval_k", 8)),
        )
        actor = Actor(
            embed_dim=embed_dim,
            action_dim=adapter.get_action_dim(),
            num_heads=4,
            num_layers=3,
            mlp_ratio=2.0,
            dropout=0.1,
            num_ctx_tokens=self.config.num_ctx_tokens,
            num_candidates=int(self.config.logging.get("num_candidates", 6)),
            path_length=int(self.config.logging.get("path_length", 3)),
        )
        cost_fns, weights = zip(*adapter.get_intrinsic_cost_fns(), strict=False)
        cost_module = CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.1,
                num_ctx_tokens=self.config.num_ctx_tokens,
                horizon=self.config.rollout_horizon,
            ),
        )
        latent_action = LatentActionEncoder(
            action_dim=adapter.get_action_dim(),
            skill_dim=embed_dim,
            max_path_length=actor.path_length,
        )
        if domain_cfg.world_model_backend == "mamba":
            world_model = MambaActionConditionedWorldModel(
                embed_dim=embed_dim,
                action_dim=adapter.get_action_dim(),
                posture_dim=actor.posture_dim,
                max_horizon=max(self.config.rollout_horizon, actor.path_length),
            )
        else:
            world_model = ActionConditionedWorldModel(
                embed_dim=embed_dim,
                action_dim=adapter.get_action_dim(),
                posture_dim=actor.posture_dim,
                max_horizon=max(self.config.rollout_horizon, actor.path_length),
            )
        sleep = SleepCoordinator(
            episodic_memory=memory,
            procedural_memory=procedural,
            latent_action_encoder=latent_action,
            world_model=world_model,
            cost_module=cost_module,
            domain_index=None,
            sleep_interval_steps=max(1, int(self.config.sleep.get("interval_steps", 64))),
        )
        retrieval_scorer = MemoryRelevanceScorer(
            embed_dim=embed_dim,
            posture_dim=actor.posture_dim,
        )
        procedural_reranker = ProceduralRelevanceScorer(
            embed_dim=embed_dim,
            retrieval_dim=procedural.retrieval_dim,
        )
        high_level_planner = HighLevelPlanner(
            embed_dim=embed_dim,
            skill_dim=embed_dim,
        )
        mcts = MCTSSearch(
            actor=actor,
            world_model=world_model,
            cost_module=cost_module,
            high_level_planner=high_level_planner,
            simulations=domain_cfg.mcts_simulations,
            max_depth=domain_cfg.mcts_depth,
            rollout_horizon=domain_cfg.mcts_rollout_horizon,
        )
        model = Chamelia(
            hjepa=hjepa,
            configurator=configurator,
            actor=actor,
            cost=cost_module,
            memory=memory,
            domain=adapter,
            procedural_memory=procedural,
            mcts_search=mcts,
            domain_index=None,
            sleep_coordinator=sleep,
            world_model=world_model,
            world_model_backend=domain_cfg.world_model_backend,
            retrieval_scorer=retrieval_scorer,
            procedural_reranker=procedural_reranker,
            latent_action_encoder=latent_action,
            iob_encoder=iob,
            csr_encoder=csr,
            skill_codec=codec,
            embed_dim=embed_dim,
            action_dim=adapter.get_action_dim(),
            num_ctx_tokens=self.config.num_ctx_tokens,
            rollout_horizon=self.config.rollout_horizon,
            reasoning_steps=self.config.reasoning_steps,
            planner_backend=domain_cfg.planner_backend,
            model_version=f"{domain_cfg.name}-{domain_cfg.family}",
        ).to(self.device)
        adapter_bank = LoRAAdapterBank(
            model,
            rank=int(procedural_cfg.get("lora_rank", 8)),
        )
        domain_index = DomainIndex(
            domain_root / "cognitive",
            adapter_bank=adapter_bank,
        )
        sleep.domain_index = domain_index
        model.domain_index = domain_index
        model.set_domain(adapter)
        return model

    def _parameter_groups(
        self,
        model: Chamelia,
        domain_cfg: DomainRunConfig,
        phase_cfg: DomainPhaseConfig,
    ) -> list[dict[str, Any]]:
        hjepa_params = list(model.hjepa.parameters())
        tokenizer = model.get_domain_tokenizer()
        tokenizer_params = list(tokenizer.parameters()) if tokenizer is not None else []
        tracked = {id(param) for param in hjepa_params + tokenizer_params}
        transfer_modules = [
            getattr(model, "iob_encoder", None),
            getattr(model, "csr_encoder", None),
            getattr(model, "skill_codec", None),
            getattr(model, "latent_action_encoder", None),
            getattr(model, "procedural_reranker", None),
        ]
        transfer_param_ids = {
            id(param)
            for module in transfer_modules
            if module is not None
            for param in module.parameters()
        }
        module_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in tracked and id(param) not in transfer_param_ids
        ]
        groups = []
        if phase_cfg.train_hjepa:
            groups.append({"params": hjepa_params + tokenizer_params, "lr": phase_cfg.hjepa_lr})
        groups.append({"params": module_params, "lr": phase_cfg.module_lr})
        return groups

    def _encode_latent(
        self,
        model: Chamelia,
        adapter: InteractiveDomainAdapter,
        observation: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = adapter.tokenize_observation(observation)
        tokens = tokenized.tokens.to(self.device)
        mask = _build_zero_mask(tokens)
        with torch.no_grad():
            hjepa_out = forward_hjepa(model.hjepa, tokens, mask, input_kind="embedded_tokens")
            z = hjepa_out["target_features"][:, 0, :]
        return tokens.detach().cpu(), z.squeeze(0).detach().cpu()

    def _pretrain_hjepa(
        self,
        family_name: str,
        adapter: InteractiveDomainAdapter,
        raw_observations: list[Any],
        *,
        steps: int,
        batch_size: int,
        mask_ratio: float,
        lr: float,
    ) -> None:
        if not raw_observations or steps <= 0:
            return
        hjepa = self.backbones.get(family_name)
        tokenizer = adapter.get_tokenizer()
        if isinstance(tokenizer, torch.nn.Module):
            tokenizer.to(self.device)
        params = list(hjepa.parameters())
        if isinstance(tokenizer, torch.nn.Module):
            params.extend(tokenizer.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        hjepa.train()
        if isinstance(tokenizer, torch.nn.Module):
            tokenizer.train()
        family_loss = self._build_representation_loss(family_name)
        for _ in range(steps):
            batch_obs = random.sample(raw_observations, min(batch_size, len(raw_observations)))
            prepared = [adapter.prepare_bridge_observation(obs) for obs in batch_obs]
            collated = tokenizer.collate(prepared) if hasattr(tokenizer, "collate") else torch.stack(prepared, dim=0)
            collated = collated.to(self.device)
            tokenized = tokenizer(collated)
            mask = _build_random_mask(
                tokenized.tokens.shape[0],
                tokenized.tokens.shape[1],
                mask_ratio,
                device=self.device,
            )
            outputs = forward_hjepa(hjepa, tokenized.tokens, mask, input_kind="embedded_tokens")
            optimizer.zero_grad(set_to_none=True)
            loss_dict = family_loss(
                outputs["predictions"],
                outputs["targets"],
                masks=outputs["masks_valid"],
                context_features=outputs.get("context_features"),
                vq_commitment_loss=outputs.get("vq_commitment_loss"),
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

    def _extract_search_policy(
        self,
        outputs: dict[str, Any],
        *,
        batch_idx: int,
        action_dim: int,
    ) -> torch.Tensor | None:
        candidate_actions = outputs["candidate_actions"][batch_idx].detach().cpu()
        target = torch.zeros(action_dim, dtype=torch.float32)
        mcts_traces = outputs.get("mcts_traces") or []
        if batch_idx < len(mcts_traces) and mcts_traces[batch_idx] is not None:
            children = list(mcts_traces[batch_idx].get("children", []))
            for child_idx, child in enumerate(children[: candidate_actions.shape[0]]):
                action_idx = int(candidate_actions[child_idx].argmax().item())
                target[action_idx] += float(child.get("visit_count", 0))
            if float(target.sum().item()) > 0.0:
                return target / target.sum()
        candidate_costs = outputs["candidate_costs"]["total"][batch_idx].detach().cpu()
        weights = torch.softmax(-candidate_costs, dim=0)
        for child_idx, weight in enumerate(weights[: candidate_actions.shape[0]]):
            action_idx = int(candidate_actions[child_idx].argmax().item())
            target[action_idx] += float(weight.item())
        if float(target.sum().item()) <= 0.0:
            return None
        return target / target.sum()

    def _collect_random_bootstrap(self, adapter: InteractiveDomainAdapter, domain_cfg: DomainRunConfig) -> list[Any]:
        observations: list[Any] = []
        for episode_idx in range(domain_cfg.bootstrap_random_episodes):
            observation, info = adapter.reset(seed=self.config.seed + episode_idx)
            for _ in range(domain_cfg.max_episode_steps):
                observations.append(_clone_nested_cpu(observation))
                action = adapter.baseline_action("random", observation, info)
                observation, _reward, terminated, truncated, info = adapter.step(action)
                if terminated or truncated:
                    break
        return observations

    def _compute_actor_loss(
        self,
        model: Chamelia,
        records: list[ReplayRecord],
        action_space_type: str,
    ) -> torch.Tensor | None:
        _ = action_space_type
        valid = [
            record
            for record in records
            if record.latent_z is not None and record.ctx_tokens is not None
        ]
        if not valid:
            return None
        states = torch.stack([record.latent_z for record in valid], dim=0)
        ctx_tokens = torch.stack([record.ctx_tokens for record in valid], dim=0)
        states = states.to(self.device)
        ctx_tokens = ctx_tokens.to(self.device)
        domain_state = _move_nested_to_device(
            _collate_domain_state_batch([record.domain_state for record in valid]),
            self.device,
        )
        simple_baseline_path = model.domain.build_simple_baseline_path(
            domain_state,
            model.actor.path_length,
            model.actor.action_dim,
        )
        proposal = model.actor.propose(
            states,
            ctx_tokens,
            simple_baseline_path=simple_baseline_path,
        )
        rollout = model.world_model(
            z=states,
            actions=proposal["candidate_paths"],
            ctx_tokens=ctx_tokens,
            candidate_postures=proposal["candidate_postures"],
            reasoning_states=proposal["reasoning_states"],
            horizon=min(model.actor.path_length, model.world_model.max_horizon),
        )
        candidate_costs = model.cost.score_candidates(
            z=states,
            actions=proposal["candidate_paths"],
            ctx_tokens=ctx_tokens,
            domain_state=domain_state,
            future_z=rollout["terminal_latents"],
            future_trajectory=rollout["trajectory"],
            imagined_domain_state_builder=model.domain.build_imagined_domain_state,
        )
        candidate_paths = proposal["candidate_paths"]
        selection_logits = proposal["candidate_selection_logits"]
        target_weights: torch.Tensor | None = None
        path_loss: torch.Tensor | None = None
        target_paths: list[torch.Tensor] = []
        have_target_paths = True
        for record in valid:
            selected_path = record.selected_path
            if selected_path is None:
                have_target_paths = False
                break
            target_path = selected_path.float().to(self.device)
            if target_path.dim() == 1:
                target_path = target_path.unsqueeze(0)
            if target_path.shape[-1] != candidate_paths.shape[-1]:
                action_delta = int(candidate_paths.shape[-1]) - int(target_path.shape[-1])
                if action_delta < 0:
                    target_path = target_path[:, : candidate_paths.shape[-1]]
                else:
                    target_path = F.pad(target_path, (0, action_delta))
            if target_path.shape[0] != candidate_paths.shape[2]:
                if target_path.shape[0] > candidate_paths.shape[2]:
                    target_path = target_path[: candidate_paths.shape[2], :]
                else:
                    pad_steps = int(candidate_paths.shape[2]) - int(target_path.shape[0])
                    pad_value = (
                        target_path[-1:, :]
                        if target_path.shape[0] > 0
                        else torch.zeros(
                            1,
                            candidate_paths.shape[-1],
                            device=self.device,
                            dtype=candidate_paths.dtype,
                        )
                    )
                    target_path = torch.cat(
                        [target_path, pad_value.expand(pad_steps, -1)],
                        dim=0,
                    )
            target_paths.append(target_path.to(candidate_paths.dtype))
        if have_target_paths and target_paths:
            target_path_tensor = torch.stack(target_paths, dim=0)
            expanded_targets = target_path_tensor.unsqueeze(1).expand(
                -1,
                candidate_paths.shape[1],
                -1,
                -1,
            )
            if action_space_type == "discrete":
                target_actions = target_path_tensor.argmax(dim=-1)
                flat_logits = candidate_paths.reshape(
                    candidate_paths.shape[0] * candidate_paths.shape[1] * candidate_paths.shape[2],
                    candidate_paths.shape[3],
                )
                flat_targets = target_actions.unsqueeze(1).expand(
                    -1,
                    candidate_paths.shape[1],
                    -1,
                ).reshape(-1)
                per_step_errors = F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    reduction="none",
                ).view(
                    candidate_paths.shape[0],
                    candidate_paths.shape[1],
                    candidate_paths.shape[2],
                )
                path_errors = per_step_errors.mean(dim=-1)
            else:
                path_errors = F.smooth_l1_loss(
                    candidate_paths,
                    expanded_targets,
                    reduction="none",
                ).mean(dim=(-1, -2))
            path_loss = path_errors.min(dim=1).values.mean()
            target_weights = torch.softmax((-path_errors.detach()) / 0.25, dim=1)
        if target_weights is None:
            target_weights = torch.softmax(
                (-candidate_costs["total"].detach()) / 0.25,
                dim=1,
            )
        selection_loss = -(target_weights * F.log_softmax(selection_logits, dim=1)).sum(dim=1).mean()
        diversity_loss = model.actor.compute_posture_diversity_loss(
            proposal["candidate_postures"],
            candidate_paths,
        )
        total_loss = selection_loss + (0.02 * diversity_loss)
        if path_loss is not None:
            total_loss = total_loss + path_loss
        return total_loss

    def _count_actor_valid(self, records: list[ReplayRecord]) -> int:
        return sum(
            1
            for record in records
            if record.latent_z is not None and record.ctx_tokens is not None
        )

    def _compute_world_model_loss(self, model: Chamelia, windows: list[ReplayWindow]) -> torch.Tensor | None:
        valid = [
            window
            for window in windows
            if window.records
            and window.records[0].latent_z is not None
            and window.records[0].ctx_tokens is not None
            and all(record.next_latent_z is not None for record in window.records)
        ]
        if not valid:
            return None
        horizon = min(
            len(valid[0].records),
            model.world_model.max_horizon,
        )
        z_t = torch.stack([window.records[0].latent_z for window in valid], dim=0).to(self.device)
        ctx_tokens = torch.stack([window.records[0].ctx_tokens for window in valid], dim=0).to(self.device)
        action_paths = torch.stack(
            [
                torch.stack(
                    [record.action_logits_or_vec for record in window.records[:horizon]],
                    dim=0,
                )
                for window in valid
            ],
            dim=0,
        ).to(self.device).unsqueeze(1)
        target_trajectory = torch.stack(
            [
                torch.stack(
                    [record.next_latent_z for record in window.records[:horizon]],
                    dim=0,
                )
                for window in valid
            ],
            dim=0,
        ).to(self.device)
        step_weights = self._planner_discount_weights(
            horizon,
            float(model.cost.gamma),
            dtype=target_trajectory.dtype,
            device=target_trajectory.device,
        )
        world_model_loss, predicted_trajectory = model.world_model.compute_trajectory_loss(
            z_t=z_t,
            actions=action_paths,
            target_trajectory=target_trajectory,
            ctx_tokens=ctx_tokens,
            step_weights=step_weights,
        )
        decoder_losses: list[torch.Tensor] = []
        calibration_losses: list[torch.Tensor] = []
        predicted_steps = predicted_trajectory[:, 0, :, :]
        terminal_transition_loss = F.smooth_l1_loss(
            predicted_steps[:, -1, :],
            target_trajectory[:, -1, :].detach(),
        )
        world_model_loss = world_model_loss + (0.5 * terminal_transition_loss)
        for step_idx in range(horizon):
            target_domain_state = _move_nested_to_device(
                _collate_domain_state_batch(
                    [window.records[step_idx].next_domain_state for window in valid]
                ),
                self.device,
            )
            decoder_loss = model.domain.compute_latent_state_decoder_loss(
                predicted_steps[:, step_idx, :],
                target_domain_state,
            )
            if decoder_loss is not None:
                decoder_losses.append(step_weights[step_idx] * decoder_loss)
            calibration_loss = model.domain.compute_imagined_state_calibration_loss(
                predicted_steps[:, step_idx, :],
                action_paths[:, 0, step_idx, :],
                target_domain_state,
                step_idx,
            )
            if calibration_loss is not None:
                calibration_losses.append(step_weights[step_idx] * calibration_loss)
        if decoder_losses:
            world_model_loss = world_model_loss + (0.25 * torch.stack(decoder_losses).sum())
        if calibration_losses:
            world_model_loss = world_model_loss + (0.5 * torch.stack(calibration_losses).sum())
        return world_model_loss

    def _count_world_model_valid(self, windows: list[ReplayWindow]) -> int:
        return sum(
            1
            for window in windows
            if window.records
            and window.records[0].latent_z is not None
            and window.records[0].ctx_tokens is not None
            and all(record.next_latent_z is not None for record in window.records)
        )

    def _compute_critic_loss(
        self,
        model: Chamelia,
        windows: list[ReplayWindow],
        *,
        fallback_records: list[ReplayRecord] | None = None,
    ) -> torch.Tensor | None:
        valid_windows = [
            window
            for window in windows
            if len(window.records) >= 2
            and window.records[0].next_latent_z is not None
            and window.records[0].ctx_tokens is not None
        ]
        if valid_windows:
            gamma = float(model.cost.gamma)
            keys = torch.stack(
                [window.records[0].next_latent_z for window in valid_windows],
                dim=0,
            ).to(self.device)
            ctx_tokens = torch.stack(
                [window.records[0].ctx_tokens for window in valid_windows],
                dim=0,
            ).to(self.device)
            targets = torch.zeros(len(valid_windows), dtype=torch.float32, device=self.device)
            bootstrap_indices: list[int] = []
            bootstrap_latents: list[torch.Tensor] = []
            bootstrap_ctx_tokens: list[torch.Tensor] = []
            bootstrap_scales: list[float] = []

            for window_idx, window in enumerate(valid_windows):
                future_records = window.records[1:]
                discounted_future = 0.0
                for step_offset, record in enumerate(future_records):
                    discounted_future += (gamma ** step_offset) * float(record.cost)
                targets[window_idx] = discounted_future
                last_record = window.records[-1]
                if (
                    not last_record.done
                    and last_record.next_latent_z is not None
                    and last_record.ctx_tokens is not None
                ):
                    bootstrap_indices.append(window_idx)
                    bootstrap_latents.append(last_record.next_latent_z)
                    bootstrap_ctx_tokens.append(last_record.ctx_tokens)
                    bootstrap_scales.append(gamma ** len(future_records))

            if bootstrap_indices:
                bootstrap_values = model.cost.trainable_critic(
                    torch.stack(bootstrap_latents, dim=0).to(self.device),
                    torch.stack(bootstrap_ctx_tokens, dim=0).to(self.device),
                ).detach()
                for target_idx, scale, bootstrap_value in zip(
                    bootstrap_indices,
                    bootstrap_scales,
                    bootstrap_values,
                    strict=False,
                ):
                    targets[target_idx] += float(scale) * float(bootstrap_value.item())

            predicted = model.cost.trainable_critic(keys, ctx_tokens)
            return model.cost.trainable_critic.compute_critic_loss(predicted, targets)

        valid_records = [
            record
            for record in (fallback_records or [])
            if record.next_latent_z is not None and record.ctx_tokens is not None
        ]
        if not valid_records:
            return None
        keys = torch.stack([record.next_latent_z for record in valid_records], dim=0).to(self.device)
        ctx_tokens = torch.stack([record.ctx_tokens for record in valid_records], dim=0).to(self.device)
        costs = torch.tensor([record.cost for record in valid_records], dtype=torch.float32, device=self.device)
        predicted = model.cost.trainable_critic(keys, ctx_tokens)
        return model.cost.trainable_critic.compute_critic_loss(predicted, costs)

    def _count_critic_valid(
        self,
        windows: list[ReplayWindow],
        *,
        fallback_records: list[ReplayRecord] | None = None,
    ) -> int:
        valid_window_count = sum(
            1
            for window in windows
            if len(window.records) >= 2
            and window.records[0].next_latent_z is not None
            and window.records[0].ctx_tokens is not None
        )
        if valid_window_count > 0:
            return valid_window_count
        return sum(
            1
            for record in (fallback_records or [])
            if record.next_latent_z is not None and record.ctx_tokens is not None
        )

    def _compute_iob_loss(
        self,
        model: Chamelia,
        *,
        domain_name: str,
        batch_size: int = 32,
        temperature: float = 0.1,
    ) -> torch.Tensor | None:
        if model.iob_encoder is None:
            return None
        candidates = [
            record
            for record in model.memory.records[: model.memory.size]
            if record.outcome_key is not None and record.domain_name == domain_name
        ]
        if len(candidates) < 3:
            return None
        sampled = random.sample(candidates, min(batch_size, len(candidates)))
        keys = torch.stack([record.key for record in sampled], dim=0).to(self.device)
        outcomes = torch.stack([record.outcome_key for record in sampled if record.outcome_key is not None], dim=0).to(self.device)
        if outcomes.shape[0] != keys.shape[0]:
            return None
        outcome_norm = F.normalize(outcomes, dim=-1)
        outcome_similarity = outcome_norm @ outcome_norm.T
        identity = torch.eye(outcome_similarity.shape[0], dtype=torch.bool, device=self.device)
        positive_mask = (outcome_similarity >= 0.8) & (~identity)
        negative_mask = (outcome_similarity <= 0.2) & (~identity)
        if not positive_mask.any() or not negative_mask.any():
            return None
        encoded = model.iob_encoder(keys)
        widths = model.memory._resolve_iob_widths()
        if not widths:
            widths = (model.iob_encoder.bottleneck_dim,)
        losses: list[torch.Tensor] = []
        for width in widths:
            truncated = model.iob_encoder.truncate(encoded, width)
            normalized = F.normalize(truncated, dim=-1)
            logits = (normalized @ normalized.T) / max(float(temperature), 1.0e-4)
            logits = logits.masked_fill(identity, float("-inf"))
            valid_mask = positive_mask | negative_mask
            masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
            pos_logsumexp = torch.logsumexp(masked_logits.masked_fill(~positive_mask, float("-inf")), dim=1)
            denom_logsumexp = torch.logsumexp(masked_logits, dim=1)
            anchor_mask = positive_mask.any(dim=1) & negative_mask.any(dim=1)
            if anchor_mask.any():
                losses.append((-(pos_logsumexp - denom_logsumexp))[anchor_mask].mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compute_latent_action_loss(
        self,
        model: Chamelia,
        windows: list[ReplayWindow],
    ) -> torch.Tensor | None:
        if model.latent_action_encoder is None:
            return None
        valid = [
            window
            for window in windows
            if window.records
            and window.records[0].latent_z is not None
            and window.records[-1].next_latent_z is not None
        ]
        losses: list[torch.Tensor] = []
        if valid:
            action_paths = torch.stack(
                [
                    torch.stack([record.action_logits_or_vec for record in window.records], dim=0)
                    for window in valid
                ],
                dim=0,
            ).to(self.device)
            target_delta = torch.stack(
                [
                    window.records[-1].next_latent_z - window.records[0].latent_z
                    for window in valid
                ],
                dim=0,
            ).to(self.device)
            losses.append(
                model.latent_action_encoder.compute_loss(
                    action_paths,
                    target_delta,
                    path_length=action_paths.shape[1],
                )
            )
        if model.procedural_memory is not None and model.procedural_memory.records:
            skill_records = [
                record
                for record in model.procedural_memory.records.values()
                if record.extras is not None and record.extras.get("target_delta") is not None
            ]
            if skill_records:
                sampled = random.sample(skill_records, min(16, len(skill_records)))
                skill_paths = torch.stack([record.action_path for record in sampled], dim=0).to(self.device)
                target_delta = torch.stack(
                    [
                        record.extras["target_delta"]
                        if torch.is_tensor(record.extras["target_delta"])
                        else torch.tensor(record.extras["target_delta"], dtype=torch.float32)
                        for record in sampled
                    ],
                    dim=0,
                ).to(self.device)
                losses.append(
                    model.latent_action_encoder.compute_loss(
                        skill_paths,
                        target_delta,
                        path_length=skill_paths.shape[1],
                    )
                )
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _compute_codec_loss(self, model: Chamelia) -> torch.Tensor | None:
        if model.skill_codec is None or model.procedural_memory is None or not model.procedural_memory.records:
            return None
        sampled = random.sample(
            list(model.procedural_memory.records.values()),
            min(32, len(model.procedural_memory.records)),
        )
        embeddings = torch.stack([record.embedding for record in sampled], dim=0).to(self.device)
        return model.skill_codec(embeddings)["loss"]

    def _compute_procedural_reranker_loss(self, model: Chamelia) -> torch.Tensor | None:
        if (
            model.procedural_reranker is None
            or model.sleep_coordinator is None
            or model.procedural_memory is None
        ):
            return None
        examples = model.sleep_coordinator.sample_reranker_examples(batch_size=8)
        if not examples:
            return None
        losses: list[torch.Tensor] = []
        for example in examples:
            query_latent = example.query_latent.unsqueeze(0).to(self.device)
            skill_embeddings = example.skill_embeddings.unsqueeze(0).to(self.device)
            if model.csr_encoder is not None:
                retrieval_vectors = model.csr_encoder(skill_embeddings.squeeze(0)).unsqueeze(0)
            else:
                retrieval_vectors = skill_embeddings
            retrieval_similarity = F.cosine_similarity(
                F.normalize(query_latent.unsqueeze(1).expand_as(skill_embeddings), dim=-1),
                F.normalize(skill_embeddings, dim=-1),
                dim=-1,
            )
            confidence = example.confidence.unsqueeze(0).to(self.device)
            scores = model.procedural_reranker(
                query_latent=query_latent,
                skill_embeddings=skill_embeddings,
                retrieval_vectors=retrieval_vectors,
                retrieval_similarity=retrieval_similarity,
                confidence=confidence,
            )
            loss = compute_procedural_reranker_loss(
                scores["scores"],
                example.evaluation_scores.unsqueeze(0).to(self.device),
            )
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _optimize_transfer_modules(
        self,
        model: Chamelia,
        *,
        domain_cfg: DomainRunConfig,
        phase_name: str,
        optimizer: torch.optim.Optimizer | None,
        global_step: int,
    ) -> None:
        if optimizer is None or global_step % 16 != 0:
            return
        windows = self.replay.sample_windows(
            batch_size=64,
            horizon=min(self.config.rollout_horizon, model.world_model.max_horizon),
            domain_id=domain_cfg.name,
        )
        losses = {
            "iob": self._compute_iob_loss(model, domain_name=domain_cfg.name),
            "latent_action": self._compute_latent_action_loss(model, windows),
            "codec": self._compute_codec_loss(model),
            "procedural_reranker": self._compute_procedural_reranker_loss(model),
        }
        active = {key: value for key, value in losses.items() if value is not None}
        if not active:
            return
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.stack(list(active.values())).sum()
        total_loss.backward()
        optimizer.step()
        if model.memory.iob_encoder is not None:
            model.memory.refresh_iob_keys()
        if model.procedural_memory is not None:
            model.procedural_memory.refresh_representations()
        for key, value in active.items():
            self._transfer_loss_window[key].append(float(value.detach().item()))
        self._transfer_steps += 1
        if self._transfer_steps % 10 == 0:
            summary = " ".join(
                f"{key}_loss={mean(self._transfer_loss_window[key]):.4f}"
                for key in sorted(self._transfer_loss_window)
                if self._transfer_loss_window[key]
            )
            print(
                f"[{domain_cfg.name}] transfer_step={self._transfer_steps} {summary}",
                flush=True,
            )
            self._append_diagnostic_event(
                domain_name=domain_cfg.name,
                phase_name=phase_name,
                kind="transfer_step",
                payload={
                    "global_step": global_step,
                    "transfer_step": self._transfer_steps,
                    **{
                        f"{key}_loss": float(mean(self._transfer_loss_window[key]))
                        for key in self._transfer_loss_window
                        if self._transfer_loss_window[key]
                    },
                },
            )

    def _compute_representation_loss(
        self,
        loss_fn: torch.nn.Module,
        outputs: dict[str, Any],
    ) -> torch.Tensor:
        hjepa_out = outputs["hjepa_out"]
        return loss_fn(
            hjepa_out["predictions"],
            hjepa_out["targets"],
            masks=hjepa_out["masks_valid"],
            context_features=hjepa_out.get("context_features"),
            vq_commitment_loss=hjepa_out.get("vq_commitment_loss"),
        )["loss"]

    def _optimize_from_replay(
        self,
        model: Chamelia,
        outputs: dict[str, Any],
        replay_record: ReplayRecord,
        *,
        phase_name: str,
        phase_cfg: DomainPhaseConfig,
        optimizer: torch.optim.Optimizer,
        representation_loss_fn: torch.nn.Module,
        domain_cfg: DomainRunConfig,
        global_step: int,
    ) -> dict[str, float]:
        self._optimization_calls += 1
        losses: dict[str, torch.Tensor] = {}
        actor_batch = self.replay.sample(32, domain_id=domain_cfg.name, require_next_latent=False)
        actor_valid = self._count_actor_valid(actor_batch)
        actor_loss = self._compute_actor_loss(
            model,
            actor_batch,
            action_space_type=model.domain.action_space_type if isinstance(model.domain, InteractiveDomainAdapter) else "discrete",
        )
        if actor_loss is not None:
            losses["actor"] = phase_cfg.actor_loss_weight * actor_loss
        world_model_windows = self.replay.sample_windows(
            32,
            horizon=min(self.config.rollout_horizon, model.world_model.max_horizon),
            domain_id=domain_cfg.name,
        )
        world_model_valid = self._count_world_model_valid(world_model_windows)
        world_model_loss = self._compute_world_model_loss(model, world_model_windows)
        if world_model_loss is not None:
            losses["world_model"] = phase_cfg.world_model_loss_weight * world_model_loss
        critic_batch = self.replay.sample(32, domain_id=domain_cfg.name)
        critic_windows: list[ReplayWindow] = []
        max_critic_horizon = max(
            2,
            min(
                self.config.rollout_horizon + 1,
                model.world_model.max_horizon + 1,
            ),
        )
        for critic_horizon in range(max_critic_horizon, 1, -1):
            critic_windows = self.replay.sample_windows(
                32,
                horizon=critic_horizon,
                domain_id=domain_cfg.name,
            )
            if critic_windows:
                break
        critic_valid = self._count_critic_valid(
            critic_windows,
            fallback_records=critic_batch,
        )
        critic_loss = self._compute_critic_loss(
            model,
            critic_windows,
            fallback_records=critic_batch,
        )
        if critic_loss is not None:
            losses["critic"] = phase_cfg.critic_loss_weight * critic_loss
        rep_active = phase_cfg.train_hjepa
        if phase_cfg.train_hjepa:
            losses["rep"] = phase_cfg.representation_loss_weight * self._compute_representation_loss(
                representation_loss_fn,
                outputs,
            )
        retrieval_active = False
        if phase_cfg.use_memory and global_step % max(1, self.config.memory.get("retrieval_train_interval", 8)) == 0:
            retrieval_active = True
            retrieval_loss = model.train_retrieval_from_memory()
            if retrieval_loss is not None:
                losses["retrieval"] = phase_cfg.retrieval_loss_weight * retrieval_loss
        if not losses:
            if self._optimization_calls % 50 == 0:
                print(
                    f"[{domain_cfg.name}] opt_call={self._optimization_calls} optimizer_skipped=1 "
                    f"actor_batch={len(actor_batch)} actor_valid={actor_valid} "
                    f"wm_batch={len(world_model_windows)} wm_valid={world_model_valid} "
                    f"critic_valid={critic_valid} retrieval_active={int(retrieval_active)} "
                    f"rep_active={int(rep_active)} replay={len(self.replay)}",
                    flush=True,
                )
            return {}
        total = torch.stack(list(losses.values())).sum()
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = self._gradient_norm(model)
        optimizer.step()
        detached_losses = {key: float(value.detach().item()) for key, value in losses.items()}
        for key, value in detached_losses.items():
            self._loss_window[key].append(value)
        self._optimizer_steps += 1
        if self._optimizer_steps % 50 == 0:
            loss_parts = []
            for key in ("world_model", "actor", "critic", "retrieval", "rep"):
                window = self._loss_window.get(key)
                if window:
                    loss_parts.append(f"{key}_loss={mean(window):.4f}")
            if loss_parts:
                print(
                    f"[{domain_cfg.name}] opt_call={self._optimization_calls} opt_step={self._optimizer_steps} "
                    + " ".join(loss_parts)
                    + (
                        f" grad_norm={grad_norm:.4f} replay={len(self.replay)} actor_valid={actor_valid} "
                        f"wm_valid={world_model_valid} critic_valid={critic_valid}"
                    ),
                    flush=True,
                )
        self._append_diagnostic_event(
            domain_name=domain_cfg.name,
            phase_name=phase_name,
            kind="optimizer",
            payload={
                "opt_call": self._optimization_calls,
                "opt_step": self._optimizer_steps,
                "global_step": global_step,
                "grad_norm": grad_norm,
                "replay_size": len(self.replay),
                "actor_valid": actor_valid,
                "world_model_valid": world_model_valid,
                "critic_valid": critic_valid,
                "retrieval_active": int(retrieval_active),
                **{f"{key}_loss": value for key, value in detached_losses.items()},
            },
        )
        return detached_losses

    def _episode_record(
        self,
        *,
        domain_cfg: DomainRunConfig,
        episode_id: int,
        step_idx: int,
        observation: Any,
        tokenized: Any,
        domain_state: dict[str, Any],
        outputs: dict[str, Any],
        next_observation: Any,
        next_tokens: torch.Tensor,
        next_z: torch.Tensor,
        reward: float,
        cost: float,
        done: bool,
        legal_actions_mask: torch.Tensor | None,
    ) -> ReplayRecord:
        return ReplayRecord(
            domain_id=domain_cfg.name,
            modality_family=domain_cfg.family,
            episode_id=episode_id,
            step_idx=step_idx,
            obs_raw=_clone_nested_cpu(observation),
            tokenizer_input=_clone_nested_cpu(domain_state),
            tokens=tokenized.tokens.detach().cpu().squeeze(0),
            mask=_build_zero_mask(tokenized.tokens).detach().cpu().squeeze(0),
            domain_state=_clone_nested_cpu(domain_state),
            action=_clone_nested_cpu(outputs["action"][0] if torch.is_tensor(outputs["action"]) else outputs["action"]),
            action_logits_or_vec=outputs["action_vec"].detach().cpu()[0],
            legal_actions_mask=None if legal_actions_mask is None else legal_actions_mask.detach().cpu().reshape(-1),
            reward=float(reward),
            cost=float(cost),
            done=bool(done),
            next_obs_raw=_clone_nested_cpu(next_observation),
            next_tokens=next_tokens.detach().cpu().squeeze(0),
            next_mask=torch.zeros(next_tokens.shape[1], dtype=torch.float32),
            next_domain_state=_clone_nested_cpu({}),
            latent_z=outputs["z"].detach().cpu()[0],
            next_latent_z=next_z.detach().cpu(),
            ctx_tokens=outputs["ctx_tokens"].detach().cpu()[0],
            search_policy=self._extract_search_policy(
                outputs,
                batch_idx=0,
                action_dim=outputs["action_vec"].shape[-1],
            ),
            search_value=float(-outputs["cost"]["total"][0].detach().item()),
            selected_path=outputs["selected_path"].detach().cpu()[0],
            selected_posture=outputs["selected_posture"].detach().cpu()[0],
            memory_hits=0 if outputs["retrieved_episode_scores"] is None else int(outputs["retrieved_episode_scores"][0].numel()),
            procedural_skill_ids=tuple(outputs.get("skill_traces", [()])[0] or ()),
            reasoning_trace_id=f"{domain_cfg.name}:{episode_id}:{step_idx}",
        )

    @contextmanager
    def _ablation_context(self, model: Chamelia, ablation: str):
        original_memory = model.memory
        original_procedural = model.procedural_memory
        original_sleep = model.sleep_coordinator
        if ablation == "no_memory":
            model.memory = LatentMemory(
                embed_dim=original_memory.embed_dim,
                max_episodes=original_memory.max_episodes,
                retrieval_k=original_memory.retrieval_k,
                device=original_memory.device,
            )
            model.procedural_memory = None
            model.sleep_coordinator = None
        elif ablation == "no_sleep":
            model.procedural_memory = None
            model.sleep_coordinator = None
        try:
            yield
        finally:
            model.memory = original_memory
            model.procedural_memory = original_procedural
            model.sleep_coordinator = original_sleep

    def _evaluate_baseline(
        self,
        adapter: InteractiveDomainAdapter,
        domain_cfg: DomainRunConfig,
        *,
        episodes: int,
        kind: str,
    ) -> dict[str, float]:
        summaries: list[dict[str, Any]] = []
        if hasattr(adapter, "set_eval_opponent_depth") and domain_cfg.name == "connect4":
            adapter.set_eval_opponent_depth(4)
        for episode_idx in range(episodes):
            observation, info = adapter.reset(seed=self.config.seed + 10_000 + episode_idx)
            reward_total = 0.0
            winner = 0
            step_count = 0
            for step_count in range(1, domain_cfg.max_episode_steps + 1):
                action = adapter.baseline_action(kind, observation, info)
                observation, reward, terminated, truncated, info = adapter.step(action)
                reward_total += float(reward)
                if terminated or truncated:
                    winner = int(info.get("winner", 0))
                    break
            summaries.append(
                {
                    "episode_reward": reward_total,
                    "episode_length": step_count,
                    "winner": winner,
                }
            )
        if hasattr(adapter, "set_eval_opponent_depth") and domain_cfg.name == "connect4":
            adapter.set_eval_opponent_depth(0)
        metrics = adapter.compute_metrics(summaries)
        metrics["episodes"] = float(episodes)
        return metrics

    def _evaluate_model(
        self,
        model: Chamelia,
        adapter: InteractiveDomainAdapter,
        domain_cfg: DomainRunConfig,
        *,
        episodes: int,
        ablation: str,
        use_train_mode: bool = False,
    ) -> dict[str, float]:
        summaries: list[dict[str, Any]] = []
        with self._ablation_context(model, ablation):
            was_training = model.training
            if use_train_mode:
                model.train()
            else:
                model.eval()
            if hasattr(adapter, "set_eval_opponent_depth") and domain_cfg.name == "connect4":
                adapter.set_eval_opponent_depth(4)
            with torch.no_grad():
                for episode_idx in range(episodes):
                    observation, info = adapter.reset(seed=self.config.seed + 20_000 + episode_idx)
                    reward_total = 0.0
                    step_count = 0
                    winner = 0
                    for step_count in range(1, domain_cfg.max_episode_steps + 1):
                        tokenized = adapter.tokenize_observation(observation)
                        tokens = tokenized.tokens.to(self.device)
                        outputs = model(
                            tokens=tokens,
                            mask=_build_zero_mask(tokens),
                            domain_state=_move_nested_to_device(adapter.build_domain_state(observation, info), self.device),
                            actor_mode="mode2",
                            store_to_memory=False,
                            input_kind="embedded_tokens",
                        )
                        action = outputs["action"]
                        if torch.is_tensor(action):
                            chosen_action = action[0]
                        else:
                            chosen_action = action
                        observation, reward, terminated, truncated, info = adapter.step(chosen_action)
                        reward_total += float(reward)
                        if terminated or truncated:
                            winner = int(info.get("winner", 0))
                            break
                    summaries.append(
                        {
                            "episode_reward": reward_total,
                            "episode_length": step_count,
                            "winner": winner,
                        }
                    )
            if hasattr(adapter, "set_eval_opponent_depth") and domain_cfg.name == "connect4":
                adapter.set_eval_opponent_depth(0)
            model.train(was_training)
        metrics = adapter.compute_metrics(summaries)
        metrics["episodes"] = float(episodes)
        return metrics

    def _save_checkpoint(
        self,
        *,
        domain_cfg: DomainRunConfig,
        phase_name: str,
        episode_idx: int,
        model: Chamelia,
        optimizer: torch.optim.Optimizer,
        evaluation: dict[str, Any],
    ) -> Path:
        checkpoint_dir = self.run_dir / domain_cfg.name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{phase_name}-ep{episode_idx:05d}.pt"
        model.procedural_memory.save()
        payload = {
            "domain_name": domain_cfg.name,
            "family_name": domain_cfg.family,
            "phase": phase_name,
            "episode_idx": episode_idx,
            "config": {
                "domain": domain_cfg.__dict__,
                "orchestrator": self.config.__dict__,
            },
            "backbones": self.backbones.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episodic_memory": model.memory.state_dict(),
            "replay": self.replay.state_dict(),
            "procedural_root": str(model.procedural_memory.storage.paths.root),
            "sleep_metadata": {
                "last_report": model.sleep_coordinator.last_report if model.sleep_coordinator is not None else None,
            },
            "evaluation": evaluation,
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        domain_cfg: DomainRunConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[Chamelia, dict[str, Any]]:
        """Restore one orchestrator checkpoint into a fresh domain model."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.backbones.load_state_dict(payload["backbones"])
        adapter = self._build_adapter(domain_cfg)
        model = self._build_model(domain_cfg, adapter)
        model.load_state_dict(payload["model_state_dict"])
        model.memory.load_state_dict(payload["episodic_memory"])
        self.replay.load_state_dict(payload["replay"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return model, payload

    def _promote_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        domain_cfg: DomainRunConfig,
        phase_name: str,
        is_best: bool,
    ) -> None:
        """Maintain stable latest/best checkpoint aliases per phase."""
        checkpoint_dir = checkpoint_path.parent
        latest_path = checkpoint_dir / f"{phase_name}-latest.pt"
        self._link_or_copy_checkpoint(checkpoint_path, latest_path)
        if is_best:
            best_path = checkpoint_dir / f"{phase_name}-best.pt"
            self._link_or_copy_checkpoint(checkpoint_path, best_path)

    def _link_or_copy_checkpoint(self, source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
        try:
            target.symlink_to(source.name)
            return
        except OSError:
            pass
        shutil.copy2(source, target)

    def _phase_passed(self, metrics: dict[str, Any], domain_cfg: DomainRunConfig) -> bool:
        primary = float(metrics["full"].get(domain_cfg.primary_metric, 0.0))
        parity_threshold = domain_cfg.parity_threshold
        memory_threshold = domain_cfg.memory_gain_threshold
        if parity_threshold is not None and primary < float(parity_threshold):
            return False
        if memory_threshold is not None:
            memoryless = float(metrics["no_memory"].get(domain_cfg.primary_metric, 0.0))
            if primary - memoryless < float(memory_threshold):
                return False
        return True

    def _run_phase(
        self,
        *,
        phase_name: str,
        phase_cfg: DomainPhaseConfig,
        domain_cfg: DomainRunConfig,
        adapter: InteractiveDomainAdapter,
        model: Chamelia,
        representation_loss_fn: torch.nn.Module,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self._parameter_groups(model, domain_cfg, phase_cfg))
        transfer_optimizer = self._build_transfer_optimizer(model)
        self._loss_window.clear()
        self._transfer_loss_window.clear()
        self._optimization_calls = 0
        self._optimizer_steps = 0
        self._transfer_steps = 0
        best_metric = float("-inf")
        phase_metrics: dict[str, Any] = {}
        episode_rewards: deque[float] = deque(maxlen=100)
        episode_lengths: deque[int] = deque(maxlen=100)
        print(
            f"[{domain_cfg.name}] starting phase={phase_name} "
            f"episodes={phase_cfg.episodes} memory={phase_cfg.use_memory} "
            f"sleep={phase_cfg.use_sleep} train_hjepa={phase_cfg.train_hjepa}",
            flush=True,
        )
        if phase_cfg.use_sleep and model.sleep_coordinator is not None:
            model.sleep_coordinator.start()
        global_step = 0
        progress_interval = max(
            1,
            min(
                domain_cfg.checkpoint_interval_episodes,
                max(phase_cfg.episodes // 10, 10),
            ),
        )
        for episode_idx in range(1, phase_cfg.episodes + 1):
            observation, info = adapter.reset(seed=self.config.seed + episode_idx)
            episode_reward = 0.0
            winner = 0
            for step_idx in range(domain_cfg.max_episode_steps):
                tokenized = adapter.tokenize_observation(observation)
                tokens = tokenized.tokens.to(self.device)
                domain_state = adapter.build_domain_state(observation, info)
                outputs = model(
                    tokens=tokens,
                    mask=_build_zero_mask(tokens),
                    domain_state=_move_nested_to_device(domain_state, self.device),
                    actor_mode="mode2",
                    store_to_memory=phase_cfg.use_memory,
                    input_kind="embedded_tokens",
                )
                action = outputs["action"]
                chosen_action = action[0] if torch.is_tensor(action) else action
                next_observation, reward, terminated, truncated, info = adapter.step(chosen_action)
                next_tokens, next_z = self._encode_latent(model, adapter, next_observation)
                realized_cost = adapter.compute_realized_cost(
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                if phase_cfg.use_memory:
                    model.fill_outcome(
                        ic_realized=realized_cost,
                        outcome_z=next_z.unsqueeze(0).to(self.device),
                    )
                record = self._episode_record(
                    domain_cfg=domain_cfg,
                    episode_id=episode_idx,
                    step_idx=step_idx,
                    observation=observation,
                    tokenized=tokenized,
                    domain_state=domain_state,
                    outputs=outputs,
                    next_observation=next_observation,
                    next_tokens=next_tokens,
                    next_z=next_z,
                    reward=reward,
                    cost=realized_cost,
                    done=bool(terminated or truncated),
                    legal_actions_mask=adapter.legal_action_mask(observation, info),
                )
                record.next_domain_state = _clone_nested_cpu(adapter.build_domain_state(next_observation, info))
                self.replay.add(record)
                global_step += 1
                if global_step % max(1, domain_cfg.optimizer_interval) == 0:
                    self._optimize_from_replay(
                        model,
                        outputs,
                        record,
                        phase_name=phase_name,
                        phase_cfg=phase_cfg,
                        optimizer=optimizer,
                        representation_loss_fn=representation_loss_fn,
                        domain_cfg=domain_cfg,
                        global_step=global_step,
                    )
                self._optimize_transfer_modules(
                    model,
                    domain_cfg=domain_cfg,
                    phase_name=phase_name,
                    optimizer=transfer_optimizer,
                    global_step=global_step,
                )
                episode_reward += float(reward)
                observation = next_observation
                if terminated or truncated:
                    winner = int(info.get("winner", 0))
                    break
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(int(step_idx + 1))
            if episode_idx % progress_interval == 0 or episode_idx == 1:
                output_summary = self._summarize_outputs(outputs)
                planner_debug = self._extract_planner_diagnostics(outputs, batch_idx=0)
                print(
                    f"[{domain_cfg.name}][{phase_name}] episode={episode_idx}/{phase_cfg.episodes} "
                    f"reward={episode_reward:.3f} winner={winner} replay={len(self.replay)} "
                    f"memory_size={model.memory.size} "
                    f"reward_mean100={mean(episode_rewards):.3f} "
                    f"len_mean100={mean(episode_lengths):.2f} "
                    f"candidate_total_min={float(output_summary.get('candidate_total_min', 0.0)):.4f} "
                    f"candidate_total_mean={float(output_summary.get('candidate_total_mean', 0.0)):.4f} "
                    f"candidate_total_max={float(output_summary.get('candidate_total_max', 0.0)):.4f} "
                    f"memory_hits={int(output_summary.get('memory_hits', 0))} "
                    f"skills={int(output_summary.get('procedural_skill_count', 0))}"
                    + (
                        f" mcts_reason={output_summary.get('planner_selection_reason', 'n/a')}"
                        f" sel_vs_base_pred={float(output_summary.get('selected_vs_baseline_predicted', 0.0)):.4f}"
                        f" pred_ic={float(output_summary.get('selected_vs_baseline_predicted_ic', 0.0)):.4f}"
                        f" pred_tc_tail={float(output_summary.get('selected_vs_baseline_predicted_discounted_tc', 0.0)):.4f}"
                        f" sel_vs_base_actual={float(output_summary.get('selected_vs_baseline_actual_cost', 0.0)):.4f}"
                        f" sel_term_mae={float(output_summary.get('selected_terminal_state_mae', 0.0)):.4f}"
                        f" rank_src={str(output_summary.get('predicted_advantage_source', 'n/a'))}"
                        f" harmful_src={str(output_summary.get('harmful_pick_source', 'n/a'))}"
                        f" pred_guard={float(output_summary.get('required_predicted_improvement', 0.0)):.4f}"
                        f" pred_std={float(output_summary.get('root_predicted_cost_std', 0.0)):.4f}"
                        if planner_debug is not None
                        else ""
                    ),
                    flush=True,
                )
                self._append_diagnostic_event(
                    domain_name=domain_cfg.name,
                    phase_name=phase_name,
                    kind="episode_progress",
                    payload={
                        "episode_idx": episode_idx,
                        "episode_reward": float(episode_reward),
                        "winner": winner,
                        "replay_size": len(self.replay),
                        "memory_size": model.memory.size,
                        "reward_mean100": mean(episode_rewards),
                        "episode_length_mean100": mean(episode_lengths),
                        "planner_debug": planner_debug,
                        **output_summary,
                    },
                )
            if phase_cfg.use_sleep and episode_idx % max(1, domain_cfg.sleep_interval_episodes) == 0 and model.sleep_coordinator is not None:
                print(
                    f"[{domain_cfg.name}][{phase_name}] sleep requested at episode={episode_idx}",
                    flush=True,
                )
                model.sleep_coordinator.request_run()
            if episode_idx % max(1, domain_cfg.checkpoint_interval_episodes) == 0 or episode_idx == phase_cfg.episodes:
                metrics = {
                    "full": self._evaluate_model(
                        model,
                        adapter,
                        domain_cfg,
                        episodes=domain_cfg.evaluation_episodes,
                        ablation="full",
                    ),
                    "no_memory": self._evaluate_model(
                        model,
                        adapter,
                        domain_cfg,
                        episodes=domain_cfg.evaluation_episodes,
                        ablation="no_memory",
                    ),
                    "no_sleep": self._evaluate_model(
                        model,
                        adapter,
                        domain_cfg,
                        episodes=domain_cfg.evaluation_episodes,
                        ablation="no_sleep",
                    ),
                    "full_train_mode": self._evaluate_model(
                        model,
                        adapter,
                        domain_cfg,
                        episodes=domain_cfg.evaluation_episodes,
                        ablation="full",
                        use_train_mode=True,
                    ),
                    "baselines": {
                        baseline: self._evaluate_baseline(
                            adapter,
                            domain_cfg,
                            episodes=domain_cfg.evaluation_episodes,
                            kind=baseline,
                        )
                        for baseline in domain_cfg.baselines
                    },
                }
                checkpoint_path = self._save_checkpoint(
                    domain_cfg=domain_cfg,
                    phase_name=phase_name,
                    episode_idx=episode_idx,
                    model=model,
                    optimizer=optimizer,
                    evaluation=metrics,
                )
                phase_metrics = metrics
                metric_value = float(metrics["full"].get(domain_cfg.primary_metric, 0.0))
                print(
                    f"[{domain_cfg.name}][{phase_name}] checkpoint episode={episode_idx} "
                    + self._summarize_metrics(metrics, primary_metric=domain_cfg.primary_metric)
                    + f" checkpoint={checkpoint_path.name}",
                    flush=True,
                )
                self._append_diagnostic_event(
                    domain_name=domain_cfg.name,
                    phase_name=phase_name,
                    kind="checkpoint",
                    payload={
                        "episode_idx": episode_idx,
                        "checkpoint_path": str(checkpoint_path),
                        "primary_metric": domain_cfg.primary_metric,
                        "phase_pass_candidate": self._phase_passed(metrics, domain_cfg),
                        "metrics": metrics,
                    },
                )
                if domain_cfg.primary_mode == "min":
                    metric_value = -metric_value
                is_best = metric_value >= best_metric
                best_metric = max(best_metric, metric_value)
                self._promote_checkpoint(
                    checkpoint_path,
                    domain_cfg=domain_cfg,
                    phase_name=phase_name,
                    is_best=is_best,
                )
        if phase_cfg.use_sleep and model.sleep_coordinator is not None:
            model.sleep_coordinator.stop()
            if model.sleep_coordinator.last_report is not None:
                report = model.sleep_coordinator.last_report
                print(
                    f"[{domain_cfg.name}][{phase_name}] sleep report promotions={len(report.promotions)} "
                    f"segments={report.decomposed_segments} dream={report.dream_candidates} "
                    f"rsd={report.rsd_candidates} bodegen={report.bodegen_candidates}",
                    flush=True,
                )
                self._append_diagnostic_event(
                    domain_name=domain_cfg.name,
                    phase_name=phase_name,
                    kind="sleep_report",
                    payload={
                        "promotions": len(report.promotions),
                        "segments": report.decomposed_segments,
                        "dream_candidates": report.dream_candidates,
                        "rsd_candidates": report.rsd_candidates,
                        "bodegen_candidates": report.bodegen_candidates,
                    },
                )
        phase_metrics["phase_passed"] = self._phase_passed(phase_metrics, domain_cfg) if phase_metrics else False
        phase_metrics["best_metric"] = best_metric
        print(
            f"[{domain_cfg.name}] completed phase={phase_name} "
            f"passed={phase_metrics['phase_passed']} best_metric={best_metric:.4f}",
            flush=True,
        )
        self._append_diagnostic_event(
            domain_name=domain_cfg.name,
            phase_name=phase_name,
            kind="phase_complete",
            payload={
                "phase_passed": phase_metrics["phase_passed"],
                "best_metric": best_metric,
                "primary_metric": domain_cfg.primary_metric,
            },
        )
        return phase_metrics

    def _run_domain(self, domain_cfg: DomainRunConfig) -> dict[str, Any]:
        adapter = self._build_adapter(domain_cfg)
        model: Chamelia | None = None
        try:
            print(
                f"[{domain_cfg.name}] bootstrap_random_episodes={domain_cfg.bootstrap_random_episodes} "
                f"bootstrap_pretrain_steps={domain_cfg.bootstrap_pretrain_steps}",
                flush=True,
            )
            bootstrap_observations = self._collect_random_bootstrap(adapter, domain_cfg)
            self._pretrain_hjepa(
                domain_cfg.family,
                adapter,
                bootstrap_observations,
                steps=domain_cfg.bootstrap_pretrain_steps,
                batch_size=domain_cfg.bootstrap_batch_size,
                mask_ratio=domain_cfg.mask_ratio,
                lr=1.0e-4,
            )
            model = self._build_model(domain_cfg, adapter)
            representation_loss_fn = self._build_representation_loss(domain_cfg.family)
            domain_results: dict[str, Any] = {}
            ordered_phases = ("core_control", "episodic_memory", "sleep")
            for phase_name in ordered_phases:
                phase_cfg = domain_cfg.phases.get(phase_name)
                if phase_cfg is None or phase_cfg.episodes <= 0:
                    continue
                domain_results[phase_name] = self._run_phase(
                    phase_name=phase_name,
                    phase_cfg=phase_cfg,
                    domain_cfg=domain_cfg,
                    adapter=adapter,
                    model=model,
                    representation_loss_fn=representation_loss_fn,
                )
            print(f"[{domain_cfg.name}] domain complete", flush=True)
            return domain_results
        finally:
            self._cleanup_model(model, adapter)
