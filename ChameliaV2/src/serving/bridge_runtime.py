"""Session-scoped runtime for the Julia ↔ Python Chamelia bridge."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.hjepa_adapter import forward_hjepa
from src.chamelia.memory import LatentMemory
from src.chamelia.memory import EpisodeRecord, RetrievalTraceStep
from src.chamelia.plugins import DomainRegistry, InSiteBridgeDomain, ProteinDTIDomain
from src.chamelia.plugins.base import AbstractDomain
from src.chamelia.retrieval import MemoryRelevanceScorer
from src.chamelia.world_model import ActionConditionedWorldModel
from src.models.hjepa import HJEPA
from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain
from training.curriculum.domains.stage5_health import HealthCurriculumDomain


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _StubSequenceHJEPA(torch.nn.Module):
    """Small token-input backbone for bridge-mode local serving."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def _apply_fpn(self, features: torch.Tensor, is_prediction: bool = False) -> list[torch.Tensor]:
        _ = is_prediction
        return [features, features[:, ::2, :], features.mean(dim=1, keepdim=True)]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        masked_tokens = tokens * (1.0 - mask.unsqueeze(-1))
        cls = masked_tokens.mean(dim=1, keepdim=True)
        target_features = torch.cat([cls, tokens], dim=1)
        predictions = self._apply_fpn(masked_tokens, is_prediction=True)
        targets = self._apply_fpn(tokens, is_prediction=False)
        masks_valid = [
            torch.ones(tokens.shape[0], level.shape[1], dtype=torch.bool, device=tokens.device)
            for level in predictions
        ]
        return {
            "predictions": predictions,
            "targets": targets,
            "masks_valid": masks_valid,
            "context_features": target_features,
            "target_features": target_features,
        }


@dataclass
class BridgeSession:
    """Per-session bridge runtime state."""

    session_id: str
    domain_name: str
    model_version: str
    model: Chamelia
    domain: AbstractDomain


@dataclass(frozen=True)
class PlannerModeProfile:
    """Shared-planner profile for bridge operating modes."""

    mode: str
    max_candidates: int
    use_retrieved_postures: bool
    planner_profile: str


class BridgeRuntime:
    """Build and serve repaired Chamelia sessions for bridge calls."""

    def __init__(
        self,
        config_path: str | None = None,
        device: str | None = None,
        backbone_mode: str | None = None,
        checkpoint_path: str | None = None,
        model_version: str | None = None,
    ) -> None:
        resolved_checkpoint = checkpoint_path if checkpoint_path is not None else os.getenv("CHAMELIA_BRIDGE_CHECKPOINT")
        resolved_model_version = model_version if model_version is not None else os.getenv("CHAMELIA_BRIDGE_MODEL_VERSION")
        self.config_path = Path(config_path) if config_path is not None else _project_root() / "configs" / "chamelia.yaml"
        self.device = torch.device(device or os.getenv("CHAMELIA_BRIDGE_DEVICE", "cpu"))
        self.backbone_mode = backbone_mode or os.getenv("CHAMELIA_BRIDGE_BACKBONE_MODE", "stub")
        self.checkpoint_path = resolved_checkpoint.strip() if isinstance(resolved_checkpoint, str) and resolved_checkpoint.strip() else None
        self.model_version = (
            resolved_model_version.strip()
            if isinstance(resolved_model_version, str) and resolved_model_version.strip()
            else None
        )
        self.sessions: dict[tuple[str, str], BridgeSession] = {}
        self._config = yaml.safe_load(self.config_path.read_text())
        self._checkpoint_blob: dict[str, Any] | None = None
        if self.model_version is None:
            checkpoint_blob = self._get_checkpoint_blob()
            if checkpoint_blob is not None and isinstance(checkpoint_blob.get("model_version"), str):
                self.model_version = str(checkpoint_blob["model_version"]).strip() or None
            if self.model_version is None and self.checkpoint_path:
                self.model_version = Path(self.checkpoint_path).stem
            if self.model_version is None:
                self.model_version = f"{self.backbone_mode}-local"

    def reset_sessions(self) -> None:
        self.sessions.clear()

    def _get_checkpoint_blob(self) -> dict[str, Any] | None:
        if self.checkpoint_path is None:
            return None
        if self._checkpoint_blob is None:
            raw = torch.load(self.checkpoint_path, map_location=self.device)
            self._checkpoint_blob = raw if isinstance(raw, dict) else {"model_state_dict": raw}
        return self._checkpoint_blob

    def _resolve_model_config(self) -> dict[str, Any]:
        checkpoint_blob = self._get_checkpoint_blob()
        if checkpoint_blob is not None:
            checkpoint_cfg = checkpoint_blob.get("config")
            if isinstance(checkpoint_cfg, dict):
                nested_cfg = checkpoint_cfg.get("model")
                if isinstance(nested_cfg, dict):
                    return nested_cfg
                return checkpoint_cfg
        if self.backbone_mode == "stub":
            stub_cfg = self._config.get("stub_model")
            if stub_cfg is not None:
                return stub_cfg
            return {
                "embed_dim": 64,
                "configurator": {
                    "num_ctx_tokens": 4,
                    "num_heads": 4,
                    "num_layers": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "memory_read_k": 4,
                },
                "actor": {"num_heads": 4, "num_layers": 2, "mlp_ratio": 2.0, "dropout": 0.0},
                "cost": {
                    "critic_num_heads": 4,
                    "critic_num_layers": 2,
                    "critic_mlp_ratio": 2.0,
                    "critic_dropout": 0.0,
                    "critic_horizon": 30,
                },
                "memory": {"max_episodes": 512, "retrieval_k": 4, "device": "cpu"},
            }
        return self._config["model"]

    def _infer_encoder_embed_dim(self, encoder_type: str, configured_embed_dim: int) -> int:
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

    def _build_domain(self, domain_name: str, embed_dim: int) -> AbstractDomain:
        if domain_name == "protein_dti":
            max_candidate_drugs = int(os.getenv("CHAMELIA_PROTEIN_DTI_MAX_CANDIDATES", "20"))
            action_dim = int(
                os.getenv("CHAMELIA_PROTEIN_DTI_ACTION_DIM", str(max_candidate_drugs))
            )
            return ProteinDTIDomain(
                db_path=os.getenv("CHAMELIA_PROTEIN_DTI_DB_PATH"),
                data_base_dir=os.getenv("CHAMELIA_PROTEIN_DTI_DATA_DIR"),
                embed_dim=embed_dim,
                max_candidate_drugs=max_candidate_drugs,
                action_dim=action_dim,
                affinity_type=os.getenv("CHAMELIA_PROTEIN_DTI_AFFINITY_TYPE", "Kd"),
                split=os.getenv("CHAMELIA_PROTEIN_DTI_SPLIT", "train"),
                split_strategy=os.getenv(
                    "CHAMELIA_PROTEIN_DTI_SPLIT_STRATEGY",
                    "protein_family",
                ),
                seed=int(os.getenv("CHAMELIA_PROTEIN_DTI_SEED", "42")),
            )
        if domain_name in DomainRegistry.list_domains():
            return DomainRegistry.get(domain_name)
        if domain_name == "insite_t1d":
            domain = InSiteBridgeDomain(embed_dim=embed_dim, action_dim=8)
        elif domain_name == "language":
            domain = LanguageCurriculumDomain(batch_size=1, seq_len=32).build_runtime_domain(embed_dim)
        elif domain_name == "basic_arithmetic":
            domain = ReasoningCurriculumDomain(
                domain_variant="basic_arithmetic", batch_size=1, seq_len=32
            ).build_runtime_domain(embed_dim)
        elif domain_name == "basic_arithmetic_patterns":
            domain = PatternCurriculumDomain(
                domain_variant="basic_arithmetic_patterns", batch_size=1, seq_len=32
            ).build_runtime_domain(embed_dim)
        elif domain_name == "chess":
            domain = GamesCurriculumDomain(domain_variant="chess", batch_size=1, seq_len=64).build_runtime_domain(embed_dim)
        elif domain_name == "collaborative":
            domain = CollaborativeCurriculumDomain(batch_size=1, seq_len=32).build_runtime_domain(embed_dim)
        elif domain_name in {"synthetic_patients", "health"}:
            domain = HealthCurriculumDomain(batch_size=1, seq_len=48).build_runtime_domain(embed_dim)
        else:
            raise KeyError(f"Unsupported bridge domain '{domain_name}'.")
        if domain is None:
            raise ValueError(f"Domain '{domain_name}' did not build a runtime plugin.")
        if domain_name not in DomainRegistry.list_domains():
            DomainRegistry.register(domain)
        return domain

    def _build_model(self, domain_name: str) -> tuple[Chamelia, AbstractDomain]:
        model_cfg = self._resolve_model_config()
        configured_embed_dim = int(model_cfg["embed_dim"])
        if self.backbone_mode == "hjepa":
            embed_dim = self._infer_encoder_embed_dim(
                str(model_cfg.get("encoder_type", "vit_base_patch16_224")),
                configured_embed_dim,
            )
        else:
            embed_dim = configured_embed_dim

        domain = self._build_domain(domain_name, embed_dim)
        action_dim = domain.get_action_dim()
        num_ctx_tokens = int(model_cfg["configurator"]["num_ctx_tokens"])

        if self.backbone_mode == "stub":
            hjepa: torch.nn.Module = _StubSequenceHJEPA(embed_dim=embed_dim)
        elif self.backbone_mode == "hjepa":
            hjepa = HJEPA(
                encoder_type=str(model_cfg.get("encoder_type", "vit_base_patch16_224")),
                img_size=int(model_cfg.get("img_size", 224)),
                embed_dim=embed_dim,
                predictor_depth=int(model_cfg.get("predictor_depth", 6)),
                predictor_num_heads=int(model_cfg.get("predictor_num_heads", 12)),
                predictor_mlp_ratio=float(model_cfg.get("predictor_mlp_ratio", 4.0)),
                num_hierarchies=int(model_cfg.get("num_hierarchies", 3)),
                pretrained=bool(model_cfg.get("pretrained", False)),
                drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
                use_fpn=bool(model_cfg.get("use_fpn", False)),
                fpn_feature_dim=embed_dim if bool(model_cfg.get("use_fpn", False)) else None,
                fpn_fusion_method=str(model_cfg.get("fpn_fusion_method", "add")),
                use_gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", False)),
                use_layerscale=bool(model_cfg.get("use_layerscale", False)),
                layerscale_init=float(model_cfg.get("layerscale_init", 1.0e-5)),
                use_flash_attention=bool(model_cfg.get("use_flash_attention", True)),
            )
        else:
            raise ValueError(f"Unsupported bridge backbone mode '{self.backbone_mode}'.")

        configurator = Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=int(model_cfg["configurator"]["num_heads"]),
            num_layers=int(model_cfg["configurator"]["num_layers"]),
            mlp_ratio=float(model_cfg["configurator"]["mlp_ratio"]),
            dropout=float(model_cfg["configurator"]["dropout"]),
            memory_read_k=int(model_cfg["configurator"]["memory_read_k"]),
        )
        actor = Actor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_heads=int(model_cfg["actor"]["num_heads"]),
            num_layers=int(model_cfg["actor"]["num_layers"]),
            mlp_ratio=float(model_cfg["actor"]["mlp_ratio"]),
            dropout=float(model_cfg["actor"]["dropout"]),
            num_ctx_tokens=num_ctx_tokens,
        )
        cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
        cost_module = CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=int(model_cfg["cost"]["critic_num_heads"]),
                num_layers=int(model_cfg["cost"]["critic_num_layers"]),
                mlp_ratio=float(model_cfg["cost"]["critic_mlp_ratio"]),
                dropout=float(model_cfg["cost"]["critic_dropout"]),
                num_ctx_tokens=num_ctx_tokens,
                horizon=int(model_cfg["cost"]["critic_horizon"]),
            ),
        )
        memory_cfg = model_cfg["memory"]
        model = Chamelia(
            hjepa=hjepa,  # type: ignore[arg-type]
            configurator=configurator,
            actor=actor,
            cost=cost_module,
            memory=LatentMemory(
                embed_dim=embed_dim,
                max_episodes=int(memory_cfg["max_episodes"]),
                retrieval_k=int(memory_cfg["retrieval_k"]),
                device=str(memory_cfg.get("device", "cpu")),
            ),
            domain=domain,
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_ctx_tokens=num_ctx_tokens,
            model_version=self.model_version,
        ).to(self.device)
        model.eval()

        checkpoint_blob = self._get_checkpoint_blob()
        if checkpoint_blob is not None:
            state_dict = checkpoint_blob.get("model_state_dict", checkpoint_blob)
            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to load bridge checkpoint '{self.checkpoint_path}' into runtime model for "
                    f"domain '{domain_name}'."
                ) from exc
            model.eval()
        return model, domain

    def get_session(self, session_id: str, domain_name: str) -> BridgeSession:
        key = (domain_name, session_id)
        if key not in self.sessions:
            model, domain = self._build_model(domain_name)
            self.sessions[key] = BridgeSession(
                session_id=session_id,
                domain_name=domain_name,
                model_version=self.model_version,
                model=model,
                domain=domain,
            )
        return self.sessions[key]


def _float_tensor(data: Any, device: torch.device, *, min_dim: int = 0) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
    while tensor.dim() < min_dim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _maybe_float_tensor(data: Any, device: torch.device, *, min_dim: int = 0) -> torch.Tensor | None:
    if data is None:
        return None
    if isinstance(data, list) and not data:
        return None
    return _float_tensor(data, device, min_dim=min_dim)


def _as_transport(value: torch.Tensor | None) -> Any:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _normalized_bridge_mode(mode: str) -> str:
    normalized = mode.strip()
    if normalized not in {"v1.1", "v1.5", "v3"}:
        raise ValueError(f"Unsupported bridge mode '{mode}'.")
    return normalized


def _planner_mode_profile(model: Chamelia, mode: str) -> PlannerModeProfile:
    normalized = _normalized_bridge_mode(mode)
    full_candidates = int(model.actor.num_candidates)
    if full_candidates < 2:
        raise ValueError("Bridge planner requires at least two candidates including the explicit baseline.")
    if normalized == "v1.1":
        return PlannerModeProfile(
            mode=normalized,
            max_candidates=min(2, full_candidates),
            use_retrieved_postures=False,
            planner_profile="conservative_shared_planner",
        )
    if normalized == "v1.5":
        return PlannerModeProfile(
            mode=normalized,
            max_candidates=min(4, full_candidates),
            use_retrieved_postures=True,
            planner_profile="lightweight_shared_planner",
        )
    return PlannerModeProfile(
        mode=normalized,
        max_candidates=full_candidates,
        use_retrieved_postures=True,
        planner_profile="default_shared_planner",
    )


def _build_retrieval_trace(
    raw_trace: Any,
) -> tuple[RetrievalTraceStep, ...] | None:
    if not isinstance(raw_trace, list) or not raw_trace:
        return None

    trace_steps: list[RetrievalTraceStep] = []
    for raw_step in raw_trace:
        if not isinstance(raw_step, dict):
            continue
        query_key = raw_step.get("query_key")
        memory_keys = raw_step.get("memory_keys")
        memory_summaries = raw_step.get("memory_summaries")
        base_quality_scores = raw_step.get("base_quality_scores")
        if query_key is None or memory_keys is None or memory_summaries is None or base_quality_scores is None:
            continue
        trace_steps.append(
            RetrievalTraceStep(
                query_key=torch.as_tensor(query_key, dtype=torch.float32),
                memory_keys=torch.as_tensor(memory_keys, dtype=torch.float32),
                memory_summaries=torch.as_tensor(memory_summaries, dtype=torch.float32),
                base_quality_scores=torch.as_tensor(base_quality_scores, dtype=torch.float32),
                query_posture=(
                    torch.as_tensor(raw_step["query_posture"], dtype=torch.float32)
                    if raw_step.get("query_posture") is not None
                    else None
                ),
                memory_postures=(
                    torch.as_tensor(raw_step["memory_postures"], dtype=torch.float32)
                    if raw_step.get("memory_postures") is not None
                    else None
                ),
                base_scores=(
                    torch.as_tensor(raw_step["base_scores"], dtype=torch.float32)
                    if raw_step.get("base_scores") is not None
                    else None
                ),
                relevance_scores=(
                    torch.as_tensor(raw_step["relevance_scores"], dtype=torch.float32)
                    if raw_step.get("relevance_scores") is not None
                    else None
                ),
                relevance_weights=(
                    torch.as_tensor(raw_step["relevance_weights"], dtype=torch.float32)
                    if raw_step.get("relevance_weights") is not None
                    else None
                ),
            )
        )
    return tuple(trace_steps) if trace_steps else None


def encode_session(
    session: BridgeSession,
    *,
    input_kind: str,
    tokens: Any | None = None,
    mask: Any | None = None,
    observation: Any | None = None,
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device

    if input_kind == "plugin_observation":
        if observation is None:
            raise ValueError("plugin_observation input requires an observation payload.")
        tokenizer_input = session.domain.prepare_bridge_observation(observation)
        tokenized = session.domain.get_tokenizer()(tokenizer_input)
        tokens_tensor = tokenized.tokens.to(device)
        if tokenized.padding_mask is None:
            mask_tensor = torch.zeros(
                tokens_tensor.shape[0],
                tokens_tensor.shape[1],
                device=device,
                dtype=torch.float32,
            )
        else:
            mask_tensor = tokenized.padding_mask.to(device=device, dtype=torch.float32)
        hjepa_input_kind = "embedded_tokens"
    else:
        if tokens is None:
            raise ValueError(f"{input_kind} input requires tokens.")
        tokens_tensor = _float_tensor(tokens, device, min_dim=3 if input_kind != "image" else 4)
        if mask is None:
            if input_kind == "image":
                mask_tensor = torch.zeros(
                    tokens_tensor.shape[0],
                    1,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                mask_tensor = torch.zeros(
                    tokens_tensor.shape[0],
                    tokens_tensor.shape[1],
                    device=device,
                    dtype=torch.float32,
                )
        else:
            mask_tensor = _float_tensor(mask, device, min_dim=2)
        hjepa_input_kind = input_kind

    with torch.no_grad():
        hjepa_out = forward_hjepa(model.hjepa, tokens_tensor, mask_tensor, input_kind=hjepa_input_kind)
        z = model._get_scene_summary(hjepa_out)
        level_feats = model._extract_level_features(hjepa_out)

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "z_t": _as_transport(z.squeeze(0)),
        "hierarchy_tokens": {
            "level0": _as_transport(level_feats[0].squeeze(0)),
            "level1": _as_transport(level_feats[1].squeeze(0)),
            "level2": _as_transport(level_feats[2].squeeze(0)),
        },
        "encoder_diagnostics": {
            "token_count": int(level_feats[0].shape[1]),
            "embed_dim": int(z.shape[-1]),
            "input_kind": input_kind,
        },
    }


def retrieve_session(
    session: BridgeSession,
    *,
    z_t: Any,
    query_posture: Any | None = None,
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device
    z = _float_tensor(z_t, device, min_dim=2)
    posture = _maybe_float_tensor(query_posture, device, min_dim=2)

    with torch.no_grad():
        retrieved_keys, episodes = model.memory.retrieve(z)
        bundle = model._rerank_retrieved_memory(
            query_key=z,
            episodes=episodes,
            retrieved_keys=retrieved_keys,
            query_posture=posture,
        )

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "retrieved_keys": (
            _as_transport(retrieved_keys.squeeze(0)) if retrieved_keys is not None else []
        ),
        "retrieved_episode_summaries": (
            _as_transport(bundle["episode_summaries"].squeeze(0))
            if bundle["episode_summaries"] is not None
            else []
        ),
        "retrieved_episode_scores": (
            _as_transport(bundle["episode_scores"].squeeze(0))
            if bundle["episode_scores"] is not None
            else []
        ),
        "retrieved_postures": (
            _as_transport(bundle["postures"].squeeze(0)) if bundle["postures"] is not None else None
        ),
        "retrieved_posture_scores": (
            _as_transport(bundle["posture_scores"].squeeze(0))
            if bundle["posture_scores"] is not None
            else None
        ),
        "retrieval_base_scores": (
            _as_transport(bundle["base_scores"].squeeze(0)) if bundle["base_scores"] is not None else None
        ),
        "retrieval_base_quality_scores": (
            _as_transport(bundle["base_quality_scores"].squeeze(0))
            if bundle["base_quality_scores"] is not None
            else None
        ),
        "retrieval_relevance_scores": (
            _as_transport(bundle["relevance_scores"].squeeze(0))
            if bundle["relevance_scores"] is not None
            else None
        ),
        "retrieval_relevance_weights": (
            _as_transport(bundle["relevance_weights"].squeeze(0))
            if bundle["relevance_weights"] is not None
            else None
        ),
    }


def ingest_replay_examples(
    session: BridgeSession,
    *,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    model = session.model
    existing_replay_ids = {
        (
            str(record.metadata.get("source_patient_id", "")),
            int(record.metadata.get("bridge_replay_record_id", -1)),
            str(record.model_version or ""),
        )
        for record in model.memory.records[: model.memory.size]
        if record.metadata is not None and "bridge_replay_record_id" in record.metadata
    }

    ingested = 0
    skipped = 0
    duplicates = 0
    for example in examples:
        domain_name = example.get("domain_name")
        model_version = example.get("model_version")
        if domain_name != session.domain_name or model_version != session.model_version:
            skipped += 1
            continue

        record_id = example.get("record_id")
        if not isinstance(record_id, int):
            skipped += 1
            continue

        source_patient_id = str(example.get("source_patient_id", ""))
        dedupe_key = (source_patient_id, record_id, session.model_version)
        if dedupe_key in existing_replay_ids:
            duplicates += 1
            continue

        z_t = example.get("z_t")
        ctx_tokens = example.get("ctx_tokens")
        selected_action_vec = example.get("selected_action_vec")
        selected_path = example.get("selected_path")
        outcome_z_tH = example.get("outcome_z_tH")
        realized_ic = example.get("realized_ic")
        if (
            z_t is None
            or ctx_tokens is None
            or selected_action_vec is None
            or selected_path is None
            or outcome_z_tH is None
            or not isinstance(realized_ic, (int, float))
        ):
            skipped += 1
            continue

        retrieval_trace = _build_retrieval_trace(example.get("retrieval_trace"))
        record = EpisodeRecord(
            key=torch.as_tensor(z_t, dtype=torch.float32),
            action=torch.as_tensor(selected_action_vec, dtype=torch.float32),
            ctx_tokens=torch.as_tensor(ctx_tokens, dtype=torch.float32),
            ic_at_decision=float(example.get("selected_candidate_ic", 0.0) or 0.0),
            ic_realized=float(realized_ic),
            tc_predicted=float(example.get("selected_candidate_tc", 0.0) or 0.0),
            outcome_key=torch.as_tensor(outcome_z_tH, dtype=torch.float32),
            step=int(example.get("day", 0) or 0),
            domain_name=session.domain_name,
            model_version=session.model_version,
            candidate_postures=_maybe_float_tensor(example.get("candidate_postures"), torch.device("cpu"), min_dim=2),
            selected_posture=_maybe_float_tensor(example.get("selected_posture"), torch.device("cpu"), min_dim=1),
            candidate_reasoning_states=_maybe_float_tensor(
                example.get("candidate_reasoning_states"),
                torch.device("cpu"),
                min_dim=2,
            ),
            candidate_paths=_maybe_float_tensor(example.get("candidate_paths"), torch.device("cpu"), min_dim=3),
            selected_path=torch.as_tensor(selected_path, dtype=torch.float32),
            candidate_actions=_maybe_float_tensor(example.get("candidate_actions"), torch.device("cpu"), min_dim=2),
            candidate_ic=_maybe_float_tensor(example.get("candidate_ic"), torch.device("cpu"), min_dim=1),
            candidate_tc=_maybe_float_tensor(example.get("candidate_tc"), torch.device("cpu"), min_dim=1),
            candidate_total=_maybe_float_tensor(example.get("candidate_total"), torch.device("cpu"), min_dim=1),
            selected_candidate_idx=(
                int(example["selected_candidate_idx"])
                if isinstance(example.get("selected_candidate_idx"), int)
                else None
            ),
            retrieval_trace=retrieval_trace,
            metadata={
                "source_patient_id": source_patient_id,
                "bridge_replay_record_id": record_id,
                "selected_candidate_slot": example.get("selected_candidate_slot"),
                "julia_selection": example.get("julia_selection"),
                "selected_candidate": example.get("selected_candidate"),
            },
        )
        model.memory.store(record)
        existing_replay_ids.add(dedupe_key)
        ingested += 1

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "ingested": ingested,
        "skipped": skipped,
        "duplicates": duplicates,
        "memory_size": model.memory.size,
    }


def configure_session(
    session: BridgeSession,
    *,
    encoded_state: dict[str, Any],
    retrieved_memory: dict[str, Any],
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device
    level0 = _float_tensor(encoded_state["hierarchy_tokens"]["level0"], device, min_dim=3)
    level1 = _float_tensor(encoded_state["hierarchy_tokens"]["level1"], device, min_dim=3)
    level2 = _float_tensor(encoded_state["hierarchy_tokens"]["level2"], device, min_dim=3)
    memory_tokens = _maybe_float_tensor(retrieved_memory.get("retrieved_episode_summaries"), device, min_dim=3)
    memory_scores = _maybe_float_tensor(retrieved_memory.get("retrieved_episode_scores"), device, min_dim=2)

    with torch.no_grad():
        ctx_tokens = model.configurator(
            hjepa_outputs={"target_features_per_level": [level0, level1, level2]},
            memory_tokens=memory_tokens,
            memory_scores=memory_scores,
        )

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "ctx_tokens": _as_transport(ctx_tokens.squeeze(0)),
        "config_diagnostics": {
            "num_ctx_tokens": int(ctx_tokens.shape[1]),
            "embed_dim": int(ctx_tokens.shape[-1]),
        },
    }


def propose_session(
    session: BridgeSession,
    *,
    mode: str,
    encoded_state: dict[str, Any],
    configurator_output: dict[str, Any],
    retrieved_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device
    mode_profile = _planner_mode_profile(model, mode)
    z = _float_tensor(encoded_state["z_t"], device, min_dim=2)
    ctx_tokens = _float_tensor(configurator_output["ctx_tokens"], device, min_dim=3)
    retrieved_postures = None
    retrieved_posture_scores = None
    if retrieved_memory is not None:
        retrieved_postures = _maybe_float_tensor(retrieved_memory.get("retrieved_postures"), device, min_dim=3)
        retrieved_posture_scores = _maybe_float_tensor(
            retrieved_memory.get("retrieved_posture_scores"),
            device,
            min_dim=2,
        )

    with torch.no_grad():
        proposal = model.actor.propose(
            z=z,
            ctx_tokens=ctx_tokens,
            retrieved_postures=(
                retrieved_postures if mode_profile.use_retrieved_postures else None
            ),
            retrieved_posture_scores=(
                retrieved_posture_scores if mode_profile.use_retrieved_postures else None
            ),
        )
        candidate_paths = proposal["candidate_paths"][:, : mode_profile.max_candidates, :, :]
        candidate_actions = proposal["candidate_actions"][:, : mode_profile.max_candidates, :]
        candidate_postures = proposal["candidate_postures"][:, : mode_profile.max_candidates, :]
        reasoning_states = proposal["reasoning_states"][:, : mode_profile.max_candidates, :]

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "candidate_paths": _as_transport(candidate_paths.squeeze(0)),
        "candidate_actions": _as_transport(candidate_actions.squeeze(0)),
        "candidate_postures": _as_transport(candidate_postures.squeeze(0)),
        "candidate_reasoning_states": _as_transport(reasoning_states.squeeze(0)),
        "proposal_diagnostics": {
            "mode": mode_profile.mode,
            "planner_profile": mode_profile.planner_profile,
            "shared_planner_substrate": True,
            "num_candidates": int(candidate_paths.shape[1]),
            "candidate_limit_applied": int(mode_profile.max_candidates),
            "full_candidate_budget": int(model.actor.num_candidates),
            "path_length": int(candidate_paths.shape[2]),
            "action_dim": int(candidate_paths.shape[-1]),
            "uses_retrieved_postures": bool(mode_profile.use_retrieved_postures),
            "contains_explicit_baseline": bool(candidate_paths.shape[1] > 0),
        },
    }


def rollout_session(
    session: BridgeSession,
    *,
    encoded_state: dict[str, Any],
    configurator_output: dict[str, Any],
    proposal_bundle: dict[str, Any],
    rollout_horizon: int,
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device
    z = _float_tensor(encoded_state["z_t"], device, min_dim=2)
    ctx_tokens = _float_tensor(configurator_output["ctx_tokens"], device, min_dim=3)
    candidate_paths = _float_tensor(proposal_bundle["candidate_paths"], device, min_dim=4)
    candidate_postures = _maybe_float_tensor(proposal_bundle.get("candidate_postures"), device, min_dim=3)
    reasoning_states = _maybe_float_tensor(
        proposal_bundle.get("candidate_reasoning_states"),
        device,
        min_dim=3,
    )

    with torch.no_grad():
        rollout = model.world_model(
            z=z,
            actions=candidate_paths,
            ctx_tokens=ctx_tokens,
            candidate_postures=candidate_postures,
            reasoning_states=reasoning_states,
            horizon=rollout_horizon,
        )

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "trajectory": _as_transport(rollout["trajectory"].squeeze(0)),
        "terminal_latents": _as_transport(rollout["terminal_latents"].squeeze(0)),
        "summary_tokens": _as_transport(rollout["summary_tokens"].squeeze(0)),
        "rollout_diagnostics": {
            "horizon": int(rollout["trajectory"].shape[2]),
            "rollout_dim": int(rollout["trajectory"].shape[-1]),
        },
    }


def critic_session(
    session: BridgeSession,
    *,
    encoded_state: dict[str, Any],
    configurator_output: dict[str, Any],
    proposal_bundle: dict[str, Any],
    rollout_bundle: dict[str, Any],
    domain_state: Any,
) -> dict[str, Any]:
    model = session.model
    device = next(model.parameters()).device
    z = _float_tensor(encoded_state["z_t"], device, min_dim=2)
    ctx_tokens = _float_tensor(configurator_output["ctx_tokens"], device, min_dim=3)
    candidate_paths = _float_tensor(proposal_bundle["candidate_paths"], device, min_dim=4)
    terminal_latents = _float_tensor(rollout_bundle["terminal_latents"], device, min_dim=3)
    trajectory = _float_tensor(rollout_bundle["trajectory"], device, min_dim=4)
    domain_state_tensorized = session.domain.get_domain_state(domain_state)
    domain_state_tensorized = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in domain_state_tensorized.items()
    }

    with torch.no_grad():
        scores = model.cost.score_candidates(
            z=z,
            actions=candidate_paths,
            ctx_tokens=ctx_tokens,
            domain_state=domain_state_tensorized,
            future_z=terminal_latents,
            future_trajectory=trajectory,
        )

    return {
        "bridge_version": "v1",
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "candidate_ic": _as_transport(scores["ic"].squeeze(0)),
        "candidate_tc": _as_transport(scores["tc"].squeeze(0)),
        "candidate_total": _as_transport(scores["total"].squeeze(0)),
        "critic_diagnostics": {
            "used_ctx_tokens": True,
            "path_level_ic": True,
        },
    }
