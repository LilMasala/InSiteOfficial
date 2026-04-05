"""Curriculum training loop scaffold."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.chamelia.chamelia import Chamelia
from src.chamelia.retrieval import compute_retrieval_relevance_loss
from training.curriculum.domains.base import CurriculumDomain
from training.curriculum.graduation import GraduationManager
from training.curriculum.batch import ChameliaStepBatch, CurriculumBatch


class CurriculumStageRunner:
    """Orchestrate curriculum stages, probes, and checkpointing.

    This is a scaffold runner: it manages stage/domain progression, masking, probe checks,
    and checkpoint state, but leaves the actual model-specific optimization step to the
    injected ``train_step`` callback.
    """

    def __init__(
        self,
        model: Any,
        stages: list[list[CurriculumDomain]],
        graduation_manager: GraduationManager,
        config: dict[str, Any],
        device: torch.device | str,
        train_step: callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        representation_loss_fn: Any | None = None,
        representation_loss_weight: float = 1.0,
        store_to_memory: bool = False,
        critic_train_interval: int = 0,
        critic_loss_weight: float = 1.0,
        world_model_train_interval: int = 0,
        world_model_loss_weight: float = 1.0,
        path_baseline_loss_weight: float = 0.0,
        path_baseline_margin: float = 0.05,
        posture_diversity_loss_weight: float = 0.05,
        posture_similarity_cap: float = 0.90,
        posture_path_similarity_cap: float = 0.95,
        posture_specialization_loss_weight: float = 0.05,
        posture_specialization_temperature: float = 0.25,
        retrieval_relevance_loss_weight: float = 0.05,
        retrieval_relevance_temperature: float = 0.25,
        retrieval_train_interval: int = 0,
        retrieval_replay_loss_weight: float = 1.0,
        mode1_distill_interval: int = 0,
        mode1_distill_weight: float = 0.0,
        export_model_config: dict[str, Any] | None = None,
        export_backbone_mode: str | None = None,
        export_model_version: str | None = None,
    ) -> None:
        """Initialize the curriculum runner.

        Args:
            model: Model under training.
            stages: List of stages, each containing domain objects.
            graduation_manager: Graduation manager.
            config: Curriculum config dictionary.
            device: Target device.
            train_step: Optional callback consuming ``(model, batch, domain, level, device)``.
            optimizer: Optional optimizer for default gradient updates.
            representation_loss_fn: Optional JEPA/combined representation loss module.
            representation_loss_weight: Scalar weight for representation loss.
            store_to_memory: Whether Chamelia should store training episodes to memory.
            critic_train_interval: Interval in steps for critic updates from memory.
            critic_loss_weight: Scalar weight for memory critic loss.
            world_model_train_interval: Interval in steps for world-model updates from memory.
            world_model_loss_weight: Scalar weight for memory world-model loss.
            path_baseline_loss_weight: Scalar weight for path-vs-baseline ranking loss.
            path_baseline_margin: Minimum realized improvement required before preferring
                a non-baseline path over the simple/null baseline.
            posture_diversity_loss_weight: Scalar weight for candidate-posture diversity loss.
            posture_similarity_cap: Maximum allowed posture similarity across
                non-baseline candidate postures.
            posture_path_similarity_cap: Maximum allowed path similarity across
                non-baseline candidate postures.
            posture_specialization_loss_weight: Scalar weight for delayed-outcome
                specialization pressure across non-baseline postures.
            posture_specialization_temperature: Softmax temperature used when converting
                realized non-baseline path advantages into specialization targets.
            retrieval_relevance_loss_weight: Scalar weight for direct delayed-outcome
                supervision on the learned retrieval reranker.
            retrieval_relevance_temperature: Softmax temperature used when converting
                shortlist memory targets into retrieval relevance targets.
            retrieval_train_interval: Interval in steps for replaying stored retrieval
                decisions against realized outcomes from memory.
            retrieval_replay_loss_weight: Scalar weight for replayed retrieval-reranker loss.
            mode1_distill_interval: Interval in steps for distilling mode1 from mode2.
            mode1_distill_weight: Scalar weight for mode1 distillation loss.
            export_model_config: Optional bridge-loadable model config to bundle into
                exported curriculum checkpoints.
            export_backbone_mode: Optional bridge backbone mode associated with exported
                model artifacts, such as ``stub`` or ``hjepa``.
            export_model_version: Optional explicit model version metadata for exported
                bridge artifacts.
        """
        self.model = model
        self.stages = stages
        self.graduation_manager = graduation_manager
        self.config = config
        self.device = torch.device(device)
        self.train_step = train_step
        self.optimizer = optimizer
        if isinstance(representation_loss_fn, torch.nn.Module):
            representation_loss_fn = representation_loss_fn.to(self.device)
        self.representation_loss_fn = representation_loss_fn
        self.representation_loss_weight = representation_loss_weight
        self.store_to_memory = store_to_memory
        self.critic_train_interval = critic_train_interval
        self.critic_loss_weight = critic_loss_weight
        self.world_model_train_interval = world_model_train_interval
        self.world_model_loss_weight = world_model_loss_weight
        self.path_baseline_loss_weight = path_baseline_loss_weight
        self.path_baseline_margin = path_baseline_margin
        self.posture_diversity_loss_weight = posture_diversity_loss_weight
        self.posture_similarity_cap = posture_similarity_cap
        self.posture_path_similarity_cap = posture_path_similarity_cap
        self.posture_specialization_loss_weight = posture_specialization_loss_weight
        self.posture_specialization_temperature = posture_specialization_temperature
        self.retrieval_relevance_loss_weight = retrieval_relevance_loss_weight
        self.retrieval_relevance_temperature = retrieval_relevance_temperature
        self.retrieval_train_interval = retrieval_train_interval
        self.retrieval_replay_loss_weight = retrieval_replay_loss_weight
        self.mode1_distill_interval = mode1_distill_interval
        self.mode1_distill_weight = mode1_distill_weight
        self.export_model_config = deepcopy(export_model_config) if export_model_config is not None else None
        self.export_backbone_mode = export_backbone_mode
        self.export_model_version = export_model_version
        self.stage_idx = int(config.get("curriculum", {}).get("start_stage", 0))
        self.global_step = 0
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.bridge_artifact_dir = self.checkpoint_dir / "bridge_artifacts"
        self.bridge_artifact_dir.mkdir(parents=True, exist_ok=True)
        self._runtime_domains: dict[str, Any] = {}

    def _optimizer_param_ids(self) -> set[int]:
        """Return ids of parameters already tracked by the optimizer.

        Args:
            None.

        Returns:
            Set of parameter object ids.
        """
        if self.optimizer is None:
            return set()
        param_ids: set[int] = set()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param_ids.add(id(param))
        return param_ids

    def _ensure_optimizer_tracks_model(self) -> None:
        """Add newly-registered trainable model parameters to the optimizer.

        Args:
            None.

        Returns:
            None.
        """
        if self.optimizer is None:
            return
        existing_ids = self._optimizer_param_ids()
        missing = [
            param
            for param in self.model.parameters()
            if param.requires_grad and id(param) not in existing_ids
        ]
        if missing:
            self.optimizer.add_param_group({"params": missing})

    def _runtime_domain_for(self, domain: CurriculumDomain) -> Any:
        """Return a cached runtime domain plugin for a curriculum domain.

        Args:
            domain: Curriculum domain.

        Returns:
            Runtime domain plugin.
        """
        key = domain.domain_name()
        if key not in self._runtime_domains:
            if not isinstance(self.model, Chamelia):
                raise ValueError("Runtime-domain resolution requires a Chamelia model.")
            runtime_domain = domain.build_runtime_domain(self.model.embed_dim)
            if runtime_domain is None:
                raise ValueError(
                    f"Domain '{domain.domain_name()}' does not yet provide a runtime Chamelia plugin."
                )
            self._runtime_domains[key] = runtime_domain
            self.model.set_domain(runtime_domain)
            self._ensure_optimizer_tracks_model()
        return self._runtime_domains[key]

    def _domain_iterator(self, domain: CurriculumDomain) -> Iterator[CurriculumBatch]:
        """Create an iterator over a domain's current training loader."""
        loader = domain.get_data_loader(domain.cost.current_level, split="train")
        while True:
            for batch in loader:
                yield batch

    def _to_step_batch(
        self,
        batch: CurriculumBatch,
        domain: CurriculumDomain,
    ) -> ChameliaStepBatch:
        """Convert a curriculum batch to a normalized Chamelia step batch.

        Args:
            batch: Curriculum batch.
            domain: Owning domain.

        Returns:
            ``ChameliaStepBatch``.
        """
        tokens = batch.tokens
        embedded_tokens = batch.embedded_tokens
        if tokens is not None:
            masked_tokens, mask = domain.get_masking_strategy(domain.cost.current_level).apply(
                tokens,
                domain.cost.current_level,
            )
            return ChameliaStepBatch(
                domain_name=batch.domain_name,
                model_inputs=masked_tokens,
                input_mask=mask.to(self.device),
                input_kind="token_ids",
                targets=batch.targets,
                domain_state=batch.domain_state,
                metadata=batch.metadata,
            )
        if embedded_tokens is None:
            raise ValueError("CurriculumBatch must contain tokens or embedded_tokens.")
        return ChameliaStepBatch(
            domain_name=batch.domain_name,
            model_inputs=embedded_tokens,
            input_mask=batch.input_mask,
            input_kind="embedded_tokens",
            targets=batch.targets,
            domain_state=batch.domain_state,
            metadata=batch.metadata,
        )

    def _default_train_step(
        self,
        step_batch: ChameliaStepBatch,
        domain: CurriculumDomain,
    ) -> torch.Tensor:
        """Default train step for Chamelia-backed curriculum domains.

        Args:
            step_batch: Normalized model-step batch.
            domain: Active curriculum domain.

        Returns:
            Scalar loss tensor.
        """
        if not isinstance(self.model, Chamelia):
            raise ValueError("Default curriculum train step requires a Chamelia model.")
        runtime_domain = self._runtime_domain_for(domain)
        self.model.set_domain(runtime_domain)
        if step_batch.input_kind == "token_ids":
            tokenized = runtime_domain.get_tokenizer()(step_batch.model_inputs.long())
            model_inputs = tokenized.tokens
            input_kind = "embedded_tokens"
        else:
            model_inputs = step_batch.model_inputs
            input_kind = step_batch.input_kind
        outputs = self.model(
            tokens=model_inputs,
            mask=step_batch.input_mask,
            domain_state=step_batch.domain_state,
            actor_mode="mode2",
            store_to_memory=self.store_to_memory,
            input_kind=input_kind,
        )
        loss = outputs["cost"]["total"].mean()
        if self.representation_loss_fn is not None:
            hjepa_out = outputs["hjepa_out"]
            masks = hjepa_out.get("masks_valid", hjepa_out.get("mask_valid"))
            if isinstance(masks, torch.Tensor):
                masks = [masks for _ in range(len(hjepa_out["predictions"]))]
            rep_loss_dict = self.representation_loss_fn(
                hjepa_out["predictions"],
                hjepa_out["targets"],
                masks=masks,
                context_features=hjepa_out.get("context_features"),
            )
            loss = loss + self.representation_loss_weight * rep_loss_dict["loss"]

        if self.path_baseline_loss_weight > 0.0:
            path_baseline_loss = self._compute_path_baseline_loss(
                outputs=outputs,
                step_batch=step_batch,
                runtime_domain=runtime_domain,
            )
            if path_baseline_loss is not None:
                loss = loss + self.path_baseline_loss_weight * path_baseline_loss

        if self.posture_diversity_loss_weight > 0.0:
            posture_diversity_loss = self._compute_posture_diversity_loss(outputs=outputs)
            if posture_diversity_loss is not None:
                loss = loss + self.posture_diversity_loss_weight * posture_diversity_loss
        if self.posture_specialization_loss_weight > 0.0:
            posture_specialization_loss = self._compute_posture_specialization_loss(
                outputs=outputs,
                step_batch=step_batch,
                runtime_domain=runtime_domain,
            )
            if posture_specialization_loss is not None:
                loss = loss + self.posture_specialization_loss_weight * posture_specialization_loss

        outcome_observation, realized_ic = self._resolve_selected_outcome(
            outputs=outputs,
            step_batch=step_batch,
            runtime_domain=runtime_domain,
        )

        if self.retrieval_relevance_loss_weight > 0.0:
            retrieval_relevance_loss = self._compute_retrieval_relevance_loss(
                outputs=outputs,
                realized_ic=realized_ic,
            )
            if retrieval_relevance_loss is not None:
                loss = loss + self.retrieval_relevance_loss_weight * retrieval_relevance_loss

        if self.store_to_memory:
            if outcome_observation is not None and realized_ic is not None:
                self.model.fill_outcome(
                    ic_realized=realized_ic,
                    outcome_observation=outcome_observation,
                )

        next_step = self.global_step + 1
        if self.critic_train_interval > 0 and next_step % self.critic_train_interval == 0:
            critic_loss = self.model.train_critic_from_memory()
            if critic_loss is not None:
                loss = loss + self.critic_loss_weight * critic_loss

        if self.world_model_train_interval > 0 and next_step % self.world_model_train_interval == 0:
            world_model_loss = self.model.train_world_model_from_memory()
            if world_model_loss is not None:
                loss = loss + self.world_model_loss_weight * world_model_loss

        if self.retrieval_train_interval > 0 and next_step % self.retrieval_train_interval == 0:
            retrieval_replay_loss = self.model.train_retrieval_from_memory(
                temperature=self.retrieval_relevance_temperature
            )
            if retrieval_replay_loss is not None:
                loss = loss + self.retrieval_replay_loss_weight * retrieval_replay_loss

        if self.mode1_distill_interval > 0 and next_step % self.mode1_distill_interval == 0:
            distill_loss = self.model.actor.distill_from_mode2(
                states=outputs["z"],
                ctx_tokens=outputs["ctx_tokens"],
                mode2_actions=outputs["action_vec"],
            )
            loss = loss + self.mode1_distill_weight * distill_loss
        return loss

    def _simulate_selected_outcome(
        self,
        outputs: dict[str, Any],
        step_batch: ChameliaStepBatch,
        runtime_domain: Any,
    ) -> dict[str, torch.Tensor] | None:
        """Simulate the selected path outcome when the runtime domain supports it."""
        if hasattr(runtime_domain, "simulate_path_outcome") and "selected_path" in outputs:
            simulated = runtime_domain.simulate_path_outcome(
                outputs["selected_path"].detach(),
                step_batch.domain_state,
            )
            if simulated is not None:
                return simulated
        if hasattr(runtime_domain, "simulate_delayed_outcome"):
            return runtime_domain.simulate_delayed_outcome(
                outputs["action_vec"].detach(),
                step_batch.domain_state,
            )
        return None

    def _resolve_selected_outcome(
        self,
        outputs: dict[str, Any],
        step_batch: ChameliaStepBatch,
        runtime_domain: Any,
    ) -> tuple[Any, Any]:
        """Resolve a delayed outcome payload for the selected path when available."""
        outcome_observation = step_batch.domain_state.get("outcome_observation")
        realized_ic = step_batch.domain_state.get("realized_intrinsic_cost")
        if outcome_observation is not None and realized_ic is not None:
            return outcome_observation, realized_ic
        if hasattr(runtime_domain, "simulate_delayed_outcome"):
            simulated = self._simulate_selected_outcome(
                outputs=outputs,
                step_batch=step_batch,
                runtime_domain=runtime_domain,
            )
            if simulated is not None:
                outcome_observation = simulated.get("outcome_observation", outcome_observation)
                realized_ic = simulated.get("realized_intrinsic_cost", realized_ic)
        return outcome_observation, realized_ic

    def _expand_domain_state_for_candidates(
        self,
        domain_state: dict[str, Any],
        num_candidates: int,
    ) -> dict[str, Any]:
        """Repeat per-batch tensor fields so candidate-path simulation can score all slots."""
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1.")
        batch_size = None
        for value in domain_state.values():
            if torch.is_tensor(value) and value.dim() >= 1:
                batch_size = value.shape[0]
                break
        if batch_size is None or num_candidates == 1:
            return dict(domain_state)

        expanded: dict[str, Any] = {}
        for key, value in domain_state.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == batch_size:
                expanded[key] = value.unsqueeze(1).expand(
                    -1, num_candidates, *value.shape[1:]
                ).reshape(batch_size * num_candidates, *value.shape[1:])
            else:
                expanded[key] = value
        return expanded

    def _compute_path_baseline_loss(
        self,
        outputs: dict[str, Any],
        step_batch: ChameliaStepBatch,
        runtime_domain: Any,
    ) -> torch.Tensor | None:
        """Compute a path-ranking loss against the simple/null baseline path."""
        candidate_paths = outputs.get("candidate_paths")
        candidate_costs = outputs.get("candidate_costs")
        if not isinstance(candidate_paths, torch.Tensor):
            return None
        if not isinstance(candidate_costs, dict):
            return None
        total_costs = candidate_costs.get("total")
        if not isinstance(total_costs, torch.Tensor):
            return None
        if candidate_paths.dim() != 4 or candidate_paths.shape[1] < 2:
            return None
        if not hasattr(runtime_domain, "simulate_path_outcome"):
            return None

        baseline_path = candidate_paths[:, 0, :, :].detach()
        nonbaseline_totals = total_costs[:, 1:]
        best_rel_idx = nonbaseline_totals.argmin(dim=1)
        gather_index = best_rel_idx.view(-1, 1, 1, 1).expand(
            -1,
            1,
            candidate_paths.shape[2],
            candidate_paths.shape[3],
        )
        best_nonbaseline_path = candidate_paths[:, 1:, :, :].detach().gather(1, gather_index).squeeze(1)
        baseline_outcome = runtime_domain.simulate_path_outcome(
            baseline_path,
            step_batch.domain_state,
        )
        candidate_outcome = runtime_domain.simulate_path_outcome(
            best_nonbaseline_path,
            step_batch.domain_state,
        )
        if baseline_outcome is None or candidate_outcome is None:
            return None

        baseline_realized = baseline_outcome["realized_intrinsic_cost"].to(self.device).flatten()
        candidate_realized = candidate_outcome["realized_intrinsic_cost"].to(self.device).flatten()
        baseline_pred = total_costs[:, 0]
        candidate_pred = nonbaseline_totals.gather(1, best_rel_idx.unsqueeze(1)).squeeze(1)
        target = torch.where(
            candidate_realized + self.path_baseline_margin < baseline_realized,
            torch.ones_like(candidate_realized),
            -torch.ones_like(candidate_realized),
        )
        return F.margin_ranking_loss(
            baseline_pred,
            candidate_pred,
            target,
            margin=self.path_baseline_margin,
        )

    def _compute_posture_diversity_loss(
        self,
        outputs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Compute diversity pressure across non-baseline candidate postures."""
        if not isinstance(self.model, Chamelia):
            return None
        candidate_postures = outputs.get("candidate_postures")
        candidate_paths = outputs.get("candidate_paths")
        if not isinstance(candidate_postures, torch.Tensor):
            return None
        if not isinstance(candidate_paths, torch.Tensor):
            return None
        if candidate_postures.dim() != 3 or candidate_paths.dim() != 4:
            return None
        return self.model.actor.compute_posture_diversity_loss(
            candidate_postures=candidate_postures,
            candidate_paths=candidate_paths,
            max_posture_similarity=self.posture_similarity_cap,
            max_path_similarity=self.posture_path_similarity_cap,
        )

    def _compute_retrieval_relevance_loss(
        self,
        outputs: dict[str, Any],
        realized_ic: Any,
    ) -> torch.Tensor | None:
        """Directly supervise the learned retrieval reranker from delayed outcomes."""
        if realized_ic is None:
            return None
        learned_scores = outputs.get("retrieval_relevance_scores")
        retrieved_postures = outputs.get("retrieved_postures")
        selected_posture = outputs.get("selected_posture")
        base_quality_scores = outputs.get("retrieval_base_quality_scores")
        if not isinstance(learned_scores, torch.Tensor):
            return None
        if not isinstance(retrieved_postures, torch.Tensor):
            return None
        if not isinstance(selected_posture, torch.Tensor):
            return None
        if not isinstance(base_quality_scores, torch.Tensor):
            return None
        realized_tensor = torch.as_tensor(realized_ic, dtype=torch.float32, device=self.device).flatten()
        if learned_scores.dim() != 2 or retrieved_postures.dim() != 3 or selected_posture.dim() != 2:
            return None
        return compute_retrieval_relevance_loss(
            learned_scores=learned_scores,
            retrieved_postures=retrieved_postures,
            selected_posture=selected_posture,
            base_quality_scores=base_quality_scores,
            realized_ic=realized_tensor,
            temperature=self.retrieval_relevance_temperature,
        )

    def _compute_posture_specialization_loss(
        self,
        outputs: dict[str, Any],
        step_batch: ChameliaStepBatch,
        runtime_domain: Any,
    ) -> torch.Tensor | None:
        """Use realized path outcomes to specialize non-baseline postures without naming them."""
        candidate_postures = outputs.get("candidate_postures")
        candidate_paths = outputs.get("candidate_paths")
        candidate_costs = outputs.get("candidate_costs")
        if not isinstance(candidate_postures, torch.Tensor):
            return None
        if not isinstance(candidate_paths, torch.Tensor):
            return None
        if not isinstance(candidate_costs, dict):
            return None
        predicted_total = candidate_costs.get("total")
        if not isinstance(predicted_total, torch.Tensor):
            return None
        if candidate_postures.dim() != 3 or candidate_paths.dim() != 4:
            return None
        if candidate_paths.shape[1] < 3:
            return None
        if not hasattr(runtime_domain, "simulate_path_outcome"):
            return None

        batch_size, num_candidates, path_length, action_dim = candidate_paths.shape
        flat_paths = candidate_paths.detach().reshape(batch_size * num_candidates, path_length, action_dim)
        expanded_state = self._expand_domain_state_for_candidates(
            step_batch.domain_state,
            num_candidates=num_candidates,
        )
        simulated = runtime_domain.simulate_path_outcome(flat_paths, expanded_state)
        if simulated is None:
            return None
        realized_ic = simulated.get("realized_intrinsic_cost")
        if not isinstance(realized_ic, torch.Tensor):
            return None
        realized_ic = realized_ic.to(self.device).reshape(batch_size, num_candidates)
        if num_candidates <= 1:
            return None

        nonbaseline_realized = realized_ic[:, 1:]
        baseline_realized = realized_ic[:, :1]
        realized_advantage = baseline_realized - nonbaseline_realized
        temperature = max(float(self.posture_specialization_temperature), 1e-4)
        target_distribution = F.softmax(realized_advantage / temperature, dim=1).detach()
        predicted_logits = -predicted_total[:, 1:] / temperature
        predicted_log_probs = F.log_softmax(predicted_logits, dim=1)
        return -(target_distribution * predicted_log_probs).sum(dim=1).mean()

    def run(self) -> None:
        """Run the curriculum scaffold loop."""
        eval_interval = int(self.config.get("curriculum", {}).get("eval_interval", 5000))
        while self.stage_idx < len(self.stages):
            stage_domains = self.stages[self.stage_idx]
            iterators = {domain.domain_name(): self._domain_iterator(domain) for domain in stage_domains}
            while not all(domain.cost.current_level >= len(domain.get_cost_schedule()) - 1 for domain in stage_domains):
                for domain in stage_domains:
                    batch = next(iterators[domain.domain_name()]).to_device(self.device)
                    batch.domain_state["level"] = domain.cost.current_level
                    if batch.tokens is not None:
                        batch.domain_state["tokens"] = batch.tokens
                    step_batch = self._to_step_batch(batch, domain)
                    if self.train_step is not None:
                        loss = self.train_step(self.model, step_batch, domain, self.device)
                    else:
                        loss = self._default_train_step(step_batch, domain)
                    if self.optimizer is not None:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()
                    latent_state = step_batch.model_inputs.float()
                    if latent_state.dim() == 2:
                        latent_state = latent_state.mean(dim=1, keepdim=True).repeat(1, 8)
                    elif latent_state.dim() == 3:
                        latent_state = latent_state.mean(dim=1)
                    action = torch.zeros(latent_state.shape[0], domain.action_dim, device=self.device)
                    domain.cost(latent_state.float(), action, batch.domain_state)
                    self.global_step += 1

                    if self.global_step % eval_interval == 0:
                        stage_results = self.graduation_manager.run_stage_probe(self.model, self.stage_idx)
                        domain_results = stage_results[domain.domain_name()]
                        if domain.cost.maybe_advance(domain_results):
                            self.save_stage_checkpoint(
                                self.stage_idx,
                                {"event": "level_advancement", "domain": domain.domain_name(), "metrics": domain_results},
                            )
            passed, metrics = self.graduation_manager.check_stage_graduation(self.model, self.stage_idx)
            if passed:
                self.save_stage_checkpoint(self.stage_idx, {"event": "stage_graduated", "metrics": metrics})
                self.stage_idx += 1
            else:
                break

    def save_stage_checkpoint(self, stage_idx: int, metrics: dict[str, Any]) -> None:
        """Save an immutable stage or level checkpoint.

        Args:
            stage_idx: Stage index.
            metrics: Metrics and metadata to store.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"chamelia_stage{stage_idx}_{metrics.get('event', 'checkpoint')}_{timestamp}.pt"
        payload = {
            "stage_idx": stage_idx,
            "global_step": self.global_step,
            "metrics": metrics,
            "status": self.graduation_manager.get_status_report(),
        }
        status_path = self.checkpoint_dir / filename
        bridge_artifact_path = self._save_bridge_artifact(stage_idx, metrics, timestamp)
        if bridge_artifact_path is not None:
            payload["bridge_artifact_path"] = str(bridge_artifact_path)
        torch.save(payload, status_path)

    def _save_bridge_artifact(
        self,
        stage_idx: int,
        metrics: dict[str, Any],
        timestamp: str,
    ) -> Path | None:
        """Persist a bridge-loadable model artifact alongside stage status.

        Args:
            stage_idx: Stage index.
            metrics: Metrics/event metadata.
            timestamp: Shared checkpoint timestamp.

        Returns:
            Path to the saved bridge artifact, or ``None`` if export is unavailable.
        """
        if not isinstance(self.model, torch.nn.Module):
            return None
        if self.export_model_config is None:
            return None

        event = str(metrics.get("event", "checkpoint"))
        artifact_name = f"bridge_stage{stage_idx}_{event}_{timestamp}.pth"
        artifact_payload = {
            "stage_idx": stage_idx,
            "global_step": self.global_step,
            "event": event,
            "metrics": metrics,
            "status": self.graduation_manager.get_status_report(),
            "model_state_dict": self.model.state_dict(),
            "config": deepcopy(self.export_model_config),
            "bridge_backbone_mode": self.export_backbone_mode,
            "model_version": self.export_model_version,
        }
        artifact_path = self.bridge_artifact_dir / artifact_name
        torch.save(artifact_payload, artifact_path)
        return artifact_path

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restore runner counters from a saved scaffold checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.stage_idx = int(payload["stage_idx"])
        self.global_step = int(payload["global_step"])
