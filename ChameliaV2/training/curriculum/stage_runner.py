"""Curriculum training loop scaffold."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.chamelia.chamelia import Chamelia
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
        """
        self.model = model
        self.stages = stages
        self.graduation_manager = graduation_manager
        self.config = config
        self.device = torch.device(device)
        self.train_step = train_step
        self.optimizer = optimizer
        self.stage_idx = int(config.get("curriculum", {}).get("start_stage", 0))
        self.global_step = 0
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
            store_to_memory=False,
            input_kind=input_kind,
        )
        runtime_costs = runtime_domain.get_intrinsic_cost_fns()
        loss = torch.zeros((), device=self.device)
        for cost_fn, weight in runtime_costs:
            loss = loss + float(weight) * cost_fn(
                outputs["z"],
                outputs["action_vec"],
                step_batch.domain_state,
            ).mean()
        return loss

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
        torch.save(payload, self.checkpoint_dir / filename)

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restore runner counters from a saved scaffold checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.stage_idx = int(payload["stage_idx"])
        self.global_step = int(payload["global_step"])
