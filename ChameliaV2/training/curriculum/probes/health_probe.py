"""Behavioral health probe for Chamelia planning quality."""

from __future__ import annotations

import copy
from typing import Any

import torch

from src.chamelia.memory import EpisodeRecord
from training.curriculum.domains.stage5_health import (
    HEALTH_ACTION_LABELS,
    HEALTH_FIELDS,
    HealthRuntimeDomain,
    _health_realized_cost,
    _state_to_tensor,
    _state_to_tokens,
)


def _infer_device(model: Any) -> torch.device:
    """Infer the active device for a model-like object."""
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except (StopIteration, TypeError):
            pass
    return torch.device(getattr(model, "device", "cpu"))


class HealthProbe:
    """Probe simple-path preference, fragile support, and memory-sensitive replanning."""

    def __init__(self, simplicity_margin: float = 0.05) -> None:
        self.simplicity_margin = simplicity_margin

    def evaluate(self, model: Any, batch: dict[str, Any] | None = None) -> dict[str, float]:
        """Evaluate planner behavior on health-specific counterfactual scenarios."""
        batch = batch or {}
        runtime_domain = batch.get("runtime_domain")
        if runtime_domain is None:
            runtime_domain = getattr(model, "domain", None)
        if not isinstance(runtime_domain, HealthRuntimeDomain):
            raise TypeError("HealthProbe requires a HealthRuntimeDomain runtime plugin.")

        if hasattr(model, "set_domain"):
            model.set_domain(runtime_domain)

        cases = batch.get("cases", self._default_cases())
        if not cases:
            raise ValueError("HealthProbe requires at least one probe case.")

        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()

        simplicity_hits = 0
        simplicity_eligible = 0
        fragile_supportive = 0
        fragile_non_aggressive = 0
        fragile_count = 0
        realized_advantages: list[float] = []
        memory_shifts: list[float] = []

        try:
            for case in cases:
                outputs, domain_state = self._run_case(
                    model=model,
                    runtime_domain=runtime_domain,
                    case=case,
                )
                chosen_cost = self._realized_path_cost(
                    runtime_domain=runtime_domain,
                    path=outputs["selected_path"],
                    domain_state=domain_state,
                )
                baseline_path = outputs["candidate_paths"][:, 0, :, :]
                baseline_cost = self._realized_path_cost(
                    runtime_domain=runtime_domain,
                    path=baseline_path,
                    domain_state=domain_state,
                )
                realized_advantages.append(float((baseline_cost - chosen_cost).mean().item()))

                selected_idx = outputs.get("selected_candidate_idx")
                if not isinstance(selected_idx, torch.Tensor):
                    selected_idx = torch.zeros(1, dtype=torch.long, device=chosen_cost.device)
                simplicity_mask = (baseline_cost - chosen_cost).abs() <= self.simplicity_margin
                simplicity_eligible += int(simplicity_mask.sum().item())
                simplicity_hits += int(
                    (simplicity_mask & (selected_idx.to(simplicity_mask.device) == 0)).sum().item()
                )

                first_action = outputs["selected_path"][:, 0, :]
                decoded = runtime_domain.decode_action(first_action)
                labels = [decoded] if isinstance(decoded, str) else list(decoded)
                if bool(case.get("fragile", False)):
                    fragile_count += len(labels)
                    fragile_supportive += sum(
                        label in {"hold", "stabilize", "support"} for label in labels
                    )
                    fragile_non_aggressive += sum(
                        label != "aggressive_optimize" for label in labels
                    )

                if bool(case.get("memory_relevant", False)):
                    memory_shifts.append(
                        self._measure_memory_plan_shift(
                            model=model,
                            runtime_domain=runtime_domain,
                            case=case,
                        )
                    )
        finally:
            if hasattr(model, "train"):
                model.train(was_training)

        return {
            "simple_path_preference": (
                float(simplicity_hits) / float(max(1, simplicity_eligible))
            ),
            "baseline_competitive_rate": float(simplicity_eligible) / float(len(cases)),
            "fragile_supportive_rate": (
                float(fragile_supportive) / float(max(1, fragile_count))
            ),
            "fragile_aggressive_avoidance": (
                float(fragile_non_aggressive) / float(max(1, fragile_count))
            ),
            "mean_realized_advantage_over_baseline": (
                float(sum(realized_advantages) / max(1, len(realized_advantages)))
            ),
            "memory_plan_shift_rate": (
                float(sum(memory_shifts) / max(1, len(memory_shifts)))
                if memory_shifts
                else 0.0
            ),
        }

    def _default_cases(self) -> list[dict[str, Any]]:
        """Return a compact set of canonical health planning scenarios."""
        return [
            {
                "name": "stable_low_burden",
                "state": {
                    "bg": 112.0,
                    "mood": 0.25,
                    "engagement": 0.82,
                    "trust": 0.88,
                    "burnout": 0.08,
                    "burden": 0.12,
                },
                "fragile": False,
                "memory_relevant": False,
            },
            {
                "name": "fragile_high_burden",
                "state": {
                    "bg": 178.0,
                    "mood": -0.35,
                    "engagement": 0.42,
                    "trust": 0.38,
                    "burnout": 0.34,
                    "burden": 0.62,
                },
                "fragile": True,
                "memory_relevant": True,
            },
            {
                "name": "stress_regime",
                "state": {
                    "bg": 168.0,
                    "mood": -0.18,
                    "engagement": 0.56,
                    "trust": 0.54,
                    "burnout": 0.24,
                    "burden": 0.48,
                },
                "fragile": True,
                "memory_relevant": True,
            },
        ]

    def _case_state(self, case: dict[str, Any]) -> dict[str, float]:
        """Normalize a probe case into a plain health-state dictionary."""
        state = case.get("state", case.get("patient_state"))
        if isinstance(state, dict):
            return {field: float(state[field]) for field in HEALTH_FIELDS}
        if torch.is_tensor(state):
            values = state.detach().cpu().flatten().tolist()
            return {field: float(value) for field, value in zip(HEALTH_FIELDS, values, strict=True)}
        raise TypeError("Probe case must provide a health state as dict or tensor.")

    def _build_domain_state(
        self,
        case: dict[str, Any],
        runtime_domain: HealthRuntimeDomain,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build embedded tokens, mask, and domain state for one probe case."""
        state_dict = self._case_state(case)
        tokens = _state_to_tokens(
            state_dict,
            runtime_domain.owner.spec.seq_len,
            runtime_domain.vocab_size,
        ).unsqueeze(0)
        tokenized = runtime_domain.get_tokenizer()(tokens.to(device))
        mask = torch.zeros(
            tokenized.tokens.shape[0],
            tokenized.tokens.shape[1],
            device=device,
            dtype=torch.float32,
        )
        crisis = float(
            state_dict["bg"] > 180.0
            or state_dict["trust"] < 0.55
            or state_dict["burden"] > 0.45
        )
        domain_state = {
            "patient_state": _state_to_tensor(state_dict).unsqueeze(0).to(device),
            "target": tokens.to(device),
            "crisis": torch.tensor([crisis], dtype=torch.float32, device=device),
        }
        return tokenized.tokens, mask, domain_state

    def _run_case(
        self,
        model: Any,
        runtime_domain: HealthRuntimeDomain,
        case: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the planner on one health probe case."""
        device = _infer_device(model)
        tokens, mask, domain_state = self._build_domain_state(case, runtime_domain, device)
        with torch.no_grad():
            outputs = model(
                tokens=tokens,
                mask=mask,
                domain_state=domain_state,
                actor_mode="mode2",
                store_to_memory=False,
                input_kind="embedded_tokens",
            )
        return outputs, domain_state

    def _realized_path_cost(
        self,
        runtime_domain: HealthRuntimeDomain,
        path: torch.Tensor,
        domain_state: dict[str, Any],
    ) -> torch.Tensor:
        """Simulate and return realized path cost for a selected or baseline path."""
        simulated = runtime_domain.simulate_path_outcome(path.detach(), domain_state)
        if simulated is None:
            patient_state = domain_state["patient_state"]
            costs = []
            for idx in range(patient_state.shape[0]):
                state_dict = {
                    field: float(patient_state[idx, field_idx].item())
                    for field_idx, field in enumerate(HEALTH_FIELDS)
                }
                costs.append(_health_realized_cost(state_dict))
            return torch.tensor(costs, dtype=torch.float32, device=path.device)
        return simulated["realized_intrinsic_cost"].to(path.device).flatten()

    def _select_support_seed(
        self,
        outputs: dict[str, Any],
        runtime_domain: HealthRuntimeDomain,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick a supportive/stabilizing seed from the current candidate set when possible."""
        candidate_actions = outputs["candidate_actions"][0]
        candidate_paths = outputs["candidate_paths"][0]
        candidate_postures = outputs["candidate_postures"][0]
        decoded = runtime_domain.decode_action(candidate_actions)
        labels = [decoded] if isinstance(decoded, str) else list(decoded)
        preferred = {"support", "stabilize", "hold"}
        for idx, label in enumerate(labels):
            if label in preferred and idx < candidate_postures.shape[0]:
                return (
                    candidate_actions[idx].detach().cpu(),
                    candidate_paths[idx].detach().cpu(),
                    candidate_postures[idx].detach().cpu(),
                )
        return (
            outputs["action_vec"][0].detach().cpu(),
            outputs["selected_path"][0].detach().cpu(),
            outputs["selected_posture"][0].detach().cpu(),
        )

    def _snapshot_memory(self, model: Any) -> tuple[torch.Tensor, list[Any], int, int] | None:
        """Capture the current memory state so probe seeding can be reverted."""
        memory = getattr(model, "memory", None)
        if memory is None:
            return None
        return (
            memory.keys.clone(),
            copy.deepcopy(memory.records),
            int(memory.size),
            int(memory.head),
        )

    def _restore_memory(
        self,
        model: Any,
        snapshot: tuple[torch.Tensor, list[Any], int, int] | None,
    ) -> None:
        """Restore a previously snapshotted memory state."""
        if snapshot is None or not hasattr(model, "memory"):
            return
        keys, records, size, head = snapshot
        model.memory.keys.copy_(keys)
        model.memory.records = records
        model.memory.size = size
        model.memory.head = head

    def _measure_memory_plan_shift(
        self,
        model: Any,
        runtime_domain: HealthRuntimeDomain,
        case: dict[str, Any],
    ) -> float:
        """Measure whether a high-quality retrieved episode changes the chosen plan."""
        memory = getattr(model, "memory", None)
        if memory is None:
            return 0.0

        base_outputs, _ = self._run_case(model, runtime_domain, case)
        snapshot = self._snapshot_memory(model)
        try:
            seed_action, seed_path, seed_posture = self._select_support_seed(
                base_outputs,
                runtime_domain,
            )
            seed_record = EpisodeRecord(
                key=base_outputs["z"][0].detach().cpu(),
                action=seed_action,
                ctx_tokens=base_outputs["ctx_tokens"][0].detach().cpu(),
                ic_at_decision=0.1,
                ic_realized=0.0,
                tc_predicted=0.0,
                outcome_key=base_outputs["z"][0].detach().cpu(),
                step=-1,
                domain_name=runtime_domain.domain_name,
                selected_posture=seed_posture,
                selected_path=seed_path,
            )
            memory.store(seed_record)
            seeded_outputs, _ = self._run_case(model, runtime_domain, case)
        finally:
            self._restore_memory(model, snapshot)

        label_before = runtime_domain.decode_action(base_outputs["action_vec"])
        label_after = runtime_domain.decode_action(seeded_outputs["action_vec"])
        before_labels = [label_before] if isinstance(label_before, str) else list(label_before)
        after_labels = [label_after] if isinstance(label_after, str) else list(label_after)
        if before_labels != after_labels:
            return 1.0
        if not torch.allclose(
            base_outputs["selected_path"],
            seeded_outputs["selected_path"],
            atol=1.0e-5,
        ):
            return 1.0
        if not torch.allclose(
            base_outputs["selected_posture"],
            seeded_outputs["selected_posture"],
            atol=1.0e-5,
        ):
            return 1.0
        return 0.0
