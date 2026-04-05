"""Stage 5: health, diagnosis, and care curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import (
    BaseCurriculumDomain,
    DomainSpec,
    MaskingStrategy,
    SequenceRuntimeDomain,
    make_level,
)
from training.curriculum.generators.health_sim import PatientState, SyntheticPatientEnv


HEALTH_FIELDS = ("bg", "mood", "engagement", "trust", "burnout", "burden")
HEALTH_ACTION_LABELS = ("hold", "stabilize", "support", "aggressive_optimize")


def _clip01(value: float) -> float:
    """Clamp a float into [0, 1]."""
    return max(0.0, min(1.0, value))


def _state_to_tensor(state: dict[str, float]) -> torch.Tensor:
    """Convert a patient-state dict to a compact tensor."""
    return torch.tensor([float(state[field]) for field in HEALTH_FIELDS], dtype=torch.float32)


def _tensor_to_state_dict(state_tensor: torch.Tensor) -> dict[str, float]:
    """Convert a compact patient-state tensor back to a dict."""
    values = state_tensor.detach().cpu().tolist()
    return {field: float(value) for field, value in zip(HEALTH_FIELDS, values, strict=True)}


def _normalize_state_value(field: str, value: float) -> float:
    """Normalize a patient-state value into [0, 1] for tokenization."""
    if field == "bg":
        return _clip01((value - 40.0) / 260.0)
    if field == "mood":
        return _clip01((value + 1.0) / 2.0)
    return _clip01(value)


def _state_to_tokens(state: dict[str, float], seq_len: int, vocab_size: int) -> torch.Tensor:
    """Encode a patient state into a fixed-length token sequence."""
    base_tokens: list[int] = []
    max_token = max(2, vocab_size - 1)
    for field in HEALTH_FIELDS:
        normalized = _normalize_state_value(field, float(state[field]))
        token = 1 + int(round(normalized * (max_token - 1)))
        complement = 1 + int(round((1.0 - normalized) * (max_token - 1)))
        base_tokens.extend([token, complement])

    base = torch.tensor(base_tokens, dtype=torch.long)
    repeats = (seq_len + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:seq_len]


def _health_realized_cost(state: dict[str, float]) -> float:
    """Compute realized health cost from an outcome state."""
    bg_cost = abs(float(state["bg"]) - 110.0) / 100.0
    burden_cost = max(0.0, float(state["burden"]))
    trust_cost = max(0.0, 1.0 - float(state["trust"]))
    burnout_cost = max(0.0, float(state["burnout"]))
    crisis_penalty = 0.5 if float(state["bg"]) < 70.0 or float(state["bg"]) > 220.0 else 0.0
    return float(bg_cost + burden_cost + trust_cost + 0.5 * burnout_cost + crisis_penalty)


class HealthMaskingStrategy(MaskingStrategy):
    """Mask physiological, psychological, and outcome fields."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply health-stage masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active health level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        window = min(2 + level, max(2, tokens.shape[1] // 4))
        start = tokens.shape[1] // 3
        mask[:, start : start + window] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _health_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic physiological and psychosocial sequences."""
    generator = torch.Generator().manual_seed(500 + level + (0 if split == "train" else 4000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        env = SyntheticPatientEnv()
        current_state = env.reset()
        event_draw = int(torch.randint(0, 3, (1,), generator=generator).item())
        if event_draw == 1:
            env.inject_event("illness", 0.3 + 0.7 * float(torch.rand(1, generator=generator).item()))
        elif event_draw == 2:
            env.inject_event("acute_stress", 0.3 + 0.7 * float(torch.rand(1, generator=generator).item()))
        current_state = env._to_dict()
        heuristic_action = "support" if current_state["trust"] < 0.6 or current_state["burden"] > 0.35 else "stabilize"
        next_state, _, _, _ = env.step(heuristic_action, {})
        tokens = _state_to_tokens(current_state, spec.seq_len, spec.vocab_size)
        target = _state_to_tokens(next_state, spec.seq_len, spec.vocab_size)
        crisis = torch.tensor(
            float(
                level >= 3
                or current_state["bg"] > 180.0
                or current_state["trust"] < 0.55
                or current_state["burden"] > 0.45
            ),
            dtype=torch.float32,
        )
        samples.append(
            {
                "tokens": tokens,
                "target": target,
                "crisis": crisis,
                "patient_state": _state_to_tensor(current_state),
            }
        )
    return samples


class HealthRuntimeDomain(SequenceRuntimeDomain):
    """Runtime health domain with simulated delayed outcomes."""

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        """Decode continuous actions into named care interventions."""
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        logits = action_vec[:, : len(HEALTH_ACTION_LABELS)]
        indices = logits.argmax(dim=-1).tolist()
        labels = [HEALTH_ACTION_LABELS[idx] for idx in indices]
        return labels[0] if len(labels) == 1 else labels

    def simulate_delayed_outcome(
        self,
        action_vec: torch.Tensor,
        domain_state: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """Simulate the next patient state from the selected action."""
        if action_vec.dim() == 2:
            action_vec = action_vec.unsqueeze(1)
        return self.simulate_path_outcome(action_vec, domain_state)

    def simulate_path_outcome(
        self,
        action_path: torch.Tensor,
        domain_state: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """Simulate a delayed outcome for a whole candidate path."""
        patient_state = domain_state.get("patient_state")
        if not torch.is_tensor(patient_state):
            return None
        if patient_state.dim() == 1:
            patient_state = patient_state.unsqueeze(0)
        if action_path.dim() == 2:
            action_path = action_path.unsqueeze(1)

        outcome_tokens: list[torch.Tensor] = []
        realized_costs: list[float] = []
        patient_state_cpu = patient_state.detach().cpu()
        for idx in range(action_path.shape[0]):
            env = SyntheticPatientEnv()
            state_dict = _tensor_to_state_dict(patient_state_cpu[idx])
            env.state = PatientState(**state_dict)
            labels = self.decode_action(action_path[idx])
            if isinstance(labels, str):
                labels = [labels]
            next_state = state_dict
            cumulative_cost = 0.0
            for label in labels:
                next_state, _, _, _ = env.step(label, {})
                cumulative_cost += _health_realized_cost(next_state)
            outcome_tokens.append(
                _state_to_tokens(next_state, self.owner.spec.seq_len, self.owner.vocab_size)
            )
            realized_costs.append(cumulative_cost / max(1, len(labels)))

        return {
            "outcome_observation": torch.stack(outcome_tokens, dim=0).to(action_path.device),
            "realized_intrinsic_cost": torch.tensor(
                realized_costs,
                dtype=torch.float32,
                device=action_path.device,
            ),
        }

class HealthCurriculumDomain(BaseCurriculumDomain):
    """Stage 5 health and care scaffold."""

    def __init__(self, domain_variant: str = "synthetic_patients", batch_size: int = 8, seq_len: int = 48) -> None:
        """Initialize the health curriculum domain scaffold."""

        def physiology_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        def crisis_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            crisis = domain_state["crisis"].to(action.device)
            aggressive = action.abs().mean(dim=-1)
            return crisis * aggressive

        schedule = [
            make_level(0, "physiological prediction accuracy", [(physiology_cost, 1.0)], {"health_score": 0.72}, 64),
            make_level(
                1,
                "psychological state prediction",
                [(physiology_cost, 0.8), (crisis_cost, 0.2)],
                {"health_score": 0.78, "trust_alignment": 0.70},
                64,
            ),
            make_level(
                2,
                "intervention appropriateness",
                [(physiology_cost, 0.6), (crisis_cost, 0.4)],
                {"health_score": 0.82, "trust_alignment": 0.76},
                64,
            ),
            make_level(
                3,
                "illness and crisis recognition",
                [(physiology_cost, 0.4), (crisis_cost, 0.6)],
                {"health_score": 0.88, "crisis_recognition": 0.90},
                64,
            ),
            make_level(
                4,
                "autonomy and trust management",
                [(physiology_cost, 0.4), (crisis_cost, 0.6)],
                {"health_score": 0.91, "autonomy_respect": 0.90},
                64,
            ),
            make_level(
                5,
                "personalization",
                [(physiology_cost, 0.3), (crisis_cost, 0.7)],
                {"health_score": 0.94, "personalization": 0.85},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.72 + 0.05 * level)
            return {
                "health_score": base,
                "crisis_recognition": min(0.99, 0.74 + 0.05 * level),
                "trust_alignment": min(0.99, 0.70 + 0.04 * level),
                "autonomy_respect": min(0.99, 0.74 + 0.05 * level),
                "personalization": min(0.99, 0.68 + 0.04 * level),
            }

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=5,
                action_dim=16,
                vocab_size=4096,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=HealthMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=_health_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a simple sequence-based runtime plugin for health tokens."""
        return HealthRuntimeDomain(owner=self, embed_dim=embed_dim)
