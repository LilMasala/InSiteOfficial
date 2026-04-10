"""Interactive CartPole domain adapter for unified Chamelia training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.plugins.base import InteractiveDomainAdapter
from src.chamelia.tokenizers import StateVectorTokenizer

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised via fallback tests
    gym = None


@dataclass
class _ToyCartPoleState:
    x: float = 0.0
    x_dot: float = 0.0
    theta: float = 0.0
    theta_dot: float = 0.0

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.x, self.x_dot, self.theta, self.theta_dot],
            dtype=torch.float32,
        )


class _ToyCartPoleEnv:
    """Cheap local fallback that preserves the CartPole observation/action contract."""

    def __init__(self, *, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self._state = _ToyCartPoleState()
        self._steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        values = (torch.rand(4, generator=generator) - 0.5) * 0.1
        self._state = _ToyCartPoleState(
            x=float(values[0].item()),
            x_dot=float(values[1].item()),
            theta=float(values[2].item()),
            theta_dot=float(values[3].item()),
        )
        self._steps = 0
        return self._state.as_tensor(), {}

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        force = -1.0 if int(action) == 0 else 1.0
        self._steps += 1
        self._state.x_dot = 0.90 * self._state.x_dot + 0.08 * force - 0.04 * self._state.theta
        self._state.theta_dot = (
            0.92 * self._state.theta_dot + 0.12 * force + 0.25 * self._state.theta
        )
        self._state.x += self._state.x_dot
        self._state.theta += self._state.theta_dot
        terminated = abs(self._state.x) > 2.4 or abs(self._state.theta) > 0.35
        truncated = self._steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0
        return self._state.as_tensor(), reward, terminated, truncated, {}


class CartPoleDomain(InteractiveDomainAdapter):
    """Discrete CartPole adapter backed by Gymnasium when available."""

    modality_family = "state_vector_hjepa"
    action_space_type = "discrete"

    def __init__(self, *, embed_dim: int = 128, max_steps: int = 500) -> None:
        self._tokenizer = StateVectorTokenizer(
            num_features=4,
            embed_dim=embed_dim,
            domain_name="cartpole",
        )
        self.state_decoder = nn.Sequential(
            nn.LazyLinear(64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self._env = gym.make("CartPole-v1", max_episode_steps=max_steps) if gym is not None else _ToyCartPoleEnv(max_steps=max_steps)
        self._last_observation: torch.Tensor | None = None

    def get_tokenizer(self) -> StateVectorTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return 2

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {"state_decoder": self.state_decoder}

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        return action_vec.argmax(dim=-1)

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def stability_cost(_z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            state_value = domain_state.get("state_vector")
            if state_value is None:
                state = torch.zeros(action.shape[0], 4, dtype=torch.float32, device=action.device)
            else:
                state = state_value.float().to(action.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
            centered = state[:, 0].pow(2) + 4.0 * state[:, 2].pow(2)
            velocity = 0.1 * (state[:, 1].pow(2) + state[:, 3].pow(2))
            chosen_action = action.argmax(dim=-1).float()
            preference = domain_state.get("center_preference")
            if preference is None:
                preference = torch.full_like(chosen_action, 0.5)
            else:
                preference = preference.to(action.device)
            action_bias = 0.02 * torch.abs(chosen_action - preference)
            return centered + velocity + action_bias

        return [(stability_cost, 1.0)]

    def decode_state_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.state_decoder(z.float())

    def build_imagined_domain_state(
        self,
        current_domain_state: dict[str, Any],
        future_z: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        _ = step_idx
        approx_state = self.decode_state_from_latent(future_z)
        center_preference = torch.where(
            approx_state[:, 0] > 0.0,
            torch.zeros_like(approx_state[:, 0]),
            torch.ones_like(approx_state[:, 0]),
        )
        imagined_state = dict(current_domain_state)
        imagined_state["state_vector"] = approx_state
        imagined_state["center_preference"] = center_preference
        return imagined_state

    def build_simple_baseline_path(
        self,
        domain_state: dict[str, Any],
        path_length: int,
        action_dim: int,
    ) -> torch.Tensor | None:
        state = domain_state.get("state_vector")
        if state is None:
            return None
        state = state.float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] < 4 or action_dim < 2:
            return None
        steering = state[:, 2] + (0.25 * state[:, 3]) + (0.05 * state[:, 0]) + (0.02 * state[:, 1])
        preferred_action = (steering >= 0.0).long()
        logits = torch.full(
            (state.shape[0], path_length, action_dim),
            fill_value=-6.0,
            dtype=state.dtype,
            device=state.device,
        )
        logits.scatter_(
            dim=-1,
            index=preferred_action.view(-1, 1, 1).expand(-1, path_length, 1),
            src=torch.full(
                (state.shape[0], path_length, 1),
                fill_value=6.0,
                dtype=state.dtype,
                device=state.device,
            ),
        )
        return logits

    def compute_goal_latent(
        self,
        domain_state: dict[str, Any],
        z: torch.Tensor,
    ) -> torch.Tensor | None:
        _ = domain_state
        _ = z
        return None

    def compute_latent_state_decoder_loss(
        self,
        predicted_future_z: torch.Tensor,
        target_domain_state: dict[str, Any],
    ) -> torch.Tensor | None:
        state_value = target_domain_state.get("state_vector")
        if state_value is None:
            return None
        target_state = state_value.float().to(predicted_future_z.device)
        if target_state.dim() == 1:
            target_state = target_state.unsqueeze(0)
        decoded = self.decode_state_from_latent(predicted_future_z)
        return F.smooth_l1_loss(decoded, target_state.detach())

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.prepare_bridge_observation(observation).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        center_preference = torch.where(state[:, 0] > 0.0, torch.zeros_like(state[:, 0]), torch.ones_like(state[:, 0]))
        payload = {
            "state_vector": state,
            "center_preference": center_preference,
        }
        if info:
            payload["info"] = dict(info)
        return payload

    def get_domain_state(self, observation: Any) -> dict:
        return self.build_domain_state(observation, None)

    def prepare_bridge_observation(self, observation: Any) -> torch.Tensor:
        if torch.is_tensor(observation):
            return observation.float()
        return torch.tensor(observation, dtype=torch.float32)

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        state = domain_state["state_vector"].float()
        if state.dim() == 2:
            return state.mean(dim=0)
        return state

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        observation, info = self._env.reset(seed=seed)
        self._last_observation = self.prepare_bridge_observation(observation)
        return self._last_observation.clone(), dict(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chosen = int(action.item()) if torch.is_tensor(action) else int(action)
        observation, reward, terminated, truncated, info = self._env.step(chosen)
        self._last_observation = self.prepare_bridge_observation(observation)
        return self._last_observation.clone(), float(reward), bool(terminated), bool(truncated), dict(info)

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        _ = observation
        _ = info
        return torch.ones(2, dtype=torch.bool)

    def compute_realized_cost(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> float:
        _ = info
        state = self.prepare_bridge_observation(observation)
        penalty = float(state[0].item() ** 2 + 4.0 * state[2].item() ** 2)
        if terminated:
            penalty += 5.0
        if truncated:
            penalty *= 0.5
        return penalty - float(reward)

    def compute_metrics(self, episode_records: list[dict[str, Any]]) -> dict[str, float]:
        if not episode_records:
            return {"episode_reward_mean": 0.0, "episode_length_mean": 0.0}
        rewards = [float(record.get("episode_reward", 0.0)) for record in episode_records]
        lengths = [float(record.get("episode_length", 0.0)) for record in episode_records]
        return {
            "episode_reward_mean": sum(rewards) / len(rewards),
            "episode_length_mean": sum(lengths) / len(lengths),
        }

    @property
    def domain_name(self) -> str:
        return "cartpole"

    @property
    def vocab_size(self) -> int:
        return 0
