"""Interactive CartPole domain adapter for unified Chamelia training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

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
        self._env = gym.make("CartPole-v1", max_episode_steps=max_steps) if gym is not None else _ToyCartPoleEnv(max_steps=max_steps)
        self._last_observation: torch.Tensor | None = None

    def get_tokenizer(self) -> StateVectorTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return 2

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
