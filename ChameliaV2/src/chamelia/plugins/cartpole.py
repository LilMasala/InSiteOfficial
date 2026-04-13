"""Interactive CartPole domain adapter for unified Chamelia training."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    _X_THRESHOLD = 2.4
    _TRUE_THETA_THRESHOLD = 12 * 2 * math.pi / 360
    _TOY_THETA_THRESHOLD = 0.35

    def __init__(self, *, embed_dim: int = 128, max_steps: int = 500) -> None:
        self.max_steps = max_steps
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
        self._episode_steps = 0

    def get_tokenizer(self) -> StateVectorTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return 2

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {"state_decoder": self.state_decoder}

    def _termination_thresholds(self) -> tuple[float, float]:
        theta_threshold = (
            self._TRUE_THETA_THRESHOLD if gym is not None else self._TOY_THETA_THRESHOLD
        )
        return self._X_THRESHOLD, theta_threshold

    def _derive_state_features(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x_threshold, theta_threshold = self._termination_thresholds()
        x = state[:, 0]
        theta = state[:, 2]
        x_limit = torch.full_like(x, float(x_threshold))
        theta_limit = torch.full_like(theta, float(theta_threshold))
        termination_ratio = torch.maximum(
            x.abs() / x_limit.clamp_min(1.0e-6),
            theta.abs() / theta_limit.clamp_min(1.0e-6),
        )
        stability_margin = torch.minimum(
            x_limit - x.abs(),
            theta_limit - theta.abs(),
        )
        center_preference = torch.where(
            x > 0.0,
            torch.zeros_like(x),
            torch.ones_like(x),
        )
        return {
            "center_preference": center_preference,
            "termination_ratio": termination_ratio,
            "stability_margin": stability_margin,
        }

    def _compute_stability_cost_from_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        state = state.float().to(action.device)
        action = action.float()
        features = self._derive_state_features(state)
        centered = state[:, 0].pow(2) + 4.0 * state[:, 2].pow(2)
        velocity = 0.1 * (state[:, 1].pow(2) + state[:, 3].pow(2))
        chosen_action = action.argmax(dim=-1).float()
        action_bias = 0.02 * torch.abs(chosen_action - features["center_preference"].to(action.device))
        termination_ratio = features["termination_ratio"].to(action.device)
        stability_margin = features["stability_margin"].to(action.device)
        risk_buffer_penalty = 0.5 * F.relu(termination_ratio - 0.75).pow(2)
        terminal_risk_penalty = 5.0 * torch.sigmoid(24.0 * (termination_ratio - 1.0))
        margin_penalty = 0.25 * F.relu(0.05 - stability_margin).pow(2)
        return centered + velocity + action_bias + risk_buffer_penalty + terminal_risk_penalty + margin_penalty

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
            return self._compute_stability_cost_from_state(state, action)

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
        imagined_state = dict(current_domain_state)
        imagined_state["state_vector"] = approx_state
        imagined_state.update(self._derive_state_features(approx_state))
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
        preferred_action = (steering < 0.0).long()
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
        feature_weights = torch.tensor(
            [1.5, 0.5, 3.0, 0.75],
            dtype=decoded.dtype,
            device=decoded.device,
        )
        per_feature_error = F.smooth_l1_loss(
            decoded,
            target_state.detach(),
            reduction="none",
        )
        state_loss = (per_feature_error * feature_weights.view(1, -1)).mean()
        predicted_features = self._derive_state_features(decoded)
        target_features = self._derive_state_features(target_state)
        ratio_loss = F.smooth_l1_loss(
            predicted_features["termination_ratio"],
            target_features["termination_ratio"].detach(),
        )
        margin_loss = F.smooth_l1_loss(
            predicted_features["stability_margin"],
            target_features["stability_margin"].detach(),
        )
        return state_loss + (0.5 * ratio_loss) + (0.25 * margin_loss)

    def compute_imagined_state_calibration_loss(
        self,
        predicted_future_z: torch.Tensor,
        action: torch.Tensor,
        target_domain_state: dict[str, Any],
        step_idx: int,
    ) -> torch.Tensor | None:
        _ = step_idx
        state_value = target_domain_state.get("state_vector")
        if state_value is None:
            return None
        target_state = state_value.float().to(predicted_future_z.device)
        if target_state.dim() == 1:
            target_state = target_state.unsqueeze(0)
        decoded = self.decode_state_from_latent(predicted_future_z)
        predicted_cost = self._compute_stability_cost_from_state(decoded, action)
        target_cost = self._compute_stability_cost_from_state(target_state, action)
        predicted_features = self._derive_state_features(decoded)
        target_features = self._derive_state_features(target_state)
        cost_loss = F.smooth_l1_loss(predicted_cost, target_cost.detach())
        ratio_loss = F.smooth_l1_loss(
            predicted_features["termination_ratio"],
            target_features["termination_ratio"].detach(),
        )
        return cost_loss + (0.25 * ratio_loss)

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.prepare_bridge_observation(observation).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        payload = {
            "state_vector": state,
        }
        payload.update(self._derive_state_features(state))
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
        self._episode_steps = 0
        return self._last_observation.clone(), dict(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chosen = int(action.item()) if torch.is_tensor(action) else int(action)
        observation, reward, terminated, truncated, info = self._env.step(chosen)
        self._last_observation = self.prepare_bridge_observation(observation)
        self._episode_steps += 1
        return self._last_observation.clone(), float(reward), bool(terminated), bool(truncated), dict(info)

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        _ = observation
        _ = info
        return torch.ones(2, dtype=torch.bool)

    def baseline_action(
        self,
        kind: str,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> Any:
        if kind == "simple":
            domain_state = self.build_domain_state(observation, info)
            baseline_path = self.build_simple_baseline_path(
                domain_state,
                path_length=1,
                action_dim=self.get_action_dim(),
            )
            if baseline_path is not None:
                return baseline_path[:, 0, :].argmax(dim=-1)
        return super().baseline_action(kind, observation, info)

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

    def _simulate_true_cartpole_step(
        self,
        state: torch.Tensor,
        action_idx: int,
        *,
        step_count: int,
    ) -> tuple[torch.Tensor, float, bool, bool]:
        x, x_dot, theta, theta_dot = [float(value) for value in state.tolist()]
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masscart + masspole
        length = 0.5
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02
        x_threshold = 2.4
        theta_threshold_radians = 12 * 2 * math.pi / 360

        force = force_mag if int(action_idx) == 1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + (polemass_length * theta_dot * theta_dot * sintheta)) / total_mass
        thetaacc = (gravity * sintheta - (costheta * temp)) / (
            length * ((4.0 / 3.0) - ((masspole * costheta * costheta) / total_mass))
        )
        xacc = temp - ((polemass_length * thetaacc * costheta) / total_mass)

        x = x + (tau * x_dot)
        x_dot = x_dot + (tau * xacc)
        theta = theta + (tau * theta_dot)
        theta_dot = theta_dot + (tau * thetaacc)

        next_state = torch.tensor([x, x_dot, theta, theta_dot], dtype=torch.float32)
        terminated = bool(
            x < -x_threshold
            or x > x_threshold
            or theta < -theta_threshold_radians
            or theta > theta_threshold_radians
        )
        truncated = bool(step_count + 1 >= self.max_steps)
        reward = 1.0
        return next_state, reward, terminated, truncated

    def _simulate_toy_step(
        self,
        state: torch.Tensor,
        action_idx: int,
        *,
        step_count: int,
    ) -> tuple[torch.Tensor, float, bool, bool]:
        force = -1.0 if int(action_idx) == 0 else 1.0
        x, x_dot, theta, theta_dot = [float(value) for value in state.tolist()]
        x_dot = 0.90 * x_dot + 0.08 * force - 0.04 * theta
        theta_dot = 0.92 * theta_dot + 0.12 * force + 0.25 * theta
        x = x + x_dot
        theta = theta + theta_dot
        next_state = torch.tensor([x, x_dot, theta, theta_dot], dtype=torch.float32)
        terminated = bool(abs(x) > 2.4 or abs(theta) > 0.35)
        truncated = bool(step_count + 1 >= self.max_steps)
        reward = 1.0 if not terminated else 0.0
        return next_state, reward, terminated, truncated

    def _simulate_counterfactual_path(
        self,
        initial_state: torch.Tensor,
        action_path: torch.Tensor,
        *,
        gamma: float,
        start_step: int,
    ) -> dict[str, Any]:
        state = initial_state.detach().clone().float()
        discounted_cost = 0.0
        reward_total = 0.0
        step_costs: list[float] = []
        step_rewards: list[float] = []
        effective_actions: list[int] = []
        terminated = False
        truncated = False
        for path_idx in range(action_path.shape[0]):
            action_idx = int(action_path[path_idx].argmax(dim=-1).item())
            effective_actions.append(action_idx)
            if gym is not None:
                next_state, reward, terminated, truncated = self._simulate_true_cartpole_step(
                    state,
                    action_idx,
                    step_count=start_step + path_idx,
                )
            else:
                next_state, reward, terminated, truncated = self._simulate_toy_step(
                    state,
                    action_idx,
                    step_count=start_step + path_idx,
                )
            step_cost = self.compute_realized_cost(
                next_state,
                reward,
                terminated,
                truncated,
                None,
            )
            step_costs.append(float(step_cost))
            step_rewards.append(float(reward))
            discounted_cost += float((gamma ** path_idx) * step_cost)
            reward_total += float(reward)
            state = next_state
            if terminated or truncated:
                break
        return {
            "actions": effective_actions,
            "step_costs": step_costs,
            "step_rewards": step_rewards,
            "discounted_cost": float(discounted_cost),
            "reward_total": float(reward_total),
            "steps_executed": len(step_costs),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "final_state": [float(value) for value in state.tolist()],
        }

    def analyze_planner_candidates(
        self,
        *,
        candidate_paths: torch.Tensor,
        candidate_ic: torch.Tensor | None,
        candidate_tc: torch.Tensor | None,
        candidate_total: torch.Tensor | None,
        candidate_terminal_latents: torch.Tensor | None,
        selected_candidate_idx: int | None,
        domain_state: dict[str, Any],
        gamma: float,
        planner_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        state_value = domain_state.get("state_vector")
        if state_value is None:
            return None
        state = state_value.detach().clone().float()
        if state.dim() == 2:
            state = state[0]
        if state.numel() < 4 or candidate_paths.dim() != 3:
            return None

        predicted_total = (
            candidate_total.detach().cpu().float()
            if candidate_total is not None
            else torch.zeros(candidate_paths.shape[0], dtype=torch.float32)
        )
        predicted_ic = (
            candidate_ic.detach().cpu().float()
            if candidate_ic is not None
            else torch.zeros_like(predicted_total)
        )
        predicted_tc = (
            candidate_tc.detach().cpu().float()
            if candidate_tc is not None
            else torch.zeros_like(predicted_total)
        )
        tail_discount = float(gamma ** int(candidate_paths.shape[1]))
        predicted_discounted_tc = predicted_tc * tail_discount

        decoded_terminal = None
        if candidate_terminal_latents is not None:
            decoded_terminal = self.decode_state_from_latent(candidate_terminal_latents.detach().float()).cpu()

        candidate_debug: list[dict[str, Any]] = []
        actual_costs: list[float] = []
        actual_rewards: list[float] = []
        terminal_state_maes: list[float | None] = []
        for candidate_idx in range(candidate_paths.shape[0]):
            rollout = self._simulate_counterfactual_path(
                state,
                candidate_paths[candidate_idx].detach().cpu().float(),
                gamma=gamma,
                start_step=self._episode_steps,
            )
            actual_cost = float(rollout["discounted_cost"])
            actual_reward = float(rollout["reward_total"])
            actual_costs.append(actual_cost)
            actual_rewards.append(actual_reward)
            terminal_state_mae: float | None = None
            if decoded_terminal is not None and candidate_idx < decoded_terminal.shape[0]:
                actual_final = torch.tensor(rollout["final_state"], dtype=torch.float32)
                terminal_state_mae = float(
                    torch.mean(torch.abs(decoded_terminal[candidate_idx] - actual_final)).item()
                )
            terminal_state_maes.append(terminal_state_mae)
            candidate_debug.append(
                {
                    "candidate_idx": candidate_idx,
                    "predicted_total": float(predicted_total[candidate_idx].item()),
                    "predicted_ic": float(predicted_ic[candidate_idx].item()),
                    "predicted_tc": float(predicted_tc[candidate_idx].item()),
                    "predicted_discounted_tc": float(predicted_discounted_tc[candidate_idx].item()),
                    "actual_discounted_cost": actual_cost,
                    "actual_reward": actual_reward,
                    "steps_executed": int(rollout["steps_executed"]),
                    "actions": rollout["actions"],
                    "terminated": bool(rollout["terminated"]),
                    "truncated": bool(rollout["truncated"]),
                    "step_costs": rollout["step_costs"],
                    "step_rewards": rollout["step_rewards"],
                    "final_state": rollout["final_state"],
                    "terminal_state_mae": terminal_state_mae,
                }
            )

        baseline_idx = 0
        predicted_best_idx = int(predicted_total.argmin().item())
        actual_cost_tensor = torch.tensor(actual_costs, dtype=torch.float32)
        actual_reward_tensor = torch.tensor(actual_rewards, dtype=torch.float32)
        best_actual_idx = int(actual_cost_tensor.argmin().item())
        selected_idx = int(selected_candidate_idx) if selected_candidate_idx is not None else predicted_best_idx
        selection_debug = planner_trace.get("selection_debug", {}) if planner_trace is not None else {}
        selected_minus_baseline_predicted_ic = float(
            predicted_ic[selected_idx].item() - predicted_ic[baseline_idx].item()
        )
        selected_minus_baseline_predicted_tc = float(
            predicted_tc[selected_idx].item() - predicted_tc[baseline_idx].item()
        )
        selected_minus_baseline_predicted_discounted_tc = float(
            predicted_discounted_tc[selected_idx].item() - predicted_discounted_tc[baseline_idx].item()
        )
        selected_minus_baseline_predicted_total = float(
            predicted_total[selected_idx].item() - predicted_total[baseline_idx].item()
        )
        selected_minus_baseline_actual_cost = float(
            actual_cost_tensor[selected_idx].item() - actual_cost_tensor[baseline_idx].item()
        )

        predicted_advantage_source = "none"
        if selected_minus_baseline_predicted_total < 0.0:
            ic_improves = selected_minus_baseline_predicted_ic < 0.0
            tc_improves = selected_minus_baseline_predicted_discounted_tc < 0.0
            if ic_improves and tc_improves:
                ic_magnitude = abs(selected_minus_baseline_predicted_ic)
                tc_magnitude = abs(selected_minus_baseline_predicted_discounted_tc)
                if ic_magnitude > (1.25 * tc_magnitude):
                    predicted_advantage_source = "ic_dominant"
                elif tc_magnitude > (1.25 * ic_magnitude):
                    predicted_advantage_source = "tc_tail_dominant"
                else:
                    predicted_advantage_source = "shared"
            elif ic_improves:
                predicted_advantage_source = "ic_only"
            elif tc_improves:
                predicted_advantage_source = "tc_tail_only"

        harmful_pick_source = "not_harmful"
        if selected_minus_baseline_predicted_total < 0.0 and selected_minus_baseline_actual_cost > 0.0:
            ic_improves = selected_minus_baseline_predicted_ic < 0.0
            tc_improves = selected_minus_baseline_predicted_discounted_tc < 0.0
            if tc_improves and not ic_improves:
                harmful_pick_source = "tc_tail_only_flip"
            elif ic_improves and not tc_improves:
                harmful_pick_source = "ic_path_only_flip"
            elif ic_improves and tc_improves:
                if abs(selected_minus_baseline_predicted_ic) > (
                    1.25 * abs(selected_minus_baseline_predicted_discounted_tc)
                ):
                    harmful_pick_source = "ic_path_dominant"
                elif abs(selected_minus_baseline_predicted_discounted_tc) > (
                    1.25 * abs(selected_minus_baseline_predicted_ic)
                ):
                    harmful_pick_source = "tc_tail_dominant"
                else:
                    harmful_pick_source = "shared"
            else:
                harmful_pick_source = "unclear"
        return {
            "baseline_candidate_idx": baseline_idx,
            "predicted_best_idx": predicted_best_idx,
            "best_actual_idx": best_actual_idx,
            "selected_candidate_idx": selected_idx,
            "selection_reason": str(selection_debug.get("reason", "unknown")),
            "tail_discount": tail_discount,
            "root_candidate_mean_costs": list(selection_debug.get("root_candidate_mean_costs", [])),
            "root_candidate_visit_counts": list(selection_debug.get("root_candidate_visit_counts", [])),
            "baseline_mean_cost": selection_debug.get("baseline_mean_cost"),
            "selected_mean_cost": selection_debug.get("selected_mean_cost"),
            "root_mean_cost_std": selection_debug.get("root_mean_cost_std"),
            "root_predicted_cost_std": selection_debug.get("root_predicted_cost_std"),
            "required_predicted_improvement": selection_debug.get("required_predicted_improvement"),
            "actual_predicted_improvement": selection_debug.get("actual_predicted_improvement"),
            "predicted_advantage_source": predicted_advantage_source,
            "harmful_pick_source": harmful_pick_source,
            "selected_predicted_ic": float(predicted_ic[selected_idx].item()),
            "baseline_predicted_ic": float(predicted_ic[baseline_idx].item()),
            "selected_predicted_tc": float(predicted_tc[selected_idx].item()),
            "baseline_predicted_tc": float(predicted_tc[baseline_idx].item()),
            "selected_predicted_discounted_tc": float(predicted_discounted_tc[selected_idx].item()),
            "baseline_predicted_discounted_tc": float(predicted_discounted_tc[baseline_idx].item()),
            "selected_predicted_total": float(predicted_total[selected_idx].item()),
            "baseline_predicted_total": float(predicted_total[baseline_idx].item()),
            "selected_actual_discounted_cost": float(actual_cost_tensor[selected_idx].item()),
            "baseline_actual_discounted_cost": float(actual_cost_tensor[baseline_idx].item()),
            "selected_actual_reward": float(actual_reward_tensor[selected_idx].item()),
            "baseline_actual_reward": float(actual_reward_tensor[baseline_idx].item()),
            "selected_minus_baseline_predicted": selected_minus_baseline_predicted_total,
            "selected_minus_baseline_predicted_ic": selected_minus_baseline_predicted_ic,
            "selected_minus_baseline_predicted_tc": selected_minus_baseline_predicted_tc,
            "selected_minus_baseline_predicted_discounted_tc": (
                selected_minus_baseline_predicted_discounted_tc
            ),
            "selected_minus_baseline_actual_cost": selected_minus_baseline_actual_cost,
            "selected_minus_baseline_actual_reward": float(
                actual_reward_tensor[selected_idx].item() - actual_reward_tensor[baseline_idx].item()
            ),
            "selected_terminal_state_mae": terminal_state_maes[selected_idx],
            "baseline_terminal_state_mae": terminal_state_maes[baseline_idx],
            "candidate_debug": candidate_debug,
        }

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
