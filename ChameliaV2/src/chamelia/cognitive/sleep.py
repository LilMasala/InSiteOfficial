"""Sleep-phase skill discovery and consolidation pipeline."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
import math
import random
import threading
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.cognitive.clustering import DomainIndex
from src.chamelia.cognitive.latent_action import LatentActionEncoder, LatentSkillCandidate
from src.chamelia.cognitive.procedural import ProceduralMemory, SkillRecord

try:
    import stitch_core
except ImportError:  # pragma: no cover - optional dependency at import time.
    stitch_core = None

try:
    from botorch.acquisition.analytic import ExpectedImprovement
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:  # pragma: no cover - optional dependency at import time.
    ExpectedImprovement = None
    LogExpectedImprovement = None
    SingleTaskGP = None
    fit_gpytorch_mll = None
    optimize_acqf = None
    ExactMarginalLogLikelihood = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency at import time.
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass(frozen=True)
class DecomposedSegment:
    """LOVE-style segment candidate."""

    symbolic_codes: torch.Tensor
    prototype_path: torch.Tensor
    source_episodes: tuple[int, ...]
    score: float
    symbolic_program: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SkillPromotion:
    """Promoted skill plus bookkeeping."""

    record: SkillRecord
    evaluation_score: float
    source: str


@dataclass(frozen=True)
class ProceduralRerankerExample:
    """Sleep-scored shortlist used to warm up the procedural reranker."""

    query_latent: torch.Tensor
    skill_embeddings: torch.Tensor
    evaluation_scores: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class SleepCycleReport:
    """Summary of one sleep cycle."""

    promotions: tuple[SkillPromotion, ...]
    pruned_episodes: tuple[int, ...]
    decomposed_segments: int
    dream_candidates: int
    rsd_candidates: int
    bodegen_candidates: int
    elapsed_s: float


class LOVEDecomposer:
    """Partition traces with a compression-style objective inspired by LOVE."""

    @dataclass
    class _FragmentStats:
        count: int = 0
        total_quality: float = 0.0
        total_utility: float = 0.0
        episode_ids: set[int] = field(default_factory=set)
        paths: list[torch.Tensor] = field(default_factory=list)

    @dataclass(frozen=True)
    class _TraceExample:
        record_id: int
        action_path: torch.Tensor
        program: tuple[int, ...]
        quality: float

    def __init__(
        self,
        *,
        beta: float = 0.25,
        min_frequency: int = 2,
        max_segment_length: int = 3,
    ) -> None:
        self.beta = beta
        self.min_frequency = min_frequency
        self.max_segment_length = max_segment_length

    def _path_signature(self, path: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(torch.round(path.detach().cpu() * 4.0), min=-8, max=8).to(torch.int64)
        weights = torch.arange(1, clipped.shape[-1] + 1, dtype=torch.int64)
        return (clipped * weights.unsqueeze(0)).sum(dim=-1)

    def _episode_quality(self, record: Any) -> float:
        realized = getattr(record, "ic_realized", None)
        if realized is not None:
            return -float(realized)
        return -float(getattr(record, "ic_at_decision", 0.0) + getattr(record, "tc_predicted", 0.0))

    def _is_solved(self, record: Any) -> bool:
        realized = getattr(record, "ic_realized", None)
        return realized is not None and float(realized) <= 0.0

    def _trace_corpus(self, records: list[Any]) -> list[_TraceExample]:
        usable = [
            record
            for record in records
            if getattr(record, "selected_path", None) is not None and self._is_solved(record)
        ]
        if not usable:
            usable = [
                record
                for record in records
                if getattr(record, "selected_path", None) is not None
            ]
        corpus: list[LOVEDecomposer._TraceExample] = []
        for record in usable:
            path = getattr(record, "selected_path").detach().float()
            program = tuple(int(value) for value in self._path_signature(path).tolist())
            if not program:
                continue
            corpus.append(
                LOVEDecomposer._TraceExample(
                    record_id=int(getattr(record, "record_id", 0)),
                    action_path=path,
                    program=program,
                    quality=self._episode_quality(record),
                )
            )
        return corpus

    def _candidate_library(
        self,
        corpus: list[_TraceExample],
    ) -> dict[tuple[int, ...], _FragmentStats]:
        library: dict[tuple[int, ...], LOVEDecomposer._FragmentStats] = {}
        for trace in corpus:
            max_length = min(self.max_segment_length, len(trace.program))
            for length in range(1, max_length + 1):
                for start in range(len(trace.program) - length + 1):
                    fragment = trace.program[start : start + length]
                    stats = library.setdefault(fragment, LOVEDecomposer._FragmentStats())
                    stats.count += 1
                    stats.total_quality += trace.quality
                    stats.episode_ids.add(trace.record_id)
                    stats.paths.append(trace.action_path[start : start + length].clone())
        return library

    def _segment_utility(
        self,
        fragment: tuple[int, ...],
        stats: _FragmentStats,
        *,
        num_traces: int,
    ) -> float:
        avg_quality = stats.total_quality / max(stats.count, 1)
        avg_utility = stats.total_utility / max(stats.count, 1)
        cross_episode_coverage = len(stats.episode_ids) / max(1.0, float(num_traces))
        frequency_surrogate = math.log1p(stats.count)
        compression_gain = max(0.0, float(len(fragment) - 1))
        description_length = float(len(fragment))
        predictive_usefulness = max(avg_quality, avg_utility)
        return (
            (0.40 * avg_quality)
            + (0.25 * predictive_usefulness)
            + (0.20 * cross_episode_coverage)
            + (0.10 * frequency_surrogate)
            + (0.05 * compression_gain)
            - (self.beta * description_length)
        )

    def _partition_trace(
        self,
        trace: _TraceExample,
        library: dict[tuple[int, ...], _FragmentStats],
        *,
        num_traces: int,
    ) -> list[tuple[int, int, tuple[int, ...], float]]:
        program = trace.program
        n = len(program)
        dp = [-float("inf")] * (n + 1)
        choice: list[tuple[int, int, tuple[int, ...], float] | None] = [None] * (n + 1)
        dp[0] = 0.0
        for end in range(1, n + 1):
            max_length = min(self.max_segment_length, end)
            for length in range(1, max_length + 1):
                start = end - length
                fragment = program[start:end]
                stats = library.get(fragment)
                if stats is None:
                    continue
                utility = self._segment_utility(fragment, stats, num_traces=num_traces)
                candidate_score = dp[start] + utility
                if candidate_score > dp[end]:
                    dp[end] = candidate_score
                    choice[end] = (start, end, fragment, utility)
        partition: list[tuple[int, int, tuple[int, ...], float]] = []
        cursor = n
        while cursor > 0:
            selected = choice[cursor]
            if selected is None:
                start = cursor - 1
                fragment = (program[start],)
                stats = library[fragment]
                selected = (
                    start,
                    cursor,
                    fragment,
                    self._segment_utility(fragment, stats, num_traces=num_traces),
                )
            partition.append(selected)
            cursor = selected[0]
        partition.reverse()
        return partition

    def decompose(self, records: list[Any]) -> list[DecomposedSegment]:
        corpus = self._trace_corpus(records)
        if not corpus:
            return []
        library = self._candidate_library(corpus)
        aggregated: dict[tuple[int, ...], LOVEDecomposer._FragmentStats] = {}
        for trace in corpus:
            for start, end, fragment, utility in self._partition_trace(
                trace,
                library,
                num_traces=len(corpus),
            ):
                stats = aggregated.setdefault(fragment, LOVEDecomposer._FragmentStats())
                stats.count += 1
                stats.total_quality += trace.quality
                stats.total_utility += utility
                stats.episode_ids.add(trace.record_id)
                stats.paths.append(trace.action_path[start:end].clone())
        promoted: list[DecomposedSegment] = []
        for fragment, stats in aggregated.items():
            if len(fragment) <= 1 or stats.count < self.min_frequency:
                continue
            score = stats.total_utility / max(stats.count, 1)
            if score <= 0.0:
                continue
            prototype = torch.stack(stats.paths, dim=0).mean(dim=0)
            promoted.append(
                DecomposedSegment(
                    symbolic_codes=torch.tensor(fragment, dtype=torch.long),
                    prototype_path=prototype,
                    source_episodes=tuple(sorted(stats.episode_ids)),
                    score=float(score),
                )
            )
        promoted.sort(key=lambda item: item.score, reverse=True)
        if promoted:
            return promoted
        fallback = max(aggregated.items(), key=lambda item: item[1].total_utility, default=None)
        if fallback is None:
            return []
        fragment, stats = fallback
        prototype = torch.stack(stats.paths, dim=0).mean(dim=0)
        return [
            DecomposedSegment(
                symbolic_codes=torch.tensor(fragment, dtype=torch.long),
                prototype_path=prototype,
                source_episodes=tuple(sorted(stats.episode_ids)),
                score=float(stats.total_utility / max(stats.count, 1)),
            )
        ]


class StitchProgramCodec:
    """Translate discrete segment signatures into Stitch-compatible programs."""

    @staticmethod
    def encode(symbolic_codes: torch.Tensor) -> str:
        tokens = [f"tok_{int(value)}" for value in symbolic_codes.reshape(-1).tolist()]
        program = "end"
        for token in reversed(tokens):
            program = f"(seq {token} {program})"
        return program


class StitchCompressor:
    """Actual Stitch-backed abstraction discovery over symbolic segment programs."""

    def __init__(
        self,
        *,
        iterations: int = 8,
        max_arity: int = 2,
        threads: int = 1,
    ) -> None:
        self.iterations = iterations
        self.max_arity = max_arity
        self.threads = threads

    def _fallback_dedup(self, segments: list[DecomposedSegment]) -> list[DecomposedSegment]:
        deduped: dict[tuple[int, ...], DecomposedSegment] = {}
        for segment in segments:
            key = tuple(int(value) for value in segment.symbolic_codes.reshape(-1).tolist())
            current = deduped.get(key)
            if current is None or segment.score > current.score:
                deduped[key] = segment
        return list(deduped.values())

    def compress(self, segments: list[DecomposedSegment]) -> list[DecomposedSegment]:
        if not segments:
            return []
        deduped = self._fallback_dedup(segments)
        if stitch_core is None:
            return deduped
        programs = [StitchProgramCodec.encode(segment.symbolic_codes) for segment in deduped]
        result = stitch_core.compress(
            programs=programs,
            iterations=self.iterations,
            max_arity=self.max_arity,
            threads=self.threads,
            silent=True,
        )
        if not result.abstractions:
            return deduped
        promoted: list[DecomposedSegment] = []
        for abstraction in result.abstractions:
            member_indices = [
                index
                for index, rewritten in enumerate(result.rewritten)
                if abstraction.name in rewritten
            ]
            if not member_indices:
                continue
            member_segments = [deduped[index] for index in member_indices]
            max_length = max(segment.prototype_path.shape[0] for segment in member_segments)
            padded_paths = []
            masks = []
            for segment in member_segments:
                path = segment.prototype_path
                if path.shape[0] < max_length:
                    pad = torch.zeros(
                        max_length - path.shape[0],
                        path.shape[1],
                        dtype=path.dtype,
                        device=path.device,
                    )
                    padded_paths.append(torch.cat([path, pad], dim=0))
                    mask = torch.cat(
                        [
                            torch.ones(path.shape[0], dtype=path.dtype, device=path.device),
                            torch.zeros(max_length - path.shape[0], dtype=path.dtype, device=path.device),
                        ],
                        dim=0,
                    )
                else:
                    padded_paths.append(path)
                    mask = torch.ones(max_length, dtype=path.dtype, device=path.device)
                masks.append(mask)
            stacked_paths = torch.stack(padded_paths, dim=0)
            stacked_masks = torch.stack(masks, dim=0).unsqueeze(-1).clamp_min(1.0e-6)
            prototype = (stacked_paths * stacked_masks).sum(dim=0) / stacked_masks.sum(dim=0)
            source_episodes = tuple(
                sorted(
                    {
                        episode_id
                        for segment in member_segments
                        for episode_id in segment.source_episodes
                    }
                )
            )
            promoted.append(
                DecomposedSegment(
                    symbolic_codes=member_segments[0].symbolic_codes,
                    prototype_path=prototype,
                    source_episodes=source_episodes,
                    score=float(sum(segment.score for segment in member_segments)),
                    symbolic_program=(abstraction.name, abstraction.body),
                )
            )
        if promoted:
            promoted.sort(key=lambda item: item.score, reverse=True)
            return promoted
        return deduped


class BODEGenOptimizer:
    """Bayesian optimisation over latent prompts that decode to executable skills."""

    def __init__(
        self,
        *,
        latent_prompt_dim: int | None = None,
        latent_bound: float = 2.0,
        num_initial_points: int = 8,
        num_iterations: int = 8,
        lipschitz_weight: float = 0.05,
        max_lipschitz_ratio: float = 1.0,
    ) -> None:
        self.latent_prompt_dim = latent_prompt_dim
        self.latent_bound = float(latent_bound)
        self.num_initial_points = num_initial_points
        self.num_iterations = num_iterations
        self.lipschitz_weight = float(lipschitz_weight)
        self.max_lipschitz_ratio = float(max_lipschitz_ratio)

    def optimize(
        self,
        objective: Any,
        *,
        latent_action_encoder: LatentActionEncoder,
        path_length: int,
        symbolic_codes: torch.Tensor | None = None,
        seed_prompts: list[torch.Tensor] | None = None,
        seed_paths: list[torch.Tensor] | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if (
            SingleTaskGP is None
            or ExpectedImprovement is None
            or fit_gpytorch_mll is None
            or optimize_acqf is None
            or ExactMarginalLogLikelihood is None
        ):
            raise RuntimeError("botorch and gpytorch are required for BODE-GEN optimisation.")
        requested_device = torch.device(device)
        optimization_device = (
            torch.device("cpu") if requested_device.type == "mps" else requested_device
        )
        search_dim = int(self.latent_prompt_dim or latent_action_encoder.latent_prompt_dim)
        bounds = torch.tensor(
            [[0.0] * search_dim, [1.0] * search_dim],
            dtype=torch.double,
            device=optimization_device,
        )
        low = torch.full(
            (search_dim,),
            -self.latent_bound,
            dtype=torch.double,
            device=optimization_device,
        )
        high = torch.full(
            (search_dim,),
            self.latent_bound,
            dtype=torch.double,
            device=optimization_device,
        )

        def _actualize(unit_candidate: torch.Tensor) -> torch.Tensor:
            return low + ((high - low) * unit_candidate)

        train_x = []
        if seed_prompts:
            for prompt in seed_prompts[: self.num_initial_points]:
                actual = (
                    prompt.detach()
                    .reshape(-1)
                    .float()
                    .cpu()
                    .to(device=optimization_device, dtype=torch.double)
                )
                if actual.numel() != search_dim:
                    continue
                normalized = (actual - low) / (high - low).clamp_min(1.0e-6)
                train_x.append(normalized.clamp(0.0, 1.0))
        if seed_paths:
            for path in seed_paths[: self.num_initial_points]:
                inferred = latent_action_encoder.infer_prompt(
                    path.detach().to(requested_device),
                    symbolic_codes=(
                        symbolic_codes.to(device=requested_device)
                        if symbolic_codes is not None
                        else None
                    ),
                ).squeeze(0)
                actual = (
                    inferred.detach()
                    .reshape(-1)
                    .float()
                    .cpu()
                    .to(device=optimization_device, dtype=torch.double)
                )
                normalized = (actual - low) / (high - low).clamp_min(1.0e-6)
                train_x.append(normalized.clamp(0.0, 1.0))
        sobol = torch.quasirandom.SobolEngine(search_dim, scramble=True)
        while len(train_x) < self.num_initial_points:
            sample = sobol.draw(1, dtype=torch.double).squeeze(0).to(optimization_device)
            train_x.append(sample)
        train_x_tensor = torch.stack(train_x, dim=0)
        observed_prompts: list[torch.Tensor] = []
        observed_embeddings: list[torch.Tensor] = []

        def _evaluate_prompt(prompt_vector: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
            prompt_batch = prompt_vector.to(device=requested_device, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                action_path = latent_action_encoder.decode_prompt(
                    prompt_batch,
                    path_length=path_length,
                ).squeeze(0)
                embedding = latent_action_encoder(
                    action_path,
                    symbolic_codes=(
                        symbolic_codes.to(device=requested_device)
                        if symbolic_codes is not None
                        else None
                    ),
                    prompt_vector=prompt_batch,
                    path_length=path_length,
                ).squeeze(0)
            observed_prompts.append(prompt_batch.detach())
            observed_embeddings.append(embedding.detach().unsqueeze(0))
            regularized_score = float(objective(prompt_batch.squeeze(0), action_path, embedding))
            if len(observed_prompts) > 1:
                prompt_bank = torch.cat(observed_prompts, dim=0)
                embedding_bank = torch.cat(observed_embeddings, dim=0)
                lipschitz_penalty = latent_action_encoder.lipschitz_penalty(
                    prompt_bank,
                    embedding_bank,
                    max_ratio=self.max_lipschitz_ratio,
                )
                regularized_score -= self.lipschitz_weight * float(lipschitz_penalty.item())
            return regularized_score, action_path.detach().cpu().float(), embedding.detach().cpu().float()

        initial_scores: list[float] = []
        initial_paths: list[torch.Tensor] = []
        initial_embeddings: list[torch.Tensor] = []
        for candidate in train_x_tensor:
            score, path, embedding = _evaluate_prompt(_actualize(candidate))
            initial_scores.append(score)
            initial_paths.append(path)
            initial_embeddings.append(embedding)
        train_y_tensor = torch.tensor(
            [[score] for score in initial_scores],
            dtype=torch.double,
            device=optimization_device,
        )
        best_index = int(train_y_tensor.argmax().item())
        best_x = train_x_tensor[best_index].clone()
        best_y = float(train_y_tensor[best_index].item())
        best_path = initial_paths[best_index].clone()
        best_embedding = initial_embeddings[best_index].clone()
        for _ in range(self.num_iterations):
            model = SingleTaskGP(train_x_tensor, train_y_tensor)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            if LogExpectedImprovement is not None:
                acquisition = LogExpectedImprovement(model=model, best_f=train_y_tensor.max())
            else:
                acquisition = ExpectedImprovement(model=model, best_f=train_y_tensor.max())
            candidate_x, _ = optimize_acqf(
                acquisition,
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=128,
            )
            candidate_x = candidate_x.squeeze(0)
            candidate_y, candidate_path, candidate_embedding = _evaluate_prompt(_actualize(candidate_x))
            train_x_tensor = torch.cat([train_x_tensor, candidate_x.unsqueeze(0)], dim=0)
            train_y_tensor = torch.cat(
                [
                    train_y_tensor,
                    torch.tensor([[candidate_y]], dtype=torch.double, device=optimization_device),
                ],
                dim=0,
            )
            if candidate_y > best_y:
                best_x = candidate_x.clone()
                best_y = candidate_y
                best_path = candidate_path.clone()
                best_embedding = candidate_embedding.clone()
        return _actualize(best_x).detach().cpu().float(), best_path, best_embedding, best_y


class DreamDecompiler:
    """DreamDecompiler-style chunking over latent MCTS frontiers."""

    @dataclass(frozen=True)
    class _FrontierEntry:
        program: tuple[int, ...]
        action_path: torch.Tensor
        probability: float
        log_probability: float

    @dataclass(frozen=True)
    class _TaskFrontier:
        record_id: int
        state_key: torch.Tensor
        actor_consistency: float
        entries: tuple["DreamDecompiler._FrontierEntry", ...]

    def __init__(
        self,
        *,
        max_segment_length: int = 4,
        top_k_candidates: int = 4,
        chunk_weighting: str = "raw",
        num_consolidate: int = 6,
        chunk_threshold: float = 1.0e-3,
        neighborhood_k: int = 4,
    ) -> None:
        if chunk_weighting not in {"raw", "prop", "uniform"}:
            raise ValueError("chunk_weighting must be one of raw, prop, or uniform.")
        self.max_segment_length = max_segment_length
        self.top_k_candidates = top_k_candidates
        self.chunk_weighting = chunk_weighting
        self.num_consolidate = num_consolidate
        self.chunk_threshold = chunk_threshold
        self.neighborhood_k = neighborhood_k

    def _path_signature(self, path: torch.Tensor) -> tuple[int, ...]:
        clipped = torch.clamp(torch.round(path.detach().cpu() * 4.0), min=-8, max=8).to(torch.int64)
        weights = torch.arange(1, clipped.shape[-1] + 1, dtype=torch.int64)
        signature = (clipped * weights.unsqueeze(0)).sum(dim=-1)
        return tuple(int(value) for value in signature.tolist())

    def _actor_consistency(self, candidate_total: torch.Tensor) -> float:
        probs = torch.softmax(-candidate_total.detach().float(), dim=0)
        entropy = -(probs * probs.clamp_min(1.0e-6).log()).sum()
        max_entropy = math.log(max(2, candidate_total.shape[0]))
        return max(0.0, 1.0 - float(entropy.item() / max_entropy))

    def _is_unsolved(self, record: Any) -> bool:
        realized = getattr(record, "ic_realized", None)
        if realized is None:
            return True
        return float(realized) > 0.0

    def _extract_frontiers(self, records: list[Any]) -> list[_TaskFrontier]:
        frontiers: list[DreamDecompiler._TaskFrontier] = []
        for record in records:
            candidate_total = getattr(record, "candidate_total", None)
            candidate_paths = getattr(record, "candidate_paths", None)
            if candidate_total is None or candidate_paths is None or not self._is_unsolved(record):
                continue
            total = candidate_total.detach().float()
            paths = candidate_paths.detach().float()
            if total.dim() > 1:
                total = total.reshape(-1)
            if paths.dim() != 3 or total.numel() != paths.shape[0]:
                continue
            probs = torch.softmax(-total, dim=0)
            order = probs.topk(min(self.top_k_candidates, probs.shape[0])).indices
            entries: list[DreamDecompiler._FrontierEntry] = []
            for idx in order.tolist():
                probability = float(probs[idx].item())
                entries.append(
                    DreamDecompiler._FrontierEntry(
                        program=self._path_signature(paths[idx]),
                        action_path=paths[idx],
                        probability=probability,
                        log_probability=float(math.log(max(probability, 1.0e-8))),
                    )
                )
            if not entries:
                continue
            frontiers.append(
                DreamDecompiler._TaskFrontier(
                    record_id=int(getattr(record, "record_id", 0)),
                    state_key=getattr(record, "key").detach().float(),
                    actor_consistency=self._actor_consistency(total),
                    entries=tuple(entries),
                )
            )
        return frontiers

    def _state_neighborhood_consistency(
        self,
        frontier: _TaskFrontier,
        all_frontiers: list[_TaskFrontier],
    ) -> float:
        if len(all_frontiers) <= 1:
            return frontier.actor_consistency
        peers = [item for item in all_frontiers if item.record_id != frontier.record_id]
        if not peers:
            return frontier.actor_consistency
        states = torch.stack([peer.state_key for peer in peers], dim=0)
        similarities = F.cosine_similarity(
            frontier.state_key.unsqueeze(0),
            states,
            dim=-1,
        )
        topk = similarities.topk(min(self.neighborhood_k, similarities.numel()))
        weights = torch.softmax(topk.values, dim=0)
        peer_consistency = torch.tensor(
            [peers[int(index)].actor_consistency for index in topk.indices.tolist()],
            dtype=torch.float32,
        )
        blended = float((weights * peer_consistency).sum().item())
        return 0.5 * frontier.actor_consistency + 0.5 * blended

    def _contains_fragment(self, program: tuple[int, ...], fragment: tuple[int, ...]) -> list[int]:
        matches: list[int] = []
        frag_len = len(fragment)
        if frag_len == 0 or frag_len > len(program):
            return matches
        for start in range(len(program) - frag_len + 1):
            if program[start : start + frag_len] == fragment:
                matches.append(start)
        return matches

    def _candidate_fragments(self, frontiers: list[_TaskFrontier]) -> list[tuple[int, ...]]:
        counts: dict[tuple[int, ...], int] = {}
        for frontier in frontiers:
            seen: set[tuple[int, ...]] = set()
            for entry in frontier.entries:
                program = entry.program
                for length in range(1, min(self.max_segment_length, len(program) - 1) + 1):
                    for start in range(len(program) - length + 1):
                        fragment = program[start : start + length]
                        if fragment in seen:
                            continue
                        counts[fragment] = counts.get(fragment, 0) + 1
                        seen.add(fragment)
        fragments = [fragment for fragment, count in counts.items() if count >= 2]
        fragments.sort(key=lambda fragment: (len(fragment), counts[fragment]), reverse=True)
        return fragments

    def _compression_gain(self, program: tuple[int, ...], fragment: tuple[int, ...]) -> float:
        matches = self._contains_fragment(program, fragment)
        if not matches:
            return 0.0
        return (len(matches) * len(fragment)) / max(1, len(program))

    def _chunk_given_task_probability(
        self,
        fragment: tuple[int, ...],
        frontier: _TaskFrontier,
        neighborhood_consistency: float,
    ) -> float:
        if not frontier.entries:
            return 0.0
        total_probability = sum(entry.probability for entry in frontier.entries)
        chunk_probability = 0.0
        for entry in frontier.entries:
            compression_gain = self._compression_gain(entry.program, fragment)
            if compression_gain <= 0.0:
                continue
            if self.chunk_weighting == "uniform":
                weight = 1.0 / len(frontier.entries)
            elif self.chunk_weighting == "prop":
                weight = entry.probability / max(total_probability, 1.0e-8)
            else:
                weight = entry.probability
            chunk_probability += weight * compression_gain
        return neighborhood_consistency * chunk_probability

    def _fragment_chunk_probability(
        self,
        fragment: tuple[int, ...],
        frontiers: list[_TaskFrontier],
        neighborhood_consistency: dict[int, float],
    ) -> float:
        numerator = 0.0
        denominator = 0.0
        for frontier in frontiers:
            fragment_prior = max(
                (
                    entry.probability
                    for entry in frontier.entries
                    if self._contains_fragment(entry.program, fragment)
                ),
                default=0.0,
            )
            if fragment_prior <= 0.0:
                continue
            chunk_given_task = self._chunk_given_task_probability(
                fragment,
                frontier,
                neighborhood_consistency.get(frontier.record_id, frontier.actor_consistency),
            )
            numerator += fragment_prior * chunk_given_task
            denominator += fragment_prior
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator

    def extract(self, records: list[Any]) -> list[DecomposedSegment]:
        frontiers = self._extract_frontiers(records)
        if not frontiers:
            return []
        neighborhood_consistency = {
            frontier.record_id: self._state_neighborhood_consistency(frontier, frontiers)
            for frontier in frontiers
        }
        promoted: list[DecomposedSegment] = []
        for fragment in self._candidate_fragments(frontiers):
            chunk_probability = self._fragment_chunk_probability(
                fragment,
                frontiers,
                neighborhood_consistency,
            )
            if chunk_probability < self.chunk_threshold:
                continue
            matched_paths: list[torch.Tensor] = []
            source_episodes: set[int] = set()
            for frontier in frontiers:
                for entry in frontier.entries:
                    for start in self._contains_fragment(entry.program, fragment):
                        matched_paths.append(entry.action_path[start : start + len(fragment)].clone())
                        source_episodes.add(frontier.record_id)
            if not matched_paths:
                continue
            prototype = torch.stack(matched_paths, dim=0).mean(dim=0)
            promoted.append(
                DecomposedSegment(
                    symbolic_codes=torch.tensor(fragment, dtype=torch.long),
                    prototype_path=prototype,
                    source_episodes=tuple(sorted(source_episodes)),
                    score=float(chunk_probability),
                )
            )
        promoted.sort(key=lambda item: item.score, reverse=True)
        return promoted[: self.num_consolidate]


class ChoreographerEvaluator:
    """Imagination-first skill evaluation loop."""

    def evaluate(
        self,
        *,
        world_model: Any,
        cost_module: Any,
        reference_records: list[Any],
        candidates: list[LatentSkillCandidate],
    ) -> list[tuple[LatentSkillCandidate, float]]:
        evaluated: list[tuple[LatentSkillCandidate, float]] = []
        if not reference_records:
            return evaluated
        world_device = next(iter(world_model.parameters()), None)
        world_device = world_device.device if world_device is not None else torch.device("cpu")
        for candidate in candidates:
            sample_scores: list[float] = []
            for record in reference_records[: min(8, len(reference_records))]:
                if getattr(record, "ctx_tokens", None) is None:
                    continue
                z = record.key.detach().float().unsqueeze(0).to(world_device)
                ctx_tokens = record.ctx_tokens.detach().float().unsqueeze(0).to(world_device)
                actions = candidate.action_path.detach().float().unsqueeze(0).unsqueeze(0).to(world_device)
                rollout = world_model(
                    z=z,
                    actions=actions,
                    ctx_tokens=ctx_tokens,
                    candidate_postures=None,
                    reasoning_states=None,
                    horizon=min(candidate.action_path.shape[0], world_model.max_horizon),
                )
                score = cost_module.score_candidates(
                    z=z,
                    actions=actions,
                    ctx_tokens=ctx_tokens,
                    domain_state={},
                    future_z=rollout["terminal_latents"],
                    future_trajectory=rollout["trajectory"],
                )["total"][0, 0]
                sample_scores.append(-float(score.item()) * candidate.source_weight)
            if sample_scores:
                evaluated.append((candidate, sum(sample_scores) / len(sample_scores)))
        evaluated.sort(key=lambda item: item[1], reverse=True)
        return evaluated


@dataclass(frozen=True)
class _GeneratorSnapshot:
    """One frozen RSD generator distribution snapshot."""

    mean: torch.Tensor
    log_std: torch.Tensor


class _DiagonalGaussianSkillGenerator(nn.Module):
    """Diagonal-Gaussian policy over flattened latent action paths."""

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        state_batch: torch.Tensor,
    ) -> tuple[torch.distributions.Independent, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state_batch)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(min=-5.0, max=2.0)
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mean, log_std.exp()),
            1,
        )
        return distribution, mean, log_std


class RSDAdversary:
    """Regret-aware adversarial skill discovery operating in Global-D intent space.

    Architecture
    ------------
    The generator policy is a diagonal-Gaussian MLP that maps D-dimensional
    latent states to D-dimensional *intent* vectors — one intent per episode
    in the eligible batch.  A lightweight ``intent_decoder`` (``nn.Linear``)
    then projects each intent to ``path_length × action_dim`` for world-model
    evaluation.

    Operating in D-space (rather than raw action space) has two benefits:

    1.  The novelty window stores D-dimensional snapshots regardless of A,
        eliminating the cross-domain dimension-mismatch crash on CartPole.
    2.  Skill novelty is measured in the same space as LatentMemory keys,
        so cosine distance in the window is semantically meaningful.
    """

    def __init__(
        self,
        *,
        generator_steps: int = 24,
        learning_rate: float = 3.0e-2,
        window_size: int = 5,
        window_weight: float = 0.1,
        confidence_weight: float = 0.2,
        regret_floor: float = 0.05,
        distinct_threshold: float = 0.1,
    ) -> None:
        self.generator_steps = generator_steps
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.window_weight = window_weight
        self.confidence_weight = confidence_weight
        self.regret_floor = regret_floor
        self.distinct_threshold = distinct_threshold
        # Generator operates in D-space; recreated only when D changes.
        self.generator: _DiagonalGaussianSkillGenerator | None = None
        self.generator_optim: torch.optim.Optimizer | None = None
        self._latent_dim: int | None = None
        # Intent decoder maps D → path_length * action_dim; recreated when
        # A or H changes.  Not part of any nn.Module tree — managed here.
        self.intent_decoder: nn.Linear | None = None
        self.path_length: int | None = None
        self.action_dim: int | None = None
        # Novelty window: all snapshots are D-dimensional intent vectors.
        self.window: list[_GeneratorSnapshot] = []
        self.previous_world_model: Any | None = None
        self.previous_cost_module: Any | None = None

    def _module_device(self, module: Any) -> torch.device:
        parameter = next(iter(module.parameters()), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device

    def _ensure_generator(
        self,
        *,
        state_dim: int,
        latent_dim: int,
        path_length: int,
        action_dim: int,
        device: torch.device,
    ) -> None:
        """Ensure the D-space generator and intent decoder are ready.

        The generator and decoder are treated as a single unit.  The generator
        learns to produce D-dimensional intent vectors that are meaningful only
        *relative to a specific decoder*.  If the decoder is replaced (because
        A or H changed), the generator's learned output distribution no longer
        maps to valid actions — so both must be reset together.

        Concretely, we reset the whole unit whenever any of the following
        change: ``latent_dim`` (D), ``path_length`` (H), ``action_dim`` (A).
        The novelty window is also cleared on any reset because its snapshots
        were accumulated under the old generator.
        """
        action_space_changed = (
            self.path_length != path_length or self.action_dim != action_dim
        )
        latent_space_changed = (self._latent_dim != latent_dim)
        needs_reset = (
            self.generator is None
            or action_space_changed
            or latent_space_changed
        )

        if needs_reset:
            self.generator = _DiagonalGaussianSkillGenerator(
                state_dim=state_dim,
                output_dim=latent_dim,  # always D — domain-agnostic
            ).to(device)
            self.generator_optim = torch.optim.Adam(
                self.generator.parameters(), lr=self.learning_rate
            )
            self.intent_decoder = nn.Linear(latent_dim, path_length * action_dim).to(device)
            self._latent_dim = latent_dim
            self.path_length = path_length
            self.action_dim = action_dim
            # Window snapshots are tied to the old generator's output space;
            # discard them to prevent stale KL comparisons.
            self.window.clear()
        else:
            self.generator.to(device)
            self.intent_decoder.to(device)

    def _decode_intents(self, intents: torch.Tensor) -> torch.Tensor:
        """Decode D-dimensional intent vectors into action paths.

        Args:
            intents: Tensor of shape [B, D].

        Returns:
            Tensor of shape [B, path_length, action_dim].
        """
        if self.intent_decoder is None or self.path_length is None or self.action_dim is None:
            raise RuntimeError("RSD generator has not been initialized.")
        flat = self.intent_decoder(intents)
        return flat.view(flat.shape[0], self.path_length, self.action_dim)

    def _evaluate_paths(
        self,
        *,
        world_model: Any,
        cost_module: Any,
        records: list[Any],
        paths: torch.Tensor,
    ) -> torch.Tensor:
        device = self._module_device(world_model)
        z = torch.stack([record.key.detach().float() for record in records], dim=0).to(device)
        ctx_tokens = torch.stack(
            [record.ctx_tokens.detach().float() for record in records],
            dim=0,
        ).to(device)
        candidate_paths = paths.to(device).unsqueeze(1)
        rollout = world_model(
            z=z,
            actions=candidate_paths,
            ctx_tokens=ctx_tokens,
            candidate_postures=None,
            reasoning_states=None,
            horizon=min(paths.shape[1], world_model.max_horizon),
        )
        total_cost = cost_module.score_candidates(
            z=z,
            actions=candidate_paths,
            ctx_tokens=ctx_tokens,
            domain_state={},
            future_z=rollout["terminal_latents"],
            future_trajectory=rollout["trajectory"],
        )["total"][:, 0]
        return (-total_cost).to(paths.device)

    def _window_novelty(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """KL-based novelty against the D-space intent window."""
        if not self.window:
            return torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype)
        current_var = (2.0 * log_std).exp()
        novelty_scores = []
        for snapshot in self.window:
            snapshot_mean = snapshot.mean.to(mean.device).unsqueeze(0)
            snapshot_log_std = snapshot.log_std.to(mean.device).unsqueeze(0)
            snapshot_var = (2.0 * snapshot_log_std).exp()
            kl = 0.5 * (
                ((current_var + (mean - snapshot_mean).pow(2)) / snapshot_var.clamp_min(1.0e-6))
                - 1.0
                + (2.0 * snapshot_log_std)
                - (2.0 * log_std)
            ).sum(dim=-1)
            novelty_scores.append(kl)
        return torch.stack(novelty_scores, dim=0).min(dim=0).values

    def _confidence_penalty(
        self,
        *,
        paths: torch.Tensor,
        records: list[Any],
        latent_action_encoder: LatentActionEncoder,
        procedural_memory: ProceduralMemory,
    ) -> torch.Tensor:
        penalties = []
        for path, record in zip(paths, records, strict=False):
            target_delta = (
                record.outcome_key.detach().float() - record.key.detach().float()
                if getattr(record, "outcome_key", None) is not None
                else None
            )
            embedding = latent_action_encoder.encode_candidate(
                LatentSkillCandidate(
                    action_path=path,
                    symbolic_codes=None,
                    target_delta=target_delta,
                    source_weight=1.0,
                    source_episodes=(int(getattr(record, "record_id", 0)),),
                )
            ).squeeze(0)
            retrieved = procedural_memory.retrieve(embedding.detach().cpu(), k=1)
            penalties.append(retrieved[0].similarity if retrieved else 0.0)
        return torch.tensor(
            penalties,
            dtype=paths.dtype,
            device=paths.device,
        )

    @staticmethod
    def _normalize_regret(regret: torch.Tensor) -> torch.Tensor:
        """Safely normalize regret; returns zero tensor for collapsed variance."""
        mean = regret.mean()
        std = regret.std(unbiased=False)
        if std.item() < 1.0e-8:
            return torch.zeros_like(regret)
        return (regret - mean) / std

    def _update_window(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> None:
        snapshot = _GeneratorSnapshot(
            mean=mean.detach().mean(dim=0).cpu(),
            log_std=log_std.detach().mean(dim=0).cpu(),
        )
        if self.window:
            min_distance = min(
                float((snapshot.mean - existing.mean).norm().item())
                for existing in self.window
            )
            if min_distance < self.distinct_threshold:
                return
        self.window.append(snapshot)
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def _snapshot_evaluator(self, world_model: Any, cost_module: Any) -> None:
        self.previous_world_model = copy.deepcopy(world_model).eval()
        self.previous_cost_module = copy.deepcopy(cost_module).eval()

    def propose(
        self,
        *,
        records: list[Any],
        procedural_memory: ProceduralMemory,
        latent_action_encoder: LatentActionEncoder,
        world_model: Any,
        cost_module: Any,
    ) -> list[LatentSkillCandidate]:
        eligible = [
            record
            for record in records
            if getattr(record, "ctx_tokens", None) is not None
            and getattr(record, "selected_path", None) is not None
        ]
        if not eligible:
            return []
        reference = eligible[0]
        device = reference.key.device
        path_length = int(reference.selected_path.shape[0])
        action_dim = int(reference.selected_path.shape[-1])
        # D is the state key dimension; generator and window always use D.
        latent_dim = int(reference.key.shape[-1])
        self._ensure_generator(
            state_dim=latent_dim,
            latent_dim=latent_dim,
            path_length=path_length,
            action_dim=action_dim,
            device=device,
        )
        assert self.generator is not None
        assert self.generator_optim is not None
        state_batch = torch.stack(
            [record.key.detach().float() for record in eligible], dim=0
        ).to(device)
        for _ in range(self.generator_steps):
            distribution, mean, log_std = self.generator(state_batch)
            intents = distribution.rsample()          # [B, D]
            paths = self._decode_intents(intents)     # [B, H, A]
            current_value = self._evaluate_paths(
                world_model=world_model,
                cost_module=cost_module,
                records=eligible,
                paths=paths,
            )
            if self.previous_world_model is not None and self.previous_cost_module is not None:
                previous_value = self._evaluate_paths(
                    world_model=self.previous_world_model,
                    cost_module=self.previous_cost_module,
                    records=eligible,
                    paths=paths,
                )
            else:
                previous_value = torch.zeros_like(current_value)
            regret = current_value - previous_value
            normalized_regret = self._normalize_regret(regret)
            log_prob = distribution.log_prob(intents)
            window_novelty = self._window_novelty(mean, log_std)
            confidence_penalty = self._confidence_penalty(
                paths=paths,
                records=eligible,
                latent_action_encoder=latent_action_encoder,
                procedural_memory=procedural_memory,
            )
            loss = (
                (-log_prob * normalized_regret.detach()).mean()
                - self.window_weight * window_novelty.mean()
                + self.confidence_weight * confidence_penalty.mean()
            )
            self.generator_optim.zero_grad()
            loss.backward()
            self.generator_optim.step()
        distribution, mean, log_std = self.generator(state_batch)
        intents = mean                                # deterministic decode
        paths = self._decode_intents(intents)
        current_value = self._evaluate_paths(
            world_model=world_model,
            cost_module=cost_module,
            records=eligible,
            paths=paths,
        )
        if self.previous_world_model is not None and self.previous_cost_module is not None:
            previous_value = self._evaluate_paths(
                world_model=self.previous_world_model,
                cost_module=self.previous_cost_module,
                records=eligible,
                paths=paths,
            )
        else:
            previous_value = torch.zeros_like(current_value)
        confidence_penalty = self._confidence_penalty(
            paths=paths,
            records=eligible,
            latent_action_encoder=latent_action_encoder,
            procedural_memory=procedural_memory,
        )
        effective_regret = current_value - previous_value - (self.confidence_weight * confidence_penalty)
        proposals: list[LatentSkillCandidate] = []
        for idx, record in enumerate(eligible):
            score = float(effective_regret[idx].item())
            if score <= self.regret_floor:
                continue
            proposals.append(
                LatentSkillCandidate(
                    action_path=paths[idx].detach().cpu().float(),
                    symbolic_codes=None,
                    target_delta=(
                        record.outcome_key.detach().float() - record.key.detach().float()
                        if getattr(record, "outcome_key", None) is not None
                        else None
                    ),
                    source_weight=score,
                    source_episodes=(int(getattr(record, "record_id", 0)),),
                )
            )
        proposals.sort(key=lambda item: item.source_weight, reverse=True)
        self._update_window(mean, log_std)
        self._snapshot_evaluator(world_model, cost_module)
        return proposals


class GemmaAutoDocWorker:
    """Offline Gemma worker for naming and documenting promoted skills."""

    def __init__(
        self,
        *,
        model_id: str = "google/gemma-4-E2B-it",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        generator: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._generator = generator
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._generator is not None or self._model is not None:
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required for the Gemma AutoDoc worker.")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
        )

    def _prompt(self, candidate: LatentSkillCandidate, ordinal: int) -> str:
        symbolic = (
            " ".join(str(int(value)) for value in candidate.symbolic_codes.reshape(-1))
            if candidate.symbolic_codes is not None
            else "none"
        )
        return (
            "Name this latent skill and write one short docstring.\n"
            f"ordinal: {ordinal}\n"
            f"path_length: {candidate.action_path.shape[0]}\n"
            f"source_weight: {candidate.source_weight:.4f}\n"
            f"symbolic_codes: {symbolic}\n"
            "Return exactly two lines:\n"
            "NAME: <snake_case_name>\n"
            "DOC: <single sentence>\n"
        )

    def describe(self, candidate: LatentSkillCandidate, ordinal: int) -> tuple[str, str]:
        prompt = self._prompt(candidate, ordinal)
        if self._generator is not None:
            generated = str(self._generator(prompt))
        else:
            self._load()
            assert self._tokenizer is not None
            assert self._model is not None
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        name = None
        description = None
        for line in generated.splitlines():
            if line.startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            if line.startswith("DOC:"):
                description = line.split(":", 1)[1].strip()
        if name and description:
            return name, description
        raise ValueError("Gemma AutoDoc worker returned an unparseable response.")


class LILOAutoDoc:
    """Offline naming and documentation for promoted skills."""

    def __init__(self, worker: GemmaAutoDocWorker | None = None) -> None:
        self.worker = worker

    def describe(self, candidate: LatentSkillCandidate, ordinal: int) -> tuple[str, str]:
        if self.worker is not None:
            return self.worker.describe(candidate, ordinal)
        if candidate.symbolic_codes is not None:
            code_stub = "_".join(str(int(value)) for value in candidate.symbolic_codes.reshape(-1)[:6])
        else:
            code_stub = f"path{candidate.action_path.shape[0]}"
        name = f"skill_{ordinal}_{code_stub}"
        description = (
            f"Sleep-promoted latent macro-action compiled from {len(candidate.source_episodes)} episode(s) "
            f"with source weight {candidate.source_weight:.3f}."
        )
        return name, description


class SleepCoordinator:
    """Background-capable sleep coordinator."""

    def __init__(
        self,
        *,
        episodic_memory: Any,
        procedural_memory: ProceduralMemory,
        latent_action_encoder: LatentActionEncoder,
        world_model: Any,
        cost_module: Any,
        domain_index: DomainIndex | None = None,
        sleep_interval_steps: int = 32,
        autodoc_worker: GemmaAutoDocWorker | None = None,
        fasttrack_return_threshold: float = 0.5,
        fasttrack_surprise_threshold: float = 0.35,
    ) -> None:
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.latent_action_encoder = latent_action_encoder
        self.world_model = world_model
        self.cost_module = cost_module
        self.domain_index = domain_index
        self.sleep_interval_steps = sleep_interval_steps
        self.decomposer = LOVEDecomposer()
        self.compressor = StitchCompressor()
        self.dream = DreamDecompiler()
        self.choreographer = ChoreographerEvaluator()
        self.rsd = RSDAdversary()
        self.autodoc = LILOAutoDoc(worker=autodoc_worker)
        self.fasttrack_return_threshold = float(fasttrack_return_threshold)
        self.fasttrack_surprise_threshold = float(fasttrack_surprise_threshold)
        self._pending = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self.last_report: SleepCycleReport | None = None
        self._reranker_examples: deque[ProceduralRerankerExample] = deque(maxlen=512)

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        # Keep shutdown explicit so LanceDB-backed sleep jobs fully tear down
        # before interpreter finalization.
        self._worker = threading.Thread(target=self._loop, daemon=False)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._pending.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            if not self._worker.is_alive():
                self._worker = None

    def request_run(self) -> None:
        self._pending.set()

    def maybe_trigger(self, step_count: int) -> None:
        if step_count > 0 and step_count % self.sleep_interval_steps == 0:
            self.request_run()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._pending.wait(timeout=0.5)
            if not self._pending.is_set():
                continue
            self._pending.clear()
            self.last_report = self.run_cycle()

    def _gather_records(self) -> list[Any]:
        return list(self.episodic_memory.records[: self.episodic_memory.size])

    def reranker_example_count(self) -> int:
        return len(self._reranker_examples)

    def sample_reranker_examples(self, batch_size: int) -> list[ProceduralRerankerExample]:
        if not self._reranker_examples:
            return []
        count = min(int(batch_size), len(self._reranker_examples))
        return random.sample(list(self._reranker_examples), count)

    def _candidate_from_segment(self, segment: DecomposedSegment) -> LatentSkillCandidate:
        return LatentSkillCandidate(
            action_path=segment.prototype_path.detach().float(),
            symbolic_codes=segment.symbolic_codes.detach().long(),
            target_delta=None,
            source_weight=float(segment.score),
            source_episodes=segment.source_episodes,
        )

    def maybe_fasttrack_record(
        self,
        record: Any,
        *,
        utility_score: float,
        surprise_score: float,
    ) -> SkillPromotion | None:
        selected_path = getattr(record, "selected_path", None)
        if selected_path is None:
            return None
        should_promote = (
            float(utility_score) >= self.fasttrack_return_threshold
            or (
                float(surprise_score) >= self.fasttrack_surprise_threshold
                and float(utility_score) > 0.0
            )
        )
        if not should_promote:
            return None
        candidate = LatentSkillCandidate(
            action_path=selected_path.detach().float().cpu(),
            symbolic_codes=None,
            target_delta=(
                getattr(record, "outcome_key", None).detach().float().cpu() - record.key.detach().float().cpu()
                if getattr(record, "outcome_key", None) is not None
                else None
            ),
            source_weight=max(float(utility_score), float(surprise_score)),
            source_episodes=(int(getattr(record, "record_id", 0)),),
        )
        embedding = self.latent_action_encoder.encode_candidate(candidate).detach().float().cpu().squeeze(0)
        retrieved = self.procedural_memory.retrieve(
            embedding,
            k=1,
            domain_name=getattr(record, "domain_name", None),
        )
        extras = {
            "promotion_source": "online_fasttrack",
            "utility_score": float(utility_score),
            "surprise_score": float(surprise_score),
        }
        if retrieved and float(retrieved[0].similarity) >= 0.985:
            matched = self.procedural_memory.get_skill(retrieved[0].record.skill_id)
            if matched is None:
                return None
            updated_confidence = max(float(matched.confidence), 0.35 + (0.1 * min(1.0, float(utility_score))))
            self.procedural_memory.update_confidence(matched.skill_id, updated_confidence)
            self.procedural_memory.update_extras(matched.skill_id, extras)
            refreshed = self.procedural_memory.get_skill(matched.skill_id)
            if refreshed is None:
                return None
            return SkillPromotion(
                record=refreshed,
                evaluation_score=max(float(utility_score), float(surprise_score)),
                source="online_fasttrack",
            )
        name, description = self.autodoc.describe(candidate, ordinal=1)
        promoted_record = self.procedural_memory.add_skill(
            embedding=embedding,
            action_path=selected_path.detach().float().cpu(),
            source_episodes=(int(getattr(record, "record_id", 0)),),
            constraints={"utility_score": float(utility_score)},
            confidence=0.35 + (0.1 * min(1.0, float(utility_score))),
            name=name,
            description=description,
            domain_name=getattr(record, "domain_name", None),
            extras=extras,
        )
        return SkillPromotion(
            record=promoted_record,
            evaluation_score=max(float(utility_score), float(surprise_score)),
            source="online_fasttrack",
        )

    def _bodegen_candidates(
        self,
        *,
        records: list[Any],
        rsd_candidates: list[LatentSkillCandidate],
    ) -> list[LatentSkillCandidate]:
        if not records or not rsd_candidates:
            return []
        generated: list[LatentSkillCandidate] = []
        reference_record = next(
            (record for record in records if getattr(record, "ctx_tokens", None) is not None),
            None,
        )
        if reference_record is None:
            return generated
        for candidate in rsd_candidates[:4]:
            optimizer = BODEGenOptimizer(
                latent_prompt_dim=self.latent_action_encoder.latent_prompt_dim,
                num_initial_points=4,
                num_iterations=4,
            )
            world_device = self.rsd._module_device(self.world_model)

            def objective(
                prompt_vector: torch.Tensor,
                path: torch.Tensor,
                embedding: torch.Tensor,
            ) -> float:
                z = reference_record.key.detach().float().unsqueeze(0).to(world_device)
                ctx_tokens = reference_record.ctx_tokens.detach().float().unsqueeze(0).to(world_device)
                actions = path.detach().float().unsqueeze(0).unsqueeze(0).to(world_device)
                rollout = self.world_model(
                    z=z,
                    actions=actions,
                    ctx_tokens=ctx_tokens,
                    candidate_postures=None,
                    reasoning_states=None,
                    horizon=min(path.shape[0], self.world_model.max_horizon),
                )
                score = self.cost_module.score_candidates(
                    z=z,
                    actions=actions,
                    ctx_tokens=ctx_tokens,
                    domain_state={},
                    future_z=rollout["terminal_latents"],
                    future_trajectory=rollout["trajectory"],
                )["total"][0, 0]
                objective_value = -float(score.item())
                retrieved = self.procedural_memory.retrieve(embedding.detach().cpu(), k=1)
                if retrieved:
                    objective_value -= 0.1 * retrieved[0].similarity
                if candidate.target_delta is not None and getattr(reference_record, "outcome_key", None) is not None:
                    predicted_delta = rollout["terminal_latents"][0, 0] - z[0]
                    alignment = F.cosine_similarity(
                        predicted_delta.unsqueeze(0),
                        candidate.target_delta.detach().unsqueeze(0),
                        dim=-1,
                    )
                    objective_value += float(alignment.item())
                objective_value -= 0.01 * float(prompt_vector.norm().item())
                return objective_value

            try:
                optimized_prompt, optimized_path, _optimized_embedding, best_score = optimizer.optimize(
                    objective,
                    latent_action_encoder=self.latent_action_encoder,
                    path_length=candidate.action_path.shape[0],
                    symbolic_codes=candidate.symbolic_codes,
                    seed_prompts=(
                        [candidate.prompt_vector]
                        if candidate.prompt_vector is not None
                        else None
                    ),
                    seed_paths=[candidate.action_path],
                    device=reference_record.key.device,
                )
            except RuntimeError:
                continue
            generated.append(
                LatentSkillCandidate(
                    action_path=optimized_path,
                    symbolic_codes=candidate.symbolic_codes,
                    target_delta=candidate.target_delta,
                    source_weight=max(candidate.source_weight, best_score),
                    source_episodes=candidate.source_episodes,
                    prompt_vector=optimized_prompt,
                )
            )
        return generated

    def run_cycle(self) -> SleepCycleReport:
        started = time.time()
        records = self._gather_records()
        segments = self.compressor.compress(self.decomposer.decompose(records))
        dream_segments = self.dream.extract(records)
        rsd_candidates = self.rsd.propose(
            records=records,
            procedural_memory=self.procedural_memory,
            latent_action_encoder=self.latent_action_encoder,
            world_model=self.world_model,
            cost_module=self.cost_module,
        )
        bodegen_candidates = self._bodegen_candidates(
            records=records,
            rsd_candidates=rsd_candidates,
        )
        raw_candidates = [self._candidate_from_segment(segment) for segment in segments]
        raw_candidates.extend(self._candidate_from_segment(segment) for segment in dream_segments[:8])
        raw_candidates.extend(rsd_candidates[:8])
        raw_candidates.extend(bodegen_candidates[:8])
        scored_candidates: list[LatentSkillCandidate] = []
        for candidate in raw_candidates:
            scored_candidates.append(candidate)
        evaluated = self.choreographer.evaluate(
            world_model=self.world_model,
            cost_module=self.cost_module,
            reference_records=records,
            candidates=scored_candidates,
        )
        if records and len(evaluated) >= 2:
            query_latent = torch.stack(
                [record.key.detach().float().cpu() for record in records[: min(8, len(records))]],
                dim=0,
            ).mean(dim=0)
            skill_embeddings = torch.stack(
                [
                    self.latent_action_encoder.encode_candidate(candidate).detach().float().cpu().squeeze(0)
                    for candidate, _ in evaluated[: min(8, len(evaluated))]
                ],
                dim=0,
            )
            evaluation_scores = torch.tensor(
                [float(score) for _, score in evaluated[: skill_embeddings.shape[0]]],
                dtype=torch.float32,
            )
            confidence = torch.sigmoid(evaluation_scores)
            self._reranker_examples.append(
                ProceduralRerankerExample(
                    query_latent=query_latent,
                    skill_embeddings=skill_embeddings,
                    evaluation_scores=evaluation_scores,
                    confidence=confidence,
                )
            )
        promotions: list[SkillPromotion] = []
        compiled_signatures: set[tuple[float, ...]] = set()
        for ordinal, (candidate, evaluation_score) in enumerate(evaluated[:8], start=1):
            embedding = self.latent_action_encoder.encode_candidate(candidate).squeeze(0)
            name, description = self.autodoc.describe(candidate, ordinal)
            prior_match = self.procedural_memory.retrieve(
                embedding.detach().float().cpu(),
                k=1,
                domain_name=records[0].domain_name if records else None,
            )
            record = self.procedural_memory.add_skill(
                embedding=embedding,
                action_path=candidate.action_path,
                source_episodes=candidate.source_episodes,
                constraints={"source_weight": candidate.source_weight},
                confidence=max(0.1, float(torch.sigmoid(torch.tensor(evaluation_score)).item())),
                name=name,
                description=description,
                domain_name=records[0].domain_name if records else None,
                symbolic_program=(
                    tuple(str(int(value)) for value in candidate.symbolic_codes.reshape(-1))
                    if candidate.symbolic_codes is not None
                    else None
                ),
                extras={
                    "evaluation_score": float(evaluation_score),
                    "target_delta": (
                        candidate.target_delta.detach().float().cpu()
                        if candidate.target_delta is not None
                        else None
                    ),
                },
            )
            if prior_match:
                matched = self.procedural_memory.get_skill(prior_match[0].record.skill_id)
                if (
                    matched is not None
                    and matched.skill_id != record.skill_id
                    and str(matched.extras.get("promotion_source", "")) == "online_fasttrack"
                    and float(prior_match[0].similarity) >= 0.985
                ):
                    self.procedural_memory.deprecate(matched.skill_id, record.skill_id)
            for source_episode in candidate.source_episodes:
                compiled_signatures.add((float(source_episode),))
            if self.domain_index is not None and records:
                route = self.domain_index.route(records[0].key.detach(), records[0].domain_name)
                self.domain_index.record_skill_trigger(route.cluster_id, record.skill_id, evaluation_score)
            promotions.append(
                SkillPromotion(
                    record=record,
                    evaluation_score=float(evaluation_score),
                    source="sleep",
                )
            )
        pruned_episodes: list[int] = []
        if compiled_signatures:
            for record in records:
                if (float(getattr(record, "record_id", 0)),) in compiled_signatures:
                    pruned_episodes.append(int(record.record_id))
                    if hasattr(self.episodic_memory, "prune"):
                        break
        if pruned_episodes and hasattr(self.episodic_memory, "prune"):
            self.episodic_memory.prune(lambda record: int(record.record_id) in set(pruned_episodes))
        for record in records:
            self.procedural_memory.storage.archive_episode(
                int(getattr(record, "record_id", 0)),
                {
                    "key": record.key,
                    "action": record.action,
                    "ctx_tokens": record.ctx_tokens,
                    "outcome_key": getattr(record, "outcome_key", None),
                    "selected_path": getattr(record, "selected_path", None),
                    "candidate_paths": getattr(record, "candidate_paths", None),
                    "candidate_total": getattr(record, "candidate_total", None),
                    "goal_key": getattr(record, "goal_key", None),
                    "mcts_trace": getattr(record, "mcts_trace", None),
                    "skill_trace": getattr(record, "skill_trace", None),
                    "domain_cluster_id": getattr(record, "domain_cluster_id", None),
                    "domain_name": getattr(record, "domain_name", None),
                    "metadata": getattr(record, "metadata", None),
                },
            )
        self.procedural_memory.save()
        elapsed = time.time() - started
        return SleepCycleReport(
            promotions=tuple(promotions),
            pruned_episodes=tuple(pruned_episodes),
            decomposed_segments=len(segments),
            dream_candidates=len(dream_segments),
            rsd_candidates=len(rsd_candidates),
            bodegen_candidates=len(bodegen_candidates),
            elapsed_s=elapsed,
        )
