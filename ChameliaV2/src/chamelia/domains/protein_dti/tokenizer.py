"""Tokenizer for protein-drug interaction episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from src.chamelia.tokenizers.base import AbstractTokenizer, TokenizerOutput

from .features import AnnotationVocab, build_vocab
from .graph_builder import GraphRecord, coerce_graph_record


@dataclass
class ProteinDTIObservation:
    """One ranked protein-versus-compound episode."""

    uniprot_id: str
    protein_graph: Any
    candidate_drugs: list[Any]
    candidate_ids: list[str]
    affinity_values: list[float]
    go_terms: list[str]
    cath_ids: list[str]
    affinity_type: str = "Kd"
    metadata: dict[str, Any] | None = None


@dataclass
class ProteinDTIBatch:
    """Batched protein DTI observations."""

    observations: list[ProteinDTIObservation]


class _GraphMessageLayer(nn.Module):
    """Small edge-aware message-passing layer without external scatter deps."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_embeddings
        src = edge_index[0].long()
        dst = edge_index[1].long()
        messages = self.message_mlp(node_embeddings[src] + edge_embeddings)
        aggregated = torch.zeros_like(node_embeddings)
        aggregated.index_add_(0, dst, messages)
        degree = torch.zeros(
            node_embeddings.shape[0],
            1,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        degree.index_add_(
            0,
            dst,
            torch.ones(dst.shape[0], 1, dtype=node_embeddings.dtype, device=node_embeddings.device),
        )
        aggregated = aggregated / degree.clamp_min(1.0)
        return self.norm(node_embeddings + self.update_mlp(aggregated))


class _GraphTokenEncoder(nn.Module):
    """Encode one graph into a small fixed token set."""

    def __init__(self, embed_dim: int, pooled_tokens: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pooled_tokens = int(pooled_tokens)
        self.node_proj = nn.LazyLinear(embed_dim)
        self.edge_proj = nn.LazyLinear(embed_dim)
        self.layers = nn.ModuleList([_GraphMessageLayer(embed_dim) for _ in range(3)])
        self.output_norm = nn.LayerNorm(embed_dim)

    def _pool_segments(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return torch.zeros(self.pooled_tokens, self.embed_dim, device=node_embeddings.device)
        global_token = node_embeddings.mean(dim=0, keepdim=True)
        if self.pooled_tokens == 1:
            return self.output_norm(global_token)
        segment_count = self.pooled_tokens - 1
        chunks = torch.chunk(node_embeddings, segment_count, dim=0)
        segment_tokens = []
        for chunk in chunks:
            if chunk.numel() == 0:
                segment_tokens.append(global_token.squeeze(0))
            else:
                segment_tokens.append(chunk.mean(dim=0))
        while len(segment_tokens) < segment_count:
            segment_tokens.append(global_token.squeeze(0))
        stacked = torch.vstack([global_token, torch.stack(segment_tokens[:segment_count], dim=0)])
        return self.output_norm(stacked)

    def forward(self, graph: GraphRecord) -> torch.Tensor:
        node_embeddings = self.node_proj(graph.x.float())
        if graph.edge_index.numel() == 0 or graph.edge_attr.numel() == 0:
            return self._pool_segments(node_embeddings)
        edge_embeddings = self.edge_proj(graph.edge_attr.float())
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, graph.edge_index, edge_embeddings)
        return self._pool_segments(node_embeddings)


class ProteinDrugTokenizer(AbstractTokenizer):
    """Flatten graph-and-annotation observations into standard Chamelia tokens."""

    def __init__(
        self,
        *,
        embed_dim: int = 512,
        max_candidate_drugs: int = 20,
        protein_summary_tokens: int = 4,
        max_go_terms: int = 64,
        max_cath_ids: int = 16,
        go_vocab: AnnotationVocab | dict[str, int] | None = None,
        cath_vocab: AnnotationVocab | dict[str, int] | None = None,
        go_vocab_size: int = 50_000,
        cath_vocab_size: int = 10_000,
        domain_name: str = "protein_dti",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_candidate_drugs = int(max_candidate_drugs)
        self.protein_summary_tokens = int(protein_summary_tokens)
        self.max_go_terms = int(max_go_terms)
        self.max_cath_ids = int(max_cath_ids)
        self.max_seq_len = self.protein_summary_tokens + self.max_candidate_drugs + 2
        self.domain_name = domain_name

        self.protein_encoder = _GraphTokenEncoder(embed_dim=embed_dim, pooled_tokens=protein_summary_tokens)
        self.drug_encoder = _GraphTokenEncoder(embed_dim=embed_dim, pooled_tokens=1)
        self.go_vocab = self._coerce_vocab(go_vocab, go_vocab_size)
        self.cath_vocab = self._coerce_vocab(cath_vocab, cath_vocab_size)
        self.go_embed = nn.Embedding(self.go_vocab.size, embed_dim, padding_idx=0)
        self.cath_embed = nn.Embedding(self.cath_vocab.size, embed_dim, padding_idx=0)
        self.go_pool = nn.Linear(embed_dim, embed_dim)
        self.cath_pool = nn.Linear(embed_dim, embed_dim)
        self.type_embed = nn.Embedding(4, embed_dim)
        pos = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, self.max_seq_len, embed_dim)
        pe[:, :, 0::2] = torch.sin(pos * div)
        pe[:, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("sinusoidal_pos_embed", pe, persistent=False)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def _coerce_vocab(
        self,
        vocab: AnnotationVocab | dict[str, int] | None,
        fallback_size: int,
    ) -> AnnotationVocab:
        if isinstance(vocab, AnnotationVocab):
            return vocab
        if isinstance(vocab, dict):
            size = max([0, *vocab.values()]) + 1
            return AnnotationVocab(token_to_id=dict(vocab), size=max(size, fallback_size))
        return build_vocab([], max_size=fallback_size)

    def collate(self, samples: list[Any]) -> ProteinDTIBatch:
        observations: list[ProteinDTIObservation] = []
        for sample in samples:
            if isinstance(sample, ProteinDTIObservation):
                observations.append(sample)
            else:
                raise TypeError("ProteinDrugTokenizer expects ProteinDTIObservation samples.")
        return ProteinDTIBatch(observations=observations)

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        return torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    def _annotation_token(
        self,
        annotation_ids: list[str],
        vocab: AnnotationVocab,
        embed: nn.Embedding,
        pool: nn.Linear,
        max_items: int,
        device: torch.device,
    ) -> torch.Tensor:
        indices = vocab.encode_many(annotation_ids, max_items=max_items).to(device)
        token_embeddings = embed(indices.unsqueeze(0))
        pooled = token_embeddings.mean(dim=1)
        return pool(pooled)

    def _move_graph(self, graph: GraphRecord, device: torch.device) -> GraphRecord:
        return GraphRecord(
            identifier=graph.identifier,
            x=graph.x.to(device),
            edge_index=graph.edge_index.to(device),
            edge_attr=graph.edge_attr.to(device),
            metadata=dict(graph.metadata),
        )

    def _encode_sample(self, observation: ProteinDTIObservation, device: torch.device) -> torch.Tensor:
        protein_graph = self._move_graph(
            coerce_graph_record(observation.protein_graph, identifier_fallback=observation.uniprot_id),
            device,
        )
        protein_tokens = self.protein_encoder(protein_graph)
        protein_types = self.type_embed(
            torch.zeros(protein_tokens.shape[0], dtype=torch.long, device=device)
        )
        token_parts = [protein_tokens + protein_types]

        candidate_graphs = observation.candidate_drugs[: self.max_candidate_drugs]
        for graph_payload in candidate_graphs:
            graph = self._move_graph(coerce_graph_record(graph_payload), device)
            drug_token = self.drug_encoder(graph)
            drug_type = self.type_embed(torch.ones(1, dtype=torch.long, device=device))
            token_parts.append(drug_token + drug_type)

        go_token = self._annotation_token(
            observation.go_terms,
            vocab=self.go_vocab,
            embed=self.go_embed,
            pool=self.go_pool,
            max_items=self.max_go_terms,
            device=device,
        )
        go_type = self.type_embed(torch.full((1,), 2, dtype=torch.long, device=device))
        cath_token = self._annotation_token(
            observation.cath_ids,
            vocab=self.cath_vocab,
            embed=self.cath_embed,
            pool=self.cath_pool,
            max_items=self.max_cath_ids,
            device=device,
        )
        cath_type = self.type_embed(torch.full((1,), 3, dtype=torch.long, device=device))
        token_parts.append(go_token + go_type)
        token_parts.append(cath_token + cath_type)

        return torch.cat(token_parts, dim=0)

    def forward(self, batch: ProteinDTIBatch) -> TokenizerOutput:
        if not isinstance(batch, ProteinDTIBatch):
            raise TypeError("ProteinDrugTokenizer expects a ProteinDTIBatch input.")
        if not batch.observations:
            empty_tokens = torch.empty(0, 0, self.embed_dim, dtype=torch.float32)
            empty_positions = torch.empty(0, 0, dtype=torch.long)
            empty_mask = torch.empty(0, 0, dtype=torch.bool)
            return TokenizerOutput(
                tokens=empty_tokens,
                position_ids=empty_positions,
                padding_mask=empty_mask,
                domain_name=self.domain_name,
            )

        device = next(self.parameters()).device
        sample_tokens = [self._encode_sample(observation, device) for observation in batch.observations]
        max_len = max(tokens.shape[0] for tokens in sample_tokens)
        if max_len > self.max_seq_len:
            raise ValueError(f"ProteinDrugTokenizer emitted {max_len} tokens, exceeds max_seq_len {self.max_seq_len}.")

        padded_tokens = torch.zeros(
            len(sample_tokens),
            max_len,
            self.embed_dim,
            dtype=torch.float32,
            device=device,
        )
        padding_mask = torch.ones(
            len(sample_tokens),
            max_len,
            dtype=torch.bool,
            device=device,
        )
        for index, tokens in enumerate(sample_tokens):
            padded_tokens[index, : tokens.shape[0], :] = tokens
            padding_mask[index, : tokens.shape[0]] = False

        position_ids = self.get_position_ids(len(sample_tokens), max_len, device)
        tokens = padded_tokens + self.sinusoidal_pos_embed[:, :max_len, :]
        tokens = self.output_norm(tokens)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        return TokenizerOutput(
            tokens=tokens,
            position_ids=position_ids,
            padding_mask=padding_mask,
            domain_name=self.domain_name,
        )
