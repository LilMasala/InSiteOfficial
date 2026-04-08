"""Representation upgrades for the cognitive architecture."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QuantizedSkill:
    """Container for vector-quantized latent skills."""

    quantized: torch.Tensor
    codes: torch.Tensor
    commitment_loss: torch.Tensor


class VectorQuantizer(nn.Module):
    """Standard VQ bottleneck used for skill discretisation."""

    def __init__(
        self,
        embed_dim: int,
        codebook_size: int = 256,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.beta = float(beta)
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, inputs: torch.Tensor) -> QuantizedSkill:
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embed_dim)
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat_inputs @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )
        codes = distances.argmin(dim=1)
        quantized = self.codebook(codes).view(*original_shape)
        straight_through = inputs + (quantized - inputs).detach()
        commitment = F.mse_loss(inputs, quantized.detach())
        codebook = F.mse_loss(quantized, inputs.detach())
        return QuantizedSkill(
            quantized=straight_through,
            codes=codes.view(*original_shape[:-1]),
            commitment_loss=codebook + (self.beta * commitment),
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.codebook(codes)


class InformationOrderedBottleneck(nn.Module):
    """Learn a truncatable embedding whose most important dimensions come first."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.proj = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.importance = nn.Parameter(torch.linspace(1.0, 0.1, bottleneck_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.proj(inputs)
        order = torch.argsort(self.importance.abs(), descending=True)
        ordered = projected.index_select(dim=-1, index=order)
        scales = self.importance.abs().index_select(dim=0, index=order).clamp_min(1.0e-4)
        return ordered * scales

    def truncate(self, embeddings: torch.Tensor, width: int) -> torch.Tensor:
        width = max(1, min(int(width), embeddings.shape[-1]))
        return embeddings[..., :width]

    def coarse_to_fine(
        self,
        embeddings: torch.Tensor,
        widths: list[int],
    ) -> list[torch.Tensor]:
        return [self.truncate(embeddings, width) for width in widths]


class ContrastiveSparseRepresentation(nn.Module):
    """Project dense skill latents into a sparse retrieval-friendly space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1024,
        active_dims: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.active_dims = active_dims
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        active_dims: int | None = None,
    ) -> torch.Tensor:
        active = active_dims if active_dims is not None else self.active_dims
        projected = self.norm(self.proj(inputs))
        if active <= 0 or active >= projected.shape[-1]:
            return F.normalize(projected, dim=-1)
        topk = projected.abs().topk(active, dim=-1).indices
        sparse = torch.zeros_like(projected)
        sparse.scatter_(
            dim=-1,
            index=topk,
            src=projected.gather(dim=-1, index=topk),
        )
        return F.normalize(sparse, dim=-1)


class IsotropicSkillCodec(nn.Module):
    """Compress skills to a fixed discrete token budget and reconstruct on demand."""

    def __init__(
        self,
        embed_dim: int,
        num_tokens: int = 32,
        codebook_size: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.token_dim = math.ceil(embed_dim / num_tokens)
        self.encoder = nn.Linear(embed_dim, num_tokens * self.token_dim)
        self.quantizer = VectorQuantizer(
            embed_dim=self.token_dim,
            codebook_size=codebook_size,
        )
        self.decoder = nn.Linear(num_tokens * self.token_dim, embed_dim)

    def encode(self, skills: torch.Tensor) -> QuantizedSkill:
        projected = self.encoder(skills).view(*skills.shape[:-1], self.num_tokens, self.token_dim)
        return self.quantizer(projected)

    def encode_codes(self, skills: torch.Tensor) -> torch.Tensor:
        """Encode a dense skill embedding into discrete token ids."""
        return self.encode(skills).codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes).reshape(*codes.shape[:-1], -1)
        return self.decoder(quantized)

    def reconstruct(self, skills: torch.Tensor) -> torch.Tensor:
        """Round-trip a dense skill embedding through the runtime storage format."""
        return self.decode(self.encode_codes(skills))

    def forward(self, skills: torch.Tensor) -> dict[str, torch.Tensor]:
        quantized = self.encode(skills)
        reconstructed = self.decoder(quantized.quantized.reshape(*skills.shape[:-1], -1))
        return {
            "reconstructed": reconstructed,
            "codes": quantized.codes,
            "loss": quantized.commitment_loss + F.mse_loss(reconstructed, skills),
        }
