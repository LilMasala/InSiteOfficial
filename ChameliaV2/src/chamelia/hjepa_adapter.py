"""Adapters that let Chamelia run HJEPA on images or pre-embedded token sequences."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models.hjepa import HJEPA


def _encode_target_preembedded(hjepa: HJEPA, embedded_tokens: torch.Tensor) -> torch.Tensor:
    """Run the target encoder path on pre-embedded tokens.

    Args:
        hjepa: Backing HJEPA model.
        embedded_tokens: Pre-embedded tokens [B, N, D].

    Returns:
        Target features [B, N+1, D].
    """
    target_encoder = hjepa.target_encoder
    x = embedded_tokens
    cls_token = target_encoder.vit.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    if not getattr(target_encoder, "sequence_mode", False):
        x = x + target_encoder.vit.pos_embed[:, : x.shape[1], :]
    x = target_encoder.vit.pos_drop(x)
    x = target_encoder.vit.blocks(x)
    x = target_encoder.vit.norm(x)
    return x


def _hierarchical_outputs(
    hjepa: HJEPA,
    predicted_features: torch.Tensor,
    target_masked: torch.Tensor,
    mask_valid: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build hierarchical outputs mirroring ``HJEPA.forward``.

    Args:
        hjepa: HJEPA model.
        predicted_features: Finest-level predictions [B, N_masked, D].
        target_masked: Finest-level targets [B, N_masked, D].
        mask_valid: Validity mask [B, N_masked].

    Returns:
        Tuple of hierarchical predictions, targets, and validity masks.
    """
    def safe_pyramid(features: torch.Tensor) -> list[torch.Tensor]:
        """Build a short-sequence-safe hierarchy with adaptive pooling.

        Args:
            features: Base tensor [B, N, D].

        Returns:
            List of ``num_hierarchies`` tensors with lengths >= 1.
        """
        base = rearrange(features, "b n d -> b d n")
        levels: list[torch.Tensor] = []
        base_len = features.shape[1]
        for level in range(hjepa.num_hierarchies):
            out_len = max(1, base_len // (2**level))
            pooled = F.adaptive_avg_pool1d(base, out_len)
            levels.append(rearrange(pooled, "b d n -> b n d"))
        return levels

    predictions_hierarchy: list[torch.Tensor] = []
    targets_hierarchy: list[torch.Tensor] = []
    masks_valid_hierarchy: list[torch.Tensor] = []

    if hjepa.use_fpn:
        mask_valid_float = mask_valid.unsqueeze(-1).float()
        mask_valid_expanded = mask_valid_float.expand(-1, -1, hjepa.embed_dim)
        try:
            pred_fpn_features = hjepa._apply_fpn(predicted_features, is_prediction=True)
            target_fpn_features = hjepa._apply_fpn(target_masked, is_prediction=False)
            mask_fpn_features = hjepa._apply_fpn(mask_valid_expanded, is_prediction=False)
        except RuntimeError:
            pred_fpn_features = safe_pyramid(predicted_features)
            target_fpn_features = safe_pyramid(target_masked)
            mask_fpn_features = safe_pyramid(mask_valid_expanded)

        for level in range(hjepa.num_hierarchies):
            pred_projected = hjepa.hierarchy_projections[level](pred_fpn_features[level])
            target_projected = hjepa.hierarchy_projections[level](target_fpn_features[level])
            mask_valid_level = mask_fpn_features[level].mean(dim=-1) > 0.5
            predictions_hierarchy.append(pred_projected)
            targets_hierarchy.append(target_projected)
            masks_valid_hierarchy.append(mask_valid_level)
    else:
        for level in range(hjepa.num_hierarchies):
            pred_projected = hjepa.hierarchy_projections[level](predicted_features)
            target_projected = hjepa.hierarchy_projections[level](target_masked)
            mask_valid_level = mask_valid.clone()
            if level > 0:
                pred_projected = rearrange(pred_projected, "b n d -> b d n")
                target_projected = rearrange(target_projected, "b n d -> b d n")
                pred_projected = hjepa.hierarchy_pooling[level](pred_projected)
                target_projected = hjepa.hierarchy_pooling[level](target_projected)
                pred_projected = rearrange(pred_projected, "b d n -> b n d")
                target_projected = rearrange(target_projected, "b d n -> b n d")
                mask_valid_float = rearrange(mask_valid_level.float(), "b n -> b 1 n")
                mask_valid_float = hjepa.hierarchy_pooling[level](mask_valid_float)
                mask_valid_level = rearrange(mask_valid_float, "b 1 n -> b n") > 0.5
            predictions_hierarchy.append(pred_projected)
            targets_hierarchy.append(target_projected)
            masks_valid_hierarchy.append(mask_valid_level)
    return predictions_hierarchy, targets_hierarchy, masks_valid_hierarchy


def forward_hjepa(
    hjepa: HJEPA | Any,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    input_kind: str = "auto",
) -> dict[str, Any]:
    """Run HJEPA on images or pre-embedded tokens.

    Args:
        hjepa: HJEPA-compatible model.
        inputs: Image tensor [B, C, H, W] or pre-embedded tokens [B, N, D].
        mask: Binary mask [B, N] with 1 indicating masked positions.
        input_kind: ``image``, ``embedded_tokens``, or ``auto``.

    Returns:
        HJEPA-style output dictionary.
    """
    if input_kind == "auto":
        input_kind = "image" if inputs.dim() == 4 else "embedded_tokens"

    if input_kind == "image":
        return hjepa(inputs, mask)

    if input_kind != "embedded_tokens":
        raise ValueError(f"Unsupported input_kind '{input_kind}'.")

    if inputs.dim() != 3:
        raise ValueError(f"Expected pre-embedded tokens [B, N, D], got {tuple(inputs.shape)}.")

    if not hasattr(hjepa, "context_encoder") or not hasattr(hjepa, "predictor"):
        return hjepa(inputs, mask)

    B, N, _ = inputs.shape
    context_features = hjepa.context_encoder(inputs, mask=mask, pre_embedded=True)
    with torch.no_grad():
        target_features = _encode_target_preembedded(cast(HJEPA, hjepa), inputs)

    mask_bool = mask.bool()
    num_masked_per_sample = mask_bool.sum(dim=1)
    max_masked = int(num_masked_per_sample.max().item())
    if max_masked == 0:
        max_masked = 1
        mask_indices = torch.zeros((B, 1), dtype=torch.long, device=mask.device)
        mask_valid = torch.zeros((B, 1), dtype=torch.bool, device=mask.device)
    else:
        mask_indices = torch.zeros((B, max_masked), dtype=torch.long, device=mask.device)
        mask_valid = torch.zeros((B, max_masked), dtype=torch.bool, device=mask.device)
        for batch_idx in range(B):
            sample_mask_indices = mask_bool[batch_idx].nonzero(as_tuple=True)[0]
            num_masked = len(sample_mask_indices)
            mask_indices[batch_idx, :num_masked] = sample_mask_indices
            mask_valid[batch_idx, :num_masked] = True

    pos_embed = hjepa.context_encoder.vit.pos_embed[:, 1 : N + 1, :].expand(B, -1, -1)
    predicted_features = hjepa.predictor(
        context_features=context_features[:, 1:, :],
        mask_indices=mask_indices,
        pos_embed=pos_embed,
    )
    mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, hjepa.embed_dim)
    target_masked = torch.gather(target_features[:, 1:, :], 1, mask_indices_expanded)

    predictions_hierarchy, targets_hierarchy, masks_valid_hierarchy = _hierarchical_outputs(
        cast(HJEPA, hjepa),
        predicted_features,
        target_masked,
        mask_valid,
    )

    return {
        "predictions": predictions_hierarchy,
        "targets": targets_hierarchy,
        "masks_valid": masks_valid_hierarchy,
        "context_features": context_features,
        "target_features": target_features,
    }
