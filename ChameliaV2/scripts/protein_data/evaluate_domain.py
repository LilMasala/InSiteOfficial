"""Run a small protein DTI ranking evaluation using the bridge runtime model path."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from common import default_data_dir, default_db_path
from src.chamelia.domains.protein_dti.metrics import (
    binary_auroc,
    mean_squared_error,
    spearman_rank_correlation,
)
from src.serving.bridge_runtime import BridgeRuntime


def _apply_runtime_env(
    *,
    db_path: Path,
    data_dir: Path,
    split: str,
    split_strategy: str,
    affinity_type: str,
    max_candidate_drugs: int,
    seed: int,
) -> None:
    os.environ["CHAMELIA_PROTEIN_DTI_DB_PATH"] = str(db_path)
    os.environ["CHAMELIA_PROTEIN_DTI_DATA_DIR"] = str(data_dir)
    os.environ["CHAMELIA_PROTEIN_DTI_SPLIT"] = split
    os.environ["CHAMELIA_PROTEIN_DTI_SPLIT_STRATEGY"] = split_strategy
    os.environ["CHAMELIA_PROTEIN_DTI_AFFINITY_TYPE"] = affinity_type
    os.environ["CHAMELIA_PROTEIN_DTI_MAX_CANDIDATES"] = str(max_candidate_drugs)
    os.environ["CHAMELIA_PROTEIN_DTI_ACTION_DIM"] = str(max_candidate_drugs)
    os.environ["CHAMELIA_PROTEIN_DTI_SEED"] = str(seed)


def _mask_tensor(tokenized: Any) -> torch.Tensor:
    if tokenized.padding_mask is None:
        return torch.zeros(
            tokenized.tokens.shape[0],
            tokenized.tokens.shape[1],
            device=tokenized.tokens.device,
            dtype=torch.float32,
        )
    return tokenized.padding_mask.to(device=tokenized.tokens.device, dtype=torch.float32)


def run(
    *,
    db_path: Path,
    data_dir: Path,
    split: str,
    split_strategy: str,
    affinity_type: str,
    max_candidate_drugs: int,
    num_proteins: int,
    seed: int,
    binder_threshold: float,
    backbone_mode: str,
    checkpoint_path: str | None,
    device: str,
) -> dict[str, Any]:
    _apply_runtime_env(
        db_path=db_path,
        data_dir=data_dir,
        split=split,
        split_strategy=split_strategy,
        affinity_type=affinity_type,
        max_candidate_drugs=max_candidate_drugs,
        seed=seed,
    )
    runtime = BridgeRuntime(
        backbone_mode=backbone_mode,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    session = runtime.get_session("protein-dti-eval", "protein_dti")
    domain = session.domain
    tokenizer = domain.get_tokenizer()
    rng = random.Random(seed)

    protein_ids = domain.dataset.protein_ids  # type: ignore[attr-defined]
    if num_proteins > 0 and num_proteins < len(protein_ids):
        selected_proteins = sorted(rng.sample(protein_ids, num_proteins))
    else:
        selected_proteins = protein_ids

    per_protein: list[dict[str, Any]] = []
    all_truth: list[float] = []
    all_scores: list[float] = []
    all_labels: list[int] = []

    for uniprot_id in selected_proteins:
        observation = domain.dataset.load_observation(  # type: ignore[attr-defined]
            uniprot_id,
            deterministic=True,
        )
        if observation is None:
            continue
        batch = tokenizer.collate([observation])
        tokenized = tokenizer(batch)
        with torch.no_grad():
            outputs = session.model(
                tokenized.tokens,
                _mask_tensor(tokenized),
                domain.get_domain_state(batch),
                input_kind="embedded_tokens",
                store_to_memory=False,
                advance_step=False,
            )
        candidate_count = len(observation.candidate_ids)
        scores = outputs["action_vec"][0, :candidate_count].detach().cpu().tolist()
        truth = [float(value) for value in observation.affinity_values]
        spearman = spearman_rank_correlation(truth, scores)
        mse = mean_squared_error(truth, scores)
        labels = [1 if value >= binder_threshold else 0 for value in truth]
        auroc = binary_auroc(labels, scores)
        per_protein.append(
            {
                "uniprot_id": uniprot_id,
                "candidate_count": candidate_count,
                "spearman": spearman,
                "mse": mse,
                "auroc": auroc,
                "candidate_ids": list(observation.candidate_ids),
            }
        )
        all_truth.extend(truth)
        all_scores.extend(scores)
        all_labels.extend(labels)

    spearmans = [entry["spearman"] for entry in per_protein if entry["spearman"] is not None]
    mses = [entry["mse"] for entry in per_protein if entry["mse"] is not None]
    summary = {
        "domain_name": session.domain_name,
        "model_version": session.model_version,
        "backbone_mode": backbone_mode,
        "checkpoint_path": checkpoint_path,
        "split": split,
        "split_strategy": split_strategy,
        "affinity_type": affinity_type,
        "binder_threshold": binder_threshold,
        "evaluated_proteins": len(per_protein),
        "candidate_count_mean": (
            statistics.fmean(entry["candidate_count"] for entry in per_protein)
            if per_protein
            else None
        ),
        "spearman_mean": statistics.fmean(spearmans) if spearmans else None,
        "spearman_median": statistics.median(spearmans) if spearmans else None,
        "per_protein_mse_mean": statistics.fmean(mses) if mses else None,
        "pooled_mse": mean_squared_error(all_truth, all_scores),
        "pooled_auroc": binary_auroc(all_labels, all_scores),
        "per_protein": per_protein,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate protein DTI rankings for a Chamelia session/checkpoint.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split-strategy", type=str, default="protein_family")
    parser.add_argument("--affinity-type", type=str, default="Kd")
    parser.add_argument("--max-candidate-drugs", type=int, default=20)
    parser.add_argument("--num-proteins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binder-threshold", type=float, default=7.0)
    parser.add_argument("--backbone-mode", type=str, default="stub")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run(
        db_path=Path(args.db),
        data_dir=Path(args.data_dir),
        split=str(args.split),
        split_strategy=str(args.split_strategy),
        affinity_type=str(args.affinity_type),
        max_candidate_drugs=int(args.max_candidate_drugs),
        num_proteins=int(args.num_proteins),
        seed=int(args.seed),
        binder_threshold=float(args.binder_threshold),
        backbone_mode=str(args.backbone_mode),
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
        device=str(args.device),
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"domain={summary['domain_name']} model_version={summary['model_version']}")
    print(
        f"split={summary['split']} strategy={summary['split_strategy']} "
        f"affinity_type={summary['affinity_type']}"
    )
    print(
        f"evaluated_proteins={summary['evaluated_proteins']} "
        f"candidate_count_mean={summary['candidate_count_mean']}"
    )
    print(
        f"spearman_mean={summary['spearman_mean']} "
        f"spearman_median={summary['spearman_median']}"
    )
    print(
        f"per_protein_mse_mean={summary['per_protein_mse_mean']} "
        f"pooled_mse={summary['pooled_mse']} pooled_auroc={summary['pooled_auroc']}"
    )


if __name__ == "__main__":
    main()
