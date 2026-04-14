"""Evaluation helpers for protein DTI ranking tasks."""

from __future__ import annotations

from math import sqrt


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for item_idx in range(cursor, end):
            ranks[indexed[item_idx][0]] = average_rank
        cursor = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=False))
    denom_x = sqrt(sum(x * x for x in centered_x))
    denom_y = sqrt(sum(y * y for y in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return numerator / (denom_x * denom_y)


def spearman_rank_correlation(truth: list[float], predicted: list[float]) -> float | None:
    """Compute Spearman correlation with average ranks for ties."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length.")
    if len(truth) < 2:
        return None
    truth_ranks = _average_ranks(list(truth))
    predicted_ranks = _average_ranks(list(predicted))
    return _pearson(truth_ranks, predicted_ranks)


def mean_squared_error(truth: list[float], predicted: list[float]) -> float | None:
    """Compute the mean squared error."""
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted must have the same length.")
    if not truth:
        return None
    return sum((target - guess) ** 2 for target, guess in zip(truth, predicted, strict=False)) / len(truth)


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Compute AUROC using the Mann-Whitney interpretation."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    if not labels:
        return None
    positive_count = sum(1 for label in labels if int(label) == 1)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    score_ranks = _average_ranks(list(scores))
    positive_rank_sum = sum(
        rank
        for label, rank in zip(labels, score_ranks, strict=False)
        if int(label) == 1
    )
    baseline = positive_count * (positive_count + 1) / 2.0
    return (positive_rank_sum - baseline) / (positive_count * negative_count)
