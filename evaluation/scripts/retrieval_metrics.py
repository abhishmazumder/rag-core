"""Deterministic retrieval metrics. Pure functions, no I/O and no external libraries.

Inputs are graded relevance scores (0-3) listed in retrieval order.

Conventions:
- Precision@k, Recall@k and MRR treat score > 0 as relevant.
- Precision@k always divides by k. If fewer than k results were returned, the missing
  positions count as non-relevant (standard definition).
- Recall@k and nDCG@k are None (undefined) when the query has no relevant ground truth.
- aggregate_mean ignores None values; it is None when every value is None.
"""

import math


def _hits(relevances: list[int], k: int) -> int:
    return sum(1 for score in relevances[:k] if score > 0)


def precision_at_k(relevances: list[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    return _hits(relevances, k) / k


def recall_at_k(relevances: list[int], total_relevant: int, k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    if total_relevant == 0:
        return None
    return _hits(relevances, k) / total_relevant


def reciprocal_rank(relevances: list[int]) -> float:
    for rank, score in enumerate(relevances, start=1):
        if score > 0:
            return 1 / rank
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum((2**score - 1) / math.log2(rank + 1) for rank, score in enumerate(relevances[:k], 1))


def ndcg_at_k(relevances: list[int], ground_truth_scores: list[int], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be >= 1")
    ideal = dcg_at_k(sorted(ground_truth_scores, reverse=True), k)
    if ideal == 0:
        return None
    return dcg_at_k(relevances, k) / ideal


def aggregate_mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
