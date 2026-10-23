"""Small pure-Python metric helpers, also used by the test suite."""

from __future__ import annotations

import math
from collections.abc import Iterable


def regression_metrics(pairs: Iterable[tuple[float, float]]) -> dict[str, float | int]:
    """Return MAE, RMSE, and count for (actual, predicted) pairs."""
    errors = [float(predicted) - float(actual) for actual, predicted in pairs]
    if not errors:
        raise ValueError("At least one actual/predicted pair is required")
    return {
        "count": len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def ranking_metrics(recommended: Iterable[int], relevant: Iterable[int], k: int) -> dict[str, float | int]:
    """Compute precision@k, recall@k, and hit-rate for one user's ranked list."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "hit": 0}
    selected = list(recommended)[:k]
    hits = len(set(selected) & relevant_set)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant_set),
        "hit": int(hits > 0),
    }
