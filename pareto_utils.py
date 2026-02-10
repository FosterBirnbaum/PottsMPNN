"""Pareto utilities for 2D minimization objectives."""

from __future__ import annotations

from typing import Optional

import numpy as np


def pareto_front_2d(stability_scores: np.ndarray, binding_scores: np.ndarray) -> np.ndarray:
    """Compute a 2-objective Pareto front in O(n log n) for minimization scores."""
    stability = np.asarray(stability_scores)
    binding = np.asarray(binding_scores)
    if stability.shape != binding.shape:
        raise ValueError("stability_scores and binding_scores must have identical shapes.")

    n = stability.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.lexsort((binding, stability))
    stable_sorted = stability[order]
    bind_sorted = binding[order]

    front_sorted = np.ones(n, dtype=bool)
    best_binding = np.inf

    i = 0
    while i < n:
        j = i + 1
        while j < n and stable_sorted[j] == stable_sorted[i] and bind_sorted[j] == bind_sorted[i]:
            j += 1

        current_binding = bind_sorted[i]
        is_dominated = current_binding > best_binding
        if is_dominated:
            front_sorted[i:j] = False

        if current_binding < best_binding:
            best_binding = current_binding

        i = j

    front = np.zeros(n, dtype=bool)
    front[order] = front_sorted
    return front


def pareto_prefilter_mask(
    stability_scores: np.ndarray,
    binding_scores: np.ndarray,
    percentile: Optional[float],
) -> np.ndarray:
    """Return mask for points in top percentile of either minimization objective."""
    stability = np.asarray(stability_scores)
    binding = np.asarray(binding_scores)
    if stability.shape != binding.shape:
        raise ValueError("stability_scores and binding_scores must have identical shapes.")

    n = stability.shape[0]
    if percentile is None:
        return np.ones(n, dtype=bool)
    if not (0.0 < percentile <= 100.0):
        raise ValueError("pareto_prefilter_percentile must be in (0, 100].")
    if n == 0:
        return np.zeros(0, dtype=bool)

    stability_cutoff = np.percentile(stability, percentile)
    binding_cutoff = np.percentile(binding, percentile)
    return (stability <= stability_cutoff) | (binding <= binding_cutoff)
