from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import RatingSpec
from .segmentation import Segment


def weighted_kappa(a: Sequence[float], b: Sequence[float], levels: int) -> float:
    """Quadratic-weighted Cohen's kappa between two ordinal raters."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    observed = np.zeros((levels, levels), dtype=float)
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= max(1.0, observed.sum())
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.arange(levels)
    weights = (grid[:, None] - grid[None, :]) ** 2 / max(1, (levels - 1)) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denominator)


def rater_agreement(replicates: Sequence[Sequence[Mapping[str, Any]]], spec: RatingSpec) -> Dict[str, float]:
    """Agreement between replicate raters on the scale field.

    ``exact`` and ``within_one`` are averaged over rater pairs;
    ``all_raters_exact`` is the fraction of segments every rater scored
    identically, which is the strictest summary.
    """
    scale = spec.scale
    values = [np.asarray([item[scale.name] for item in replicate], dtype=float) for replicate in replicates]
    if len(values) < 2:
        return {"n_raters": len(values)}
    pairs = list(itertools.combinations(values, 2))
    stacked = np.vstack(values)
    return {
        "n_raters": len(values),
        "exact": float(np.mean([np.mean(a == b) for a, b in pairs])),
        "within_one": float(np.mean([np.mean(np.abs(a - b) <= 1) for a, b in pairs])),
        "weighted_kappa": float(np.mean([weighted_kappa(a, b, len(scale.levels)) for a, b in pairs])),
        "all_raters_exact": float(np.mean(np.all(stacked == stacked[0], axis=0))),
    }


def consensus_table(
    segments: Sequence[Segment],
    replicates: Sequence[Sequence[Mapping[str, Any]]],
    spec: RatingSpec,
) -> pd.DataFrame:
    """Combine replicate ratings into one row per segment.

    The scale gets mean, standard deviation, range and the per-replicate
    values; flags get means; categoricals get the modal choice and how often
    raters chose it. Reasons are kept joined, since they are the main way to
    audit a rating by hand.
    """
    scale = spec.scale
    rows: List[Dict[str, Any]] = []
    for position, segment in enumerate(segments):
        values = [replicate[position] for replicate in replicates]
        scores = np.asarray([float(item[scale.name]) for item in values])
        row: Dict[str, Any] = {
            "index": segment.index,
            "onset_s": segment.onset,
            "offset_s": segment.offset,
            "first_word": segment.first_word,
            "n_words": segment.n_words,
            "text": segment.text,
            "n_raters": len(values),
            f"{scale.name}_mean": float(scores.mean()),
            f"{scale.name}_sd": float(scores.std()),
            f"{scale.name}_min": float(scores.min()),
            f"{scale.name}_max": float(scores.max()),
            f"{scale.name}_by_replicate": ";".join(str(int(value)) for value in scores),
        }
        for flag in spec.flags:
            row[f"{flag}_mean"] = float(np.mean([float(item[flag]) for item in values]))
        for categorical in spec.categoricals:
            counts = Counter(str(item[categorical]) for item in values)
            choice, count = counts.most_common(1)[0]
            row[f"{categorical}_mode"] = choice
            row[f"{categorical}_agreement"] = count / len(values)
        if spec.include_confidence:
            row["confidence_mean"] = float(np.mean([float(item["confidence"]) for item in values]))
        if spec.include_reason:
            row["reasons"] = " || ".join(str(item["reason"]) for item in values)
        rows.append(row)
    return pd.DataFrame(rows)


def rating_distribution(replicates: Sequence[Sequence[Mapping[str, Any]]], spec: RatingSpec) -> Dict[str, float]:
    """Share of all replicate ratings at each level of the scale."""
    values = [item[spec.scale.name] for replicate in replicates for item in replicate]
    total = max(1, len(values))
    counts = Counter(int(value) for value in values)
    return {str(level): counts.get(level, 0) / total for level in spec.scale.levels}
