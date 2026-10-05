"""Remove a rating from a feature space: the ablation.

How much of an encoding model's prediction depends on what a rating measures? Fit the model on the
full features and on the features with the rating's direction removed, and compare held-out
correlations voxel by voxel (Δr = r_full - r_removed).

What is removed:

1. The rating regressor (``regressors``, column ``load``) is regressed on the word rate at lags
   0-4 plus an intercept, so the removed direction is the rating beyond speech rate.
2. That residual at lags 0-4 (zero-padded per story, as the model's FIR delays are) plus an
   intercept spans the removed subspace. Lags matter: the model sees the features at delays 1-4,
   so removing lag 0 alone would leave the rating reachable through a delayed copy.
3. Each feature column is regressed on that subspace and replaced by its residual.

Both regressions are fitted on the training stories (every story but the test story) and applied
unchanged to all of them, test story included. The features do not depend on the subject, so one
removed set serves every subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import TEST_STORY

DELAYS = (0, 1, 2, 3, 4)


def lag_matrix(series: np.ndarray, delays=DELAYS) -> np.ndarray:
    """Zero-padded lagged copies of a 1-D series, one column per delay."""
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    columns = []
    for delay in delays:
        column = np.zeros_like(series)
        if delay == 0:
            column = series.copy()
        elif delay < series.size:
            column[delay:] = series[:-delay]
        columns.append(column)
    return np.column_stack(columns)


def least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    gram = design.T @ design
    return np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0]), design.T @ target)


def lagged_design(series: np.ndarray, delays=DELAYS) -> np.ndarray:
    columns = lag_matrix(series, delays)
    return np.column_stack([columns, np.ones(columns.shape[0])])


def orthogonalise(regressors: dict[str, np.ndarray], fit_stories: list[str], delays=DELAYS) -> dict[str, np.ndarray]:
    """The rating (column 0) with the word rate (column 1) at ``delays`` and an intercept regressed out."""
    design = np.vstack([lagged_design(regressors[s][:, 1], delays) for s in fit_stories])
    target = np.concatenate([regressors[s][:, 0] for s in fit_stories])
    beta = least_squares(design, target)
    return {s: regressors[s][:, 0].astype(np.float64) - lagged_design(regressors[s][:, 1], delays) @ beta
            for s in regressors}


def remove(features: dict[str, np.ndarray], series: dict[str, np.ndarray], fit_stories: list[str],
           delays=DELAYS) -> tuple[dict[str, np.ndarray], float]:
    """The features with ``series`` (lags ``delays`` + intercept) projected out, fitted on ``fit_stories``,
    and the mean share of feature variance removed per story."""
    for story, values in features.items():
        if values.shape[0] != series[story].size:
            raise ValueError(f"{story}: {values.shape[0]} feature rows, {series[story].size} regressor rows")
    design = np.vstack([lagged_design(series[s], delays) for s in fit_stories])
    target = np.vstack([features[s] for s in fit_stories]).astype(np.float64)
    beta = least_squares(design, target)
    out, removed = {}, []
    for story, values in features.items():
        original = values.astype(np.float64)
        residual = original - lagged_design(series[story], delays) @ beta
        total = float(np.var(original, axis=0).sum())
        if total > 0:
            removed.append(1.0 - float(np.var(residual, axis=0).sum()) / total)
        out[story] = residual.astype(np.float32)
    return out, float(np.mean(removed))


def build_removed_features(feature_root: Path, feature: str, regressor_dir: Path, output_root: Path,
                           suffix: str = "_removed", test_story: str = TEST_STORY) -> dict:
    """Writes ``<output_root>/<feature><suffix>/<story>.npy``: the feature space with the rating removed."""
    feature_root, regressor_dir = Path(feature_root), Path(regressor_dir)
    stories = sorted(p.stem for p in (feature_root / feature).glob("*.npy") if (regressor_dir / p.name).exists())
    fit_stories = [s for s in stories if s != test_story]
    regressors = {s: np.load(regressor_dir / f"{s}.npy").astype(np.float64) for s in stories}
    series = orthogonalise(regressors, fit_stories)
    features = {s: np.load(feature_root / feature / f"{s}.npy") for s in stories}
    removed, fraction = remove(features, series, fit_stories)
    folder = Path(output_root) / f"{feature}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    for story, values in removed.items():
        np.save(folder / f"{story}.npy", values)
    summary = {"source_feature": feature, "regressors": str(regressor_dir), "stories": len(stories),
               "fit_stories": len(fit_stories), "delays": list(DELAYS),
               "mean_fraction_variance_removed": fraction}
    (folder / "removal.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{feature}{suffix}: removed {100 * fraction:.3f}% of feature variance on average", flush=True)
    return summary
