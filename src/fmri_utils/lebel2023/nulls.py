"""Optional: a variance-matched null for the feature-space ablation.

Removing any five temporal directions from a feature space costs some prediction, so Δr alone does
not say whether the rating is special. The null removes random semantic directions instead, each
one a time course built from the features themselves and as large as the rating's:

    s = (X - mean) w,  w ~ N(0, I) normalised,

orthogonalised on the word rate (lags 0-4), the rating (lags 0-4) and an intercept (fitted on the
training stories), and kept when removing its lags 0-4 removes as much feature variance as removing
the rating's does: |f(s) / f(rating) - 1| <= 0.1, f the mean over stories of the share of feature
variance removed. 1,000 such directions, each fitted exactly like the ablation, give 1,000 null Δr
per voxel; per voxel, net = Δr - mean null Δr and z = net / sd of the null Δr.

It is expensive: each direction is a full refit (about 1,800 cluster tasks of a few hours for three
feature spaces and eight subjects). ``group.null_draws`` turns the subjects' nulls into a group
test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .ablation import DELAYS, lag_matrix, lagged_design, least_squares, orthogonalise
from .dataset import TEST_STORY


class NullSpace:
    """A feature space, the rating and its covariates: builds null series and measures what they remove."""

    def __init__(self, feature_root: Path, feature: str, regressor_dir: Path):
        feature_root, regressor_dir = Path(feature_root), Path(regressor_dir)
        self.stories = sorted(p.stem for p in (feature_root / feature).glob("*.npy") if (regressor_dir / p.name).exists())
        self.fit = [s for s in self.stories if s != TEST_STORY]
        self.X = {s: np.load(feature_root / feature / f"{s}.npy").astype(np.float64) for s in self.stories}
        regressors = {s: np.load(regressor_dir / f"{s}.npy").astype(np.float64) for s in self.stories}
        self.rating = orthogonalise(regressors, self.fit)
        self.mean = np.vstack([self.X[s] for s in self.fit]).mean(axis=0)
        self.covariates = {s: np.column_stack([lag_matrix(regressors[s][:, 1]), lag_matrix(self.rating[s]),
                                               np.ones(self.rating[s].size)]) for s in self.stories}
        self.C = np.vstack([self.covariates[s] for s in self.fit])
        self.total = {s: float(self.X[s].var(axis=0).sum()) for s in self.stories}

    def null_series(self, w: np.ndarray) -> dict[str, np.ndarray]:
        raw = {s: (self.X[s] - self.mean) @ w for s in self.stories}
        beta = least_squares(self.C, np.concatenate([raw[s] for s in self.fit]))
        return {s: raw[s] - self.covariates[s] @ beta for s in self.stories}

    def _coefficients(self, series):
        design = {s: lagged_design(series[s], DELAYS) for s in self.stories}
        return design, least_squares(np.vstack([design[s] for s in self.fit]), np.vstack([self.X[s] for s in self.fit]))

    def removed(self, series: dict[str, np.ndarray]) -> float:
        design, coef = self._coefficients(series)
        return float(np.mean([1 - (self.X[s] - design[s] @ coef).var(axis=0).sum() / self.total[s] for s in self.stories]))

    def project_out(self, series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        design, coef = self._coefficients(series)
        return {s: (self.X[s] - design[s] @ coef).astype(np.float32) for s in self.stories}


def select_directions(feature_root: Path, feature: str, regressor_dir: Path, output: Path, n: int = 1000,
                      tolerance: float = 0.10, seed: int = 0, max_candidates: int = 100000) -> dict:
    started = time.time()
    space = NullSpace(feature_root, feature, regressor_dir)
    f_rating = space.removed(space.rating)
    rng = np.random.default_rng(seed)
    accepted, fractions, drawn = [], [], 0
    while len(accepted) < n and drawn < max_candidates:
        w = rng.standard_normal(space.mean.size)
        w /= np.linalg.norm(w)
        drawn += 1
        f = space.removed(space.null_series(w))
        if abs(f / f_rating - 1) <= tolerance:
            accepted.append(w.astype(np.float32))
            fractions.append(f)
        if drawn % 500 == 0:
            print(f"{drawn} drawn, {len(accepted)} accepted ({time.time() - started:.0f} s)", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, w=np.stack(accepted), fraction=np.array(fractions), rating_fraction=f_rating,
             tolerance=tolerance, candidates_drawn=drawn, seed=seed)
    summary = {"feature": feature, "rating_fraction": f_rating, "accepted": len(accepted), "drawn": drawn,
               "seconds": round(time.time() - started)}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def score(d_obs: np.ndarray, d_null: np.ndarray) -> dict[str, np.ndarray]:
    """Per voxel against its null drops (nulls x voxels): net, z and the one-sided empirical p."""
    mean, sd = d_null.mean(axis=0), d_null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (d_obs - mean) / sd, np.nan)
    p = (1 + (d_null >= d_obs[None]).sum(axis=0)) / (d_null.shape[0] + 1)
    return {"net": d_obs - mean, "z": z, "p": p, "null_mean": mean, "null_sd": sd}


class ChunkFitter:
    """One subject's voxel chunk, loaded once, refitted on any feature set (the ablation's design)."""

    def __init__(self, dataset, subject: str, chunk: int, n_chunks: int, extra_root: Path | None = None):
        from .dataset import common_finite
        from .encoding import chunk_bounds
        self.dataset, self.subject, self.extra_root = dataset, subject, extra_root
        self.stories = dataset.training_stories(subject)
        self.start, self.stop = chunk_bounds(dataset.n_columns(subject), chunk, n_chunks)
        train = [dataset.response(subject, s, self.start, self.stop) for s in self.stories]
        test = dataset.response(subject, TEST_STORY, self.start, self.stop)
        repeats = dataset.repeats(subject, TEST_STORY, self.start, self.stop)
        valid = common_finite([*train, test, repeats])
        self.train, self.test = [r[:, valid] for r in train], test[:, valid]
        self.voxel_index = (self.start + np.flatnonzero(valid)).astype(np.int64)

    def fit(self, features: dict[str, np.ndarray]) -> np.ndarray:
        from .encoding import fit_voxelwise, prepare_design
        extra_train = extra_test = None
        if self.extra_root is not None:
            def extra(story, rows):
                values = np.load(Path(self.extra_root) / f"{story}.npy").astype(np.float32)
                return values.reshape(values.shape[0], -1)[:rows, :1]
            extra_train = [extra(s, features[s].shape[0]) for s in self.stories]
            extra_test = extra(TEST_STORY, features[TEST_STORY].shape[0])
        design = prepare_design([features[s] for s in self.stories], features[TEST_STORY],
                                extra_train=extra_train, extra_test=extra_test)
        correlation, _ = fit_voxelwise(design, self.stories, self.train, self.test, self.subject)
        return correlation


def fit_nulls(dataset, subject: str, feature_root: Path, feature: str, regressor_dir: Path, directions: Path,
              output: Path, chunk: int, n_chunks: int, start: int, stop: int, extra_root: Path | None = None,
              checkpoint: int = 25) -> Path:
    """Fits null directions start..stop-1 for one voxel chunk:
    ``<output>/<subject>/chunk_<c>_of_<n>/nulls_<start>-<stop>.npz`` (``correlation``, ``null_id``, ``voxel_index``)."""
    space = NullSpace(feature_root, feature, regressor_dir)
    fitter = ChunkFitter(dataset, subject, chunk, n_chunks, extra_root)
    w = np.load(directions)["w"]
    stop = min(stop or len(w), len(w))
    folder = Path(output) / subject / f"chunk_{chunk:03d}_of_{n_chunks:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"nulls_{start:04d}-{stop:04d}.npz"
    partial = target.with_suffix(".partial.npz")
    if target.exists():
        return target
    done, rows = [], []
    if partial.exists():
        saved = np.load(partial)
        done, rows = list(saved["null_id"]), list(saved["correlation"])
    for k in range(start, stop):
        if k in done:
            continue
        rows.append(fitter.fit(space.project_out(space.null_series(w[k].astype(np.float64)))))
        done.append(k)
        if len(done) % checkpoint == 0:
            np.savez(partial, correlation=np.stack(rows).astype(np.float32), null_id=np.array(done), voxel_index=fitter.voxel_index)
    np.savez(target, correlation=np.stack(rows).astype(np.float32), null_id=np.array(done), voxel_index=fitter.voxel_index)
    partial.unlink(missing_ok=True)
    return target


def read_nulls(folder: Path, n_columns: int, n_nulls: int) -> np.ndarray:
    """A subject's null fits (``<output>/<subject>``) as nulls x columns (raises if any is missing)."""
    import re
    block = re.compile(r"nulls_(\d+)-(\d+)\.npz$")
    out = np.full((n_nulls, n_columns), np.nan, dtype=np.float32)
    for chunk_dir in sorted(Path(folder).glob("chunk_*_of_*")):
        rows, index = {}, None
        for path in chunk_dir.glob("nulls_*.npz"):
            if ".partial" in path.name or not block.search(path.name):
                continue
            data = np.load(path)
            index = data["voxel_index"]
            rows.update({int(k): row for k, row in zip(data["null_id"].tolist(), data["correlation"])})
        missing = sorted(set(range(n_nulls)) - set(rows))
        if missing:
            raise ValueError(f"{chunk_dir}: {len(missing)} nulls missing (first {missing[:5]})")
        out[:, index] = np.stack([rows[k] for k in range(n_nulls)])
    return out
