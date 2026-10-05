"""Ablate a stimulus annotation from a feature space, and the variance-matched null.

How much of an encoding model's prediction depends on what an annotation (a rating, a label time
course) measures? Fit the model on the full features and on the features with the annotation's
direction removed, and compare held-out correlations voxel by voxel: delta r = r_full - r_removed.

**What is removed.** The annotation regressor is first regressed on a covariate (typically the word
rate) at lags 0-4 plus an intercept, so what is removed is the annotation beyond it. That residual
at lags 0-4 (zero-padded per run, like the model's FIR delays) plus an intercept spans the removed
subspace; every feature column is regressed on it and replaced by its residual. Lags matter: a
model with FIR delays would otherwise reach the annotation through a delayed copy. Both regressions
are fitted on the training runs and applied unchanged to every run, test run included.

Regressor files are ``<run>.npy`` with the annotation in column 0 and the covariate in column 1
(``fmri-story-ratings regressors`` writes them); feature spaces are ``<root>/<feature>/<run>.npy``.

**The variance-matched null (optional).** Removing any few temporal directions costs some
prediction, so delta r alone does not say whether the annotation is special. The null removes random
semantic directions of the features instead, s = (X - mean) w with w ~ N(0, I) normalised,
orthogonalised on the covariate, the annotation (lags 0-4 each) and an intercept, and kept when
removing it removes as much feature variance as removing the annotation does
(|f(s) / f(annotation) - 1| <= tolerance, f the mean share over runs). Each kept direction is fitted
exactly like the ablation, giving a null delta r distribution per voxel: net = delta r - mean, z = net / sd.
Each direction is a full refit, so 1,000 of them cost 1,000 times the ablation.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np

DELAYS = (0, 1, 2, 3, 4)


# ---- the removal ------------------------------------------------------------------------------
def lag_matrix(series: np.ndarray, delays: Sequence[int] = DELAYS) -> np.ndarray:
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


def lagged_design(series: np.ndarray, delays: Sequence[int] = DELAYS) -> np.ndarray:
    columns = lag_matrix(series, delays)
    return np.column_stack([columns, np.ones(columns.shape[0])])


def orthogonalise(regressors: dict[str, np.ndarray], fit_runs: Sequence[str], delays: Sequence[int] = DELAYS) -> dict[str, np.ndarray]:
    """The annotation (column 0) with the covariate (column 1) at ``delays`` and an intercept regressed out."""
    design = np.vstack([lagged_design(regressors[r][:, 1], delays) for r in fit_runs])
    beta = least_squares(design, np.concatenate([regressors[r][:, 0] for r in fit_runs]))
    return {r: regressors[r][:, 0].astype(np.float64) - lagged_design(regressors[r][:, 1], delays) @ beta for r in regressors}


def remove(features: dict[str, np.ndarray], series: dict[str, np.ndarray], fit_runs: Sequence[str],
           delays: Sequence[int] = DELAYS) -> tuple[dict[str, np.ndarray], float]:
    """Features with ``series`` (lags + intercept) projected out, fitted on ``fit_runs``; and the mean
    share of feature variance removed per run."""
    for run, values in features.items():
        if values.shape[0] != series[run].size:
            raise ValueError(f"{run}: {values.shape[0]} feature rows, {series[run].size} regressor rows")
    beta = least_squares(np.vstack([lagged_design(series[r], delays) for r in fit_runs]),
                         np.vstack([features[r] for r in fit_runs]).astype(np.float64))
    out, removed = {}, []
    for run, values in features.items():
        original = values.astype(np.float64)
        residual = original - lagged_design(series[run], delays) @ beta
        total = float(np.var(original, axis=0).sum())
        if total > 0:
            removed.append(1.0 - float(np.var(residual, axis=0).sum()) / total)
        out[run] = residual.astype(np.float32)
    return out, float(np.mean(removed))


def build_removed_features(feature_root: Path, feature: str, regressor_dir: Path, output_root: Path,
                           test_runs: Sequence[str], suffix: str = "_removed") -> dict:
    """Writes ``<output_root>/<feature><suffix>/<run>.npy``: the feature space with the annotation removed.

    The regressions are fitted on every run with both files except ``test_runs``."""
    feature_root, regressor_dir = Path(feature_root), Path(regressor_dir)
    runs = sorted(p.stem for p in (feature_root / feature).glob("*.npy") if (regressor_dir / p.name).exists())
    fit_runs = [r for r in runs if r not in set(test_runs)]
    series = orthogonalise({r: np.load(regressor_dir / f"{r}.npy").astype(np.float64) for r in runs}, fit_runs)
    removed, fraction = remove({r: np.load(feature_root / feature / f"{r}.npy") for r in runs}, series, fit_runs)
    folder = Path(output_root) / f"{feature}{suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    for run, values in removed.items():
        np.save(folder / f"{run}.npy", values)
    summary = {"source_feature": feature, "regressors": str(regressor_dir), "runs": len(runs), "fit_runs": len(fit_runs),
               "test_runs": list(test_runs), "delays": list(DELAYS), "mean_fraction_variance_removed": fraction}
    (folder / "removal.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{feature}{suffix}: removed {100 * fraction:.3f}% of feature variance on average", flush=True)
    return summary


# ---- the variance-matched null ----------------------------------------------------------------
class NullSpace:
    """A feature space, the annotation and its covariates: builds null series and measures what they remove."""

    def __init__(self, feature_root: Path, feature: str, regressor_dir: Path, test_runs: Sequence[str]):
        feature_root, regressor_dir = Path(feature_root), Path(regressor_dir)
        self.runs = sorted(p.stem for p in (feature_root / feature).glob("*.npy") if (regressor_dir / p.name).exists())
        self.fit = [r for r in self.runs if r not in set(test_runs)]
        self.X = {r: np.load(feature_root / feature / f"{r}.npy").astype(np.float64) for r in self.runs}
        regressors = {r: np.load(regressor_dir / f"{r}.npy").astype(np.float64) for r in self.runs}
        self.annotation = orthogonalise(regressors, self.fit)
        self.mean = np.vstack([self.X[r] for r in self.fit]).mean(axis=0)
        self.covariates = {r: np.column_stack([lag_matrix(regressors[r][:, 1]), lag_matrix(self.annotation[r]),
                                               np.ones(self.annotation[r].size)]) for r in self.runs}
        self.C = np.vstack([self.covariates[r] for r in self.fit])
        self.total = {r: float(self.X[r].var(axis=0).sum()) for r in self.runs}

    def null_series(self, w: np.ndarray) -> dict[str, np.ndarray]:
        raw = {r: (self.X[r] - self.mean) @ w for r in self.runs}
        beta = least_squares(self.C, np.concatenate([raw[r] for r in self.fit]))
        return {r: raw[r] - self.covariates[r] @ beta for r in self.runs}

    def _coefficients(self, series):
        design = {r: lagged_design(series[r]) for r in self.runs}
        return design, least_squares(np.vstack([design[r] for r in self.fit]), np.vstack([self.X[r] for r in self.fit]))

    def removed(self, series: dict[str, np.ndarray]) -> float:
        design, coef = self._coefficients(series)
        return float(np.mean([1 - (self.X[r] - design[r] @ coef).var(axis=0).sum() / self.total[r] for r in self.runs]))

    def project_out(self, series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        design, coef = self._coefficients(series)
        return {r: (self.X[r] - design[r] @ coef).astype(np.float32) for r in self.runs}


def select_directions(space: NullSpace, output: Path, n: int = 1000, tolerance: float = 0.10, seed: int = 0,
                      max_candidates: int = 100000) -> dict:
    started = time.time()
    f_annotation = space.removed(space.annotation)
    rng = np.random.default_rng(seed)
    accepted, fractions, drawn = [], [], 0
    while len(accepted) < n and drawn < max_candidates:
        w = rng.standard_normal(space.mean.size)
        w /= np.linalg.norm(w)
        drawn += 1
        f = space.removed(space.null_series(w))
        if abs(f / f_annotation - 1) <= tolerance:
            accepted.append(w.astype(np.float32))
            fractions.append(f)
        if drawn % 500 == 0:
            print(f"{drawn} drawn, {len(accepted)} accepted ({time.time() - started:.0f} s)", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, w=np.stack(accepted), fraction=np.array(fractions), annotation_fraction=f_annotation,
             tolerance=tolerance, candidates_drawn=drawn, seed=seed)
    summary = {"annotation_fraction": f_annotation, "accepted": len(accepted), "drawn": drawn, "seconds": round(time.time() - started)}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def score(d_obs: np.ndarray, d_null: np.ndarray) -> dict[str, np.ndarray]:
    """Per voxel against its null drops (nulls x voxels): net, z, one-sided empirical p."""
    mean, sd = d_null.mean(axis=0), d_null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (d_obs - mean) / sd, np.nan)
    p = (1 + (d_null >= d_obs[None]).sum(axis=0)) / (d_null.shape[0] + 1)
    return {"net": d_obs - mean, "z": z, "p": p, "null_mean": mean, "null_sd": sd}


def fit_nulls(table, subject: str, space: NullSpace, directions: Path, output: Path, chunk: int, n_chunks: int,
              start: int, stop: int, extra_root: Path | None = None, checkpoint: int = 25) -> Path:
    """Fits null directions start..stop-1 for one column chunk of a ``RunTable`` subject:
    ``<output>/<subject>/chunk_<c>_of_<n>/nulls_<start>-<stop>.npz`` (``correlation``, ``null_id``, ``voxel_index``)."""
    from fmri_utils.encoding.columns import Chunk
    data = Chunk(table, subject, chunk, n_chunks)
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
        features = space.project_out(space.null_series(w[k].astype(np.float64)))
        rows.append(data.fit(data.design(features, extra_root))[0])
        done.append(k)
        if len(done) % checkpoint == 0:
            np.savez(partial, correlation=np.stack(rows).astype(np.float32), null_id=np.array(done), voxel_index=data.voxel_index)
    np.savez(target, correlation=np.stack(rows).astype(np.float32), null_id=np.array(done), voxel_index=data.voxel_index)
    partial.unlink(missing_ok=True)
    return target


def read_nulls(folder: Path, n_columns: int, n_nulls: int) -> np.ndarray:
    """A subject's null fits (``<output>/<subject>``) as nulls x columns; raises if any is missing."""
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


def main(argv=None) -> None:
    """fmri-ablation remove | nulls-select | nulls-fit"""
    import argparse
    parser = argparse.ArgumentParser(prog="fmri-ablation",
                                     description="Ablate an annotation from a feature space; the variance-matched null.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("remove", "nulls-select", "nulls-fit"):
        p = sub.add_parser(name)
        p.add_argument("--feature-root", type=Path, required=True, help="holding <feature>/<run>.npy")
        p.add_argument("--feature", required=True)
        p.add_argument("--regressors", type=Path, required=True, help="<run>.npy: annotation, covariate")
        p.add_argument("--test-run", action="append", required=True, help="excluded from the fits (repeatable)")
        if name == "remove":
            p.add_argument("--output-root", type=Path, required=True)
            p.add_argument("--suffix", default="_removed")
        if name == "nulls-select":
            p.add_argument("--output", type=Path, required=True, help="the directions (.npz)")
            p.add_argument("--n", type=int, default=1000)
            p.add_argument("--tolerance", type=float, default=0.10)
            p.add_argument("--seed", type=int, default=0)
        if name == "nulls-fit":
            p.add_argument("--run-table", type=Path, required=True)
            p.add_argument("--subject", required=True)
            p.add_argument("--directions", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--extra-root", type=Path, default=None)
            p.add_argument("--chunk", type=int, default=0)
            p.add_argument("--n-chunks", type=int, default=5)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "remove":
        build_removed_features(args.feature_root, args.feature, args.regressors, args.output_root, args.test_run, args.suffix)
        return
    space = NullSpace(args.feature_root, args.feature, args.regressors, args.test_run)
    if args.command == "nulls-select":
        select_directions(space, args.output, n=args.n, tolerance=args.tolerance, seed=args.seed)
    else:
        from fmri_utils.encoding.run_table import RunTable
        fit_nulls(RunTable(args.run_table), args.subject, space, args.directions, args.output, args.chunk,
                  args.n_chunks, args.start, args.stop, args.extra_root)


if __name__ == "__main__":
    main()
