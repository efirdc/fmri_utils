"""Cross-participant encoding: predict a subject's voxels from the other seven subjects' brains.

The stimulus is the same for everyone, so the other subjects' responses to a story are a predictor
of a subject's response to it that carries everything shared, not only what a feature space names.

``prep``  (per subject) Each run (every training story all eight subjects heard, and the test
          story) gets an aCompCor nuisance set: the top 5 principal time courses of that run's
          white-matter and lateral-ventricle voxels (Harvard-Oxford subcortical labels 1, 3, 12,
          14 on the functional grid, eroded by one voxel, minus anything within two voxels of
          cortex), each voxel z-scored within the run. Every voxel is residualised on its run's
          set plus an intercept. The released responses are only motion-corrected, detrended and
          z-scored, and share a run-locked component across subjects that this removes. The
          predictors are the cleaned cortical voxels (Harvard-Oxford cortex), z-scored over the
          training stories and reduced to 100 principal components fitted on the training stories;
          the test story is projected onto that basis. Writes ``<work>/prep/<subject>.npz``.
``fit``   (per subject and voxel chunk) The design is the other seven subjects' components side by
          side (700 columns, lag 0: everyone is on the same clock), z-scored with the training
          statistics; the target's voxels (all of them) are cleaned with their own runs' nuisance
          sets; then the same voxelwise ridge and held-out r as the stimulus models. Controls:
          ``shifted`` (the predictors' test story rolled by half) and ``crossstory`` (the
          predictors' rows for another, training story, aligned from the run start) should both
          give r near 0; what survives them is locked to the run, not the story.

The rating ablation (``ablation_components`` and ``fit_ablation``) removes the rating from the
predictors: the orthogonalised rating at lags 0-4 plus an intercept is regressed out of every
predictor subject's cleaned cortical voxels (fitted on the training stories), each predictor PCA is
refitted on what is left, and the target is fitted on those components. Δr = r_full - r_removed.
The optional null replaces the rating with random semantic directions of the English1000
features, kept when they remove as much cortical-voxel variance as the rating (``select_directions``).

Needs the registration (``func_to_anat.mat``) and the Harvard-Oxford atlases in each subject's
anatomical space (``<atlases>/<subject>/HarvardOxford-{cort,sub}_space-T1.nii.gz``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .ablation import DELAYS, lag_matrix, lagged_design, least_squares, orthogonalise
from .dataset import SUBJECTS, TEST_STORY, Dataset, common_finite
from .encoding import (chunk_bounds, design_from_blocks, fit_voxelwise, noise_ceiling, normalise, save_chunk)
from .registration import column_labels

K = 100
N_NUISANCE = 5
WM_CSF = (1, 3, 12, 14)     # Harvard-Oxford subcortical: cerebral white matter L/R, lateral ventricles L/R
SUB_CORTEX = (2, 13)        # Harvard-Oxford subcortical: cerebral cortex L/R


def atlas_path(atlases: Path, subject: str, which: str) -> Path:
    return Path(atlases) / subject / f"HarvardOxford-{which}_space-T1.nii.gz"


def cortical_columns(dataset: Dataset, subject: str, atlases: Path, registration: Path) -> np.ndarray:
    return column_labels(dataset, subject, atlas_path(atlases, subject, "cort"), registration) > 0


def nuisance_columns(dataset: Dataset, subject: str, atlases: Path, registration: Path,
                     erode: int = 1, margin: int = 2) -> np.ndarray:
    """aCompCor source columns: white matter and lateral ventricles, eroded, away from cortex."""
    from scipy import ndimage
    order = dataset.column_volume(subject)
    inside = order > 0
    sub = column_labels(dataset, subject, atlas_path(atlases, subject, "sub"), registration)
    cort = column_labels(dataset, subject, atlas_path(atlases, subject, "cort"), registration)

    def volume(columns):
        out = np.zeros(order.shape, dtype=bool)
        out[inside] = columns[order[inside] - 1]
        return out
    source = volume(np.isin(sub, WM_CSF))
    if erode:
        source = ndimage.binary_erosion(source, iterations=erode)
    cortex = volume((cort > 0) | np.isin(sub, SUB_CORTEX))
    if margin:
        cortex = ndimage.binary_dilation(cortex, iterations=margin)
    keep = source & ~cortex & inside
    columns = np.zeros(dataset.n_columns(subject), dtype=bool)
    columns[order[keep] - 1] = True
    return columns


def nuisance_set(response: np.ndarray, source: np.ndarray, n: int = N_NUISANCE) -> np.ndarray:
    """Top ``n`` principal time courses of a run's source columns, each z-scored within the run."""
    block = response[:, source]
    block = block[:, np.isfinite(block).all(axis=0)]
    sd = block.std(axis=0)
    block = (block[:, sd > 1e-6] - block[:, sd > 1e-6].mean(axis=0)) / sd[sd > 1e-6]
    u, s, _ = np.linalg.svd(block.astype(np.float64), full_matrices=False)
    return (u[:, :n] * s[:n]).astype(np.float32)


def residualise(response: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    design = np.column_stack([nuisance.astype(np.float64), np.ones(nuisance.shape[0])])
    coef, *_ = np.linalg.lstsq(design, np.nan_to_num(response.astype(np.float64)), rcond=None)
    return (response - design @ coef).astype(np.float32)


def pca_blocks(cleaned: dict[str, np.ndarray], stories: list[str], k: int = K, dtype=np.float32) -> dict[str, np.ndarray]:
    """z-score over the training stories, PCA (seed 2023) fitted there; returns train_<story> and test."""
    from sklearn.decomposition import PCA
    lengths = [cleaned[s].shape[0] for s in stories]
    stacked = np.vstack([cleaned[s] for s in stories])
    mean, sd = stacked.mean(axis=0), stacked.std(axis=0)
    sd[sd < 1e-6] = 1.0
    pca = PCA(n_components=k, svd_solver="randomized", random_state=2023).fit((stacked - mean) / sd)
    reduced = pca.transform((stacked - mean) / sd).astype(dtype)
    test = pca.transform((cleaned[TEST_STORY] - mean) / sd).astype(dtype)
    edges = np.cumsum([0] + lengths)
    out = {f"train_{s}": reduced[edges[i]:edges[i + 1]] for i, s in enumerate(stories)}
    out["test"] = test
    out["explained"] = pca.explained_variance_ratio_.astype(np.float32)
    return out


def prep(dataset: Dataset, subject: str, atlases: Path, registration: Path, work: Path,
         k: int = K, n_nuisance: int = N_NUISANCE) -> Path:
    started = time.time()
    stories = dataset.shared_training_stories()
    n_columns = dataset.n_columns(subject)
    cortical = cortical_columns(dataset, subject, atlases, registration)
    runs = {s: dataset.response(subject, s) for s in stories + [TEST_STORY]}
    valid = common_finite(list(runs.values()))
    source = nuisance_columns(dataset, subject, atlases, registration) & valid
    nuisance = {s: nuisance_set(r, source, n_nuisance) for s, r in runs.items()}
    use = cortical & valid
    cleaned = {s: residualise(r[:, use], nuisance[s]) for s, r in runs.items()}
    del runs
    blocks = pca_blocks(cleaned, stories, k)
    out = Path(work) / "prep" / f"{subject}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, stories=np.array(stories), cortical=cortical, valid=valid, nuisance_source=source,
             **blocks, **{f"nuisance_{s}": v for s, v in nuisance.items()})
    print(f"[{subject}] {use.sum()} cortical predictor columns of {n_columns}; {n_nuisance} nuisance components "
          f"per run from {source.sum()} white-matter/ventricle columns; {k} components explain "
          f"{100 * blocks['explained'].sum():.1f}% ({time.time() - started:.0f} s)", flush=True)
    return out


def cleaned_noise_ceiling(dataset: Dataset, subject: str, work: Path, n_nuisance: int = N_NUISANCE) -> dict[str, np.ndarray]:
    """The noise ceiling of the raw and of the aCompCor-cleaned test repeats (column vectors)."""
    source = np.load(Path(work) / "prep" / f"{subject}.npz")["nuisance_source"]
    repeats = dataset.repeats(subject, TEST_STORY)
    finite = np.isfinite(repeats).all(axis=(0, 1))
    cleaned = np.stack([residualise(r, nuisance_set(r, source & finite, n_nuisance)) for r in repeats])
    out = {}
    for name, data in (("raw", repeats), ("cleaned", cleaned)):
        ceiling = np.full(repeats.shape[2], np.nan, dtype=np.float32)
        ceiling[finite] = noise_ceiling(data[:, :, finite])
        out[name] = ceiling
        dataset.save_map(subject, ceiling, Path(work) / "noise_ceiling" / subject / f"noise_ceiling_{name}.nii.gz")
    return out


def _target(dataset: Dataset, subject: str, own, stories: list[str], start: int, stop: int):
    """The target's cleaned training and test responses for one chunk, and its valid columns."""
    train = [residualise(dataset.response(subject, s, start, stop), own[f"nuisance_{s}"]) for s in stories]
    test = residualise(dataset.response(subject, TEST_STORY, start, stop), own[f"nuisance_{TEST_STORY}"])
    repeats = dataset.repeats(subject, TEST_STORY, start, stop)
    valid = common_finite([*train, test, repeats]) & own["valid"][start:stop]
    return [r[:, valid] for r in train], test[:, valid], repeats[:, :, valid], valid


def fit(dataset: Dataset, subject: str, work: Path, chunk: int = 0, n_chunks: int = 5,
        control: str = "none", k: int = K) -> Path:
    started = time.time()
    others = [s for s in SUBJECTS if s != subject]
    preps = {s: np.load(Path(work) / "prep" / f"{s}.npz") for s in SUBJECTS}
    stories = [str(s) for s in preps[subject]["stories"]]
    train_blocks = [np.hstack([preps[s][f"train_{story}"] for s in others]) for story in stories]
    test_block = np.hstack([preps[s]["test"] for s in others])
    if control == "shifted":
        test_block = np.roll(test_block, test_block.shape[0] // 2, axis=0)
    elif control == "crossstory":
        n = test_block.shape[0]
        other = next(s for s in stories if preps[others[0]][f"train_{s}"].shape[0] >= n)
        test_block = np.hstack([preps[s][f"train_{other}"][:n] for s in others])
    elif control != "none":
        raise ValueError(f"unknown control {control}")
    design = design_from_blocks(train_blocks, test_block)
    start, stop = chunk_bounds(dataset.n_columns(subject), chunk, n_chunks)
    train, test, repeats, valid = _target(dataset, subject, preps[subject], stories, start, stop)
    correlation, alpha = fit_voxelwise(design, stories, train, test, subject)
    ceiling = noise_ceiling(repeats)   # the released (uncleaned) repeats
    model = f"xsub_pc{k}" + ("" if control == "none" else f"_{control}")
    arrays = {"correlation_raw": correlation, "correlation_noise_ceiling_normalized": normalise(correlation, ceiling),
              "noise_ceiling": ceiling, "selected_alpha": alpha}
    meta = {"subject": subject, "model": model, "control": control, "k": k, "voxel_start": start, "voxel_stop": stop,
            "n_columns": dataset.n_columns(subject), "mean_r": float(np.nanmean(correlation)),
            "seconds": round(time.time() - started)}
    path = save_chunk(Path(work) / "fits" / model / subject / "chunks", chunk, n_chunks, start + np.flatnonzero(valid), arrays, meta)
    print(f"[{subject} {model} chunk {chunk}] mean r {np.nanmean(correlation):.4f} ({time.time() - started:.0f} s)", flush=True)
    return path


# ---- the rating ablation -------------------------------------------------------------------------
class RatingSeries:
    """The orthogonalised rating per story, and (for the optional null) random semantic directions."""

    def __init__(self, stories: list[str], regressor_dir: Path, feature_dir: Path | None = None):
        self.fit = list(stories)
        self.stories = self.fit + [TEST_STORY]
        regressors = {s: np.load(Path(regressor_dir) / f"{s}.npy").astype(np.float64) for s in self.stories}
        self.rating = orthogonalise(regressors, self.fit)
        self.covariates = {s: np.column_stack([lag_matrix(regressors[s][:, 1]), lag_matrix(self.rating[s]),
                                               np.ones(self.rating[s].size)]) for s in self.stories}
        self.C = np.vstack([self.covariates[s] for s in self.fit])
        if feature_dir is not None:
            self.F = {s: np.load(Path(feature_dir) / f"{s}.npy").astype(np.float64) for s in self.stories}
            self.F_mean = np.vstack([self.F[s] for s in self.fit]).mean(axis=0)

    def null(self, w: np.ndarray) -> dict[str, np.ndarray]:
        """s = (F - mean) w, orthogonalised on word rate (lags 0-4), the rating (lags 0-4) and an intercept."""
        raw = {s: (self.F[s] - self.F_mean) @ w for s in self.stories}
        beta = least_squares(self.C, np.concatenate([raw[s] for s in self.fit]))
        return {s: raw[s] - self.covariates[s] @ beta for s in self.stories}

    def design(self, series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {s: lagged_design(series[s]) for s in self.stories}


def cleaned_cortical(dataset: Dataset, subject: str, work: Path, stories: list[str], columns=None) -> dict[str, np.ndarray]:
    prep_file = np.load(Path(work) / "prep" / f"{subject}.npz")
    index = np.flatnonzero(prep_file["cortical"] & prep_file["valid"])
    if columns is not None:
        index = index[columns]
    return {s: residualise(dataset.response(subject, s)[:, index], prep_file[f"nuisance_{s}"]) for s in stories + [TEST_STORY]}


def remove_series(Z: dict[str, np.ndarray], D: dict[str, np.ndarray], fit_stories: list[str]):
    """Z with D's span removed (fitted on the training stories), and each column's R^2 there."""
    coef = least_squares(np.vstack([D[s] for s in fit_stories]), np.vstack([Z[s] for s in fit_stories]).astype(np.float64))
    out = {s: (Z[s] - D[s] @ coef).astype(np.float32) for s in Z}
    train, left = np.vstack([Z[s] for s in fit_stories]), np.vstack([out[s] for s in fit_stories])
    total = train.var(axis=0)
    r2 = np.where(total > 0, 1 - left.var(axis=0) / np.where(total > 0, total, 1), 0.0)
    return out, r2.astype(np.float32)


def ablation_components(dataset: Dataset, subject: str, work: Path, name: str, regressor_dir: Path,
                        ids=(-1,), directions: Path | None = None, feature_dir: Path | None = None, k: int = K) -> None:
    """For each series id (-1: the rating; 0, 1, ...: null directions), the predictor components of
    ``subject`` with that series removed: ``<work>/ablation/<name>/components/<subject>/<id>.npz``."""
    started = time.time()
    stories = [str(s) for s in np.load(Path(work) / "prep" / f"{subject}.npz")["stories"]]
    series = RatingSeries(stories, regressor_dir, feature_dir)
    w = np.load(directions)["w"] if directions is not None else None
    Z = cleaned_cortical(dataset, subject, work, stories)
    out = Path(work) / "ablation" / name / "components" / subject
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        target = out / f"{i}.npz"
        if target.exists():
            continue
        s = series.rating if i == -1 else series.null(w[i].astype(np.float64))
        left, _ = remove_series(Z, series.design(s), series.fit)
        blocks = pca_blocks(left, stories, k, dtype=np.float16)
        blocks.pop("explained")
        np.savez(target, **blocks)
        print(f"[{subject}] components without series {i} ({time.time() - started:.0f} s)", flush=True)


def fit_ablation(dataset: Dataset, subject: str, work: Path, name: str, chunk: int = 0, n_chunks: int = 5,
                 ids=(-1,), checkpoint: int = 25) -> Path:
    """The target fitted on the other subjects' components with each series removed:
    ``<work>/ablation/<name>/fits/<subject>/chunk_<c>_of_<n>/series_<first>-<last>.npz``
    (``correlation`` rows per series id, ``series_id``, ``voxel_index``)."""
    started = time.time()
    others = [s for s in SUBJECTS if s != subject]
    own = np.load(Path(work) / "prep" / f"{subject}.npz")
    stories = [str(s) for s in own["stories"]]
    start, stop = chunk_bounds(dataset.n_columns(subject), chunk, n_chunks)
    train, test, _, valid = _target(dataset, subject, own, stories, start, stop)
    folder = Path(work) / "ablation" / name / "fits" / subject / f"chunk_{chunk:03d}_of_{n_chunks:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    ids = list(ids)
    target = folder / f"series_{ids[0]}-{ids[-1]}.npz"
    partial = target.with_suffix(".partial.npz")
    done, rows = [], []
    if partial.exists():
        saved = np.load(partial)
        done, rows = list(saved["series_id"]), list(saved["correlation"])
    index = (start + np.flatnonzero(valid)).astype(np.int64)
    components = Path(work) / "ablation" / name / "components"
    for i in ids:
        if i in done:
            continue
        comps = {s: np.load(components / s / f"{i}.npz") for s in others}
        blocks = [np.hstack([comps[s][f"train_{x}"].astype(np.float32) for s in others]) for x in stories]
        test_block = np.hstack([comps[s]["test"].astype(np.float32) for s in others])
        correlation, _ = fit_voxelwise(design_from_blocks(blocks, test_block), stories, train, test, subject)
        done.append(i)
        rows.append(correlation.astype(np.float32))
        if len(done) % checkpoint == 0:
            np.savez(partial, correlation=np.stack(rows), series_id=np.array(done), voxel_index=index)
    np.savez(target, correlation=np.stack(rows).astype(np.float32), series_id=np.array(done), voxel_index=index)
    partial.unlink(missing_ok=True)
    print(f"[{subject} chunk {chunk}] series {ids[0]}..{ids[-1]} ({time.time() - started:.0f} s)", flush=True)
    return target


def read_ablation(folder: Path, n_columns: int, ids) -> dict[int, np.ndarray]:
    """Every chunk's fits under ``fits/<subject>`` as column vectors per series id."""
    out = {i: np.full(n_columns, np.nan, dtype=np.float32) for i in ids}
    for path in Path(folder).glob("chunk_*/series_*.npz"):
        if ".partial" in path.name:
            continue
        data = np.load(path)
        for i, row in zip(data["series_id"].tolist(), data["correlation"]):
            if int(i) in out:
                out[int(i)][data["voxel_index"]] = row
    return out


def select_directions(dataset: Dataset, work: Path, regressor_dir: Path, feature_dir: Path, output: Path,
                      n: int = 1000, tolerance: float = 0.10, sample: int = 2000, seed: int = 0,
                      max_candidates: int = 100000) -> dict:
    """Random English1000 directions that remove as much predictor-voxel variance as the rating.

    The share a series removes from the cortical voxels of all subjects together (``sample`` random
    columns each) is trace((D'D)^-1 (D'Z)(D'Z)') / trace(Z'Z), D the series' centred lags 0-4 over
    the training stories; a direction is kept when |share / share_rating - 1| <= tolerance."""
    started = time.time()
    stories = [str(s) for s in np.load(Path(work) / "prep" / f"{SUBJECTS[0]}.npz")["stories"]]
    series = RatingSeries(stories, regressor_dir, feature_dir)
    rng = np.random.default_rng(seed)
    stacked = {}
    for subject in SUBJECTS:
        prep_file = np.load(Path(work) / "prep" / f"{subject}.npz")
        n_use = int((prep_file["cortical"] & prep_file["valid"]).sum())
        pick = np.sort(rng.choice(n_use, size=min(sample, n_use), replace=False))
        Z = cleaned_cortical(dataset, subject, work, stories, pick)
        train = np.vstack([Z[x] for x in series.fit]).astype(np.float64)
        stacked[subject] = train - train.mean(axis=0)
    total = sum(float((z ** 2).sum()) for z in stacked.values())

    def share(s):
        D = np.vstack([lag_matrix(s[x], DELAYS) for x in series.fit])
        D -= D.mean(axis=0)
        gram_inv = np.linalg.inv(D.T @ D + 1e-8 * np.eye(D.shape[1]))
        return sum(float(np.einsum("ij,ik,jk->", gram_inv, D.T @ z, D.T @ z)) for z in stacked.values()) / total

    f_rating = share(series.rating)
    accepted, fractions, drawn = [], [], 0
    while len(accepted) < n and drawn < max_candidates:
        w = rng.standard_normal(series.F_mean.size)
        w /= np.linalg.norm(w)
        drawn += 1
        f = share(series.null(w))
        if abs(f / f_rating - 1) <= tolerance:
            accepted.append(w.astype(np.float32))
            fractions.append(f)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, w=np.stack(accepted), fraction=np.array(fractions), rating_fraction=f_rating,
             tolerance=tolerance, candidates_drawn=drawn, sample=sample)
    summary = {"rating_fraction": f_rating, "accepted": len(accepted), "drawn": drawn, "seconds": round(time.time() - started)}
    print(json.dumps(summary), flush=True)
    return summary
