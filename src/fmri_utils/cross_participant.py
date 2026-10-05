"""Cross-participant encoding: predict a subject's voxels from other subjects' brains.

When every subject hears or sees the same stimulus, the others' responses predict a subject's
response with everything shared, not only what a feature space names. Works on a ``RunTable``
(responses as column matrices per run); runs with the same id share a stimulus.

``prep``  (per subject) Each run (the training runs every subject has, and the test run) gets an
          aCompCor nuisance set: the top ``n_nuisance`` principal time courses of that run's
          nuisance-source columns (each z-scored within the run). Every column is residualised on
          its run's set plus an intercept: released responses are often only motion-corrected,
          detrended and z-scored, and can share a run-locked component across subjects that this
          removes. The predictor columns (cortex, typically) are then z-scored over the training
          runs and reduced to ``k`` principal components fitted there; the test run is projected
          onto that basis. Writes ``<work>/prep/<subject>.npz``.
``fit``   (per subject and column chunk) The design is the other subjects' components side by
          side (lag 0: everyone is on the same clock), z-scored with the training statistics; the
          target's columns are cleaned with their own runs' nuisance sets; then voxelwise ridge with
          a fixed test run (``encoding.columns``). Controls: ``shifted`` (the predictors' test run
          rolled by half) and ``crossstory`` (the predictors' rows of another, training run, from its
          start) should give r near 0; what survives them is locked to the run, not the stimulus.

``acompcor_source`` picks the nuisance-source columns from atlas labels per column: white matter
and ventricle labels, eroded, minus anything within a margin of cortex.

**Ablation.** ``ablation_components`` and ``fit_ablation`` remove an annotation from the predictors:
the orthogonalised annotation (``ablation.orthogonalise``) at lags 0-4 plus an intercept is regressed
out of every predictor subject's cleaned predictor columns (fitted on the training runs), each PCA is
refitted on what is left, and the target is refitted on those components: delta r = r_full - r_removed.
The optional null (``select_directions``) replaces the annotation by random semantic directions of a
feature space, kept when they remove as much predictor-column variance as the annotation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .ablation import DELAYS, lag_matrix, lagged_design, least_squares, orthogonalise
from .encoding.columns import WIDE_ALPHAS, chunk_bounds, fit_voxelwise, save_chunk
from .encoding.design import design_from_blocks
from .encoding.noise_ceiling import noise_ceiling, normalise
from .encoding.run_table import RunTable, common_finite


def acompcor_source(table: RunTable, subject: str, source_labels: np.ndarray, cortex_labels: np.ndarray,
                    erode: int = 1, margin: int = 2) -> np.ndarray:
    """Columns for the nuisance set: ``source_labels`` (bool per column: e.g. white matter and
    ventricles) eroded by ``erode`` voxels on the functional grid, minus anything within ``margin``
    voxels of ``cortex_labels`` (bool per column)."""
    from scipy import ndimage
    order = table.column_volume(subject)
    inside = order > 0

    def volume(columns):
        out = np.zeros(order.shape, dtype=bool)
        out[inside] = np.asarray(columns, dtype=bool)[order[inside] - 1]
        return out
    source = volume(source_labels)
    if erode:
        source = ndimage.binary_erosion(source, iterations=erode)
    cortex = volume(cortex_labels)
    if margin:
        cortex = ndimage.binary_dilation(cortex, iterations=margin)
    keep = source & ~cortex & inside
    columns = np.zeros(table.n_columns(subject), dtype=bool)
    columns[order[keep] - 1] = True
    return columns


def nuisance_set(response: np.ndarray, source: np.ndarray, n: int = 5) -> np.ndarray:
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


def pca_blocks(cleaned: dict[str, np.ndarray], runs: Sequence[str], test_run: str, k: int = 100, dtype=np.float32) -> dict:
    """z-score over the training runs, PCA (seed 2023) fitted there: ``train_<run>``, ``test``, ``explained``."""
    from sklearn.decomposition import PCA
    lengths = [cleaned[r].shape[0] for r in runs]
    stacked = np.vstack([cleaned[r] for r in runs])
    mean, sd = stacked.mean(axis=0), stacked.std(axis=0)
    sd[sd < 1e-6] = 1.0
    pca = PCA(n_components=k, svd_solver="randomized", random_state=2023).fit((stacked - mean) / sd)
    reduced = pca.transform((stacked - mean) / sd).astype(dtype)
    edges = np.cumsum([0] + lengths)
    out = {f"train_{r}": reduced[edges[i]:edges[i + 1]] for i, r in enumerate(runs)}
    out["test"] = pca.transform((cleaned[test_run] - mean) / sd).astype(dtype)
    out["explained"] = pca.explained_variance_ratio_.astype(np.float32)
    return out


def prep(table: RunTable, subject: str, work: Path, predictor: np.ndarray, source: np.ndarray,
         runs: Sequence[str] | None = None, k: int = 100, n_nuisance: int = 5) -> Path:
    """Clean every run and build the predictor components (``predictor``/``source``: bool per column)."""
    started = time.time()
    runs = list(runs or table.shared_runs())
    test_run = table.test_run(subject)
    responses = {r: table.response(subject, r) for r in runs + [test_run]}
    valid = common_finite(list(responses.values()))
    source = np.asarray(source, dtype=bool) & valid
    nuisance = {r: nuisance_set(x, source, n_nuisance) for r, x in responses.items()}
    use = np.asarray(predictor, dtype=bool) & valid
    cleaned = {r: residualise(x[:, use], nuisance[r]) for r, x in responses.items()}
    del responses
    blocks = pca_blocks(cleaned, runs, test_run, k)
    out = Path(work) / "prep" / f"{subject}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, runs=np.array(runs), test_run=test_run, predictor=np.asarray(predictor, dtype=bool), valid=valid,
             nuisance_source=source, **blocks, **{f"nuisance_{r}": v for r, v in nuisance.items()})
    print(f"[{subject}] {use.sum()} predictor columns; {n_nuisance} nuisance components per run from {source.sum()} "
          f"columns; {k} components explain {100 * blocks['explained'].sum():.1f}% ({time.time() - started:.0f} s)", flush=True)
    return out


def _load_prep(work: Path, subject: str):
    data = np.load(Path(work) / "prep" / f"{subject}.npz")
    return data, [str(r) for r in data["runs"]], str(data["test_run"])


def cleaned_noise_ceiling(table: RunTable, subject: str, work: Path, n_nuisance: int = 5) -> dict[str, np.ndarray]:
    """The noise ceiling of the raw and of the aCompCor-cleaned test repeats (column vectors)."""
    data, _, test_run = _load_prep(work, subject)
    repeats = table.repeats(subject, test_run)
    if repeats is None:
        raise ValueError(f"{subject}: the run table has no repeats for {test_run}")
    finite = np.isfinite(repeats).all(axis=(0, 1))
    cleaned = np.stack([residualise(r, nuisance_set(r, data["nuisance_source"] & finite, n_nuisance)) for r in repeats])
    out = {}
    for name, values in (("raw", repeats), ("cleaned", cleaned)):
        ceiling = np.full(repeats.shape[2], np.nan, dtype=np.float32)
        ceiling[finite] = noise_ceiling(values[:, :, finite])
        out[name] = ceiling
        table.save_map(subject, ceiling, Path(work) / "noise_ceiling" / subject / f"noise_ceiling_{name}.nii.gz")
    return out


def _target(table: RunTable, subject: str, own, runs, test_run, start: int, stop: int):
    train = [residualise(table.response(subject, r, start, stop), own[f"nuisance_{r}"]) for r in runs]
    test = residualise(table.response(subject, test_run, start, stop), own[f"nuisance_{test_run}"])
    repeats = table.repeats(subject, test_run, start, stop)
    valid = common_finite([*train, test, repeats]) & own["valid"][start:stop]
    return [r[:, valid] for r in train], test[:, valid], None if repeats is None else repeats[:, :, valid], valid


def fit(table: RunTable, subject: str, work: Path, chunk: int = 0, n_chunks: int = 5, control: str = "none",
        subjects: Sequence[str] | None = None, alphas=WIDE_ALPHAS) -> Path:
    started = time.time()
    subjects = list(subjects or table.subjects())
    others = [s for s in subjects if s != subject]
    preps = {s: _load_prep(work, s)[0] for s in subjects}
    _, runs, test_run = _load_prep(work, subject)
    train_blocks = [np.hstack([preps[s][f"train_{r}"] for s in others]) for r in runs]
    test_block = np.hstack([preps[s]["test"] for s in others])
    if control == "shifted":
        test_block = np.roll(test_block, test_block.shape[0] // 2, axis=0)
    elif control == "crossstory":
        n = test_block.shape[0]
        other = next(r for r in runs if preps[others[0]][f"train_{r}"].shape[0] >= n)
        test_block = np.hstack([preps[s][f"train_{other}"][:n] for s in others])
    elif control != "none":
        raise ValueError(f"unknown control {control}")
    start, stop = chunk_bounds(table.n_columns(subject), chunk, n_chunks)
    train, test, repeats, valid = _target(table, subject, preps[subject], runs, test_run, start, stop)
    correlation, alpha = fit_voxelwise(design_from_blocks(train_blocks, test_block), runs, train, test, test_run, subject, alphas)
    k = preps[subject]["test"].shape[1]
    model = f"xsub_pc{k}" + ("" if control == "none" else f"_{control}")
    arrays = {"correlation_raw": correlation, "selected_alpha": alpha}
    if repeats is not None:   # the released (uncleaned) repeats
        ceiling = noise_ceiling(repeats)
        arrays.update(noise_ceiling=ceiling, correlation_noise_ceiling_normalized=normalise(correlation, ceiling))
    meta = {"subject": subject, "model": model, "control": control, "voxel_start": start, "voxel_stop": stop,
            "n_columns": table.n_columns(subject), "mean_r": float(np.nanmean(correlation)), "seconds": round(time.time() - started)}
    path = save_chunk(Path(work) / "fits" / model / subject / "chunks", chunk, n_chunks, start + np.flatnonzero(valid), arrays, meta)
    print(f"[{subject} {model} chunk {chunk}] mean r {np.nanmean(correlation):.4f} ({time.time() - started:.0f} s)", flush=True)
    return path


# ---- the ablation --------------------------------------------------------------------------------
class AnnotationSeries:
    """The orthogonalised annotation per run, and (for the optional null) random semantic directions."""

    def __init__(self, runs: Sequence[str], test_run: str, regressor_dir: Path, feature_dir: Path | None = None):
        self.fit = list(runs)
        self.runs = self.fit + [test_run]
        regressors = {r: np.load(Path(regressor_dir) / f"{r}.npy").astype(np.float64) for r in self.runs}
        self.annotation = orthogonalise(regressors, self.fit)
        self.covariates = {r: np.column_stack([lag_matrix(regressors[r][:, 1]), lag_matrix(self.annotation[r]),
                                               np.ones(self.annotation[r].size)]) for r in self.runs}
        self.C = np.vstack([self.covariates[r] for r in self.fit])
        if feature_dir is not None:
            self.F = {r: np.load(Path(feature_dir) / f"{r}.npy").astype(np.float64) for r in self.runs}
            self.F_mean = np.vstack([self.F[r] for r in self.fit]).mean(axis=0)

    def null(self, w: np.ndarray) -> dict[str, np.ndarray]:
        raw = {r: (self.F[r] - self.F_mean) @ w for r in self.runs}
        beta = least_squares(self.C, np.concatenate([raw[r] for r in self.fit]))
        return {r: raw[r] - self.covariates[r] @ beta for r in self.runs}

    def design(self, series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {r: lagged_design(series[r]) for r in self.runs}


def cleaned_predictors(table: RunTable, subject: str, work: Path, columns=None) -> dict[str, np.ndarray]:
    data, runs, test_run = _load_prep(work, subject)
    index = np.flatnonzero(data["predictor"] & data["valid"])
    if columns is not None:
        index = index[columns]
    return {r: residualise(table.response(subject, r)[:, index], data[f"nuisance_{r}"]) for r in runs + [test_run]}


def remove_series(Z: dict[str, np.ndarray], D: dict[str, np.ndarray], fit_runs: Sequence[str]):
    """Z with D's span removed (fitted on the training runs), and each column's R^2 there."""
    coef = least_squares(np.vstack([D[r] for r in fit_runs]), np.vstack([Z[r] for r in fit_runs]).astype(np.float64))
    out = {r: (Z[r] - D[r] @ coef).astype(np.float32) for r in Z}
    train, left = np.vstack([Z[r] for r in fit_runs]), np.vstack([out[r] for r in fit_runs])
    total = train.var(axis=0)
    r2 = np.where(total > 0, 1 - left.var(axis=0) / np.where(total > 0, total, 1), 0.0)
    return out, r2.astype(np.float32)


def ablation_components(table: RunTable, subject: str, work: Path, name: str, regressor_dir: Path, ids=(-1,),
                        directions: Path | None = None, feature_dir: Path | None = None) -> None:
    """For each series id (-1: the annotation; 0, 1, ...: null directions), the subject's predictor
    components with that series removed: ``<work>/ablation/<name>/components/<subject>/<id>.npz``."""
    started = time.time()
    data, runs, test_run = _load_prep(work, subject)
    k = data["test"].shape[1]
    series = AnnotationSeries(runs, test_run, regressor_dir, feature_dir)
    w = np.load(directions)["w"] if directions is not None else None
    Z = cleaned_predictors(table, subject, work)
    out = Path(work) / "ablation" / name / "components" / subject
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        target = out / f"{i}.npz"
        if target.exists():
            continue
        s = series.annotation if i == -1 else series.null(w[i].astype(np.float64))
        left, _ = remove_series(Z, series.design(s), series.fit)
        blocks = pca_blocks(left, runs, test_run, k, dtype=np.float16)
        blocks.pop("explained")
        np.savez(target, **blocks)
        print(f"[{subject}] components without series {i} ({time.time() - started:.0f} s)", flush=True)


def fit_ablation(table: RunTable, subject: str, work: Path, name: str, chunk: int = 0, n_chunks: int = 5, ids=(-1,),
                 subjects: Sequence[str] | None = None, checkpoint: int = 25, alphas=WIDE_ALPHAS) -> Path:
    """The target refitted on the others' components with each series removed:
    ``<work>/ablation/<name>/fits/<subject>/chunk_<c>_of_<n>/series_<first>-<last>.npz``."""
    started = time.time()
    others = [s for s in (subjects or table.subjects()) if s != subject]
    own, runs, test_run = _load_prep(work, subject)
    start, stop = chunk_bounds(table.n_columns(subject), chunk, n_chunks)
    train, test, _, valid = _target(table, subject, own, runs, test_run, start, stop)
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
        blocks = [np.hstack([comps[s][f"train_{r}"].astype(np.float32) for s in others]) for r in runs]
        test_block = np.hstack([comps[s]["test"].astype(np.float32) for s in others])
        correlation, _ = fit_voxelwise(design_from_blocks(blocks, test_block), runs, train, test, test_run, subject, alphas)
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


def select_directions(table: RunTable, work: Path, regressor_dir: Path, feature_dir: Path, output: Path,
                      n: int = 1000, tolerance: float = 0.10, sample: int = 2000, seed: int = 0, max_candidates: int = 100000,
                      subjects: Sequence[str] | None = None) -> dict:
    """Random semantic directions of ``feature_dir`` that remove as much predictor-column variance as
    the annotation, from all subjects' predictors together (``sample`` random columns each): the share
    a series removes is trace((D'D)^-1 (D'Z)(D'Z)') / trace(Z'Z), D its centred lags over the training runs."""
    started = time.time()
    subjects = list(subjects or table.subjects())
    _, runs, test_run = _load_prep(work, subjects[0])
    series = AnnotationSeries(runs, test_run, regressor_dir, feature_dir)
    rng = np.random.default_rng(seed)
    stacked = {}
    for subject in subjects:
        data = _load_prep(work, subject)[0]
        n_use = int((data["predictor"] & data["valid"]).sum())
        pick = np.sort(rng.choice(n_use, size=min(sample, n_use), replace=False))
        Z = cleaned_predictors(table, subject, work, pick)
        train = np.vstack([Z[r] for r in series.fit]).astype(np.float64)
        stacked[subject] = train - train.mean(axis=0)
    total = sum(float((z ** 2).sum()) for z in stacked.values())

    def share(s):
        D = np.vstack([lag_matrix(s[r], DELAYS) for r in series.fit])
        D -= D.mean(axis=0)
        gram_inv = np.linalg.inv(D.T @ D + 1e-8 * np.eye(D.shape[1]))
        return sum(float(np.einsum("ij,ik,jk->", gram_inv, D.T @ z, D.T @ z)) for z in stacked.values()) / total

    f_annotation = share(series.annotation)
    accepted, fractions, drawn = [], [], 0
    while len(accepted) < n and drawn < max_candidates:
        w = rng.standard_normal(series.F_mean.size)
        w /= np.linalg.norm(w)
        drawn += 1
        f = share(series.null(w))
        if abs(f / f_annotation - 1) <= tolerance:
            accepted.append(w.astype(np.float32))
            fractions.append(f)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, w=np.stack(accepted), fraction=np.array(fractions), annotation_fraction=f_annotation,
             tolerance=tolerance, candidates_drawn=drawn, sample=sample)
    summary = {"annotation_fraction": f_annotation, "accepted": len(accepted), "drawn": drawn, "seconds": round(time.time() - started)}
    print(json.dumps(summary), flush=True)
    return summary


def _ids(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            out.extend(range(int(a), int(b)))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None) -> None:
    """fmri-cross-participant columns | prep | fit | noise-ceiling | components | ablation-fit | select-directions"""
    import argparse
    parser = argparse.ArgumentParser(prog="fmri-cross-participant", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("columns", help="predictor and aCompCor-source columns from atlases in each anatomy")
    p.add_argument("--run-table", type=Path, required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--cortical-atlas", type=Path, required=True, help="labels above 0 are cortex (the predictors)")
    p.add_argument("--subcortical-atlas", type=Path, required=True)
    p.add_argument("--func-to-anat", type=Path, required=True, help="FLIRT-convention functional-to-anatomy matrix")
    p.add_argument("--source-labels", default="1,3,12,14",
                   help="subcortical labels of the nuisance source (Harvard-Oxford: white matter, lateral ventricles)")
    p.add_argument("--cortex-labels", default="2,13", help="subcortical labels that also count as cortex (Harvard-Oxford)")
    p.add_argument("--erode", type=int, default=1)
    p.add_argument("--margin", type=int, default=2)
    p.add_argument("--output", type=Path, required=True, help="writes <output>/<subject>.npz (predictor, source)")
    for name in ("prep", "fit", "noise-ceiling", "components", "ablation-fit"):
        p = sub.add_parser(name)
        p.add_argument("--run-table", type=Path, required=True)
        p.add_argument("--subject", required=True)
        p.add_argument("--work", type=Path, required=True)
        if name == "prep":
            p.add_argument("--columns", type=Path, required=True, help="the columns step's output folder")
            p.add_argument("--k", type=int, default=100)
            p.add_argument("--n-nuisance", type=int, default=5)
        if name in ("fit", "ablation-fit"):
            p.add_argument("--chunk", type=int, default=0)
            p.add_argument("--n-chunks", type=int, default=5)
        if name == "fit":
            p.add_argument("--control", choices=("none", "shifted", "crossstory"), default="none")
        if name in ("components", "ablation-fit"):
            p.add_argument("--name", required=True)
            p.add_argument("--ids", default="-1", help="-1: the annotation; 0:1000: null directions")
        if name == "components":
            p.add_argument("--regressors", type=Path, required=True)
            p.add_argument("--directions", type=Path, default=None)
            p.add_argument("--null-features", type=Path, default=None, help="folder of <run>.npy for the null")
    p = sub.add_parser("select-directions")
    p.add_argument("--run-table", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--regressors", type=Path, required=True)
    p.add_argument("--null-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    args = parser.parse_args(argv)
    table = RunTable(args.run_table)
    if args.command == "columns":
        from .fsl_transforms import column_labels
        cort = column_labels(table, args.subject, args.cortical_atlas, args.func_to_anat)
        subc = column_labels(table, args.subject, args.subcortical_atlas, args.func_to_anat)
        cortex = (cort > 0) | np.isin(subc, _ids(args.cortex_labels))
        source = acompcor_source(table, args.subject, np.isin(subc, _ids(args.source_labels)), cortex, args.erode, args.margin)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez(args.output / f"{args.subject}.npz", predictor=cort > 0, source=source)
        print(f"[{args.subject}] {int((cort > 0).sum())} predictor columns, {int(source.sum())} source columns", flush=True)
    elif args.command == "prep":
        columns = np.load(args.columns / f"{args.subject}.npz")
        prep(table, args.subject, args.work, columns["predictor"], columns["source"], k=args.k, n_nuisance=args.n_nuisance)
    elif args.command == "fit":
        fit(table, args.subject, args.work, args.chunk, args.n_chunks, args.control)
    elif args.command == "noise-ceiling":
        cleaned_noise_ceiling(table, args.subject, args.work)
    elif args.command == "components":
        ablation_components(table, args.subject, args.work, args.name, args.regressors, _ids(args.ids),
                            args.directions, args.null_features)
    elif args.command == "ablation-fit":
        fit_ablation(table, args.subject, args.work, args.name, args.chunk, args.n_chunks, _ids(args.ids))
    elif args.command == "select-directions":
        select_directions(table, args.work, args.regressors, args.null_features, args.output, n=args.n)


if __name__ == "__main__":
    main()
