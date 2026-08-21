from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .config import EncodingConfig
from .features import build_lagged_features, load_feature_matrix
from .preprocessing import residualize_run


REQUIRED_MANIFEST_COLUMNS = {
    "subject_id",
    "run_id",
    "bold_path",
    "mask_path",
    "features_path",
}


@dataclass(frozen=True)
class RunSpec:
    subject_id: str
    run_id: str
    bold_path: Path
    mask_path: Path
    features_path: Path
    confounds_path: Optional[Path] = None
    task: str = ""
    bold_start: int = 0
    feature_start: int = 0
    n_samples: Optional[int] = None
    preprocessing_source: str = ""


@dataclass
class EncodingRun:
    run_id: str
    features: np.ndarray
    bold: np.ndarray
    nuisance: np.ndarray
    source_rows: np.ndarray


@dataclass
class SubjectData:
    subject_id: str
    runs: List[EncodingRun]
    mask: np.ndarray
    reference_image: nib.spatialimages.SpatialImage
    voxel_indices: np.ndarray
    input_provenance: Optional[List[Dict[str, object]]] = None


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_run_manifest(path: Path, subject_id: Optional[str] = None) -> List[RunSpec]:
    """Read the preprocessing-neutral run manifest used by the fitter."""

    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Run manifest is missing columns: {}".format(sorted(missing)))
        rows = list(reader)

    specs = []
    for row in rows:
        if subject_id and row["subject_id"] != subject_id:
            continue
        empty_required = [
            name for name in REQUIRED_MANIFEST_COLUMNS if not str(row.get(name, "")).strip()
        ]
        if empty_required:
            raise ValueError(
                "Manifest row has empty required values for {}: {}".format(
                    row.get("run_id", "<unknown>"), sorted(empty_required)
                )
            )
        confounds_value = str(row.get("confounds_path", "")).strip()
        n_samples_value = str(row.get("n_samples", "")).strip()
        specs.append(
            RunSpec(
                subject_id=row["subject_id"],
                run_id=row["run_id"],
                task=str(row.get("task", "")),
                bold_path=_resolve_path(row["bold_path"], path),
                mask_path=_resolve_path(row["mask_path"], path),
                features_path=_resolve_path(row["features_path"], path),
                confounds_path=(
                    _resolve_path(confounds_value, path) if confounds_value else None
                ),
                bold_start=int(row.get("bold_start", 0) or 0),
                feature_start=int(row.get("feature_start", 0) or 0),
                n_samples=int(n_samples_value) if n_samples_value else None,
                preprocessing_source=str(row.get("preprocessing_source", "")).strip(),
            )
        )
    if not specs:
        raise ValueError("No manifest rows matched subject {}".format(subject_id or "*"))
    run_keys = [(spec.subject_id, spec.run_id) for spec in specs]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Manifest contains duplicate subject_id/run_id rows")
    return sorted(specs, key=lambda item: item.run_id)


def _same_grid(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
) -> bool:
    return first.shape[:3] == second.shape[:3] and np.allclose(
        first.affine, second.affine, atol=1.0e-4
    )


def _build_common_mask(
    specs: Sequence[RunSpec],
    config: EncodingConfig,
) -> Tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    mask_images = [nib.load(str(spec.mask_path)) for spec in specs]
    reference = mask_images[0]
    for image in mask_images[1:]:
        if not _same_grid(reference, image):
            raise ValueError(
                "Run masks do not share a grid. Resample runs to a common space or "
                "supply a preprocessing adapter that does so."
            )

    masks = [np.asanyarray(image.dataobj) > 0 for image in mask_images]
    if config.mask_strategy == "run_mask_union":
        common = np.logical_or.reduce(masks)
    elif config.mask_strategy == "run_mask_intersection":
        common = np.logical_and.reduce(masks)
    else:
        analysis_image = nib.load(str(Path(str(config.analysis_mask_path)).expanduser()))
        if not _same_grid(reference, analysis_image):
            raise ValueError("analysis_mask_path does not match the BOLD spatial grid")
        common = np.asanyarray(analysis_image.dataobj) > 0
        reference = analysis_image
    if not np.any(common):
        raise ValueError("The selected analysis mask is empty")
    return common, reference


def _select_nuisance(
    path: Optional[Path],
    requested_columns: Tuple[str, ...],
    n_rows: int,
) -> np.ndarray:
    if not requested_columns:
        return np.empty((n_rows, 0), dtype=np.float32)
    if path is None:
        raise ValueError("Nuisance columns were requested but confounds_path is empty")
    frame = pd.read_csv(path, sep="\t")
    selected = []
    for pattern in requested_columns:
        matches = [name for name in frame.columns if fnmatch.fnmatchcase(name, pattern)]
        if not matches and not any(char in pattern for char in "*?["):
            raise ValueError("Missing nuisance column '{}' in {}".format(pattern, path))
        selected.extend(name for name in matches if name not in selected)
    if not selected:
        raise ValueError("No nuisance columns matched {}".format(requested_columns))
    nuisance = (
        frame[selected]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if nuisance.shape[0] != n_rows:
        raise ValueError(
            "Confound rows ({}) do not match BOLD rows ({}) in {}".format(
                nuisance.shape[0], n_rows, path
            )
        )
    return nuisance


def load_subject_data(
    specs: Sequence[RunSpec],
    config: EncodingConfig,
) -> SubjectData:
    """Load aligned features, masked BOLD, and nuisance arrays for one subject."""

    config.validate()
    if not specs:
        raise ValueError("At least one run is required")
    subjects = {spec.subject_id for spec in specs}
    if len(subjects) != 1:
        raise ValueError("load_subject_data expects rows for exactly one subject")

    common_mask, reference_image = _build_common_mask(specs, config)
    voxel_indices = np.flatnonzero(common_mask.ravel(order="C"))
    runs = []
    input_provenance = []
    for spec in specs:
        for path in (spec.bold_path, spec.mask_path, spec.features_path):
            if not path.exists():
                raise FileNotFoundError(path)
        bold_image = nib.load(str(spec.bold_path))
        if not _same_grid(reference_image, bold_image):
            raise ValueError("BOLD grid does not match the selected mask: {}".format(spec.bold_path))
        bold_volume = np.asanyarray(bold_image.dataobj, dtype=np.float32)
        if bold_volume.ndim != 4:
            raise ValueError("BOLD image must be 4D: {}".format(spec.bold_path))
        bold = bold_volume[common_mask, :].T
        features, feature_valid = load_feature_matrix(spec.features_path)
        nuisance = _select_nuisance(
            spec.confounds_path,
            config.nuisance_columns,
            bold.shape[0],
        )
        if not np.all(np.isfinite(bold)):
            raise ValueError("BOLD contains non-finite values in run {}".format(spec.run_id))
        if nuisance.shape[1] or config.add_run_intercept:
            bold = residualize_run(
                bold,
                nuisance,
                add_intercept=config.add_run_intercept,
            )

        available = min(bold.shape[0] - spec.bold_start, features.shape[0] - spec.feature_start)
        n_samples = available if spec.n_samples is None else spec.n_samples
        if n_samples <= 0 or n_samples > available:
            raise ValueError(
                "Invalid aligned sample count for {}: requested {}, available {}".format(
                    spec.run_id, n_samples, available
                )
            )
        bold_rows = np.arange(spec.bold_start, spec.bold_start + n_samples)
        feature_rows = np.arange(spec.feature_start, spec.feature_start + n_samples)
        run_bold = np.asarray(bold[bold_rows], dtype=np.float32)
        run_nuisance = np.asarray(nuisance[bold_rows], dtype=np.float32)
        aligned_features = np.asarray(features[feature_rows], dtype=np.float32)
        aligned_valid = np.asarray(feature_valid[feature_rows], dtype=bool)
        lagged, lag_valid = build_lagged_features(
            aligned_features,
            config.lags,
            mode=config.lag_mode,
            source_valid=aligned_valid,
        )
        keep = lag_valid & np.all(np.isfinite(run_bold), axis=1)
        if np.count_nonzero(keep) < 4:
            raise ValueError("Too few valid aligned rows in run {}".format(spec.run_id))

        run_bold = run_bold[keep]
        run_nuisance = run_nuisance[keep]
        lagged = lagged[keep]
        runs.append(
            EncodingRun(
                run_id=spec.run_id,
                features=lagged,
                bold=run_bold,
                nuisance=run_nuisance,
                source_rows=bold_rows[keep],
            )
        )
        input_provenance.append(
            {
                "run_id": spec.run_id,
                "task": spec.task,
                "preprocessing_source": spec.preprocessing_source,
                "bold_path": str(spec.bold_path),
                "mask_path": str(spec.mask_path),
                "features_path": str(spec.features_path),
                "confounds_path": (
                    str(spec.confounds_path) if spec.confounds_path is not None else None
                ),
                "bold_start": spec.bold_start,
                "feature_start": spec.feature_start,
                "n_aligned_samples": int(n_samples),
                "n_model_samples": int(np.count_nonzero(keep)),
            }
        )
    feature_widths = {run.features.shape[1] for run in runs}
    voxel_widths = {run.bold.shape[1] for run in runs}
    if len(feature_widths) != 1:
        raise ValueError("Feature width differs across runs: {}".format(sorted(feature_widths)))
    if len(voxel_widths) != 1:
        raise ValueError("Masked BOLD voxel count differs across runs")
    return SubjectData(
        subject_id=next(iter(subjects)),
        runs=runs,
        mask=common_mask,
        reference_image=reference_image,
        voxel_indices=voxel_indices,
        input_provenance=input_provenance,
    )


_BIDS_TASK = re.compile(r"(?:^|_)task-([^_]+)")
_BIDS_RUN = re.compile(r"(?:^|_)run-([^_]+)")
_BIDS_SUBJECT = re.compile(r"(?:^|_)(sub-[^_]+)")


def _entity(pattern: re.Pattern, name: str, default: str = "") -> str:
    match = pattern.search(name)
    return match.group(1) if match else default


def discover_fmriprep_runs(
    fmriprep_root: Path,
    *,
    task: Optional[str] = None,
    space: Optional[str] = "T1w",
    feature_template: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Discover preprocessed BOLD runs and matching masks/confounds.

    ``feature_template`` may use ``{subject_id}``, ``{task}``, ``{run}``, and
    ``{run_id}``. Discovery intentionally does not infer stimulus alignment.
    """

    root = Path(fmriprep_root).resolve()
    pattern = "*_desc-preproc_bold.nii.gz"
    rows = []
    for bold_path in sorted(root.rglob(pattern)):
        name = bold_path.name
        if space and "_space-{}_".format(space) not in name:
            continue
        subject_id = _entity(_BIDS_SUBJECT, name)
        task_name = _entity(_BIDS_TASK, name)
        run = _entity(_BIDS_RUN, name, "1")
        if not subject_id or not task_name or (task and task_name != task):
            continue
        mask_path = Path(str(bold_path).replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"))
        confound_stem = re.sub(r"_space-[^_]+", "", name)
        confound_name = confound_stem.replace(
            "_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv"
        )
        confounds_path = bold_path.parent / confound_name
        if not mask_path.exists():
            raise FileNotFoundError("Missing fMRIPrep mask for {}".format(bold_path))
        if not confounds_path.exists():
            raise FileNotFoundError("Missing fMRIPrep confounds for {}".format(bold_path))
        run_id = "task-{}_run-{}".format(task_name, run)
        features_path = ""
        if feature_template:
            features_path = feature_template.format(
                subject_id=subject_id,
                subject=subject_id.replace("sub-", ""),
                task=task_name,
                run=run,
                run_id=run_id,
            )
        rows.append(
            {
                "subject_id": subject_id,
                "task": task_name,
                "run_id": run_id,
                "bold_path": str(bold_path),
                "mask_path": str(mask_path),
                "confounds_path": str(confounds_path),
                "features_path": features_path,
                "bold_start": "0",
                "feature_start": "0",
                "n_samples": "",
                "preprocessing_source": "fmriprep",
            }
        )
    if not rows:
        raise FileNotFoundError(
            "No fMRIPrep BOLD files found under {} for task={} space={}".format(
                root, task, space
            )
        )
    return rows


def write_run_manifest(rows: Iterable[Dict[str, str]], output_path: Path) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty run manifest")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
