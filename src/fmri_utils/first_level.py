from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from scipy.stats import norm

from .image_io import save_float32_image


@dataclass(frozen=True)
class FirstLevelConfig:
    """Configuration for one run-level Nilearn GLM."""

    t_r: float
    hrf_model: str = "spm"
    drift_model: str | None = "cosine"
    high_pass: float | None = 0.01
    drift_order: int = 1
    noise_model: str = "ar1"
    slice_time_ref: float = 0.5
    smoothing_fwhm: float | None = None
    signal_scaling: int | bool = 0
    standardize: bool = False
    minimize_memory: bool = False
    n_jobs: int = 1


@dataclass(frozen=True)
class FirstLevelContrastOutputs:
    effect_size: Path
    effect_variance: Path
    stat: Path
    z_score: Path
    p_value: Path


@dataclass(frozen=True)
class FirstLevelOutputs:
    output_dir: Path
    design_matrix_csv: Path
    design_matrix_png: Path
    contrasts: dict[str, FirstLevelContrastOutputs]


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"onset", "duration", "trial_type"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Events are missing columns: {sorted(missing)}")
    result = events.copy()
    result["onset"] = pd.to_numeric(result["onset"], errors="raise")
    result["duration"] = pd.to_numeric(result["duration"], errors="raise")
    if not np.isfinite(result[["onset", "duration"]].to_numpy(float)).all():
        raise ValueError("Events contain non-finite onset or duration values")
    if (result["duration"] <= 0).any():
        raise ValueError("Every event duration must be positive")
    result["trial_type"] = result["trial_type"].astype(str)
    return result


def _validate_confounds(confounds: pd.DataFrame | None, n_scans: int) -> pd.DataFrame | None:
    if confounds is None:
        return None
    if len(confounds) != n_scans:
        raise ValueError(f"Confound rows ({len(confounds)}) do not match scans ({n_scans})")
    result = confounds.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Confounds contain non-finite values")
    if "constant" in result.columns or "intercept" in result.columns:
        raise ValueError("Do not provide an intercept; Nilearn adds a run intercept")
    return result


def fit_first_level_run(
    run_img: str | Path | nib.spatialimages.SpatialImage,
    events: pd.DataFrame,
    contrasts: Mapping[str, str | np.ndarray],
    output_dir: str | Path,
    *,
    config: FirstLevelConfig,
    confounds: pd.DataFrame | None = None,
    mask_img: str | Path | nib.spatialimages.SpatialImage | None = None,
    subject_label: str | None = None,
    run_label: str | None = None,
    overwrite: bool = False,
) -> FirstLevelOutputs:
    """Fit one run-level voxelwise GLM and save complete contrast statistics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(run_img)) if isinstance(run_img, (str, Path)) else run_img
    if len(image.shape) != 4:
        raise ValueError(f"run_img must be 4D; got shape {image.shape}")
    checked_events = _validate_events(events)
    checked_confounds = _validate_confounds(confounds, image.shape[3])

    model = FirstLevelModel(
        t_r=float(config.t_r),
        slice_time_ref=float(config.slice_time_ref),
        hrf_model=config.hrf_model,
        drift_model=config.drift_model,
        high_pass=config.high_pass,
        drift_order=int(config.drift_order),
        mask_img=mask_img,
        smoothing_fwhm=config.smoothing_fwhm,
        standardize=bool(config.standardize),
        signal_scaling=config.signal_scaling,
        noise_model=config.noise_model,
        minimize_memory=bool(config.minimize_memory),
        n_jobs=int(config.n_jobs),
        subject_label=subject_label,
    ).fit(image, events=checked_events, confounds=checked_confounds)

    design = model.design_matrices_[0]
    design_csv = output_dir / "design_matrix.csv"
    design_png = output_dir / "design_matrix.png"
    design.to_csv(design_csv, index=True)
    figure, axis = plt.subplots(figsize=(max(9, 0.42 * len(design.columns)), 6.5))
    plot_design_matrix(design, rescale=False, axes=axis)
    axis.set_title(run_label or "First-level design matrix")
    figure.tight_layout()
    figure.savefig(design_png, dpi=180, bbox_inches="tight")
    plt.close(figure)

    contrast_outputs: dict[str, FirstLevelContrastOutputs] = {}
    for name, definition in contrasts.items():
        contrast_dir = output_dir / "contrasts" / name
        contrast_dir.mkdir(parents=True, exist_ok=True)
        paths = FirstLevelContrastOutputs(
            effect_size=contrast_dir / "effect_size.nii.gz",
            effect_variance=contrast_dir / "effect_variance.nii.gz",
            stat=contrast_dir / "stat_t.nii.gz",
            z_score=contrast_dir / "z_score.nii.gz",
            p_value=contrast_dir / "p_value.nii.gz",
        )
        if overwrite or not all(Path(path).exists() for path in asdict(paths).values()):
            maps = model.compute_contrast(definition, output_type="all")
            for key, path in asdict(paths).items():
                save_float32_image(maps[key], path)
        contrast_outputs[name] = paths

    metadata = {
        "subject_label": subject_label,
        "run_label": run_label,
        "run_image": str(run_img) if isinstance(run_img, (str, Path)) else "in_memory",
        "n_scans": int(image.shape[3]),
        "n_events": int(len(checked_events)),
        "event_types": sorted(checked_events["trial_type"].unique().tolist()),
        "confound_columns": [] if checked_confounds is None else checked_confounds.columns.tolist(),
        "config": asdict(config),
        "contrasts": {
            name: definition.tolist() if isinstance(definition, np.ndarray) else definition
            for name, definition in contrasts.items()
        },
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    checked_events.to_csv(output_dir / "events.tsv", sep="\t", index=False)
    if checked_confounds is not None:
        checked_confounds.to_csv(output_dir / "confounds.tsv", sep="\t", index=False)
    return FirstLevelOutputs(output_dir, design_csv, design_png, contrast_outputs)


def combine_fixed_effects(
    effect_maps: Sequence[str | Path],
    variance_maps: Sequence[str | Path],
    output_dir: str | Path,
    *,
    mask_img: str | Path | nib.spatialimages.SpatialImage | None = None,
    precision_weighted: bool = True,
) -> FirstLevelContrastOutputs:
    """Combine run-level contrasts into one subject map with fixed effects."""

    if len(effect_maps) != len(variance_maps) or not effect_maps:
        raise ValueError("effect_maps and variance_maps must have equal non-zero lengths")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effect, variance, stat, z_score = compute_fixed_effects(
        [str(path) for path in effect_maps],
        [str(path) for path in variance_maps],
        mask=mask_img,
        precision_weighted=bool(precision_weighted),
    )
    p_data = 2.0 * norm.sf(np.abs(np.asarray(z_score.dataobj, dtype=float)))
    p_value = nib.Nifti1Image(p_data.astype(np.float32), z_score.affine, z_score.header)
    paths = FirstLevelContrastOutputs(
        effect_size=output_dir / "effect_size.nii.gz",
        effect_variance=output_dir / "effect_variance.nii.gz",
        stat=output_dir / "stat_t.nii.gz",
        z_score=output_dir / "z_score.nii.gz",
        p_value=output_dir / "p_value.nii.gz",
    )
    for image, path in zip((effect, variance, stat, z_score, p_value), asdict(paths).values(), strict=True):
        save_float32_image(image, path)
    (output_dir / "fixed_effects_metadata.json").write_text(
        json.dumps(
            {
                "effect_maps": [str(path) for path in effect_maps],
                "variance_maps": [str(path) for path in variance_maps],
                "precision_weighted": bool(precision_weighted),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
