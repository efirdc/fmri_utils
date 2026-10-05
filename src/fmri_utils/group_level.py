from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map

from .image_io import save_float32_image


@dataclass(frozen=True)
class GroupContrastOutputs:
    effect_size: Path
    effect_variance: Path
    stat: Path
    z_score: Path
    p_value: Path
    fdr_thresholded_z: Path
    uncorrected_thresholded_z: Path
    z_score_png: Path
    fdr_thresholded_png: Path
    uncorrected_thresholded_png: Path


@dataclass(frozen=True)
class GroupLevelOutputs:
    output_dir: Path
    design_matrix_csv: Path
    inputs_csv: Path
    contrasts: dict[str, GroupContrastOutputs]


def _validate_geometry(paths: Sequence[Path]) -> None:
    reference = nib.load(str(paths[0]))
    for path in paths[1:]:
        image = nib.load(str(path))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise ValueError(f"Input geometry differs from {paths[0]}: {path}")


def fit_second_level_glm(
    map_paths: Sequence[str | Path],
    design_matrix: pd.DataFrame,
    contrasts: Mapping[str, str | np.ndarray],
    output_dir: str | Path,
    *,
    subject_ids: Sequence[str] | None = None,
    mask_img: str | Path | nib.spatialimages.SpatialImage | None = None,
    smoothing_fwhm: float | None = None,
    fdr_alpha: float = 0.05,
    uncorrected_p: float = 0.001,
    cluster_threshold: int = 0,
    two_sided: bool = True,
    overwrite: bool = False,
) -> GroupLevelOutputs:
    """Fit an explicit Nilearn group GLM and threshold z-statistic maps."""

    paths = [Path(path) for path in map_paths]
    if len(paths) < 2:
        raise ValueError("At least two subject maps are required")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing group inputs: {missing[:5]}")
    if len(design_matrix) != len(paths):
        raise ValueError("Design-matrix rows must match map paths")
    design = design_matrix.apply(pd.to_numeric, errors="raise").astype(float).reset_index(drop=True)
    values = design.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Group design contains non-finite values")
    rank = int(np.linalg.matrix_rank(values))
    if rank != design.shape[1]:
        raise ValueError(f"Group design is rank deficient: rank={rank}, columns={design.shape[1]}")
    _validate_geometry(paths)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "design_matrix.csv"
    inputs_path = output_dir / "inputs.csv"
    design.to_csv(design_path, index=False)
    pd.DataFrame(
        {
            "subject_id": list(subject_ids) if subject_ids is not None else [f"row-{i:03d}" for i in range(len(paths))],
            "map_path": [str(path) for path in paths],
        }
    ).to_csv(inputs_path, index=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"\[NiftiMasker\.fit\] Generation of a mask.*")
        model = SecondLevelModel(
            mask_img=mask_img,
            smoothing_fwhm=smoothing_fwhm,
            minimize_memory=False,
        ).fit([str(path) for path in paths], design_matrix=design)

    outputs: dict[str, GroupContrastOutputs] = {}
    for name, definition in contrasts.items():
        contrast_dir = output_dir / "contrasts" / name
        contrast_dir.mkdir(parents=True, exist_ok=True)
        current = GroupContrastOutputs(
            effect_size=contrast_dir / "effect_size.nii.gz",
            effect_variance=contrast_dir / "effect_variance.nii.gz",
            stat=contrast_dir / "stat_t.nii.gz",
            z_score=contrast_dir / "z_score.nii.gz",
            p_value=contrast_dir / "p_value.nii.gz",
            fdr_thresholded_z=contrast_dir / f"z_fdr_alpha-{fdr_alpha:g}.nii.gz",
            uncorrected_thresholded_z=contrast_dir / f"z_uncorrected_p-{uncorrected_p:g}.nii.gz",
            z_score_png=contrast_dir / "z_score_mosaic.png",
            fdr_thresholded_png=contrast_dir / f"z_fdr_alpha-{fdr_alpha:g}_mosaic.png",
            uncorrected_thresholded_png=contrast_dir / f"z_uncorrected_p-{uncorrected_p:g}_mosaic.png",
        )
        if overwrite or not all(Path(path).exists() for path in asdict(current).values()):
            maps = model.compute_contrast(definition, output_type="all")
            for key in ("effect_size", "effect_variance", "stat", "z_score", "p_value"):
                save_float32_image(maps[key], getattr(current, key))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"The given float value must not exceed.*")
                fdr_map, _ = threshold_stats_img(
                    maps["z_score"], alpha=float(fdr_alpha), height_control="fdr",
                    cluster_threshold=int(cluster_threshold), two_sided=bool(two_sided), mask_img=mask_img,
                )
                unc_map, _ = threshold_stats_img(
                    maps["z_score"], alpha=float(uncorrected_p), height_control="fpr",
                    cluster_threshold=int(cluster_threshold), two_sided=bool(two_sided), mask_img=mask_img,
                )
            save_float32_image(fdr_map, current.fdr_thresholded_z)
            save_float32_image(unc_map, current.uncorrected_thresholded_z)
            for image, path, title in (
                (maps["z_score"], current.z_score_png, f"{name}: unthresholded z"),
                (fdr_map, current.fdr_thresholded_png, f"{name}: FDR q<{fdr_alpha:g}"),
                (unc_map, current.uncorrected_thresholded_png, f"{name}: uncorrected p<{uncorrected_p:g}"),
            ):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="empty mask")
                    display = plot_stat_map(
                        image,
                        display_mode="z",
                        cut_coords=9,
                        colorbar=True,
                        symmetric_cbar=True,
                        cmap="RdBu_r",
                        title=title,
                    )
                display.savefig(path, dpi=180)
                display.close()
        outputs[name] = current

    metadata = {
        "n_subjects": len(paths),
        "design_columns": design.columns.tolist(),
        "design_rank": rank,
        "residual_degrees_of_freedom": len(paths) - rank,
        "contrasts": {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in contrasts.items()
        },
        "thresholding": {
            "input_statistic": "z_score",
            "fdr_alpha": fdr_alpha,
            "uncorrected_p": uncorrected_p,
            "cluster_threshold": cluster_threshold,
            "two_sided": two_sided,
        },
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return GroupLevelOutputs(output_dir, design_path, inputs_path, outputs)
