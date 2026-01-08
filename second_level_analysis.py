from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import warnings
import inspect

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.datasets import load_mni152_template
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel

from .transformations import warp_to_mni_with_fmriprep_transform


@dataclass(frozen=True)
class SecondLevelOutputs:
    out_dir: Path
    t_unc_path: Path
    thresholded_path: Path
    mosaic_png: Path
    # One-sided (directional) outputs, using positive t-values for both directions:
    # - gt: one-sided on t
    # - lt: one-sided on -t (sign-flipped), so output t-values are positive for < direction
    thresholded_gt_path: Path
    thresholded_lt_path: Path
    mosaic_gt_png: Path
    mosaic_lt_png: Path
    used_subject_ids: Optional[List[int]] = None
    warped_maps: Optional[List[Path]] = None


def _voxelwise_count_mean_std(
    imgs: Sequence[Union[str, Path]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute voxelwise count/mean/std across images using a streaming algorithm.

    - Ignores non-finite values (NaN/inf)
    - Does not stack all subjects into memory at once
    """
    if len(imgs) == 0:
        raise ValueError("imgs must be non-empty")

    ref_img = nib.load(str(imgs[0]))
    ref_shape = ref_img.shape
    count = np.zeros(ref_shape, dtype=np.int32)
    mean = np.zeros(ref_shape, dtype=np.float64)
    m2 = np.zeros(ref_shape, dtype=np.float64)

    for p in imgs:
        img = nib.load(str(p))
        data = np.asanyarray(img.dataobj)
        if data.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {p}: {data.shape} != {ref_shape}")
        finite = np.isfinite(data)
        if not np.any(finite):
            continue

        x = data[finite].astype(np.float64, copy=False)
        n1 = count[finite].astype(np.float64, copy=False)
        n = n1 + 1.0
        delta = x - mean[finite]
        mean[finite] = mean[finite] + delta / n
        delta2 = x - mean[finite]
        m2[finite] = m2[finite] + delta * delta2
        count[finite] = n.astype(np.int32)

    std = np.full(ref_shape, np.nan, dtype=np.float64)
    valid = count > 1
    std[valid] = np.sqrt(m2[valid] / (count[valid] - 1))
    return count, mean, std


def second_level_group_ttest_mni(
    *,
    subject_ids: Sequence[int],
    contrast_name: str,
    out_dir: Union[str, Path],
    fmriprep_subject_dir_template: Union[str, Path],
    # Supply exactly one of these:
    subject_map_template: Optional[Union[str, Path]] = None,
    subject_pair_templates: Optional[Tuple[Union[str, Path], Union[str, Path]]] = None,
    subtract_constant: Optional[float] = None,
    interpolation: str = "linear",
    mni_voxel_size: Optional[Sequence[float]] = (2.0, 2.0, 2.0),
    smoothing_fwhm: Optional[float] = None,
    alpha: float = 0.05,
    height_control: str = "fdr",
    cluster_threshold: int = 20,
    cmap: str = "RdBu_r",
    # caching:
    cache_root: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    # artifact filtering:
    min_subjects_per_voxel: Optional[int] = None,
    min_between_subject_std: float = 1e-6,
    max_abs_t: float = 1e4,
    **threshold_kwargs,
) -> SecondLevelOutputs:
    """
    Unified second-level group analysis (one-sample t-test) in MNI space.

    Supports either:
    - per-subject precomputed maps (`subject_map_template.format(subject_id=...)`)
    - per-subject pairs to difference (`subject_pair_templates=(a_tmpl, b_tmpl)`; each `.format(subject_id=...)`)

    Workflow
    --------
    For each subject:
    - obtain unwarped image (either directly or via A-B difference)
    - warp to MNI using fMRIPrep transform (cached)
    - optionally subtract a constant in MNI space (cached)
    Then:
    - fit second-level model and compute t-map (cached)
    - threshold with nilearn `threshold_stats_img`
    - save NIfTIs + mosaic visualization PNGs
    """
    if (subject_map_template is None) == (subject_pair_templates is None):
        raise ValueError("Provide exactly one of subject_map_template or subject_pair_templates.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root_path = Path(cache_root) if cache_root is not None else (out_dir / "cache")
    cache_root_path.mkdir(parents=True, exist_ok=True)
    warped_root = cache_root_path

    warped_maps: List[Path] = []
    used_subject_ids: List[int] = []

    for subject_id in subject_ids:
        fmriprep_subject_dir = Path(str(fmriprep_subject_dir_template).format(subject_id=subject_id))
        subject_name = fmriprep_subject_dir.stem
        subj_out_dir = warped_root / subject_name
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        warped_path = subj_out_dir / f"{contrast_name}.nii.gz"
        centered_path = subj_out_dir / f"{contrast_name}_minus_constant.nii.gz"

        map_path_for_group = warped_path
        if subtract_constant is not None:
            map_path_for_group = centered_path

        if map_path_for_group.exists() and not overwrite:
            warped_maps.append(map_path_for_group)
            used_subject_ids.append(subject_id)
            continue

        # Resolve unwarped input for this subject
        if subject_map_template is not None:
            in_path = Path(str(subject_map_template).format(subject_id=subject_id))
            if not in_path.exists():
                continue
            unwarped_path = in_path
        else:
            a_tmpl, b_tmpl = subject_pair_templates  # type: ignore[misc]
            a_path = Path(str(a_tmpl).format(subject_id=subject_id))
            b_path = Path(str(b_tmpl).format(subject_id=subject_id))
            if not a_path.exists() or not b_path.exists():
                continue
            # Cache unwarped contrast to avoid recomputation if downstream warp is overwritten
            unwarped_path = subj_out_dir / f"{contrast_name}_unwarped.nii.gz"
            if overwrite or not unwarped_path.exists():
                contrast_img = image.math_img("img1 - img2", img1=a_path, img2=b_path)
                contrast_img.to_filename(unwarped_path)

        # Warp (cached)
        if overwrite or not warped_path.exists():
            try:
                warp_to_mni_with_fmriprep_transform(
                    unwarped_path,
                    fmriprep_subject_dir=fmriprep_subject_dir,
                    out_img_path=warped_path,
                    interpolation=interpolation,
                    mni_voxel_size=mni_voxel_size,
                )
            except FileNotFoundError as e:
                warnings.warn(str(e))
                continue
            except RuntimeError as e:
                warnings.warn(str(e))
                continue

        # Optional constant subtraction in MNI space (cached)
        if subtract_constant is not None:
            if overwrite or not centered_path.exists():
                img = image.load_img(str(warped_path))
                data = img.get_fdata()
                # Important: many maps are exactly 0 outside the brain.
                # If we subtract a constant everywhere (e.g. ROC AUC chance 0.5),
                # we turn the background into a constant non-zero value across subjects,
                # which can create absurd t-values at borders. Only subtract where voxel is present.
                present = np.isfinite(data) & (np.abs(data) > 0)
                centered = np.zeros_like(data, dtype=np.float32)
                centered[present] = (data[present] - float(subtract_constant)).astype(
                    np.float32, copy=False
                )
                centered_img = image.new_img_like(
                    img, centered, copy_header=True
                )
                centered_img.to_filename(str(centered_path))

        warped_maps.append(map_path_for_group)
        used_subject_ids.append(subject_id)

    if len(warped_maps) == 0:
        raise ValueError(f"No subject maps found for contrast '{contrast_name}'.")

    # Build a voxelwise validity mask to suppress absurd t-values caused by near-zero variance
    # at coverage borders / resampling artifacts.
    mask_img = None
    if min_subjects_per_voxel is None:
        # Strict default: only test voxels where *all* included subjects contribute.
        # This avoids spurious border effects from partial coverage after warping/resampling.
        min_subjects_per_voxel = int(len(warped_maps))

    if min_subjects_per_voxel > 1:
        count, mean, std = _voxelwise_count_mean_std(warped_maps)
        valid_mask = (count >= int(min_subjects_per_voxel)) & np.isfinite(std) & (std >= float(min_between_subject_std))
        # If the mask would wipe out everything, warn and fall back to no mask
        if int(valid_mask.sum()) == 0:
            warnings.warn(
                f"[second_level] Computed an empty valid_mask for {contrast_name}; "
                "disabling artifact mask. Consider lowering min_subjects_per_voxel/min_between_subject_std."
            )
        else:
            ref_img = image.load_img(str(warped_maps[0]))
            mask_img = image.new_img_like(ref_img, valid_mask.astype(np.uint8), copy_header=True)

    # Group-level GLM (one-sample t-test)
    design_matrix = pd.DataFrame({"intercept": np.ones(len(warped_maps))})
    t_unc_path = out_dir / f"{contrast_name}_tmap_unc.nii.gz"
    if t_unc_path.exists() and not overwrite:
        t_scores_img = image.load_img(str(t_unc_path))
        # If we have a validity mask, enforce it even on cached outputs
        if mask_img is not None:
            t_data = np.asarray(t_scores_img.dataobj)
            m = np.asarray(mask_img.dataobj).astype(bool)
            if t_data.shape == m.shape:
                t_data_masked = np.where(m, t_data, 0.0)
                # Write back the cleaned map so downstream uses the corrected cache.
                t_scores_img = image.new_img_like(t_scores_img, t_data_masked, copy_header=True)
                t_scores_img.to_filename(str(t_unc_path))
    else:
        slm = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=mask_img)
        slm = slm.fit([str(p) for p in warped_maps], design_matrix=design_matrix)
        t_scores_img = slm.compute_contrast(
            second_level_contrast="intercept", output_type="stat"
        )
        # Ensure masked-out voxels are zero (helps downstream thresholding/plotting consistency)
        if mask_img is not None:
            t_data = np.asarray(t_scores_img.dataobj)
            m = np.asarray(mask_img.dataobj).astype(bool)
            if t_data.shape == m.shape:
                t_data = np.where(m, t_data, 0.0)
                t_scores_img = image.new_img_like(t_scores_img, t_data, copy_header=True)
        t_scores_img.to_filename(str(t_unc_path))

    # Last-resort filtering: zero out absurd t-values (typically border/variance artifacts).
    # Zeroing (not clipping) avoids creating artificial high-t clusters.
    if max_abs_t is not None and np.isfinite(max_abs_t):
        t_data = np.asarray(t_scores_img.dataobj)
        too_big = np.isfinite(t_data) & (np.abs(t_data) > float(max_abs_t))
        if np.any(too_big):
            t_scores_img = image.new_img_like(
                t_scores_img, np.where(too_big, 0.0, t_data), copy_header=True
            )
            t_scores_img.to_filename(str(t_unc_path))

    # Ensure we can explicitly request two-sided vs one-sided thresholding in nilearn
    sig = inspect.signature(threshold_stats_img)
    if "two_sided" not in sig.parameters:
        raise TypeError(
            "Your nilearn version does not support threshold_stats_img(..., two_sided=...). "
            "Please upgrade nilearn to use explicit one-sided/two-sided thresholding."
        )

    def _threshold_and_cache(stat_img, *, out_path: Path, two_sided: bool):
        if out_path.exists() and not overwrite:
            thr_img = image.load_img(str(out_path))
        else:
            thr_img, _ = threshold_stats_img(
                stat_img,
                alpha=alpha,
                height_control=height_control,
                cluster_threshold=int(cluster_threshold),
                two_sided=bool(two_sided),
                **threshold_kwargs,
            )
            thr_img.to_filename(str(out_path))
        return thr_img

    # 1) Two-sided thresholded map (main output)
    thresholded_path = out_dir / f"{contrast_name}_tmap_{height_control}_k{int(cluster_threshold)}.nii.gz"
    thresholded_map = _threshold_and_cache(t_scores_img, out_path=thresholded_path, two_sided=True)

    if subtract_constant is not None:
        label = str(subtract_constant).replace(".", "p")
        gt_tag = f"gt{label}"
        lt_tag = f"lt{label}"
    else:
        gt_tag = "gt"
        lt_tag = "lt"

    # 2) One-sided greater-than (positive tail on t)
    thresholded_gt_path = out_dir / f"{contrast_name}_tmap_{height_control}_k{int(cluster_threshold)}_{gt_tag}_onesided.nii.gz"
    thresholded_gt_map = _threshold_and_cache(t_scores_img, out_path=thresholded_gt_path, two_sided=False)

    # 3) One-sided less-than (positive tail on -t), output is positive t-values for < direction
    t_flipped = image.math_img("-img", img=t_scores_img)
    thresholded_lt_path = out_dir / f"{contrast_name}_tmap_{height_control}_k{int(cluster_threshold)}_{lt_tag}_onesided.nii.gz"
    thresholded_lt_map = _threshold_and_cache(t_flipped, out_path=thresholded_lt_path, two_sided=False)

    # Visualizations (single mosaic PNG per map)
    bg_img = load_mni152_template()
    # Write the MNI template NIfTI into the group output folder for convenience.
    mni_template_path = out_dir / "mni152_template_T1w.nii.gz"
    if not mni_template_path.exists():
        try:
            bg_img.to_filename(str(mni_template_path))
        except Exception:
            pass

    def _render_mosaic(stat_img, *, file_stem: str, title: str) -> Path:
        mosaic = out_dir / f"{file_stem}_mosaic.png"
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_img,
            display_mode="mosaic",
            colorbar=True,
            cmap=cmap,
            title=title,
        )
        display.savefig(str(mosaic))
        display.close()
        return mosaic

    base_stem = f"{contrast_name}_tmap_{height_control}_k{int(cluster_threshold)}"
    base_title = f"{contrast_name} ({height_control}, k>={int(cluster_threshold)})"
    mosaic_png = _render_mosaic(thresholded_map, file_stem=base_stem, title=base_title)

    if subtract_constant is not None:
        gt_title = f"{contrast_name} > {subtract_constant} (one-sided, {height_control}, k>={int(cluster_threshold)})"
        lt_title = f"{contrast_name} < {subtract_constant} (one-sided, {height_control}, k>={int(cluster_threshold)})"
    else:
        gt_title = f"{contrast_name} positive (one-sided, {height_control}, k>={int(cluster_threshold)})"
        lt_title = f"{contrast_name} negative (one-sided, {height_control}, k>={int(cluster_threshold)})"

    mosaic_gt_png = _render_mosaic(
        thresholded_gt_map,
        file_stem=f"{base_stem}_{gt_tag}_onesided",
        title=gt_title,
    )
    mosaic_lt_png = _render_mosaic(
        thresholded_lt_map,
        file_stem=f"{base_stem}_{lt_tag}_onesided",
        title=lt_title,
    )

    return SecondLevelOutputs(
        out_dir=out_dir,
        t_unc_path=t_unc_path,
        thresholded_path=thresholded_path,
        mosaic_png=mosaic_png,
        thresholded_gt_path=thresholded_gt_path,
        thresholded_lt_path=thresholded_lt_path,
        mosaic_gt_png=mosaic_gt_png,
        mosaic_lt_png=mosaic_lt_png,
        used_subject_ids=used_subject_ids,
        warped_maps=warped_maps,
    )


