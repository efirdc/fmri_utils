from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import json
import warnings
import inspect

import numpy as np
import pandas as pd
import nibabel as nib

# Nilearn may warn on HPC/headless nodes when it detects a non-interactive backend (e.g. 'agg').
# This warning is noisy but harmless for batch runs that only save figures to disk.
warnings.filterwarnings(
    "ignore",
    message=r"You are using the 'agg' matplotlib backend that is non-interactive\..*",
    category=UserWarning,
)

from nilearn import image, plotting
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import non_parametric_inference
from nilearn.glm.second_level import SecondLevelModel


@dataclass(frozen=True)
class SecondLevelOutputs:
    maps_dir: Path
    group_dir: Path
    template_image_path: Optional[Path]
    t_unc_path: Path
    thresholded_two_sided_path: Path
    mosaic_two_sided_png: Path
    # One-sided (directional) outputs, using positive t-values for both directions:
    # - gt: one-sided on t
    # - lt: one-sided on -t (sign-flipped), so output t-values are positive for < direction
    thresholded_gt_path: Path
    thresholded_lt_path: Path
    mosaic_gt_png: Path
    mosaic_lt_png: Path
    used_map_paths: List[Path]


def _iter_nifti_files(maps_dir: Path) -> List[Path]:
    """Return NIfTI files in maps_dir (non-recursive)."""
    files: List[Path] = []
    for p in sorted(maps_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        files.append(p)
    return files


def _as_binary_mask(mask_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Ensure a mask is binary (0/1) by thresholding > 0.

    This allows passing continuous images (e.g. MNI T1 template) as a mask.
    """
    data = np.asarray(mask_img.dataobj)
    mask = np.isfinite(data) & (data > 0)
    return image.new_img_like(mask_img, mask.astype(np.uint8), copy_header=True)


def _voxelwise_count_mean_std(
    imgs: List[Path],
    *,
    base_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Streaming voxelwise count/mean/std across images (Welford), ignoring non-finite values.

    Parameters
    ----------
    imgs:
        List of NIfTI paths. All images must have identical shape.
    base_mask:
        Optional boolean array restricting which voxels are updated. If provided, only voxels where
        base_mask is True are included in the statistics.
    """
    if len(imgs) == 0:
        raise ValueError("imgs must be non-empty")

    ref_img = nib.load(str(imgs[0]))
    ref_shape = ref_img.shape
    if base_mask is not None and base_mask.shape != ref_shape:
        raise ValueError(f"base_mask shape mismatch: {base_mask.shape} != {ref_shape}")

    count = np.zeros(ref_shape, dtype=np.int32)
    mean = np.zeros(ref_shape, dtype=np.float64)
    m2 = np.zeros(ref_shape, dtype=np.float64)

    for p in imgs:
        img = nib.load(str(p))
        data = np.asanyarray(img.dataobj)
        if data.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {p}: {data.shape} != {ref_shape}")
        finite = np.isfinite(data)
        if base_mask is not None:
            finite &= base_mask
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


def _jsonable(val):
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (set, tuple)):
        return list(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def second_level_one_sample_ttest(
    maps_dir: Union[str, Path],
    *,
    out_dir: Optional[Union[str, Path]] = None,
    smoothing_fwhm: Optional[float] = None,
    template_image: Optional[Union[str, Path, nib.Nifti1Image]] = None,
    mask_image: Optional[Union[str, Path, nib.Nifti1Image]] = None,
    use_mask_in_thresholding: bool = True,
    min_between_subject_std: Optional[float] = None,
    min_subjects_per_voxel: Optional[int] = None,
    inference: str = "parametric",
    n_perm: int = 10000,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    tfce: bool = False,
    cluster_forming_p_threshold: Optional[float] = None,
    alpha: float = 0.05,
    height_control: str = "fdr",
    cluster_threshold: int = 20,
    cmap: str = "RdBu_r",
    overwrite: bool = False,
    **threshold_kwargs,
) -> SecondLevelOutputs:
    """
    Second-level group t-test (one-sample) from a directory of subject-level maps.

    Assumptions
    -----------
    - `maps_dir` contains ONLY the subject-level NIfTI maps you want to enter into the group model.
      (Non-recursive; subfolders are ignored.)
    - Output is written to a `group/` subfolder inside `maps_dir`.
    - Produces:
      - Two-sided thresholded map
      - One-sided gt map (positive tail)
      - One-sided lt map computed by sign-flipping the stat image (so lt map has positive t-values)
    """
    maps_dir = Path(maps_dir)
    if not maps_dir.exists():
        raise FileNotFoundError(f"maps_dir does not exist: {maps_dir}")
    if not maps_dir.is_dir():
        raise NotADirectoryError(f"maps_dir is not a directory: {maps_dir}")

    group_dir = Path(out_dir) if out_dir is not None else (maps_dir / "group")
    group_dir.mkdir(parents=True, exist_ok=True)

    map_paths = _iter_nifti_files(maps_dir)
    if len(map_paths) == 0:
        raise ValueError(f"No subject NIfTI maps found in {maps_dir}")
    if len(map_paths) < 2:
        raise ValueError(f"Need at least 2 maps for a group t-test; found {len(map_paths)} in {maps_dir}")

    # Save run arguments for reproducibility
    args_path = group_dir / "args.json"
    if overwrite or not args_path.exists():
        args_to_save = {
            "maps_dir": maps_dir,
            "out_dir": group_dir,
            "inference": inference,
            "smoothing_fwhm": smoothing_fwhm,
            "template_image": (
                str(template_image)
                if isinstance(template_image, (str, Path))
                else ("in_memory" if template_image is not None else None)
            ),
            "mask_image": (
                mask_image
                if isinstance(mask_image, str)
                else (str(mask_image) if isinstance(mask_image, (str, Path)) else ("in_memory" if mask_image is not None else None))
            ),
            "use_mask_in_thresholding": bool(use_mask_in_thresholding),
            "min_between_subject_std": (
                float(min_between_subject_std) if min_between_subject_std is not None else None
            ),
            "min_subjects_per_voxel": (int(min_subjects_per_voxel) if min_subjects_per_voxel is not None else None),
            "n_perm": int(n_perm),
            "random_state": random_state,
            "n_jobs": int(n_jobs),
            "verbose": int(verbose),
            "tfce": bool(tfce),
            "cluster_forming_p_threshold": cluster_forming_p_threshold,
            "alpha": float(alpha),
            "height_control": height_control,
            "cluster_threshold": int(cluster_threshold),
            "cmap": cmap,
            "overwrite": bool(overwrite),
            "threshold_kwargs": threshold_kwargs,
        }
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump({k: _jsonable(v) for k, v in args_to_save.items()}, f, indent=2)

    # Default mask:
    # - If caller provides a mask, we use it.
    # - Otherwise, we default to the MNI152 *brain mask* (not the T1 template volume).
    # - If caller provides `template_image` and no explicit `mask_image`, we still default
    #   to the MNI brain mask (template_image controls plotting/background, not analysis mask).
    resolved_mask_img: Optional[nib.Nifti1Image] = None
    if mask_image is None:
        resolved_mask_img = load_mni152_brain_mask()
    else:
        if isinstance(mask_image, str) and mask_image.lower() in ("mni", "mni_template", "mni152", "mni_brain", "mni_brain_mask"):
            resolved_mask_img = load_mni152_brain_mask()
        elif isinstance(mask_image, (str, Path)):
            resolved_mask_img = image.load_img(str(mask_image))
        else:
            resolved_mask_img = mask_image
        resolved_mask_img = _as_binary_mask(resolved_mask_img)

    # Ensure the analysis mask matches the map grid to avoid implicit upsampling/downsampling.
    # Nilearn's SecondLevelModel will resample inputs to the mask_img grid if they differ.
    if resolved_mask_img is not None:
        first_img = image.load_img(str(map_paths[0]))
        try:
            sig = inspect.signature(image.resample_to_img)
            kwargs = {"interpolation": "nearest"}
            if "force_resample" in sig.parameters:
                kwargs["force_resample"] = True
            if "copy_header" in sig.parameters:
                kwargs["copy_header"] = True
            resolved_mask_img = image.resample_to_img(resolved_mask_img, first_img, **kwargs)
        except Exception:
            # If resampling isn't available for some reason, continue with the provided mask.
            pass

    # Optional: exclude near-zero between-subject variance voxels (often border/coverage artifacts).
    validity_mask_img: Optional[nib.Nifti1Image] = None
    if min_between_subject_std is not None:
        thr = float(min_between_subject_std)
        if thr < 0:
            raise ValueError("min_between_subject_std must be >= 0")

        base_mask_arr: Optional[np.ndarray] = None
        if resolved_mask_img is not None:
            base_mask_arr = np.asarray(resolved_mask_img.dataobj).astype(bool)

        req_n = int(min_subjects_per_voxel) if min_subjects_per_voxel is not None else int(len(map_paths))
        if req_n < 1:
            raise ValueError("min_subjects_per_voxel must be >= 1")

        count, _, std = _voxelwise_count_mean_std(map_paths, base_mask=base_mask_arr)
        valid = (count >= req_n) & np.isfinite(std) & (std >= thr)
        if base_mask_arr is not None and base_mask_arr.shape == valid.shape:
            valid &= base_mask_arr

        first_img = image.load_img(str(map_paths[0]))
        validity_mask_img = image.new_img_like(first_img, valid.astype(np.uint8), copy_header=True)
        resolved_mask_img = validity_mask_img

    # Group-level GLM (one-sample t-test)
    design_matrix = pd.DataFrame({"intercept": np.ones(len(map_paths))})
    t_unc_path = group_dir / "tmap_unc.nii.gz"
    if t_unc_path.exists() and not overwrite:
        t_scores_img = image.load_img(str(t_unc_path))
    else:
        slm = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=resolved_mask_img)
        slm = slm.fit([str(p) for p in map_paths], design_matrix=design_matrix)
        t_scores_img = slm.compute_contrast("intercept", output_type="stat")
        t_scores_img.to_filename(str(t_unc_path))

    inference = str(inference).lower().strip()
    if inference not in ("parametric", "non_parametric"):
        raise ValueError("inference must be 'parametric' or 'non_parametric'")

    sig = inspect.signature(threshold_stats_img)
    if "two_sided" not in sig.parameters:
        raise TypeError(
            "Your nilearn version does not support threshold_stats_img(..., two_sided=...). "
            "Please upgrade nilearn."
        )

    def _threshold_and_cache(
        stat_img,
        *,
        out_path: Path,
        two_sided: bool,
        mask_img: Optional[nib.Nifti1Image] = None,
    ):
        if out_path.exists() and not overwrite:
            return image.load_img(str(out_path))
        kwargs_local = {
            "alpha": alpha,
            "height_control": height_control,
            "cluster_threshold": int(cluster_threshold),
            "two_sided": bool(two_sided),
            **threshold_kwargs,
        }
        if bool(use_mask_in_thresholding) and mask_img is not None:
            if "mask_img" in sig.parameters:
                kwargs_local["mask_img"] = mask_img
            else:
                raise TypeError(
                    "Your nilearn threshold_stats_img does not support mask_img=. "
                    "Upgrade nilearn to enable tail-masked one-sided thresholding."
                )
        thr_img, _ = threshold_stats_img(stat_img, **kwargs_local)
        thr_img.to_filename(str(out_path))
        return thr_img

    # Thresholding outputs
    if inference == "parametric":
        alpha_tag = str(alpha).replace(".", "p")
        # --- Save masks used for analysis / QC ---
        masks_dir = group_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        if resolved_mask_img is not None:
            resolved_mask_img.to_filename(str(masks_dir / "analysis_mask_resolved.nii.gz"))
        if validity_mask_img is not None:
            validity_mask_img.to_filename(str(masks_dir / "analysis_mask_validity_std.nii.gz"))

        t_data = np.asarray(t_scores_img.dataobj, dtype=float)
        finite = np.isfinite(t_data)
        gt_mask_arr = finite & (t_data > 0)
        lt_mask_arr = finite & (t_data < 0)
        if resolved_mask_img is not None:
            m = np.asarray(resolved_mask_img.dataobj).astype(bool)
            if m.shape == t_data.shape:
                gt_mask_arr &= m
                lt_mask_arr &= m
        # These tail masks are for QC/inspection only. We do NOT use them for inference correction,
        # because selecting voxels based on the sign of the observed t-statistic is data-driven.
        gt_mask_img = image.new_img_like(t_scores_img, gt_mask_arr.astype(np.uint8), copy_header=True)
        lt_mask_img = image.new_img_like(t_scores_img, lt_mask_arr.astype(np.uint8), copy_header=True)
        gt_mask_img.to_filename(str(masks_dir / "tail_mask_gt_tpos.nii.gz"))
        lt_mask_img.to_filename(str(masks_dir / "tail_mask_lt_tneg.nii.gz"))

        thresholded_two_sided_path = group_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}.nii.gz"
        thresholded_two_sided_map = _threshold_and_cache(
            t_scores_img,
            out_path=thresholded_two_sided_path,
            two_sided=True,
            mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
        )
        thresholded_gt_path = group_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}_gt_onesided.nii.gz"
        thresholded_gt_map = _threshold_and_cache(
            # One-sided correction is performed over the same analysis mask as the two-sided test.
            # (Directional inference changes the tail, not the voxel inclusion set.)
            t_scores_img,
            out_path=thresholded_gt_path,
            two_sided=False,
            mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
        )
        t_flipped = image.math_img("-img", img=t_scores_img)
        thresholded_lt_path = group_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}_lt_onesided.nii.gz"
        thresholded_lt_map = _threshold_and_cache(
            t_flipped,
            out_path=thresholded_lt_path,
            two_sided=False,
            mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
        )
        base_stem = f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}"
        base_title = f"t-map ({height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
        gt_title = f"t-map > 0 (one-sided, {height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
        lt_title = f"t-map < 0 (one-sided, {height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
    else:
        def _np_logp(two_sided_test: bool, second_level_contrast):
            return non_parametric_inference(
                second_level_input=[str(p) for p in map_paths],
                design_matrix=design_matrix,
                second_level_contrast=second_level_contrast,
                mask=resolved_mask_img,
                smoothing_fwhm=smoothing_fwhm,
                model_intercept=False,
                n_perm=int(n_perm),
                two_sided_test=bool(two_sided_test),
                random_state=random_state,
                n_jobs=int(n_jobs),
                verbose=int(verbose),
                threshold=cluster_forming_p_threshold,
                tfce=bool(tfce),
            )

        def _extract_t(ret):
            if isinstance(ret, dict):
                if "t" in ret:
                    return ret["t"]
                return None
            return None

        def _select_logp(ret):
            """
            Pick the appropriate corrected logp image depending on the inference mode:
            - TFCE: use logp_max_tfce when available
            - Cluster-size inference (threshold not None): use logp_max_size when available
            - Otherwise: default to max-T FWER logp_max_t
            """
            if not isinstance(ret, dict):
                return ret
            if bool(tfce) and "logp_max_tfce" in ret:
                return ret["logp_max_tfce"]
            if cluster_forming_p_threshold is not None and "logp_max_size" in ret:
                return ret["logp_max_size"]
            if "logp_max_t" in ret:
                return ret["logp_max_t"]
            keys = sorted(ret.keys())
            raise ValueError(
                "Unsupported non_parametric_inference outputs; expected one of "
                "logp_max_t / logp_max_size / logp_max_tfce. "
                f"Got keys: {keys}"
            )

        logp_thr = float(-np.log10(alpha))
        alpha_tag = str(alpha).replace(".", "p")
        tfce_tag = "_tfce" if bool(tfce) else ""
        if cluster_forming_p_threshold is None:
            cfp_tag = ""
        else:
            cfp_tag = f"_cfp{str(cluster_forming_p_threshold).replace('.', 'p')}"

        # Save raw non-parametric outputs for QC (even if thresholded maps end up empty).
        raw_dir = group_dir / "nonparametric_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # gt (one-sided)
        logp_gt_ret = _np_logp(False, [1.0])
        logp_gt_img = _select_logp(logp_gt_ret)
        t_gt_img = _extract_t(logp_gt_ret) or t_scores_img
        try:
            logp_gt_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_gt.nii.gz"))
        except Exception:
            pass
        if t_gt_img is None:
            t_gt_img = t_scores_img
        gt_mask = np.asarray(logp_gt_img.dataobj) >= logp_thr
        t_gt_data = np.asarray(t_gt_img.dataobj)
        thresholded_gt_map = image.new_img_like(
            t_gt_img, np.where(gt_mask & (t_gt_data > 0), t_gt_data, 0.0), copy_header=True
        )
        thresholded_gt_path = group_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_gt_onesided.nii.gz"
        thresholded_gt_map.to_filename(str(thresholded_gt_path))

        # Two-sided
        logp_two_ret = _np_logp(True, [1.0])
        logp_two_img = _select_logp(logp_two_ret)
        t_two_img = _extract_t(logp_two_ret) or t_scores_img
        try:
            logp_two_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_two_sided.nii.gz"))
        except Exception:
            pass
        if t_two_img is None:
            t_two_img = t_scores_img
        two_mask = np.asarray(logp_two_img.dataobj) >= logp_thr
        t_two_data = np.asarray(t_two_img.dataobj)
        thresholded_two_sided_map = image.new_img_like(
            t_two_img, np.where(two_mask, t_two_data, 0.0), copy_header=True
        )
        thresholded_two_sided_path = group_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}.nii.gz"
        thresholded_two_sided_map.to_filename(str(thresholded_two_sided_path))

        
        # lt (one-sided on negative), output positive values
        logp_lt_ret = _np_logp(False, [-1.0])
        logp_lt_img = _select_logp(logp_lt_ret)
        t_lt_img = _extract_t(logp_lt_ret) or t_scores_img
        try:
            logp_lt_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_lt.nii.gz"))
        except Exception:
            pass
        if t_lt_img is None:
            t_lt_img = t_scores_img
        lt_mask = np.asarray(logp_lt_img.dataobj) >= logp_thr
        t_lt_data = np.asarray(t_lt_img.dataobj)
        thresholded_lt_map = image.new_img_like(
            t_lt_img, np.where(lt_mask & (t_lt_data < 0), -t_lt_data, 0.0), copy_header=True
        )
        thresholded_lt_path = group_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_lt_onesided.nii.gz"
        thresholded_lt_map.to_filename(str(thresholded_lt_path))

        base_stem = f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}"
        base_title = f"t-map (vfwe, alpha={alpha})"
        gt_title = f"t-map > 0 (one-sided, vfwe, alpha={alpha})"
        lt_title = f"t-map < 0 (one-sided, vfwe, alpha={alpha})"

    # Template image for plotting (and convenience output)
    template_image_path: Optional[Path] = None
    if template_image is None:
        bg_img = load_mni152_template()
        template_image_path = group_dir / "mni152_template_T1w.nii.gz"
        if not template_image_path.exists():
            try:
                bg_img.to_filename(str(template_image_path))
            except Exception:
                template_image_path = None
    else:
        if isinstance(template_image, (str, Path)):
            bg_img = image.load_img(str(template_image))
            template_image_path = Path(template_image)
        else:
            bg_img = template_image
            template_image_path = None

    def _render_mosaic(stat_img, *, file_stem: str, title: str) -> Path:
        mosaic = group_dir / f"{file_stem}_mosaic.png"
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

    mosaic_two_sided_png = _render_mosaic(thresholded_two_sided_map, file_stem=base_stem, title=base_title)
    mosaic_gt_png = _render_mosaic(
        thresholded_gt_map, file_stem=f"{base_stem}_gt_onesided", title=gt_title
    )
    mosaic_lt_png = _render_mosaic(
        thresholded_lt_map, file_stem=f"{base_stem}_lt_onesided", title=lt_title
    )

    return SecondLevelOutputs(
        maps_dir=maps_dir,
        group_dir=group_dir,
        template_image_path=template_image_path,
        t_unc_path=t_unc_path,
        thresholded_two_sided_path=thresholded_two_sided_path,
        mosaic_two_sided_png=mosaic_two_sided_png,
        thresholded_gt_path=thresholded_gt_path,
        thresholded_lt_path=thresholded_lt_path,
        mosaic_gt_png=mosaic_gt_png,
        mosaic_lt_png=mosaic_lt_png,
        used_map_paths=map_paths,
    )


def main():
    # Import lazily so importing this module doesn't require Fire unless the CLI is invoked.
    import fire

    fire.Fire({"second_level": second_level_one_sample_ttest})


if __name__ == "__main__":
    main()


