from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
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


def _fisher_z_transform_img(
    img: nib.Nifti1Image,
    *,
    eps: float = 1e-7,
    clip: bool = True,
) -> nib.Nifti1Image:
    """
    Apply Fisher z-transform (atanh) to an image assumed to contain correlation coefficients.

    Notes
    -----
    - For numerical stability, values are optionally clipped to (-1 + eps, 1 - eps).
    - Non-finite voxels remain non-finite.
    """
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    out = np.full(data.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(data)
    if not np.any(finite):
        return image.new_img_like(img, out, copy_header=True)

    x = data[finite]
    if bool(clip):
        x = np.clip(x, -1.0 + float(eps), 1.0 - float(eps))
    else:
        bad = (x <= -1.0) | (x >= 1.0)
        if np.any(bad):
            raise ValueError(
                "Fisher z-transform requires all finite values to be strictly within (-1, 1). "
                "Either clean your maps or set clip=True."
            )

    out[finite] = np.arctanh(x)
    return image.new_img_like(img, out, copy_header=True)


def _apply_transformation_img(
    img: nib.Nifti1Image,
    *,
    transformation: Optional[str],
) -> nib.Nifti1Image:
    if transformation is None:
        return img
    t = str(transformation).lower().strip()
    if t in ("", "none", "null"):
        return img
    if t in ("fisherz", "fisher_z", "fisher-z"):
        return _fisher_z_transform_img(img)
    raise ValueError(f"Unsupported transformation: {transformation!r}. Supported: None, 'fisherz'.")


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
    maps: Union[str, Path, pd.DataFrame, Sequence[Union[str, Path]], np.ndarray],
    *,
    out_dir: Optional[Union[str, Path]] = None,
    smoothing_fwhm: Optional[float] = None,
    template_image: Optional[Union[str, Path, nib.Nifti1Image]] = None,
    mask_image: Optional[Union[str, Path, nib.Nifti1Image]] = None,
    use_mask_in_thresholding: bool = True,
    transformation: Optional[str] = None,
    plot_kwargs: Optional[dict] = None,
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
) -> Dict[str, SecondLevelOutputs]:
    """
    Second-level group GLM (one-sample) from subject-level maps with optional covariates.

    Assumptions
    -----------
    Inputs (`maps`)
    --------------
    `maps` can be any of:
    - **Directory path**: contains ONLY the subject-level NIfTI maps to enter into the group model
      (non-recursive; subfolders ignored).
    - **List/array of paths**: explicit subject-level map paths.
    - **CSV/TSV path**: loaded into a dataframe; must contain a `path` (or `map_path`) column plus optional covariate columns.
    - **DataFrame**: must contain a `path` (or `map_path`) column plus optional covariate columns.

    Output
    ------
    Outputs are written to `out_dir` (or to a default `group/` folder when `maps` is a directory path).
    If covariates are provided, the function produces one set of outputs per regressor (including `intercept`),
    written into subfolders:
        `{out_dir}/{regressor_name}/...`
    - Produces:
      - Two-sided thresholded map
      - One-sided gt map (positive tail)
      - One-sided lt map computed by sign-flipping the stat image (so lt map has positive t-values)
    """
    def _coerce_maps_to_df(
        spec: Union[str, Path, pd.DataFrame, Sequence[Union[str, Path]], np.ndarray],
    ) -> tuple[Optional[Path], pd.DataFrame]:
        """
        Returns (maps_dir_if_applicable, df with at least a 'path' column).
        """
        if isinstance(spec, pd.DataFrame):
            df0 = spec.copy()
            maps_dir0: Optional[Path] = None
        else:
            p = Path(spec) if isinstance(spec, (str, Path)) else None
            if p is not None and p.exists() and p.is_dir():
                maps_dir0 = p
                df0 = pd.DataFrame({"path": [str(pp) for pp in _iter_nifti_files(p)]})
            elif p is not None and p.exists() and p.is_file() and p.suffix.lower() in (".csv", ".tsv"):
                maps_dir0 = None
                df0 = pd.read_csv(p, sep=None, engine="python")
            else:
                maps_dir0 = None
                if isinstance(spec, (str, Path)):
                    df0 = pd.DataFrame({"path": [str(Path(spec))]})
                else:
                    arr = np.asarray(spec, dtype=object)
                    if arr.ndim != 1:
                        raise ValueError(f"`maps` must be 1D when array-like; got shape={arr.shape}")
                    df0 = pd.DataFrame({"path": [str(Path(x)) for x in arr.tolist()]})

        if "path" not in df0.columns:
            if "map_path" in df0.columns:
                df0 = df0.rename(columns={"map_path": "path"})
            else:
                raise ValueError("`maps` dataframe/CSV must include a 'path' (or 'map_path') column.")

        df0["path"] = df0["path"].astype(str)
        return maps_dir0, df0

    maps_dir, maps_df = _coerce_maps_to_df(maps)

    group_dir = (
        Path(out_dir)
        if out_dir is not None
        else ((maps_dir / "group") if maps_dir is not None else (Path.cwd() / "group"))
    )
    group_dir.mkdir(parents=True, exist_ok=True)

    map_paths = [Path(p) for p in maps_df["path"].tolist()]
    if len(map_paths) == 0:
        raise ValueError("No subject NIfTI maps provided.")
    if len(map_paths) < 2:
        raise ValueError(f"Need at least 2 maps for a group model; got {len(map_paths)}.")

    missing = [str(p) for p in map_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Some map paths do not exist (showing up to 5): {missing[:5]}")

    # Save run arguments for reproducibility
    args_path = group_dir / "args.json"
    if overwrite or not args_path.exists():
        args_to_save = {
            "maps": (str(maps_dir) if maps_dir is not None else "dataframe_or_list"),
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
            "transformation": transformation,
            "plot_kwargs": plot_kwargs,
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
            "n_maps": int(len(map_paths)),
            "covariates": [c for c in maps_df.columns if c != "path"],
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

    transformation_tag = ""
    if transformation is not None:
        t = str(transformation).lower().strip()
        if t not in ("", "none", "null"):
            transformation_tag = f"_{t}"

    second_level_input = [str(p) for p in map_paths]
    if transformation_tag != "":
        imgs = [image.load_img(str(p)) for p in map_paths]
        second_level_input = [
            _apply_transformation_img(img, transformation=transformation) for img in imgs
        ]

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

        # Guard: nilearn thresholding can crash when the provided mask is empty (0 voxels),
        # or when the stat image has no finite voxels under the mask (e.g. all-NaN / empty
        # intersection after resampling). In those cases, the correct output is an "empty"
        # thresholded map.
        if mask_img is not None:
            mask_data = np.asanyarray(mask_img.dataobj)
            if mask_data.size == 0 or int(np.count_nonzero(mask_data)) == 0:
                empty = image.new_img_like(stat_img, np.zeros(stat_img.shape, dtype=np.float32))
                empty.to_filename(str(out_path))
                return empty
            stat_data = np.asanyarray(stat_img.dataobj)
            masked = stat_data[mask_data.astype(bool)]
        else:
            stat_data = np.asanyarray(stat_img.dataobj)
            masked = stat_data.reshape(-1)

        finite = masked[np.isfinite(masked)]
        if finite.size == 0:
            empty = image.new_img_like(stat_img, np.zeros(stat_img.shape, dtype=np.float32))
            empty.to_filename(str(out_path))
            return empty
        # If all finite values are exactly zero, thresholding is meaningless and may produce inf thresholds.
        if float(np.max(np.abs(finite))) == 0.0:
            empty = image.new_img_like(stat_img, np.zeros(stat_img.shape, dtype=np.float32))
            empty.to_filename(str(out_path))
            return empty

        # Fire forwards unknown CLI args into **threshold_kwargs, so validate against nilearn's
        # threshold_stats_img signature to provide a clear error instead of a confusing TypeError.
        allowed = set(sig.parameters.keys())
        unknown = sorted([k for k in threshold_kwargs.keys() if k not in allowed])
        if unknown:
            if "transformation" in unknown:
                raise ValueError(
                    "Got an unexpected argument 'transformation' forwarded into nilearn.threshold_stats_img(). "
                    "This usually means you're running an older installed version of fmri-utils that does not "
                    "support the --transformation CLI flag, so Fire treated it as an extra kwarg. "
                    "Upgrade fmri-utils and try again (e.g. `pip install --upgrade \"git+https://github.com/efirdc/fmri_utils.git\"`)."
                )
            raise ValueError(
                "Unsupported thresholding kwargs passed via **threshold_kwargs: "
                f"{unknown}. These kwargs are forwarded to nilearn.glm.threshold_stats_img; "
                f"allowed keys for your nilearn version are: {sorted(allowed)}."
            )

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

    # ---- Design matrix (intercept + optional covariates) ----
    covariate_cols = [c for c in maps_df.columns if c != "path"]
    if covariate_cols:
        cov_df = maps_df[covariate_cols].copy()
        for c in covariate_cols:
            cov_df[c] = pd.to_numeric(cov_df[c], errors="coerce")
        if cov_df.isna().any().any():
            bad = {c: int(cov_df[c].isna().sum()) for c in covariate_cols if cov_df[c].isna().any()}
            raise ValueError(f"Covariate columns must be numeric; found NaNs after parsing: {bad}")
        design_matrix = pd.concat(
            [pd.DataFrame({"intercept": np.ones(len(map_paths))}), cov_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        design_matrix = pd.DataFrame({"intercept": np.ones(len(map_paths))})

    # Fit a parametric GLM once for (a) parametric inference and (b) QC t-maps in non-parametric mode.
    slm = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=resolved_mask_img)
    slm = slm.fit(second_level_input, design_matrix=design_matrix)

    # Plot background template (shared across contrasts)
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

    plot_kwargs_base = dict(plot_kwargs or {})
    if "output_file" in plot_kwargs_base:
        raise ValueError(
            "plot_kwargs must not include 'output_file'. "
            "This function manages output filenames itself (it always saves mosaics into the group_dir)."
        )
    if "bg_img" in plot_kwargs_base:
        raise ValueError(
            "plot_kwargs must not include 'bg_img'. "
            "Use template_image=... (or leave it as default) to control the background image."
        )

    def _render_mosaic(stat_img, *, out_dir_local: Path, file_stem: str, title: str) -> Path:
        plot_kwargs_local = dict(plot_kwargs_base)
        plot_kwargs_local.setdefault("display_mode", "mosaic")
        plot_kwargs_local.setdefault("colorbar", True)
        plot_kwargs_local.setdefault("cmap", cmap)
        plot_kwargs_local.setdefault("title", title)

        mosaic = out_dir_local / f"{file_stem}_mosaic.png"
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_img,
            **plot_kwargs_local,
        )
        display.savefig(str(mosaic))
        display.close()
        return mosaic

    regressor_names = list(design_matrix.columns)
    outputs: Dict[str, SecondLevelOutputs] = {}

    for reg_name in regressor_names:
        reg_dir = group_dir / str(reg_name)
        reg_dir.mkdir(parents=True, exist_ok=True)

        # --- Save masks used for analysis / QC ---
        masks_dir = reg_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        if resolved_mask_img is not None:
            resolved_mask_img.to_filename(str(masks_dir / "analysis_mask_resolved.nii.gz"))
        if validity_mask_img is not None:
            validity_mask_img.to_filename(str(masks_dir / "analysis_mask_validity_std.nii.gz"))

        # Unthresholded t-map
        t_unc_path = reg_dir / f"tmap_unc{transformation_tag}.nii.gz"
        if t_unc_path.exists() and not overwrite:
            t_scores_img = image.load_img(str(t_unc_path))
        else:
            t_scores_img = slm.compute_contrast(reg_name, output_type="stat")
            t_scores_img.to_filename(str(t_unc_path))

        if inference == "parametric":
            alpha_tag = str(alpha).replace(".", "p")

            thresholded_two_sided_path = reg_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}{transformation_tag}.nii.gz"
            thresholded_two_sided_map = _threshold_and_cache(
                t_scores_img,
                out_path=thresholded_two_sided_path,
                two_sided=True,
                mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
            )
            thresholded_gt_path = reg_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}{transformation_tag}_gt_onesided.nii.gz"
            thresholded_gt_map = _threshold_and_cache(
                t_scores_img,
                out_path=thresholded_gt_path,
                two_sided=False,
                mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
            )
            t_flipped = image.math_img("-img", img=t_scores_img)
            thresholded_lt_path = reg_dir / f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}{transformation_tag}_lt_onesided.nii.gz"
            thresholded_lt_map = _threshold_and_cache(
                t_flipped,
                out_path=thresholded_lt_path,
                two_sided=False,
                mask_img=(resolved_mask_img if bool(use_mask_in_thresholding) else None),
            )

            base_stem = f"tmap_{height_control}_alpha{alpha_tag}_k{int(cluster_threshold)}{transformation_tag}"
            base_title = f"{reg_name}: t-map ({height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
            gt_title = f"{reg_name}: t-map > 0 (one-sided, {height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
            lt_title = f"{reg_name}: t-map < 0 (one-sided, {height_control}, alpha={alpha}, k>={int(cluster_threshold)})"
        else:
            def _np_logp(two_sided_test: bool, second_level_contrast):
                return non_parametric_inference(
                    second_level_input=second_level_input,
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
                if isinstance(ret, dict) and "t" in ret:
                    return ret["t"]
                return None

            def _select_logp(ret):
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

            raw_dir = reg_dir / "nonparametric_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            reg_idx = regressor_names.index(reg_name)
            c_pos = [0.0] * len(regressor_names)
            c_pos[reg_idx] = 1.0
            c_neg = [0.0] * len(regressor_names)
            c_neg[reg_idx] = -1.0

            # gt (one-sided)
            logp_gt_ret = _np_logp(False, c_pos)
            logp_gt_img = _select_logp(logp_gt_ret)
            t_gt_img = _extract_t(logp_gt_ret) or t_scores_img
            try:
                logp_gt_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_gt.nii.gz"))
            except Exception:
                pass
            gt_mask = np.asarray(logp_gt_img.dataobj) >= logp_thr
            t_gt_data = np.asarray(t_gt_img.dataobj)
            thresholded_gt_map = image.new_img_like(
                t_gt_img, np.where(gt_mask & (t_gt_data > 0), t_gt_data, 0.0), copy_header=True
            )
            thresholded_gt_path = reg_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}{transformation_tag}_gt_onesided.nii.gz"
            thresholded_gt_map.to_filename(str(thresholded_gt_path))

            # Two-sided
            logp_two_ret = _np_logp(True, c_pos)
            logp_two_img = _select_logp(logp_two_ret)
            t_two_img = _extract_t(logp_two_ret) or t_scores_img
            try:
                logp_two_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_two_sided.nii.gz"))
            except Exception:
                pass
            two_mask = np.asarray(logp_two_img.dataobj) >= logp_thr
            t_two_data = np.asarray(t_two_img.dataobj)
            thresholded_two_sided_map = image.new_img_like(
                t_two_img, np.where(two_mask, t_two_data, 0.0), copy_header=True
            )
            thresholded_two_sided_path = reg_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}{transformation_tag}.nii.gz"
            thresholded_two_sided_map.to_filename(str(thresholded_two_sided_path))

            # lt (one-sided), output positive values
            logp_lt_ret = _np_logp(False, c_neg)
            logp_lt_img = _select_logp(logp_lt_ret)
            t_lt_img = _extract_t(logp_lt_ret) or t_scores_img
            try:
                logp_lt_img.to_filename(str(raw_dir / f"logp_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}_lt.nii.gz"))
            except Exception:
                pass
            lt_mask = np.asarray(logp_lt_img.dataobj) >= logp_thr
            t_lt_data = np.asarray(t_lt_img.dataobj)
            thresholded_lt_map = image.new_img_like(
                t_lt_img, np.where(lt_mask & (t_lt_data < 0), -t_lt_data, 0.0), copy_header=True
            )
            thresholded_lt_path = reg_dir / f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}{transformation_tag}_lt_onesided.nii.gz"
            thresholded_lt_map.to_filename(str(thresholded_lt_path))

            base_stem = f"tmap_vfwe_alpha{alpha_tag}{tfce_tag}{cfp_tag}{transformation_tag}"
            base_title = f"{reg_name}: t-map (vfwe, alpha={alpha})"
            gt_title = f"{reg_name}: t-map > 0 (one-sided, vfwe, alpha={alpha})"
            lt_title = f"{reg_name}: t-map < 0 (one-sided, vfwe, alpha={alpha})"

        mosaic_two_sided_png = _render_mosaic(thresholded_two_sided_map, out_dir_local=reg_dir, file_stem=base_stem, title=base_title)
        mosaic_gt_png = _render_mosaic(
            thresholded_gt_map, out_dir_local=reg_dir, file_stem=f"{base_stem}_gt_onesided", title=gt_title
        )
        mosaic_lt_png = _render_mosaic(
            thresholded_lt_map, out_dir_local=reg_dir, file_stem=f"{base_stem}_lt_onesided", title=lt_title
        )

        outputs[str(reg_name)] = SecondLevelOutputs(
            maps_dir=(maps_dir if maps_dir is not None else group_dir),
            group_dir=reg_dir,
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

    return outputs


def main():
    # Import lazily so importing this module doesn't require Fire unless the CLI is invoked.
    import fire

    fire.Fire({"second_level": second_level_one_sample_ttest})


if __name__ == "__main__":
    main()


