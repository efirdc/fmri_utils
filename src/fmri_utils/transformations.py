from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import subprocess
import shutil

import nibabel as nib
from nibabel.processing import resample_to_output


def _find_subject_ants_h5_and_ref(
    *,
    fmriprep_subject_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """
    Locate the fMRIPrep-generated ANTs transform and an MNI-space T1w reference image.

    Parameters
    ----------
    fmriprep_subject_dir:
        The subject directory that contains 'anat/' (typically the folder that also contains 'func/').
        The subject name is inferred from `Path(fmriprep_subject_dir).stem`.
    """
    subject_dir = Path(fmriprep_subject_dir)
    anat_dir = subject_dir / "anat"
    subject_name = subject_dir.stem
    h5 = anat_dir / f"{subject_name}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    if not h5.exists():
        raise FileNotFoundError(f"Missing ANTs transform: {h5}")
    # Prefer an explicit res-2 (or similar lower-res) MNI reference if present to avoid
    # accidentally selecting a high-res (e.g. 1mm) template and inflating voxel counts.
    ref_candidates = sorted(anat_dir.glob("*space-MNI152NLin2009cAsym*_T1w.nii*"))
    if len(ref_candidates) > 0:
        preferred = [p for p in ref_candidates if "res-2" in p.name]
        return h5, (preferred[0] if len(preferred) > 0 else ref_candidates[0])

    # Some fMRIPrep outputs don't include an MNI-space T1w reference in the subject folder.
    # Fall back to nilearn's bundled MNI152 template (cached by nilearn) and write it locally
    # so downstream ANTs calls have a concrete reference file.
    try:
        from nilearn.datasets import load_mni152_template  # lazy optional dependency
    except Exception as e:
        raise FileNotFoundError(
            f"Missing MNI reference T1w in {anat_dir} and nilearn is not available to provide a fallback."
        ) from e

    fallback_ref = anat_dir / "MNI152NLin2009cAsym_T1w.nii.gz"
    if not fallback_ref.exists():
        # Newer nilearn supports selecting resolution. Prefer 2mm to keep voxel counts reasonable.
        import inspect

        sig = inspect.signature(load_mni152_template)
        if "resolution" in sig.parameters:
            tmpl = load_mni152_template(resolution=2)
        else:
            tmpl = load_mni152_template()
        nib.save(tmpl, str(fallback_ref))
    return h5, fallback_ref


def _ants_interp(arg: str) -> str:
    if arg.lower() in ("linear", "lin"):
        return "Linear"
    if arg.lower() in ("nearest", "nn", "nearestneighbor"):
        return "NearestNeighbor"
    if arg.lower() in ("bspline", "spline"):
        return "BSpline"
    return "Linear"


def _antspy_interp(arg: str) -> str:
    if arg.lower() in ("linear", "lin"):
        return "linear"
    if arg.lower() in ("nearest", "nn", "nearestneighbor"):
        return "nearestNeighbor"
    if arg.lower() in ("bspline", "spline"):
        return "bSpline"
    return "linear"


def warp_to_mni_with_fmriprep_transform(
    in_img_path: Union[str, Path],
    *,
    fmriprep_subject_dir: Union[str, Path],
    out_img_path: Union[str, Path],
    interpolation: str = "linear",
    mni_voxel_size: Optional[Sequence[float]] = None,
) -> Path:
    """
    Warp an image in subject (T1w) space to MNI using the T1w->MNI transform produced by fMRIPrep.

    Parameters
    ----------
    mni_voxel_size:
        Optional voxel size (in mm) to resample the MNI reference grid to *before* applying the transform.
        If None (default), the output grid will use the voxel spacing of the **input image** being transformed.
        (This preserves native resolution while still producing an image in MNI space.)

    Notes
    -----
    Prefers ANTsPy if available (optional dependency); otherwise calls `antsApplyTransforms`.
    """
    in_img_path = Path(in_img_path)
    out_img_path = Path(out_img_path)
    h5, ref_img_path = _find_subject_ants_h5_and_ref(
        fmriprep_subject_dir=fmriprep_subject_dir,
    )
    out_img_path.parent.mkdir(parents=True, exist_ok=True)

    in_img_nib = nib.load(str(in_img_path))
    input_voxel_size = tuple(float(z) for z in in_img_nib.header.get_zooms()[:3])
    if mni_voxel_size is None:
        mni_voxel_size = input_voxel_size

    # Prefer ANTsPy if available; otherwise fall back to CLI
    use_antspy = False
    try:
        import ants  # type: ignore

        use_antspy = True
    except Exception:
        use_antspy = False

    if use_antspy:
        import ants  # type: ignore

        fixed = ants.image_read(str(ref_img_path))
        # Resample reference grid only if needed
        fixed_spacing = tuple(float(s) for s in fixed.spacing)
        if mni_voxel_size is not None and tuple(mni_voxel_size) != fixed_spacing:
            fixed = ants.resample_image(fixed, mni_voxel_size, use_voxels=False)
        moving = ants.image_read(str(in_img_path))
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[str(h5)],
            interpolator=_antspy_interp(interpolation),
        )
        ants.image_write(warped, str(out_img_path))
        return out_img_path

    # CLI path: if downsampling requested, produce a cached downsampled reference
    ref_for_cli = Path(ref_img_path)
    if mni_voxel_size is not None:
        ref_img = nib.load(str(ref_img_path))
        ref_voxel_size = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
        if tuple(mni_voxel_size) != ref_voxel_size:
            vs_tag = "_".join(f"{float(v):g}".replace(".", "p") for v in mni_voxel_size)
            cached_ref = out_img_path.parent / f"ref_MNI_{vs_tag}mm.nii.gz"
            if not cached_ref.exists():
                ref_img_ds = resample_to_output(ref_img, voxel_sizes=mni_voxel_size, order=1)
                nib.save(ref_img_ds, str(cached_ref))
            ref_for_cli = cached_ref

    ants_apply = shutil.which("antsApplyTransforms")
    if ants_apply is None:
        raise RuntimeError(
            "antsApplyTransforms was not found on PATH. To warp subject-space maps to MNI you need either:\n"
            "- ANTs available on PATH (antsApplyTransforms), or\n"
            "- ANTsPy installed (pip install antspyx)\n"
            "If you're on Windows, easiest is usually to run the second-level analysis on your cluster (DRAC) "
            "where ANTs is available."
        )

    cmd = [
        ants_apply,
        "-d",
        "3",
        "-i",
        str(in_img_path),
        "-r",
        str(ref_for_cli),
        "-o",
        str(out_img_path),
        "-t",
        str(h5),
        "-n",
        _ants_interp(interpolation),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"antsApplyTransforms failed: {result.stderr}")
    return out_img_path




