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
    ref_candidates = list(anat_dir.glob("*space-MNI152NLin2009cAsym*_T1w.nii*"))
    if len(ref_candidates) > 0:
        return h5, ref_candidates[0]

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
    mni_voxel_size: Optional[Sequence[float]] = (2.0, 2.0, 2.0),
) -> Path:
    """
    Warp an image in subject (T1w) space to MNI using the T1w->MNI transform produced by fMRIPrep.

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
        if mni_voxel_size is not None:
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
        cached_ref = out_img_path.parent / f"ref_MNI_{int(mni_voxel_size[0])}mm.nii.gz"
        if not cached_ref.exists():
            ref_img = nib.load(str(ref_img_path))
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


