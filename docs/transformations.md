# Transformations

## Warp subject-space map to MNI (fMRIPrep transform)

Function:
- `fmri_utils.transformations.warp_to_mni_with_fmriprep_transform`

This applies fMRIPrep T1w→MNI transform (`*_from-T1w_to-MNI152NLin2009cAsym*_xfm.h5`) to a 3D subject-space map.

## Example

```python
from pathlib import Path
from fmri_utils.transformations import warp_to_mni_with_fmriprep_transform

fmriprep_subject_dir = Path("/path/to/fmriprep/sub-001/")
in_img = Path("sub-001_effect_map.nii.gz")
out_img = Path("sub-001_effect_map_mni.nii.gz")

warp_to_mni_with_fmriprep_transform(
    in_img,
    fmriprep_subject_dir=fmriprep_subject_dir,
    out_img_path=out_img,
    interpolation="linear",
    mni_voxel_size=(2.0, 2.0, 2.0),
)
```

## Requirements

- ANTs available as `antsApplyTransforms` on PATH, or
- ANTsPy installed (`antspyx`).

## Notes

- Subject ID is inferred from `fmriprep_subject_dir.stem`.
- Intended for map-level transforms, not full preprocessing.
