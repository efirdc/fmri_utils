# Montage API

Module:
- `fmri_utils.montage`

## Contract

Inputs must already be aligned (same shape and affine):
- reference image
- subject effect maps
- optional include masks
- optional outline label image

The utility validates alignment and fails fast on mismatch.

## Main API

- `SubjectMap(subject_id, effect_img, include_mask_img=None)`
- `OutlineRegion(region_id, label, color)`
- `OutlineSpec(label_img, regions, alpha=1.0, linewidth=0.7)`
- `MontageConfig(...)`
- `render_axial_subject_montage_series(subject_maps, reference_img, out_dir, config, outline_spec=None)`

## Example

```python
from pathlib import Path
import nibabel as nib
from fmri_utils.montage import (
    SubjectMap,
    OutlineRegion,
    OutlineSpec,
    MontageConfig,
    render_axial_subject_montage_series,
)

ref = nib.load("reference_template.nii.gz")
sub1 = nib.load("sub-0001_effect_aligned.nii.gz")
sub2 = nib.load("sub-0002_effect_aligned.nii.gz")
label_img = nib.load("roi_labels_aligned.nii.gz")

outline = OutlineSpec(
    label_img=label_img,
    regions=[
        OutlineRegion(region_id=1, label="ROI_1", color="#1f77b4"),
        OutlineRegion(region_id=2, label="ROI_2", color="#ff7f0e"),
    ],
    alpha=1.0,
    linewidth=0.7,
)

cfg = MontageConfig(
    mode="solid",  # or "thresholded"
    slice_indices=(30, 40, 50),
    grid_rows=2,
    grid_cols=3,
    filename_prefix="mean_correlation",
    title_prefix="example_run",
    cbar_label="correlation",
    solid_center_zero=True,
)

created = render_axial_subject_montage_series(
    subject_maps=[
        SubjectMap(subject_id="sub-0001", effect_img=sub1),
        SubjectMap(subject_id="sub-0002", effect_img=sub2),
    ],
    reference_img=ref,
    out_dir=Path("out/axial_solid"),
    config=cfg,
    outline_spec=outline,
)

print(f"wrote {len(created)} images")
```

## Behavior highlights

- Supports `solid` and `thresholded` render modes.
- Optional darkening outside include masks.
- Optional ROI outlines + legend from caller-provided label map/table.
- Returns list of generated PNG paths.
