# fmri_utils

Reusable fMRI utilities (second-level group analysis, fMRIPrep-based transformations, decoding helpers).

## Install

### Option A: install directly from GitHub with pip

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git"
```

### Option B: clone with git and install in editable mode (recommended for development)

```bash
git clone https://github.com/efirdc/fmri_utils.git
cd fmri_utils
pip install -e .
```

## CLI (Fire)

This package installs a console script:

```bash
fmri-second-level second_level --maps_dir=/path/to/mni_subject_maps
```

You can also run it in module mode:

```bash
python -m fmri_utils.second_level_analysis second_level --maps_dir=/path/to/mni_subject_maps
```

### Common flags

- `--inference=parametric|non_parametric`: parametric (default) or permutation inference (voxelwise FWER)
- `--n_perm=5000`: permutations for non-parametric mode
- `--mask_image=mni_template`: use the MNI152 T1 template as a binary mask (`>0`)
- `--overwrite=True`: regenerate outputs even if files exist

Example (non-parametric):

```bash
fmri-second-level second_level \
  --maps_dir=/path/to/mni_subject_maps \
  --inference=non_parametric \
  --n_perm=5000 \
  --n_jobs=8 \
  --mask_image=mni_template \
  --overwrite=True
```

## Call from Python

```python
from fmri_utils.second_level_analysis import second_level_one_sample_ttest

out = second_level_one_sample_ttest(
    maps_dir="/path/to/mni_subject_maps",
    inference="parametric",
    height_control="fdr",
    cluster_threshold=20,
)
print(out.group_dir)
```

### Parameters

Core inputs:
- `maps_dir`: Directory containing **only** subject-level NIfTI maps (`.nii`/`.nii.gz`). Output will be written to `maps_dir/group/`.
- `overwrite`: If `True`, recompute outputs even if they already exist.

Masking / plotting:
- `mask_image`: Optional mask image. Can be:
  - a path to a NIfTI mask, or
  - `"mni_template"` / `"mni"` / `"mni152"` (uses nilearn’s bundled MNI template and binarizes it with `>0`).
  This mask is passed into the model and also used to restrict which voxels enter multiple-comparisons correction for the one-sided maps.
- `template_image`: Optional background image used for plotting. If not provided, the nilearn MNI template is used and also written into `maps_dir/group/`.
- `cmap`: Matplotlib colormap name used by nilearn plotting.

Parametric inference mode (`inference="parametric"`):
- Uses `nilearn.glm.second_level.SecondLevelModel` to compute a t-stat map, then thresholds it with `nilearn.glm.threshold_stats_img`.
- `height_control`: Currently used as passed into nilearn’s thresholding (e.g. `"fdr"`).
- `alpha`: Significance level used by nilearn’s thresholding.
- `cluster_threshold`: Cluster-extent threshold (voxel count) passed into nilearn’s thresholding.
- `smoothing_fwhm`: Passed to `SecondLevelModel`.
- Extra keyword args are forwarded to `threshold_stats_img` via `**threshold_kwargs` in the function.

Non-parametric inference mode (`inference="non_parametric"`):
- Uses `nilearn.glm.second_level.non_parametric_inference` (permutation-based, voxelwise FWER via max-T).
- `n_perm`: Number of permutations.
- `n_jobs`: Parallel jobs.
- `random_state`: Seed.
- `verbose`: Verbosity level.
- `tfce`: Whether to compute TFCE during permutations.
- `cluster_forming_p_threshold`: Passed to nilearn’s `threshold` argument (cluster-level inference in p-scale).

References:
- `nilearn.glm.second_level.non_parametric_inference` docs: [stable](https://nilearn.github.io/stable/modules/generated/nilearn.glm.second_level.non_parametric_inference.html)
- `nilearn` second-level one-sample example (includes non-parametric inference): [example](https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_second_level_one_sample_test.html)

## Apply fMRIPrep T1→MNI transform (warp a subject-space map to MNI)

`fmri_utils.transformations.warp_to_mni_with_fmriprep_transform` applies the fMRIPrep-produced ANTs transform
(`*_from-T1w_to-MNI152NLin2009cAsym*_xfm.h5`) to warp a 3D NIfTI from subject (T1w) space to MNI space.

```python
from pathlib import Path
from fmri_utils.transformations import warp_to_mni_with_fmriprep_transform

# Must contain anat/
fmriprep_subject_dir = Path("/path/to/fmriprep/sub-001/")

# Any subject/T1w-space map (example name)
in_img = Path("sub-001_effect_map.nii.gz")

# Output in MNI space
out_img = Path("sub-001_effect_map_mni.nii.gz")

warp_to_mni_with_fmriprep_transform(
    in_img,
    fmriprep_subject_dir=fmriprep_subject_dir,
    out_img_path=out_img,
    interpolation="linear",
    mni_voxel_size=(2.0, 2.0, 2.0),
)
```

Notes:
- Requires either **ANTs** (`antsApplyTransforms` on PATH) or **ANTsPy** (`pip install antspyx`).
- The subject name used to locate the `.h5` transform is inferred from `fmriprep_subject_dir.stem`.


