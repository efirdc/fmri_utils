# fmri_utils

Reusable fMRI utilities (second-level group analysis, fMRIPrep-based transformations, decoding helpers).

## Install

### (Optional but recommended) Create and activate a Python environment first

This project requires **Python >= 3.9**. Using an isolated environment helps avoid dependency conflicts.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
```

Or with conda (example):

```bash
conda create -n fmri-utils python=3.11 -y
conda activate fmri-utils
python -m pip install --upgrade pip
```

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

## Command Line Interface

This package installs a console script:

```bash
fmri-utils second_level /path/to/mni_subject_maps
```

You can also run it in module mode:

```bash
python -m fmri_utils.second_level_analysis second_level /path/to/mni_subject_maps
```

### Input modes for `second_level`

The first positional argument to `second_level` is `maps`. It can be:

- **Directory path**: a folder containing **only** subject-level NIfTI maps (`.nii`/`.nii.gz`) to enter into the model (non-recursive).
- **CSV/TSV path**: a “design file” with a required `path` (or `map_path`) column plus any number of numeric covariate columns.

#### Example: CSV/TSV with covariates

Create a CSV like:

```csv
path,expertise,is_patient
/path/to/sub-001_contrast_map_mni.nii.gz,-1.12,0
/path/to/sub-002_contrast_map_mni.nii.gz,0.34,1
/path/to/sub-003_contrast_map_mni.nii.gz,0.78,0
```

Then run:

```bash
fmri-utils second_level /path/to/design.csv \
  --inference=parametric \
  --height_control=fdr \
  --cluster_threshold=20
```

Outputs are written to `out_dir` (or a default `group/` folder), with one subfolder per regressor (including `intercept`):

- `.../group/intercept/`
- `.../group/expertise/`
- `.../group/is_patient/`

### Common flags

- `--inference=parametric|non_parametric`: parametric (default) or permutation inference (voxelwise FWER)
- `--n_perm=5000`: permutations for non-parametric mode
- `--mask_image=mni_template`: use the MNI152 T1 template as a binary mask (`>0`)
- `--transformation=fisherz`: apply a transformation to each input map in-memory before fitting/testing
- `--overwrite=True`: regenerate outputs even if files exist
- `--plot_kwargs="{...}"`: extra keyword args forwarded to `nilearn.plotting.plot_stat_map` for plotting (default `display_mode="mosaic"`)

Example (non-parametric):

```bash
fmri-utils second_level \
  /path/to/mni_subject_maps \
  --inference=non_parametric \
  --n_perm=5000 \
  --n_jobs=8 \
  --mask_image=mni_template \
  --overwrite=True
```

## Call from Python

```python
import pandas as pd
from fmri_utils.second_level_analysis import second_level_one_sample_ttest

design = pd.DataFrame(
    {
        "path": [
            "/path/to/sub-001_contrast_map_mni.nii.gz",
            "/path/to/sub-002_contrast_map_mni.nii.gz",
            "/path/to/sub-003_contrast_map_mni.nii.gz",
        ],
        # Continuous covariate (already z-scored)
        "expertise": [-1.12, 0.34, 0.78],
        # Binary categorical covariate encoded as 0/1
        "is_patient": [0, 1, 0],
    }
)

outs = second_level_one_sample_ttest(
    design,
    inference="parametric",
    height_control="fdr",
    cluster_threshold=20,
)
print(outs["intercept"].group_dir)
```

### Parameters

Core inputs:
- `maps`: One of:
  - a directory path of subject-level NIfTI maps, or
  - a CSV/TSV path with a `path` column plus covariates, or
  - a pandas DataFrame with a `path` column plus covariates.
- `out_dir`: Optional output directory. Defaults to `maps/group/` when `maps` is a directory path; otherwise defaults to `./group/`.
- `overwrite`: If `True`, recompute outputs even if they already exist.

Design / covariates:
- If covariate columns are provided, **one set of outputs is produced per regressor**, written to `{out_dir}/{regressor_name}/...` (including `intercept`).

Optional preprocessing:
- `transformation`: Optional preprocessing applied **in-memory** to each input map before running inference.
  Currently supported:
  - `"fisherz"`: Fisher z-transform (`atanh`), commonly used for correlation (\(r\)) maps prior to group stats.
  This is commonly used for correlation (\(r\)) maps prior to group statistics.

Plotting:
- `plot_kwargs`: Optional dict of keyword arguments forwarded to `nilearn.plotting.plot_stat_map` (for the saved mosaics).
  See Nilearn docs: https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_stat_map.html

Example: set explicit cut coordinates per axis (dict `<str: 1D ndarray>` style):
```bash
fmri-utils second_level /path/to/design.csv \
  --plot_kwargs="{'display_mode': 'mosaic', 'cut_coords': {'x': (-40, -20, 0, 20, 40), 'y': (-30, -10, 10, 30), 'z': (-20, 0, 20)}}"
```
Outputs:
- `args.json`: The CLI/function writes an `args.json` into the output directory containing the relevant parameters used for the run.

Masking / plotting:
- `mask_image`: Optional mask image. Can be:
  - a path to a NIfTI mask, or
  - `"mni_template"` / `"mni"` / `"mni152"` (uses nilearn’s bundled MNI template and binarizes it with `>0`).
  This mask is passed into the model and also used to restrict which voxels enter multiple-comparisons correction for the one-sided maps.
- `template_image`: Optional background image used for plotting. If not provided, the nilearn MNI template is used and written into `out_dir` when possible.
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


