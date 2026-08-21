# Voxelwise Encoding Analysis

`fmri_utils.encoding` fits stimulus-to-BOLD encoding models with nested
cross-validation. fMRIPrep derivatives are the easiest input, but the fitter
itself reads a preprocessing-neutral run manifest. FSL, AFNI, custom
preprocessing, or already-resampled arrays can use the same fitter by writing
that manifest.

## What Is Included

- fMRIPrep BOLD/mask/confounds discovery
- `.npy`, `.npz`, `.csv`, or `.tsv` feature matrices
- explicit BOLD/feature row alignment
- lag concatenation or lag averaging
- run-mask union, run-mask intersection, or an analysis mask
- per-run nuisance regression
- leave-one-run-out, grouped-run K-fold, and blocked K-fold CV
- per-voxel `PCA components x ridge alpha` tuning
- fold-local correlations and R2
- optional within-run block-shuffle permutations
- native-space NIfTI maps, fold maps, prediction traces, and JSON provenance
- voxel-chunked array execution with strict stitching validation

The package does not extract language-model features or infer stimulus timing.
Those steps are experiment-specific. The feature matrix supplied for each run
must already be aligned to that run's acquisition timeline.

## Install

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git"
```

For development:

```bash
git clone https://github.com/efirdc/fmri_utils.git
cd fmri_utils
pip install -e .
```

This installs the `fmri-encoding` command.

## Quick Start With fMRIPrep

### 1. Prepare one feature matrix per run

The preferred format is a 2D NumPy array:

```text
shape = (time points, stimulus features)
```

For example:

```text
/project/features/sub-001_task-story_run-1.npy
/project/features/sub-001_task-story_run-2.npy
/project/features/sub-001_task-story_run-3.npy
/project/features/sub-001_task-story_run-4.npy
```

### 2. Discover the fMRIPrep files

```bash
fmri-encoding discover-fmriprep \
  --fmriprep-root /project/derivatives/fmriprep \
  --task story \
  --space T1w \
  --feature-template '/project/features/{subject_id}_task-{task}_run-{run}.npy' \
  --output manifests/story_runs.csv
```

The command pairs each preprocessed BOLD file with its run brain mask and
confounds TSV. It does not guess feature/BOLD offsets.

### 3. Audit the manifest

Check at least:

- every expected subject/run is present;
- `features_path` points to the intended feature version;
- BOLD and feature row zero refer to the same acquired time point;
- discarded non-steady-state volumes are represented by `bold_start` and
  `feature_start`, not silently ignored;
- all run masks share the BOLD grid.

### 4. Fit one subject

```bash
fmri-encoding fit \
  --manifest manifests/story_runs.csv \
  --subject sub-001 \
  --output-dir results/sub-001 \
  --lags 2,3,4,5 \
  --lag-mode concat \
  --pca-components none,16,32,64,128 \
  --ridge-alphas 0.01,0.1,1,10,100,1000 \
  --cv-scheme leave_one_run_out \
  --nuisance-profile fmriprep \
  --n-acompcor 4 \
  --mask-strategy run_mask_union \
  --n-permutations 200 \
  --permutation-block-length 24
```

Run one subject per scheduler task for a cluster campaign.

## Generic Run Manifest

Required columns:

| Column | Meaning |
| --- | --- |
| `subject_id` | Subject identifier. |
| `run_id` | Unique run/fold identifier within the subject. |
| `bold_path` | 4D preprocessed BOLD NIfTI. |
| `mask_path` | 3D mask on exactly the BOLD grid. |
| `features_path` | Time x feature matrix. |

Optional columns:

| Column | Default | Meaning |
| --- | ---: | --- |
| `task` | empty | Descriptive task label. |
| `confounds_path` | empty | TSV containing nuisance regressors. |
| `bold_start` | `0` | First BOLD row included in alignment. |
| `feature_start` | `0` | Feature row paired with `bold_start`. |
| `n_samples` | maximum common length | Number of aligned rows before lag filtering. |
| `preprocessing_source` | empty | Provenance label such as `fmriprep` or `fsl`. |

Relative paths are resolved relative to the manifest file. Absolute paths are
also accepted.

Example:

```csv
subject_id,task,run_id,bold_path,mask_path,confounds_path,features_path,bold_start,feature_start,n_samples,preprocessing_source
sub-001,story,run-1,../fsl/sub-001_run-1_bold.nii.gz,../fsl/sub-001_run-1_mask.nii.gz,../fsl/sub-001_run-1_motion.tsv,../features/story-1.npy,0,5,169,fsl
```

This example states that FSL removed five acquisition volumes: BOLD row `0`
is paired with feature row `5`. The fitter records the resulting BOLD source
rows in its run data.

## Using FSL Or Another Pipeline

The custom derivative must provide:

1. one 4D BOLD file per run;
2. one mask per run on the BOLD grid;
3. a common spatial grid across runs for a subject;
4. optional nuisance TSVs;
5. an aligned feature matrix per run.

Write those paths into the generic manifest. No fMRIPrep naming convention is
used after manifest creation.

If runs are still in separate native-EPI grids, resample them into a common
subject space first. The fitter deliberately fails on mismatched grids rather
than performing an implicit resampling.

For FSL data that already received temporal high-pass filtering, a reasonable
custom nuisance specification is motion plus framewise displacement, without
fMRIPrep cosine or aCompCor columns:

```bash
fmri-encoding fit \
  --manifest manifests/fsl_story_runs.csv \
  --subject sub-001 \
  --output-dir results_fsl/sub-001 \
  --nuisance-profile custom \
  --nuisance-columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,framewise_displacement
```

Column names are not prescribed for custom inputs; they only need to match the
TSV and the command. Wildcards such as `cosine*` are supported.

## Scientific Behavior

### Alignment and lags

Features are aligned before lagging. A positive lag means:

```text
feature[t - lag] predicts BOLD[t]
```

Rows without all requested lag sources, and rows whose source feature is
marked invalid, are excluded. Lagging never crosses run boundaries.

CSV/TSV feature tables may include `semantic_valid` or `valid`. Metadata
columns such as `tr_index` and `run_id` are excluded from the feature matrix.

### Masks

- `run_mask_union`: include a voxel present in any run mask.
- `run_mask_intersection`: include only voxels present in every run mask.
- `analysis_mask_only`: use `--analysis-mask` directly.

The selected mask is applied directly to every BOLD run. The implementation
does not first apply a run mask and then remap columns into another mask.

### Nuisance regression

Nuisance regression is fit independently to every complete BOLD run before
feature alignment and lag filtering. A separate run intercept is included by
default, so each residualized voxel is run-centered.
The fMRIPrep profile uses:

- cosine high-pass regressors;
- six rigid-body motion parameters;
- framewise displacement;
- the requested number of aCompCor components.

Runwise target nuisance regression uses all BOLD rows as a preprocessing
operation, including rows later held out by CV. This is common for
nuisance-only temporal cleaning, but it is not identical to fitting nuisance
coefficients on training rows only. If the intended estimand requires strictly
fold-fitted nuisance coefficients, prepare fold-safe inputs outside this runner
or omit nuisance regression here.

### Cross-validation schemes

All schemes use nested CV. Standardization and PCA are fit only on the current
training rows.

The key distinction is whether a run is split between training and testing:

- `leave_one_run_out`: test one complete run and train on all other runs. Use
  this with a small number of runs when the goal is generalization to an unseen
  run or stimulus. Four runs produce four outer folds.
- `grouped_run_kfold`: keep runs intact but test several complete runs in each
  fold. This asks the same unseen-run question with fewer folds, which is useful
  for datasets containing many runs.
- `blocked_kfold`: divide every run into contiguous time blocks and hold out one
  block from each run per fold. Training still uses other portions of those same
  runs, so this tests unseen time intervals rather than unseen runs.

With blocked CV, use `--embargo-rows N` to remove nearby training rows that may
share contextual or lagged information with a held-out interval. Validation and
test rows are retained.

For every scheme and each outer fold:

1. the configured inner train/validation partitions are constructed;
2. feature standardization, target standardization, and PCA are fit using only
   the corresponding inner-training rows;
3. every PCA/alpha combination is scored per voxel on validation correlation;
4. correlations are averaged across inner folds in Fisher-z space;
5. the winning combination is selected independently per voxel;
6. the model is refit on all outer-training rows and evaluated on the held-out
   outer test rows.

Predictions are converted back to residualized-BOLD units before being saved.
Scientific correlations and R2 are computed within each outer fold, never by
concatenating differently standardized folds first.

The output metadata records every fold's row counts and source-row bounds by
run. For more specialized partitions, construct a `CVPlan` with the public
`OuterFold` and `InnerFold` dataclasses, then call
`fit_encoding(data, config, cv_plan=plan)`.

### Permutations

The optional null permutes the final lagged feature rows in contiguous blocks,
independently within each run. BOLD and fixed CV folds do not move. The entire
nested fitting and hyperparameter-selection procedure is repeated for each
permutation.

The block length is an exchangeability decision, not merely a performance
parameter. It should be chosen to preserve the temporal structure that would
otherwise inflate the null.

## Outputs

See [Encoding Output Reference](encoding_outputs.md) for exact definitions,
units, dimensions, formulas, and interpretation of every map.

## Cluster Execution

The CLI fits one subject at a time. The example SLURM wrapper sets the array
range to the numeric subject IDs and formats each task ID as a BIDS subject
label, such as array task `7` becoming `sub-007`. It is included at
[`examples/encoding_subject_array.sbatch`](../examples/encoding_subject_array.sbatch).
The wrapper intentionally leaves site-specific modules, environments, resource
requests, and submission policy to the caller.

Whole-subject jobs are suitable for small masks and no-permutation analyses.
For large masks or permutation tests, use voxel chunks so work can run in
parallel and stay within scheduler limits. The complete workflow is documented
in [Chunked Encoding And Stitching](encoding_chunked.md), with array and stitch
examples under `examples/`.

## Python API

```python
from pathlib import Path

from fmri_utils.encoding import (
    EncodingConfig,
    fit_encoding,
    fmriprep_nuisance_columns,
    load_run_manifest,
    load_subject_data,
    save_encoding_result,
)

config = EncodingConfig(
    lags=(2, 3, 4, 5),
    ridge_alphas=(0.01, 0.1, 1, 10, 100),
    pca_components=(None, 16, 32, 64),
    nuisance_columns=fmriprep_nuisance_columns(n_acompcor=4),
    n_permutations=200,
)

specs = load_run_manifest(Path("manifests/story_runs.csv"), "sub-001")
data = load_subject_data(specs, config)
result = fit_encoding(data, config)
save_encoding_result(result, data, Path("results/sub-001"))
```

The separation between `load_subject_data()` and `fit_encoding()` is the
extension point for a new preprocessing source. An adapter may construct
`SubjectData` directly without using the CSV manifest.

## Recommended Validation

Before a full campaign:

1. inspect feature/BOLD row correspondence for several known events;
2. plot the nuisance design and residualized BOLD for representative voxels;
3. run one subject and a small mask;
4. verify outer-fold null accuracy/effects on synthetic or permuted data;
5. inspect prediction traces, not only summary maps;
6. record preprocessing root, feature version, alignment offsets, and config in
   project-level provenance.
