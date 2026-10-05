# LeBel 2023: Encoding Models And Rating Ablations

`fmri_utils.lebel2023` reproduces, for the LeBel et al. (2023) natural-language
fMRI dataset ([OpenNeuro ds003020](https://openneuro.org/datasets/ds003020),
snapshot 3.1.1), three analyses:

1. **Stimulus encoding.** Language-model features (English1000, BERT, GPT-2 XL)
   predict each voxel's response to 26 hours of spoken stories, scored on a
   held-out story.
2. **Cross-participant encoding.** Each subject's voxels are predicted from the
   other seven subjects' brains, after cleaning a run-locked noise component.
3. **Rating ablation.** A story rating (theory of mind in the original analysis,
   any rating you like) is removed from the predictors, and the drop in held-out
   correlation (Δr) measures how much of the prediction depended on it. This works
   on the stimulus features and on the other brains alike.

An optional, expensive **variance-matched null** asks whether the rating costs
more than removing an equally large random semantic direction.

The code was ported from the analysis scripts that produced the published
results and checked against them on the real data. Features, regressors, the
removed features and a full fit chunk are bit-identical, and the MNI warp agrees
to 1e-7.

## Contents

- [What It Reproduces](#what-it-reproduces)
- [Install](#install)
- [Data](#data)
- [Pipeline: Stimulus Encoding And The Ablation](#pipeline-stimulus-encoding-and-the-ablation)
- [Pipeline: Cross-Participant Encoding And The Ablation](#pipeline-cross-participant-encoding-and-the-ablation)
- [Group Statistics](#group-statistics)
- [Optional: The Variance-Matched Null](#optional-the-variance-matched-null)
- [Running On A Cluster](#running-on-a-cluster)
- [Method Details](#method-details)
- [Python API](#python-api)

## What It Reproduces

With the theory-of-mind (ToM) rating, eight subjects (UTS01-UTS08), averaged over
each subject's best-predicted 5% of voxels:

| feature space | full r | Δr (ToM removed) | Δr as % of r |
|---|---:|---:|---:|
| English1000 | 0.274 | 0.0152 | 5.4% |
| BERT (wordctx10) | 0.324 | 0.0162 | 5.1% |
| GPT-2 XL (layer 24, wordctx10) | 0.346 | 0.0161 | 4.8% |

These are the "Lanczos + rate" models: Lanczos-summed features with a word-rate
column. Δr was positive in all 24 subject-by-feature fits. Removing ToM takes
2.95% of English1000's feature variance and 0.73% of BERT's and GPT-2 XL's.
Per subject (top-5% r / Δr):

| feature | UTS01 | UTS02 | UTS03 | UTS04 | UTS05 | UTS06 | UTS07 | UTS08 |
|---|---|---|---|---|---|---|---|---|
| English1000 | .290 / .0186 | .334 / .0228 | .389 / .0220 | .232 / .0088 | .222 / .0122 | .291 / .0165 | .228 / .0117 | .205 / .0092 |
| BERT | .347 / .0196 | .394 / .0209 | .467 / .0190 | .289 / .0129 | .258 / .0121 | .321 / .0157 | .278 / .0120 | .242 / .0174 |
| GPT-2 XL | .379 / .0190 | .421 / .0196 | .491 / .0176 | .286 / .0132 | .277 / .0104 | .356 / .0183 | .302 / .0127 | .253 / .0179 |

With the optional null, removing an equally large random semantic direction
costs 16-39% of that drop, so 61-84% of it is specific to ToM. The group t on
subject z is 11.3, and all seven Saxe ToM parcels are significant.

The cross-participant encoder reaches a group mean r of 0.16 in cortex (per
subject 0.08-0.17, top 5% 0.38-0.64). That beats the BERT stimulus model in
75-82% of voxels. Both controls sit at r ≈ 0 (mean -0.009 to 0.007).

Removing ToM from the other brains costs **no more** than a random semantic
direction: group mean top-5% z = -0.47.

## Install

```bash
pip install "fmri-utils[lebel,ratings,features] @ git+https://github.com/efirdc/fmri_utils.git"
# lebel: h5py (the responses) and fslpy (building MNI warps); ratings: the LLM SDKs;
# features: torch and transformers (BERT / GPT-2 XL; a GPU for GPT-2 XL)
```

| extra | needed for |
|---|---|
| `h5py` | reading the responses (always) |
| `torch`, `transformers` | building the BERT and GPT-2 XL features |
| `fslpy` | building the column-to-MNI warp matrices (`warp-matrix`) |
| FSL, `pycortex` | estimating registrations yourself (not needed with the shared ones) |
| `anthropic` / `openai` (`[ratings]`) | rating stories with an LLM |

## Data

### The dataset

Download ds003020 3.1.1 from OpenNeuro. Only these parts are read:

```text
derivatives/preprocessed_data/UTS01..UTS08/<story>.hf5   responses (TRs x voxels; the test story has individual_repeats)
derivatives/pycortex-db/UTS0x/transforms/*/mask_thick.nii.gz, reference.nii.gz
derivatives/pycortex-db/UTS0x/anatomicals/raw.nii.gz     (for registration and the MNI warp)
derivatives/TextGrids/<story>.TextGrid                   word timings
derivatives/english1000sm.hf5                            (or code/deep-fMRI-dataset/em_data/)
code/deep-fMRI-dataset/em_data/sess_to_story.json        the training stories
```

With the AWS CLI, `aws s3 sync --no-sign-request s3://openneuro.org/ds003020 ds003020 --exclude "sub-*"`
gets everything but the raw BOLD.

### Shared derived assets

These are dataset-specific outputs of steps that are slow or need tools beyond
Python. They are shared with the lab in three folders, and each can also be
rebuilt as the last column says.

| shared folder | contents | rebuilt by |
|---|---|---|
| `Story transcripts and ToM ratings/transcripts/<story>.json` | punctuated, time-aligned word tables (84 stories): what every rating is made on | Whisper large-v3 + alignment to the TextGrids (not in this package) |
| `Story transcripts and ToM ratings/rubrics/`, `ratings/tom-luna/` | the ToM rubric, its spec, and the ToM rating run | `fmri-story-ratings rate` |
| `Semantic features/tr_level/<feature>/<story>.npy` | english1000, bert_wordctx10, gpt2xl_l24_wordctx10 on the response clock | `fmri-lebel features` |
| `Pipeline assets (fmri_utils.lebel2023)/features/word_rate/` | the word-rate column | `fmri-lebel features --feature word_rate` |
| `Pipeline assets .../registration/UTS0x/` | `func_to_anat.mat`, `anat_to_MNI152_warpcoef.nii.gz`, `MNI_to_anat_warpcoef.nii.gz`, `T1_2mm.nii.gz` (+ `MNI152_T1_2mm.nii.gz`) | `registration.register_subject` (FSL + pycortex) |
| `Pipeline assets .../atlases/` | Harvard-Oxford in each subject's anatomy; the MNI originals; the Saxe ToM parcels (MNI, values 1-7, names) | `registration.atlas_to_anat` (FSL) |
| `Pipeline assets .../warps/UTS0x.npz` | column-to-MNI warp matrices | `fmri-lebel warp-matrix` |

The commands below use these variables:

```bash
DS=/path/to/ds003020                       # the dataset
TRANSCRIPTS=".../Story transcripts and ToM ratings/transcripts"
ASSETS=".../Pipeline assets (fmri_utils.lebel2023)"
FEATURES=$W/features                       # one folder: the three feature spaces and word_rate side by side
W=/path/to/work                            # outputs
```

Put the three `Semantic features/tr_level/` folders and `$ASSETS/features/word_rate`
together in `$FEATURES`. The pipeline addresses feature spaces as
`<root>/<feature>`, and the rating-removed spaces are written next to them.

## Pipeline: Stimulus Encoding And The Ablation

Paths use the variables defined under [Data](#data). The examples rate
`embodiment`; substitute your rating's field name.

### 1. Rate the stories

Write a rubric and a spec for your rating, then rate every story with
`fmri_utils.story_ratings`. The full guide is
[story_ratings.md](story_ratings.md). Use the shared transcripts and the same
settings as the ToM run, so the two ratings are comparable:

```bash
fmri-story-ratings rate \
  --inputs $TRANSCRIPTS \
  --prompt rubrics/embodiment.md --spec rubrics/embodiment_spec.json \
  --output-dir $W/ratings/embodiment-luna \
  --backend openai --model gpt-5.6-luna --reasoning-effort low \
  --segmentation clause --replicates 3 --chunk-size 60 --tr-seconds 2.0
```

The spec's scale `name` is the field (`embodiment` gives `embodiment_mean` per
segment). Check a few stories in the reader (`fmri-story-ratings render`) before
going on.

### 2. Build the rating regressors

```bash
fmri-lebel regressors --dataset $DS --ratings $W/ratings/embodiment-luna \
  --transcripts $TRANSCRIPTS --field embodiment --output $W/regressors/embodiment
```

This writes `<story>.npy` with columns `load` (the word ratings, Lanczos-summed
onto the response clock) and `word_rate`.

### 3. Features

Use the shared features (see [Data](#data)), or build them yourself. English1000 and
word_rate take a minute on a CPU; BERT and GPT-2 XL need a GPU and take about an
hour each for all stories.

```bash
for f in english1000 bert_wordctx10 gpt2xl_l24_wordctx10 word_rate; do
  fmri-lebel features --dataset $DS --feature $f --output $FEATURES
done
```

### 4. Remove the rating from each feature space

```bash
for f in english1000 bert_wordctx10 gpt2xl_l24_wordctx10; do
  fmri-lebel remove --feature-root $FEATURES --feature $f \
    --regressors $W/regressors/embodiment --output-root $FEATURES --suffix _no_embodiment
done
```

Each run prints the share of feature variance removed. It is worth reporting:
for ToM it was 2.95% (English1000) and 0.73% (BERT, GPT-2 XL).

### 5. Fit the full and the rating-removed models

There is one task per subject, feature space, condition (full or removed) and
voxel chunk: 8 × 3 × 2 × 5 = 240. On a desktop, a chunk of about 2,000 voxels
fits in about 20 s, so a whole subject takes about 12 minutes per model.

```bash
fmri-lebel fit --dataset $DS --subject UTS01 --feature-root $FEATURES --feature bert_wordctx10 \
  --extra-root $FEATURES/word_rate --output $W/fits --chunk 0 --n-chunks 5
fmri-lebel fit --dataset $DS --subject UTS01 --feature-root $FEATURES --feature bert_wordctx10_no_embodiment \
  --extra-root $FEATURES/word_rate --output $W/fits --chunk 0 --n-chunks 5
```

`--extra-root word_rate` is the rate column of the default "Lanczos + rate"
model. Leave it out for features only.

Outputs go to `$W/fits/<feature>/<subject>/chunks/`. To get native NIfTI maps,
run `fmri-lebel stitch --dataset $DS --folder $W/fits/<feature>/<subject> --subject <subject>`.

### 6. Summaries

```bash
fmri-lebel summary --full $W/fits/bert_wordctx10 --removed $W/fits/bert_wordctx10_no_embodiment \
  --output $W/summary_bert.json
```

This prints, per subject, the mean r and the mean Δr over the best-predicted 5%
of voxels, and Δr as a share of r: the numbers in the first table above.

## Pipeline: Cross-Participant Encoding And The Ablation

This needs the registrations and the T1-space Harvard-Oxford atlases.

```bash
# per subject: aCompCor cleaning and the 100 predictor components (a few minutes, ~16 GB)
fmri-lebel xsub-prep --dataset $DS --subject UTS01 --atlases $ASSETS/atlases --registration $ASSETS/registration --work $W/xsub
# per subject and chunk, after every prep is done: the encoder, and its two controls
fmri-lebel xsub-fit --dataset $DS --subject UTS01 --work $W/xsub --chunk 0 --n-chunks 5
fmri-lebel xsub-fit --dataset $DS --subject UTS01 --work $W/xsub --chunk 0 --n-chunks 5 --control shifted
fmri-lebel xsub-fit --dataset $DS --subject UTS01 --work $W/xsub --chunk 0 --n-chunks 5 --control crossstory
# the noise ceiling of the cleaned test repeats (optional)
fmri-lebel xsub-noise-ceiling --dataset $DS --subject UTS01 --work $W/xsub
```

Both controls should give r ≈ 0. If they don't, something run-locked survived
the cleaning.

The rating ablation removes the rating from every predictor subject's cleaned
cortical voxels, refits each PCA, and refits the target on the result:

```bash
# per subject (as predictor): components with the rating removed (series id -1)
fmri-lebel xsub-components --dataset $DS --subject UTS01 --work $W/xsub --name embodiment \
  --regressors $W/regressors/embodiment --ids -1
# per subject (as target) and chunk, after all eight components exist
fmri-lebel xsub-ablation-fit --dataset $DS --subject UTS01 --work $W/xsub --name embodiment --ids -1 --chunk 0
fmri-lebel summary --full $W/xsub/fits/xsub_pc100 --xsub-ablation $W/xsub/ablation/embodiment/fits
```

## Group Statistics

Each subject's column maps go to MNI152 2 mm through its warp matrix, `$ASSETS/warps/`.
To build them yourself (needs fslpy and the registration):

```bash
fmri-lebel warp-matrix --dataset $DS --subject UTS01 --registration $ASSETS/registration --output $ASSETS/warps
```

Then, for r or Δr:

```bash
fmri-lebel group --warps $ASSETS/warps --full $W/fits/bert_wordctx10 --removed $W/fits/bert_wordctx10_no_embodiment \
  --fsldir $ASSETS/atlases --atlas tom=$ASSETS/atlases/saxe_tom_parcels/tom_parcels_mni.nii.gz:$ASSETS/atlases/saxe_tom_parcels/tom_parcels_names.json \
  --output $W/group --prefix bert_delta_embodiment
```

This writes `<prefix>_{mean,t,neglog10p,neglog10q,neglog10fwe}_space-MNI152.nii.gz`
and `<prefix>_regions.json`:

| test | reads as |
|---|---|
| voxel t, BH q | one-sample t over the 8 subjects at each voxel, FDR over voxels |
| sign-flip FWE | family-wise p from the maximum \|t\| over all 256 sign flips. With 8 subjects the smallest possible p is 1/256, and only very large effects pass |
| region tests | the subjects' mean within each Harvard-Oxford region (and any `--atlas`), t over subjects, BH over regions. Fewer questions than voxels, so this is where 8 subjects have power |

`--fsldir` points at the Harvard-Oxford MNI images and their XML names: `$ASSETS/atlases`,
or `$FSLDIR` of an FSL install. Without it, only the `--atlas` images are tested. Use
`--xsub-ablation $W/xsub/ablation/<name>/fits` in place of `--removed` for the
cross-participant Δr.

## Optional: The Variance-Matched Null

Removing any five temporal directions costs some prediction, so Δr does not say
whether the rating is special. The null removes random semantic directions of
the same size instead. Each direction is s = (X − mean) w, with w drawn from
N(0, I) and normalised, then orthogonalised on word rate and the rating (lags
0-4 each, plus an intercept). A direction is kept when its lags remove the same
share of feature variance as the rating's, within 10%. 1,000 kept directions,
each fitted like the ablation, give a null Δr distribution per voxel.

The cost is real. Each direction is a full refit, so it is 1,000 × the ablation:
about 1,800 cluster tasks of a few hours each for three feature spaces and eight
subjects.

```bash
fmri-lebel nulls-select --feature-root $FEATURES --feature bert_wordctx10 \
  --regressors $W/regressors/embodiment --output $W/nulls/directions/bert_wordctx10.npz
# per subject x chunk x block of directions (e.g. 0-125, 125-250, ...)
fmri-lebel nulls-fit --dataset $DS --subject UTS01 --feature-root $FEATURES --feature bert_wordctx10 \
  --regressors $W/regressors/embodiment --directions $W/nulls/directions/bert_wordctx10.npz \
  --extra-root $FEATURES/word_rate --output $W/nulls/fits/bert_wordctx10 --chunk 0 --start 0 --stop 125
# the group test: 10,000 draws of one null per subject
fmri-lebel group-nulls --warps $ASSETS/warps --full $W/fits/bert_wordctx10 --removed $W/fits/bert_wordctx10_no_embodiment \
  --nulls $W/nulls/fits/bert_wordctx10 --output $W/group --prefix bert_null_embodiment
```

`group-nulls` writes the group mean drop, net (drop minus null mean), z, p, BH q
and a family-wise p from each draw's maximum z. It is a fixed-effects test: it
asks whether these subjects' drops beat the null, not whether new subjects'
would.

For the cross-participant null, the directions come from English1000 and are
matched on the share of cortical-voxel variance removed (`xsub-select-directions`).
Then `xsub-components` and `xsub-ablation-fit` take `--ids 0:1000` (with
`--directions` and `--null-features`), and `group-nulls` takes `--xsub-ablation`.

## Running On A Cluster

On Slurm, every step except `xsub-select-directions` and `nulls-select` is a job
array over subjects and chunks. A template (Alliance clusters; adjust the account
and modules):

```bash
#!/bin/bash
#SBATCH --account=def-xxx
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --array=0-39                # 8 subjects x 5 chunks
module load StdEnv/2023 python/3.11 scipy-stack
source $HOME/venvs/lebel/bin/activate
SUBJECTS=(UTS01 UTS02 UTS03 UTS04 UTS05 UTS06 UTS07 UTS08)
S=${SUBJECTS[$((SLURM_ARRAY_TASK_ID / 5))]}
C=$((SLURM_ARRAY_TASK_ID % 5))
for f in english1000 bert_wordctx10 gpt2xl_l24_wordctx10; do
  for v in "" _no_embodiment; do
    fmri-lebel fit --dataset $DS --subject $S --feature-root $FEATURES --feature $f$v \
      --extra-root $FEATURES/word_rate --output $W/fits --chunk $C --n-chunks 5
  done
done
```

| step | per task | memory |
|---|---|---|
| `features` (BERT, GPT-2 XL) | about an hour for all stories, one GPU | 40 GB |
| `fit` (one chunk, one model) | minutes | 15 GB |
| `xsub-prep` (one subject) | 5-15 min | 16 GB |
| `xsub-fit`, `xsub-ablation-fit` (one chunk, one series) | minutes | 15 GB |
| `xsub-components` (one subject, one series) | a few minutes | 16 GB |
| `nulls-fit` (one chunk, 125 directions) | a few hours | 15 GB |

Each array task loads the chunk's responses once. Small jobs are better grouped
into one task (as above) than submitted as one task per model, because each
array task waits in the queue separately.

## Method Details

| | |
|---|---|
| subjects | UTS01-UTS08 |
| training stories | each subject's canonical training stories (26; 25 shared by all eight for cross-participant) |
| test story | `wheretheressmoke`, the released mean of its repeats |
| response clock | 2 s TRs, row i centred at 2i + 11 s of story time (10 and 5 TRs trimmed by the release) |
| feature resampling | per word at its midpoint, three-lobe Lanczos sum onto the rows (unnormalised) |
| word rate column | Hann, half-width 2 TRs, words per second; appended after the PCA |
| design | z-score → PCA 256 (randomized, seed 2023) → + rate column → FIR delays 1-4 → z-score |
| ridge | per voxel, α ∈ 10…1e5 in half-decade steps, 5-fold CV over whole training stories (story i in fold i mod 5), Fisher-z-averaged r |
| score | Pearson r on the test story; noise ceiling from its repeats, normalised r = r / max(ceiling, 0.3) |
| rating regressor | word ratings (each word takes its clause's mean over 3 replicate ratings) Lanczos-summed like the features; no HRF |
| removal | the rating regressed on word rate (lags 0-4 + intercept); its residual at lags 0-4 + intercept projected out of the features; both fitted on the training stories, applied to all |
| cross-participant cleaning | 5 nuisance components per run from Harvard-Oxford white matter + lateral ventricles (subcortical labels 1, 3, 12, 14), eroded 1 voxel, ≥ 2 voxels from cortex; every voxel residualised per run |
| cross-participant predictors | cleaned Harvard-Oxford cortical voxels, z-scored, PCA 100 (seed 2023) per subject; 7 × 100 columns at lag 0 |
| MNI | FLIRT 12-DOF + FNIRT (`T1_2_MNI152_2mm`) from the pycortex anatomical; the pycortex functional-to-anatomical transform as the premat; trilinear, NaN-aware |

## Python API

```python
from fmri_utils.lebel2023 import Dataset
from fmri_utils.lebel2023 import ablation, encoding, group, regressors

ds = Dataset("ds003020")
encoding.fit_features(ds, "UTS01", "features", "bert_wordctx10", "fits",
                      extra_root="features/word_rate", chunk=0, n_chunks=5)
maps = encoding.read_chunks("fits/bert_wordctx10/UTS01")       # column vectors
ds.save_map("UTS01", maps["correlation_raw"], "r_UTS01.nii.gz")  # native NIfTI
```

`Dataset` reads responses (`response`, `repeats`), stories (`training_stories`,
`shared_training_stories`), words (`words`) and the functional grid (`unmask`,
`save_map`, `column_volume`). The modules mirror the steps above:

| module | contents |
|---|---|
| `features` | the feature builders |
| `regressors` | rating regressors |
| `ablation` | `orthogonalise`, `remove` |
| `encoding` | `prepare_design`, `fit_voxelwise`, `noise_ceiling`, chunks |
| `cross_participant` | the cross-participant prep, fits and ablation |
| `registration` | `MNIWarp`, `column_labels`, the FSL registration |
| `group` | `sign_flip`, `region_test`, `null_draws`, `top_voxel_summary` |
| `nulls` | `NullSpace`, the null directions and fits |
