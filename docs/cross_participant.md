# Cross-Participant Encoding

When every subject hears or sees the same stimulus, the other subjects'
responses predict a subject's response, with everything shared rather than
only what a feature space names. `fmri_utils.cross_participant` builds that
encoder on [column data](encoding_columns.md), cleans the run-locked noise it
would otherwise exploit, and ablates an annotation from the predictor brains.

```bash
fmri-cross-participant columns --run-table runs.csv --subject S01 \
  --cortical-atlas atlases/S01/HarvardOxford-cort_space-T1.nii.gz \
  --subcortical-atlas atlases/S01/HarvardOxford-sub_space-T1.nii.gz \
  --func-to-anat registration/S01/func_to_anat.mat --output xsub/columns
fmri-cross-participant prep --run-table runs.csv --subject S01 --work xsub --columns xsub/columns
fmri-cross-participant fit  --run-table runs.csv --subject S01 --work xsub --chunk 0 --n-chunks 5
```

## Steps

**`columns`.** For each subject, this picks the predictor columns and the
nuisance-source columns from atlases in the subject's anatomy, carried to the
functional grid through a FLIRT-convention matrix (`fsl_transforms.column_labels`):
- **predictors:** cortex, i.e. cortical-atlas labels above 0;
- **nuisance source:** white matter and lateral ventricles (Harvard-Oxford
  subcortical labels 1, 3, 12, 14 by default), eroded by one voxel, minus
  anything within two voxels of cortex (`acompcor_source`).

**`prep`.** Each run gets an aCompCor nuisance set. This covers the training
runs every subject has, plus the test run.
- The set is the top 5 principal time courses of the run's source columns, each
  z-scored within the run.
- Every column is residualised on its run's set plus an intercept. Released
  responses are often only motion-corrected, detrended and z-scored, and share
  a run-locked component across subjects that this removes.
- The cleaned predictor columns are z-scored over the training runs and reduced
  to 100 principal components fitted there. The test run is projected onto that
  basis.

**`fit`.** The design is the other subjects' components side by side (lag 0:
everyone is on the same clock). The target's columns are cleaned with their own
runs' nuisance sets, then fitted with voxelwise ridge and a fixed test run.
Two controls should both give r ≈ 0; what survives them is locked to the run,
not the stimulus:
- `--control shifted`: the predictors' test run rolled by half;
- `--control crossstory`: the predictors' rows of another, training run, from
  its start.

**`noise-ceiling`.** The ceiling of the raw and of the cleaned test repeats.

## Ablating An Annotation

```bash
fmri-cross-participant components   --run-table runs.csv --subject S01 --work xsub --name rating --regressors regressors
fmri-cross-participant ablation-fit --run-table runs.csv --subject S01 --work xsub --name rating --chunk 0
fmri-group-inference summary --full xsub/fits/xsub_pc100 --xsub-ablation xsub/ablation/rating/fits --subjects S01,...
```

The orthogonalised annotation ([ablation](ablation.md)) at lags 0-4 plus an
intercept is regressed out of every predictor subject's cleaned predictor
columns, fitted on the training runs. Each subject's PCA is refitted on what is
left, and every target is refitted on those components. Δr is the full fit
minus this one.

The optional null uses random semantic directions of a feature space (`--ids
0:1000` with `--directions` and `--null-features`). They come from
`select-directions` and are kept when they remove as much predictor-column
variance as the annotation, measured on a sample of every subject's predictor
columns.
