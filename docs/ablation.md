# Ablating An Annotation From Encoding Models

How much of an encoding model's prediction depends on what an annotation
measures? The annotation can be a rating of the stimulus, or a label time
course. Fit the model on the full features and on the features with the
annotation's direction removed, then compare held-out correlations voxel by
voxel: **Δr = r_full − r_removed**. `fmri_utils.ablation` does the removal and
the optional variance-matched null. [Cross-participant encoding](cross_participant.md)
removes the annotation from other subjects' brains instead.

```bash
fmri-story-ratings regressors --run ratings/my-run --stimuli stimuli.csv --field embodiment --output regressors
fmri-ablation remove --feature-root features --feature bert --regressors regressors --test-run test_story
fmri-encoding fit-columns ... --feature bert          # and again with --feature bert_removed
fmri-group-inference summary --full fits/bert --removed fits/bert_removed --subjects S01,S02,...
```

## What Is Removed

Regressor files are `<run>.npy` with the annotation in column 0 and a covariate
in column 1. `fmri-story-ratings regressors` writes the rating's Lanczos sum and
the word rate, on each stimulus's clock.

1. The annotation is regressed on the covariate at lags 0-4 plus an intercept,
   so what is removed is the annotation beyond the covariate (for a rating:
   beyond speech rate).
2. That residual at lags 0-4 (zero-padded per run, like the model's FIR delays)
   plus an intercept spans the removed subspace. Lags matter: a model with
   delays 1-4 would otherwise reach the annotation through a delayed copy.
3. Every feature column is regressed on that subspace and replaced by its
   residual.

Both regressions are fitted on the training runs (all but `--test-run`) and
applied unchanged to every run. The removed space is written next to the
original as `<feature>_removed`. `removal.json` records the mean share of
feature variance removed, which is worth reporting with Δr.

## The Variance-Matched Null (Optional)

Removing any few temporal directions costs some prediction, so Δr alone does
not show that the annotation is special. The null removes random semantic
directions of the features instead:

- s = (X − mean) w, with w ~ N(0, I) normalised;
- orthogonalised on the covariate, the annotation (lags 0-4 each) and an
  intercept;
- kept when removing it removes as much feature variance as removing the
  annotation: |f(s) / f(annotation) − 1| ≤ 0.1, with f the mean share over runs.

Each kept direction is fitted exactly like the ablation, so each voxel gets a
null Δr distribution: net = Δr − its mean, z = net / its sd.

```bash
fmri-ablation nulls-select --feature-root features --feature bert --regressors regressors \
  --test-run test_story --output nulls/bert.npz
fmri-ablation nulls-fit --run-table runs.csv --subject S01 --feature-root features --feature bert \
  --regressors regressors --test-run test_story --directions nulls/bert.npz \
  --extra-root features/word_rate --output nullfits/bert --chunk 0 --start 0 --stop 125
fmri-group-inference nulls --warps warps --full fits/bert --removed fits/bert_removed \
  --nulls nullfits/bert --subjects S01,S02,... --output group --prefix bert_null
```

The cost is real: each direction is a full refit, so 1,000 of them cost 1,000
times the ablation. `nulls-fit` checkpoints every 25 directions and resumes.

## Python

`ablation` provides:
- removal: `orthogonalise`, `remove`, `build_removed_features`;
- the null: `NullSpace` (`null_series`, `removed`, `project_out`),
  `select_directions`, `fit_nulls`, `read_nulls`, and `score` (per-voxel net,
  z and one-sided p against a null).
