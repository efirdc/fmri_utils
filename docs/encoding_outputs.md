# Encoding Output Reference

Encoding results use explicit descriptive names rather than a field-wide
standard. This page defines every map written by `fmri-encoding fit` and
`fmri-encoding stitch`.

## Spatial And Fold Axes

- A 3D map contains one value per modeled voxel.
- A 4D map contains one volume per outer CV fold. Volume order is recorded by
  `outer_fold_axis` or `fold_ids` in the encoding JSON.
- Voxels outside the selected model mask are `NaN`, not zero.
- Correlation maps contain Pearson `r`. Null summary maps with
  `_fisher_z` in their name contain `atanh(r)`, not `r`.

## Predictive Performance

### `mean_correlation`

One 3D map of outer-test Pearson correlations. Correlation is first computed
independently in each outer fold. Fold correlations are transformed with
Fisher `z = atanh(r)`, averaged using `n_test_rows - 3` weights, and transformed
back with `tanh`. It is therefore not the correlation of traces concatenated
across folds.

### `outer_fold_correlation`

A 4D map of Pearson correlations, one volume per outer test fold. These values
are useful for fold/run stability checks and are the inputs to
`mean_correlation`.

### `mean_r2`

One 3D map of outer-test coefficient of determination (`R2`). R2 is computed
inside each outer fold and averaged by the number of test rows. Negative values
are valid and mean that predictions performed worse than the test-fold mean
baseline.

### `outer_fold_r2`

A 4D map of fold-local outer-test R2, one volume per outer fold.

## Selected Model Settings

### `outer_selected_alpha`

A 4D map of the ridge penalty selected independently for every voxel by inner
CV, one volume per outer model. Larger alpha means stronger L2 shrinkage. Alpha
is the `alpha` parameter used by `sklearn.linear_model.Ridge`.

### `mean_selected_alpha`

One 3D map containing the arithmetic mean of `outer_selected_alpha` across
outer folds. Inspect the 4D map when selection stability matters.

### `outer_selected_pca_components`

A 4D map of the PCA dimensionality selected jointly with alpha by inner CV,
one volume per outer model. The value `0` means PCA was disabled for that grid
point; it does not mean zero components were fitted.

### `mode_selected_pca_components`

One 3D map containing the most frequent selected PCA dimensionality across
outer folds. Ties use the smallest value. This is a categorical summary, not
an average.

## Permutation Null Maps

These maps are `NaN` when `n_permutations=0`. For each permutation, feature
rows are shuffled in blocks independently within runs, while BOLD and CV
partitions remain fixed. The complete nested model-selection procedure is
repeated.

Let `z_obs = atanh(mean_correlation)` and let `z_perm` be the corresponding
value from each permutation.

### `null_mean_fisher_z`

One 3D map of `mean(z_perm)`. Units are Fisher z.

### `null_std_fisher_z`

One 3D map of the population standard deviation of `z_perm` (`ddof=0`). Units
are Fisher z.

### `null_standardized_effect_fisher_z`

One 3D map calculated as:

```text
(z_obs - mean(z_perm)) / std(z_perm)
```

This expresses displacement from the permutation null in null standard
deviations. It is not a classical regression coefficient or group-level
z-statistic.

### `empirical_p`

One 3D map of two-sided empirical permutation p-values with the +1 correction:

```text
(count(abs(z_perm) >= abs(z_obs)) + 1) / (n_permutations + 1)
```

These p-values are voxelwise and are not corrected for testing many voxels.

### `signed_z`

One 3D map converting `empirical_p` to a two-sided standard-normal magnitude,
then assigning the sign of `z_obs - null_mean_fisher_z`:

```text
sign(z_obs - null_mean) * NormalInverseSurvival(empirical_p / 2)
```

This is a signed visualization of empirical significance. It is capped in
practice by the number of permutations and is not a parametric model z-score.

## Non-Map Outputs

### `outer_fold_traces.npz`

For a non-chunked fit, this contains observed and predicted residualized-BOLD
traces for each outer fold, plus the source run and BOLD row for every test
sample. Chunked fits omit traces by default to reduce storage. Pass
`fit-chunk --save-traces` to reconstruct them during stitching.

### `encoding.json`

Records model settings, fold definitions, run order, input provenance, mask
geometry, and output paths. A stitched result also records the expected chunk
manifest, chunk files, and common configuration signature.
