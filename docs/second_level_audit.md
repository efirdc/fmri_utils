# Legacy second-level audit

This audit covers `second_level_one_sample_ttest` in `second_level_analysis.py`. The
function remains available for compatibility, but new analyses should use
`fit_second_level_glm`.

## Findings

### High: corrected thresholds treat t statistics as z statistics

The legacy function requests `output_type="stat"` from `SecondLevelModel`, producing a
t-statistic image. It then passes that image to `threshold_stats_img`, which converts
voxelwise probability thresholds on the standard-normal scale. This is not equivalent to
thresholding a t distribution with the group model's residual degrees of freedom.

The replacement requests and saves both t and z maps, then thresholds only the z map.

### High: integer mask headers can quantize statistical outputs

Nilearn can propagate a binary mask's `uint8` header into returned statistical images.
Saving those images directly allows Nibabel to encode the floating-point range into only
256 stored levels through a slope/intercept. Thresholded zeros may then reload as small
non-zero values. The replacement materializes and saves every statistical image as
unscaled `float32`; tests enforce this output invariant.

### Medium: analysis geometry can change implicitly

When no mask is supplied, the legacy function loads Nilearn's MNI mask and resamples it
to the first input map. It can also rely on `SecondLevelModel` to resample images to a mask
grid. This is convenient, but it can hide mixed image spaces or resolutions.

The replacement requires identical map geometry and fails on a mismatch. Resampling must
be an explicit preprocessing step.

### Medium: design and contrasts are implicit

For CSV/DataFrame inputs, every non-path column becomes a numeric covariate and the
function generates a contrast for every column. Covariates are not centered, so the
intercept may not represent an adjusted group mean. The design rank is reported but a
rank-deficient design is not rejected.

The replacement accepts an explicit design matrix and named contrasts, verifies finite
numeric values, and requires full column rank.

### Maintainability: fitting, thresholding, plotting, and CLI behavior are coupled

The legacy function is approximately 750 lines and handles discovery, transformation,
mask construction, parametric and permutation inference, many threshold grids, plotting,
and Fire argument forwarding. This makes the scientific model difficult to audit from a
call site.

The replacement only fits one explicit group model and writes canonical statistics and
figures. Input discovery and study-specific plotting stay in the calling project.

## Retained strengths

The legacy code performs useful path validation, records arguments, supports streaming
coverage statistics, and offers Nilearn nonparametric inference. Those features can be
migrated into focused utilities when a study requires them; they do not justify using the
legacy parametric threshold path unchanged.
