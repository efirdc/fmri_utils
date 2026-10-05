# Temporal Continuous Decoding

`fmri_utils.temporal_decoding` fits a continuous scalar target from multivoxel
or other high-dimensional time-series features. It is separate from the legacy
IID decoder because temporal neuroimaging analyses require explicit run
boundaries, blocked folds, and purging around held-out intervals.

The model is trained as follows:

1. Fit feature standardization on training rows.
2. Fit PCA on those standardized training rows.
3. Select PCA components and ridge alpha jointly using nested validation-fold
   Pearson correlation.
4. Refit on all outer-training rows and score held-out rows.
5. Combine story/fold correlations with Fisher-z weights of `n_rows - 3`.

`build_loso_plan` leaves complete runs/stories out. `build_pooled_4x3_plan`
holds out one contiguous block from every run and removes nearby training rows
using a configurable purge distance. Neither splitter shuffles time points.

The caller supplies target variants. Column zero is the observed target and all
remaining columns are null targets. For an HRF-convolved target, shuffle the
raw target within its run, convolve, then pass the variants to the decoder.
Hyperparameters are selected independently for every null variant while PCA
transformations are reused because they depend only on fixed features/folds.
