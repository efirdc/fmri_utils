# Group-level GLM

`fit_second_level_glm` wraps Nilearn's `SecondLevelModel` with an explicit design matrix
and named contrasts.

```python
from fmri_utils import fit_second_level_glm

result = fit_second_level_glm(
    subject_effect_maps,
    design_matrix,
    contrasts={
        "group_mean": "intercept",
        "group_b_minus_a": "group_b",
    },
    output_dir="results/group",
    subject_ids=subject_ids,
    mask_img="group_mask.nii.gz",
)
```

## Requirements

- Input maps must be first-level effect-size maps with identical geometry.
- Rows of the design matrix and map list must have exactly the same order.
- The design must be numeric, finite, and full rank.
- Center continuous nuisance covariates before fitting if the intercept should represent
  the adjusted group mean.

## Statistical maps

The utility saves effect size, effect variance, t, z, and p maps independently. FDR and
uncorrected voxel thresholds are applied to the **z map**. Do not pass a t map to
`nilearn.glm.threshold_stats_img`; its p-value threshold conversion assumes z statistics.
All statistical and thresholded maps are saved as unscaled `float32`, regardless of the
analysis mask's header data type.

The older `second_level_one_sample_ttest` function combines discovery, fitting,
thresholding, and plotting in one large compatibility API. New analyses should use the
explicit `fit_second_level_glm` interface.
