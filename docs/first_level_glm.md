# First-level GLM

`fit_first_level_run` is a small wrapper around Nilearn's `FirstLevelModel`. It fits one
run at a time so events, nuisance regressors, drift terms, and temporal noise are always
run-specific.

```python
from fmri_utils import FirstLevelConfig, fit_first_level_run

result = fit_first_level_run(
    run_img="sub-01_task-reading_run-1_bold.nii.gz",
    events=events_dataframe,
    confounds=confounds_dataframe,
    contrasts={"negative_content": "negative_content"},
    output_dir="results/sub-01/run-1",
    mask_img="analysis_mask.nii.gz",
    config=FirstLevelConfig(
        t_r=1.5,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ar1",
    ),
)
```

## Inputs

- `events` must contain `onset`, `duration`, and `trial_type` in seconds.
- `confounds` must contain one finite numeric row per acquired volume. Do not add an
  intercept; Nilearn creates a run intercept.
- Fit separate runs separately when nuisance and drift effects must not be shared.
- `signal_scaling=0` expresses effects relative to the run mean. Set it to `False` only
  when the input is already on a meaningful common scale.

## Outputs

Every contrast stores effect size, effect variance, t, z, and p maps. The design matrix,
events, confounds, model configuration, and a design-matrix figure are also retained.
Use `combine_fixed_effects` to combine matching run contrasts within a subject.

Statistical images are explicitly written as unscaled `float32`. This avoids quantizing
results when Nilearn propagates an integer analysis-mask header into a contrast image.

The effect-size maps, not subject z maps, are the inputs to a second-level model.
