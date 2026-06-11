# Second-level Python API

Core function:
- `fmri_utils.second_level_analysis.second_level_one_sample_ttest`

## Example

```python
import pandas as pd
from fmri_utils.second_level_analysis import second_level_one_sample_ttest

design = pd.DataFrame(
    {
        "path": [
            "/path/to/sub-001_map.nii.gz",
            "/path/to/sub-002_map.nii.gz",
            "/path/to/sub-003_map.nii.gz",
        ],
        "expertise": [-1.12, 0.34, 0.78],
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

## Key parameters

- `maps`: directory path, CSV/TSV path, DataFrame, list of paths, or ndarray
- `out_dir`: output directory
- `inference`: `parametric` or `non_parametric`
- `mask_image`: optional mask path or `mni_template`
- `template_image`: optional plotting template
- `transformation`: currently supports `fisherz`
- `plot_kwargs`: forwarded to Nilearn `plot_stat_map`

## Notes

- Covariate columns in CSV/DataFrame are modeled as regressors.
- Output includes one folder per regressor (including intercept).
- For non-parametric mode, inference uses Nilearn permutation routines.
