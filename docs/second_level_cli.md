# Second-level CLI

## Entrypoints

Console script:

```bash
fmri-utils second_level /path/to/maps_or_design
```

Module mode:

```bash
python -m fmri_utils.second_level_analysis second_level /path/to/maps_or_design
```

## Input modes

The `maps` argument can be:
- directory with subject NIfTI maps (`.nii` / `.nii.gz`, non-recursive)
- CSV/TSV with `path` (or `map_path`) plus numeric covariates

Example CSV:

```csv
path,expertise,is_patient
/path/to/sub-001_map.nii.gz,-1.12,0
/path/to/sub-002_map.nii.gz,0.34,1
/path/to/sub-003_map.nii.gz,0.78,0
```

## Common usage

Parametric:

```bash
fmri-utils second_level /path/to/design.csv \
  --inference=parametric \
  --height_control=fdr \
  --cluster_threshold=20
```

Non-parametric:

```bash
fmri-utils second_level /path/to/mni_subject_maps \
  --inference=non_parametric \
  --n_perm=5000 \
  --n_jobs=8 \
  --mask_image=mni_template \
  --overwrite=True
```

## Useful flags

- `--inference=parametric|non_parametric`
- `--n_perm=<int>`
- `--mask_image=<path|mni_template>`
- `--transformation=fisherz`
- `--overwrite=True|False`
- `--plot_kwargs="{...}"`

## Output structure

Outputs are written under `out_dir` (or default `group/`), with one subfolder per regressor including intercept:

- `.../group/intercept/`
- `.../group/<covariate>/`

An `args.json` file is also written with run settings.
