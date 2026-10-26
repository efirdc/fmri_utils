# Group Inference On Encoding Maps

`fmri_utils.group_inference` tests subjects' encoding maps (r, Δr, a
null-standardised drop) in a common template space. Each map is a vector over
the subject's response columns ([column data](encoding_columns.md)), carried to
MNI152 2 mm by a sparse warp matrix from `fsl_transforms`.

```bash
fmri-group-inference warp-matrix --run-table runs.csv --subject S01 \
  --anatomical "anat/{subject}.nii.gz" --warpcoef "reg/{subject}/anat_to_MNI152_warpcoef.nii.gz" \
  --func-to-anat "reg/{subject}/func_to_anat.mat" --template MNI152_T1_2mm.nii.gz --output warps
fmri-group-inference voxel --warps warps --full fits/bert --removed fits/bert_removed \
  --subjects S01,S02,... --fsldir $FSLDIR --output group --prefix bert_delta
fmri-group-inference summary --full fits/bert --removed fits/bert_removed --subjects S01,S02,...
```

## The Warp

`fsl_transforms.ColumnMNIWarp.build` combines:
- a FLIRT-convention functional-to-anatomy matrix;
- an anatomy-to-template FNIRT coefficient field, read with `fslpy`.

It builds one sparse trilinear matrix per subject. Values are NaN-aware, and a
template voxel is kept when its nearest column has data and it falls inside the
functional volume. This is what `applywarp --premat` gives with a separately
warped support mask, and it was checked against it to 1e-6.

Save the matrix once; `apply` and `volume` then warp any map in milliseconds.
`fsl_transforms` also wraps the FSL commands that estimate the transforms
(`register_anatomical`: FLIRT, FNIRT, invwarp), puts an MNI atlas into an
anatomy (`atlas_to_anatomical`), and exports a pycortex transform
(`pycortex_func_to_anat`).

## Tests

| command | test |
|---|---|
| `voxel` | one-sample t over subjects at each voxel, two-sided p, Benjamini-Hochberg q, and a family-wise p from the maximum \|t\| over all 2^n sign flips |
| `voxel` with `--fsldir` / `--atlas` | region tests: each subject's mean within each region, t over subjects, BH over regions |
| `summary` | per subject, over its best-predicted 5% of voxels: mean r, mean Δr, Δr as a share of r |
| `nulls --paired` | with a variance-matched null: the group mean drop against group nulls, null direction k averaged over subjects |
| `nulls` | the same against random draws of one null per subject (unpaired) |
| `group-nulls` | the group mean drop against saved group nulls (`nulls --paired --save-group-nulls`) |

Some notes on reading them:
- **Sign flips with 8 subjects:** there are only 256 sign patterns, so the
  smallest family-wise p is 1/256 and only very large effects pass.
- **Region tests** ask fewer questions than voxels, which is where a small group
  has power. `--fsldir` points at FSL's `data/atlases` (or a copy) for
  Harvard-Oxford. `--atlas NAME=IMAGE:NAMES_JSON` adds any template label image.
- **The null test** standardises the group mean at each voxel by the group
  nulls' mean and sd, z = (mean − null mean) / null sd. The family-wise cut-off
  is the 95th percentile of each group null's largest z over the brain. It also
  gives each voxel's own empirical one-sided p, (1 + group nulls at least as
  large) / (n + 1), and BH q. With 1,000 nulls p takes only the values k/1001
  (smallest 1/1001), so read it uncorrected at those levels; BH over a brain
  rarely passes. Nulls are not pooled over voxels, which would assume every
  voxel's standardised nulls have the same shape. It is a fixed-effects test: it asks
  whether these subjects' drops beat the null, not whether new subjects' would.
- **Paired or not.** One directions file serves every subject, so null k is the
  same direction in each. `--paired` averages direction k over subjects, giving
  as many group nulls as directions (FWE floor 1/1000 with 1,000), and matches
  the observed mean, which removes the same annotation from everyone. Unpaired
  draws mix different directions across subjects; if a direction's cost is
  correlated across subjects, they understate the null's spread.
- **Saved group nulls.** A null drop involves no annotation, so
  `--save-group-nulls` writes group nulls (float16, about 0.5 GB per 1,000 maps)
  that serve any annotation the directions are variance-matched to; check with
  `fmri-ablation nulls-check`, then test with `group-nulls`.

Inputs are fit folders, `<fits>/<model>` holding `<subject>/chunks`. `--removed`
turns maps into full − removed; `--xsub-ablation` uses a
[cross-participant ablation](cross_participant.md) instead.

## Python

`group_inference` provides `to_template`, `sign_flip`, `bh_adjusted`,
`region_test`, `harvard_oxford`, `null_draws` (`paired=`), `against_group_nulls`,
`save_group_nulls`, `group_nulls_test`, `top_voxel_summary` and `save_results`.
