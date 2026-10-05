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
| `nulls` | with a variance-matched null: the group mean drop against draws of one null per subject |

Some notes on reading them:
- **Sign flips with 8 subjects:** there are only 256 sign patterns, so the
  smallest family-wise p is 1/256 and only very large effects pass.
- **Region tests** ask fewer questions than voxels, which is where a small group
  has power. `--fsldir` points at FSL's `data/atlases` (or a copy) for
  Harvard-Oxford. `--atlas NAME=IMAGE:NAMES_JSON` adds any template label image.
- **Null draws:** 10,000 draws give z, one-sided p, BH q, and a family-wise p
  from each draw's largest z. This is a fixed-effects test: it asks whether
  these subjects' drops beat the null, not whether new subjects' would.

Inputs are fit folders, `<fits>/<model>` holding `<subject>/chunks`. `--removed`
turns maps into full − removed; `--xsub-ablation` uses a
[cross-participant ablation](cross_participant.md) instead.

## Python

`group_inference` provides `to_template`, `sign_flip`, `bh_adjusted`,
`region_test`, `harvard_oxford`, `null_draws`, `top_voxel_summary` and
`save_results`.
