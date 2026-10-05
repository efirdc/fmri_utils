# Registration fusion: fsaverage vertices in MNI152

`lh.MNI152NLin6Asym_rf.f32` and `rh.MNI152NLin6Asym_rf.f32` give, for every
vertex of the fsaverage (164k) mesh in nilearn's vertex order, its coordinate in
FSL's MNI152 (MNI152NLin6Asym in TemplateFlow terms), in millimetres, as
float32 x, y, z interleaved (163,842 x 3 per hemisphere).

They are the RF-ANTs average mappings of Wu et al. (2018), converted unchanged
from `lh/rh.avgMapping_allSub_RF_ANTs_MNI152_orig_to_fsaverage.mat` (variable
`ras`), downloaded 2026-10-05 from the CBIG repository,
`stable_projects/registration/Wu2017_RegistrationFusion/bin/final_warps_FS5.3`.
Each coordinate was built with `mri_vol2surf --projfrac 0.5`, so it is the
mid-thickness point, averaged over 1,490 GSP subjects' ANTs registrations.

The viewer samples an MNI152NLin6Asym map on fsaverage once per vertex at these
points (as Wu et al. project), instead of carrying fsaverage's MNI305
coordinates through a linear affine. The linear route is a median 2 mm from
these points (90th percentile 3.2-3.6 mm), about a cortical thickness.

License: MIT (CBIG repository). Cite:

Wu J, Ngo GH, Greve DN, Li J, He T, Fischl B, Eickhoff SB, Yeo BTT (2018).
Accurate nonlinear mapping between MNI volumetric and FreeSurfer surface
coordinate systems. Human Brain Mapping 39:3793-3808.
