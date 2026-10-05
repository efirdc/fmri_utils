from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_utils.first_level import FirstLevelConfig, combine_fixed_effects, fit_first_level_run
from fmri_utils.group_level import fit_second_level_glm
from fmri_utils.second_level_analysis import second_level_one_sample_ttest
from nilearn.glm import threshold_stats_img


def synthetic_run(seed: int = 1) -> tuple[nib.Nifti1Image, pd.DataFrame, nib.Nifti1Image]:
    rng = np.random.default_rng(seed)
    shape = (5, 5, 5, 48)
    data = rng.normal(100.0, 0.5, size=shape).astype(np.float32)
    data[2, 2, 2, 12:20] += 2.0
    image = nib.Nifti1Image(data, np.eye(4))
    mask = nib.Nifti1Image(np.ones(shape[:3], dtype=np.uint8), np.eye(4))
    events = pd.DataFrame(
        {"onset": [12.0], "duration": [8.0], "trial_type": ["negative_content"]}
    )
    return image, events, mask


def test_first_level_and_fixed_effects(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image, events, mask = synthetic_run()
    config = FirstLevelConfig(
        t_r=1.0, hrf_model="spm", drift_model=None, high_pass=None,
        noise_model="ols", signal_scaling=False,
    )
    first = fit_first_level_run(
        image, events, {"negative": "negative_content"}, tmp_path / "run-1",
        config=config, mask_img=mask,
    )
    contrast = first.contrasts["negative"]
    for path in contrast.__dict__.values():
        assert Path(path).exists()
        assert nib.load(path).get_data_dtype() == np.dtype(np.float32)
    assert "constant" in pd.read_csv(first.design_matrix_csv, index_col=0).columns

    fixed = combine_fixed_effects(
        [contrast.effect_size, contrast.effect_size],
        [contrast.effect_variance, contrast.effect_variance],
        tmp_path / "fixed",
        mask_img=mask,
    )
    assert fixed.z_score.exists()
    assert nib.load(fixed.z_score).get_data_dtype() == np.dtype(np.float32)


def test_second_level_saves_z_thresholds(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(6):
        data = np.full((5, 5, 5), 0.5 + index * 0.02, dtype=np.float32)
        path = tmp_path / f"sub-{index}.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)
        paths.append(path)
    mask = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))
    result = fit_second_level_glm(
        paths,
        pd.DataFrame({"intercept": np.ones(6)}),
        {"mean": "intercept"},
        tmp_path / "group",
        mask_img=mask,
    )
    outputs = result.contrasts["mean"]
    assert outputs.z_score.exists()
    assert nib.load(outputs.z_score).get_data_dtype() == np.dtype(np.float32)
    assert outputs.fdr_thresholded_z.exists()
    assert json_load(tmp_path / "group" / "model_metadata.json")["thresholding"]["input_statistic"] == "z_score"


def test_second_level_rejects_rank_deficient_design(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(3):
        path = tmp_path / f"map-{index}.nii.gz"
        nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)).to_filename(path)
        paths.append(path)
    design = pd.DataFrame({"intercept": np.ones(3), "duplicate": np.ones(3)})
    try:
        fit_second_level_glm(paths, design, {"mean": "intercept"}, tmp_path / "bad")
    except ValueError as error:
        assert "rank deficient" in str(error)
    else:
        raise AssertionError("Expected rank-deficient group design to fail")


def test_legacy_second_level_thresholds_z_and_saves_float32(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    paths = []
    for index in range(8):
        data = rng.normal(0.0, 1.0, size=(5, 5, 5)).astype(np.float32)
        data[2, 2, 2] += 1.5
        path = tmp_path / f"legacy-sub-{index}.nii.gz"
        nib.Nifti1Image(data, np.eye(4)).to_filename(path)
        paths.append(path)
    mask = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))
    out_dir = tmp_path / "legacy-group"
    second_level_one_sample_ttest(
        paths,
        out_dir=out_dir,
        mask_image=mask,
        inference="parametric",
        height_control="fpr",
        alpha=0.05,
        cluster_threshold=0,
        unc_p_threshold_grid=(0.001,),
        unc_cluster_threshold_grid=(0,),
        overwrite=True,
    )

    reg_dir = out_dir / "intercept"
    t_img = nib.load(reg_dir / "tmap_unc.nii.gz")
    z_img = nib.load(reg_dir / "zmap_unc.nii.gz")
    corrected = nib.load(reg_dir / "tmap_fpr_alpha0p05_k0.nii.gz")
    expected_z, _ = threshold_stats_img(
        z_img, alpha=0.05, height_control="fpr", cluster_threshold=0,
        two_sided=True, mask_img=mask,
    )
    expected_mask = np.asarray(expected_z.dataobj) != 0
    corrected_data = np.asarray(corrected.dataobj)
    assert np.array_equal(corrected_data != 0, expected_mask)
    assert np.allclose(corrected_data[expected_mask], np.asarray(t_img.dataobj)[expected_mask])
    assert corrected.get_data_dtype() == np.dtype(np.float32)
    assert t_img.get_data_dtype() == np.dtype(np.float32)
    assert z_img.get_data_dtype() == np.dtype(np.float32)


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
