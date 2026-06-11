from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_utils.montage import (
    MontageConfig,
    OutlineRegion,
    OutlineSpec,
    SubjectMap,
    _compute_background_with_darken,
    render_axial_subject_montage_series,
)


def _img(data: np.ndarray, affine: np.ndarray | None = None) -> nib.Nifti1Image:
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)


class MontageTests(unittest.TestCase):
    def test_alignment_validation_fails_on_affine_mismatch(self) -> None:
        ref = _img(np.ones((8, 8, 3), dtype=np.float32))
        eff = _img(np.ones((8, 8, 3), dtype=np.float32), affine=np.diag([2, 1, 1, 1]).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                render_axial_subject_montage_series(
                    [SubjectMap(subject_id="sub-0001", effect_img=eff)],
                    ref,
                    Path(tmp),
                    MontageConfig(mode="solid", slice_indices=(1,)),
                )

    def test_thresholded_mode_creates_expected_files(self) -> None:
        ref = _img(np.ones((10, 10, 4), dtype=np.float32))
        a = np.zeros((10, 10, 4), dtype=np.float32)
        b = np.zeros((10, 10, 4), dtype=np.float32)
        a[3, 4, 2] = 4.0
        b[6, 4, 2] = -4.0
        subject_maps = [SubjectMap("sub-0001", _img(a)), SubjectMap("sub-0002", _img(b))]
        cfg = MontageConfig(
            mode="thresholded",
            tail="two_sided",
            p_thresholds=(0.01,),
            slice_indices=(2,),
            grid_rows=1,
            grid_cols=2,
            filename_prefix="test_metric",
            cbar_label="effect",
            show_legend=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            created = render_axial_subject_montage_series(subject_maps, ref, Path(tmp), cfg)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())
            self.assertEqual(created[0].name, "test_metric_axial_z_002.png")

    def test_include_mask_darkening_only_affects_outside_mask(self) -> None:
        bg = np.ones((4, 4), dtype=np.float32)
        include = np.zeros((4, 4), dtype=bool)
        include[:2, :] = True
        out = _compute_background_with_darken(
            background_slice=bg,
            include_slice=include,
            darken_outside_include_mask=True,
            darken_factor=0.5,
        )
        self.assertTrue(np.allclose(out[:2, :], 1.0))
        self.assertTrue(np.allclose(out[2:, :], 0.5))

    def test_outline_missing_regions_does_not_fail(self) -> None:
        ref = _img(np.ones((9, 9, 3), dtype=np.float32))
        effect = _img(np.random.default_rng(0).normal(size=(9, 9, 3)).astype(np.float32))
        label_img = _img(np.zeros((9, 9, 3), dtype=np.float32))
        outline = OutlineSpec(
            label_img=label_img,
            regions=[OutlineRegion(region_id=99, label="Missing", color="#ff0000")],
            alpha=1.0,
            linewidth=0.7,
        )
        cfg = MontageConfig(mode="solid", slice_indices=(0, 1), grid_rows=1, grid_cols=1, filename_prefix="outline")
        with tempfile.TemporaryDirectory() as tmp:
            created = render_axial_subject_montage_series([SubjectMap("sub-0001", effect)], ref, Path(tmp), cfg, outline_spec=outline)
            self.assertEqual(len(created), 2)
            for path in created:
                self.assertTrue(path.exists())

    def test_golden_smoke_two_subjects_two_slices_both_modes(self) -> None:
        ref = _img(np.ones((10, 10, 4), dtype=np.float32))
        s1 = np.zeros((10, 10, 4), dtype=np.float32)
        s2 = np.zeros((10, 10, 4), dtype=np.float32)
        s1[2:5, 2:5, :] = 0.6
        s2[5:8, 5:8, :] = -0.4
        subject_maps = [SubjectMap("sub-0001", _img(s1)), SubjectMap("sub-0002", _img(s2))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            solid_cfg = MontageConfig(mode="solid", slice_indices=(1, 2), grid_rows=1, grid_cols=2, filename_prefix="golden_solid")
            thr_cfg = MontageConfig(
                mode="thresholded",
                p_thresholds=(0.05,),
                tail="two_sided",
                slice_indices=(1, 2),
                grid_rows=1,
                grid_cols=2,
                filename_prefix="golden_thr",
            )
            solid_created = render_axial_subject_montage_series(subject_maps, ref, out / "solid", solid_cfg)
            thr_created = render_axial_subject_montage_series(subject_maps, ref, out / "thr", thr_cfg)
            self.assertEqual(len(solid_created), 2)
            self.assertEqual(len(thr_created), 2)
            self.assertTrue(all(p.exists() for p in solid_created + thr_created))


if __name__ == "__main__":
    unittest.main()
