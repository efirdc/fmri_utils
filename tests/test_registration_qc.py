from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_utils.registration_qc import (
    RegistrationQcConfig,
    _choose_slice_indices,
    _display_plane_spacings,
    _resize_to_physical_aspect,
    _segmentation_slice_to_contour_mask,
    run_registration_qc,
)

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(img, str(path))
    return path


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class RegistrationQcTests(unittest.TestCase):
    def test_manifest_missing_required_column_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "reading",
                        "run_label": "run-1",
                        "reference_img_path": "a.nii.gz",
                        "coreg_img_path": "b.nii.gz",
                        "reference_mask_path": "c.nii.gz",
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                ],
            )
            with self.assertRaises(ValueError):
                run_registration_qc(manifest, root / "out")

    def test_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "reading",
                        "run_label": "run-1",
                        "reference_img_path": str(root / "missing_ref.nii.gz"),
                        "coreg_img_path": str(root / "missing_coreg.nii.gz"),
                        "reference_mask_path": str(root / "missing_ref_mask.nii.gz"),
                        "coreg_mask_path": str(root / "missing_coreg_mask.nii.gz"),
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                    "coreg_mask_path",
                ],
            )
            with self.assertRaises(FileNotFoundError):
                run_registration_qc(manifest, root / "out")

    def test_shape_affine_mismatch_fails_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = _save_nifti(root / "ref.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            coreg = _save_nifti(root / "coreg.nii.gz", np.ones((7, 6, 6), dtype=np.float32))
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            coreg_mask = _save_nifti(root / "coreg_mask.nii.gz", np.ones((6, 6, 6), dtype=np.float32))

            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "reading",
                        "run_label": "run-1",
                        "reference_img_path": str(ref),
                        "coreg_img_path": str(coreg),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                    }
                ],
                fieldnames=list(
                    [
                        "subject_id",
                        "task",
                        "run_label",
                        "reference_img_path",
                        "coreg_img_path",
                        "reference_mask_path",
                        "coreg_mask_path",
                    ]
                ),
            )
            with self.assertRaises(ValueError):
                run_registration_qc(
                    manifest,
                    root / "out",
                    config=RegistrationQcConfig(auto_resample_to_reference=False),
                )

    def test_shape_mismatch_auto_resamples_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = _save_nifti(root / "ref.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            coreg = _save_nifti(root / "coreg.nii.gz", np.ones((7, 6, 6), dtype=np.float32))
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            coreg_mask = _save_nifti(root / "coreg_mask.nii.gz", np.ones((7, 6, 6), dtype=np.float32))
            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "reading",
                        "run_label": "run-1",
                        "reference_img_path": str(ref),
                        "coreg_img_path": str(coreg),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                    "coreg_mask_path",
                ],
            )
            out_dir = root / "out"
            run_registration_qc(manifest, out_dir)
            with (out_dir / "run_metrics.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["auto_resampled_to_reference"]), 1)

    def test_metric_correctness_and_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_img = _save_nifti(root / "ref.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            coreg_img = _save_nifti(root / "coreg.nii.gz", np.ones((6, 6, 6), dtype=np.float32) * 2.0)

            ref_mask_data = np.zeros((6, 6, 6), dtype=np.float32)
            coreg_mask_data = np.zeros((6, 6, 6), dtype=np.float32)
            ref_mask_data[1:4, 1:4, 1:4] = 1.0
            coreg_mask_data[2:5, 1:4, 1:4] = 1.0
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", ref_mask_data)
            coreg_mask = _save_nifti(root / "coreg_mask.nii.gz", coreg_mask_data)

            transform = root / "xfm.txt"
            transform.write_text(
                "Transform: AffineTransform_double_3_3\n"
                "Parameters: 0 -1 0 1 0 0 0 0 1 3 4 0\n"
                "FixedParameters: 0 0 0\n",
                encoding="utf-8",
            )

            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "reading",
                        "run_label": "run-1",
                        "reference_img_path": str(ref_img),
                        "coreg_img_path": str(coreg_img),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                        "affine_transform_path": str(transform),
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                    "coreg_mask_path",
                    "affine_transform_path",
                ],
            )

            out_dir = root / "out"
            run_registration_qc(manifest, out_dir)

            self.assertTrue((out_dir / "run_metrics.csv").exists())
            self.assertTrue((out_dir / "subject_metrics.csv").exists())
            self.assertTrue((out_dir / "qc_summary.json").exists())
            self.assertTrue((out_dir / "overview_heatmap.png").exists())

            with (out_dir / "run_metrics.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            inter = 18.0
            union = 36.0
            expected_jaccard = inter / union
            self.assertAlmostEqual(float(row["jaccard"]), expected_jaccard, places=6)
            self.assertAlmostEqual(float(row["pct_reference_covered_by_coreg_mask"]), inter / 27.0 * 100.0, places=6)
            self.assertAlmostEqual(float(row["pct_coreg_mask_within_reference"]), inter / 27.0 * 100.0, places=6)
            self.assertAlmostEqual(float(row["transform_translation_norm_mm"]), 5.0, places=5)
            self.assertAlmostEqual(float(row["transform_rotation_deg"]), 90.0, places=5)

            metric_pngs = [p for p in out_dir.glob("*.png") if p.name != "overview_heatmap.png"]
            self.assertEqual(metric_pngs, [])

    def test_qualitative_only_row_blanks_mask_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = _save_nifti(root / "image.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            mask = _save_nifti(root / "mask.nii.gz", np.ones((6, 6, 6), dtype=np.float32))
            manifest = root / "manifest.csv"
            fields = [
                "subject_id", "task", "run_label", "reference_img_path", "coreg_img_path",
                "reference_mask_path", "coreg_mask_path", "quantitative_metrics_valid",
            ]
            _write_manifest(
                manifest,
                rows=[{
                    "subject_id": "sub-0001", "task": "video", "run_label": "",
                    "reference_img_path": str(image), "coreg_img_path": str(image),
                    "reference_mask_path": str(mask), "coreg_mask_path": str(mask),
                    "quantitative_metrics_valid": "false",
                }],
                fieldnames=fields,
            )
            out_dir = root / "out"
            summary = run_registration_qc(
                manifest,
                out_dir,
                config=RegistrationQcConfig(apng_enabled=False, write_overview_heatmap=False),
            )
            with (out_dir / "run_metrics.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["quantitative_metrics_valid"], "0")
            self.assertTrue(np.isnan(float(row["jaccard"])))
            self.assertEqual(summary["quantitative_metric_rows"], 0)

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for APNG test.")
    def test_apng_two_frames_and_optional_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (10, 10, 10)
            ref = _save_nifti(root / "ref.nii.gz", np.random.default_rng(0).normal(size=shape).astype(np.float32))
            coreg = _save_nifti(root / "coreg.nii.gz", np.random.default_rng(1).normal(size=shape).astype(np.float32))
            mask = np.zeros(shape, dtype=np.float32)
            mask[2:8, 2:8, 2:8] = 1.0
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", mask)
            coreg_mask = _save_nifti(root / "coreg_mask.nii.gz", mask)
            seg = np.zeros(shape, dtype=np.float32)
            seg[3:7, 3:7, 3:7] = 1.0
            seg_path = _save_nifti(root / "seg.nii.gz", seg)

            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "video",
                        "run_label": "run-1",
                        "reference_img_path": str(ref),
                        "coreg_img_path": str(coreg),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                        "segmentation_mask_path": str(seg_path),
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                    "coreg_mask_path",
                    "segmentation_mask_path",
                ],
            )

            out_dir = root / "out"
            run_registration_qc(manifest, out_dir)
            apng_files = list((out_dir / "qualitative" / "apng").glob("*.png"))
            self.assertEqual(len(apng_files), 1)
            with Image.open(apng_files[0]) as im:
                self.assertGreaterEqual(int(getattr(im, "n_frames", 1)), 2)
            reused = run_registration_qc(
                manifest,
                out_dir,
                config=RegistrationQcConfig(reuse_existing_apng=True),
            )
            self.assertEqual(reused["apng_created"], 0)
            self.assertEqual(reused["apng_reused"], 1)

    def test_slice_selection_is_deterministic(self) -> None:
        mask = np.zeros((20, 24, 26), dtype=bool)
        mask[2:19, 3:21, 4:22] = True
        a = _choose_slice_indices(mask, axis=2, n_slices=7, lower_pct=10.0, upper_pct=90.0)
        b = _choose_slice_indices(mask, axis=2, n_slices=7, lower_pct=10.0, upper_pct=90.0)
        np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for APNG test.")
    def test_runs_share_reference_display_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (18, 20, 22)
            ref = _save_nifti(root / "ref.nii.gz", np.random.default_rng(4).normal(size=shape))
            ref_mask_data = np.zeros(shape, dtype=np.float32)
            ref_mask_data[3:15, 4:17, 5:19] = 1.0
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", ref_mask_data)
            rows: list[dict[str, str]] = []
            for run, bounds in ((1, (1, 9)), (2, (9, 17))):
                coreg = _save_nifti(root / f"coreg_{run}.nii.gz", np.random.default_rng(run + 4).normal(size=shape))
                mask = np.zeros(shape, dtype=np.float32)
                mask[bounds[0] : bounds[1], 2:18, 2:20] = 1.0
                coreg_mask = _save_nifti(root / f"coreg_mask_{run}.nii.gz", mask)
                rows.append(
                    {
                        "subject_id": "sub-0001",
                        "task": "video",
                        "run_label": f"run-{run}",
                        "reference_img_path": str(ref),
                        "coreg_img_path": str(coreg),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                        "comparison_label": "func-to-T1w",
                    }
                )
            manifest = _write_manifest(root / "manifest.csv", rows, list(rows[0]))
            summary = run_registration_qc(manifest, root / "out")
            self.assertEqual(summary["display_geometries"], 1)
            apngs = sorted((root / "out" / "qualitative" / "apng" / "func-to-T1w").glob("*.png"))
            self.assertEqual(len(apngs), 2)
            with Image.open(apngs[0]) as first, Image.open(apngs[1]) as second:
                self.assertEqual(first.size, second.size)

    def test_comparison_label_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (6, 6, 6)
            image = _save_nifti(root / "image.nii.gz", np.ones(shape))
            mask = _save_nifti(root / "mask.nii.gz", np.ones(shape))
            rows = [
                {
                    "subject_id": "sub-0001",
                    "task": "video",
                    "run_label": "run-1",
                    "reference_img_path": str(image),
                    "coreg_img_path": str(image),
                    "reference_mask_path": str(mask),
                    "coreg_mask_path": str(mask),
                    "comparison_label": label,
                }
                for label in ("func-to-T1w", "func-to-MNI")
            ]
            summary = run_registration_qc(
                rows,
                root / "out",
                config=RegistrationQcConfig(apng_enabled=False, comparison_labels=("func-to-MNI",)),
            )
            self.assertEqual(summary["manifest_rows"], 1)
            self.assertEqual(summary["comparisons"], ["func-to-MNI"])

    def test_segmentation_contour_mask_preserves_ribbon_and_dseg_semantics(self) -> None:
        ribbon = np.zeros((6, 6), dtype=np.float32)
        ribbon[1:5, 1:5] = 1.0
        np.testing.assert_array_equal(_segmentation_slice_to_contour_mask(ribbon), ribbon > 0)

        dseg = np.zeros((6, 6), dtype=np.float32)
        dseg[1:5, 1:5] = 1.0
        dseg[1:5, 2:4] = 2.0
        dseg[2:4, 2:4] = 3.0
        out = _segmentation_slice_to_contour_mask(dseg)
        np.testing.assert_array_equal(out, dseg == 2.0)

    def test_physical_aspect_uses_voxel_sizes(self) -> None:
        affine = np.diag([0.5, 0.8, 0.5, 1.0])
        self.assertEqual(_display_plane_spacings(affine, axis=2), (0.8, 0.5))
        self.assertEqual(_display_plane_spacings(affine, axis=1), (0.5, 0.5))
        self.assertEqual(_display_plane_spacings(affine, axis=0), (0.5, 0.8))

        arr = np.ones((10, 10), dtype=np.float32)
        resized = _resize_to_physical_aspect(arr, row_spacing=0.8, col_spacing=0.5, base_spacing=0.5)
        self.assertEqual(resized.shape, (16, 10))

    @unittest.skipUnless(PIL_AVAILABLE, "Pillow is required for APNG test.")
    def test_apng_preserves_dseg_labels_until_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (12, 12, 12)
            ref = _save_nifti(root / "ref.nii.gz", np.random.default_rng(2).normal(size=shape).astype(np.float32))
            coreg = _save_nifti(root / "coreg.nii.gz", np.random.default_rng(3).normal(size=shape).astype(np.float32))
            mask = np.zeros(shape, dtype=np.float32)
            mask[2:10, 2:10, 2:10] = 1.0
            ref_mask = _save_nifti(root / "ref_mask.nii.gz", mask)
            coreg_mask = _save_nifti(root / "coreg_mask.nii.gz", mask)
            dseg = np.zeros(shape, dtype=np.float32)
            dseg[2:10, 2:10, 2:10] = 1.0
            dseg[3:9, 3:9, 3:9] = 2.0
            dseg[4:8, 4:8, 4:8] = 3.0
            seg_path = _save_nifti(root / "dseg.nii.gz", dseg)

            manifest = root / "manifest.csv"
            _write_manifest(
                manifest,
                rows=[
                    {
                        "subject_id": "sub-0001",
                        "task": "video",
                        "run_label": "run-1",
                        "reference_img_path": str(ref),
                        "coreg_img_path": str(coreg),
                        "reference_mask_path": str(ref_mask),
                        "coreg_mask_path": str(coreg_mask),
                        "segmentation_mask_path": str(seg_path),
                    }
                ],
                fieldnames=[
                    "subject_id",
                    "task",
                    "run_label",
                    "reference_img_path",
                    "coreg_img_path",
                    "reference_mask_path",
                    "coreg_mask_path",
                    "segmentation_mask_path",
                ],
            )

            out_dir = root / "out"
            run_registration_qc(manifest, out_dir)
            apng_files = list((out_dir / "qualitative" / "apng").glob("*.png"))
            self.assertEqual(len(apng_files), 1)


if __name__ == "__main__":
    unittest.main()
