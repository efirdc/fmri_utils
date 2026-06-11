from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fmri_utils.registration_qc_manifest import (
    FmriprepRegistrationManifestConfig,
    build_fmriprep_boldref_t1w_manifest,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")
    return path


class RegistrationQcManifestTests(unittest.TestCase):
    def test_builds_nested_fmriprep_manifest_and_prefers_ribbon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anat = root / "sub-001" / "sub-001" / "anat"
            func = root / "sub-001" / "sub-001" / "func"
            _touch(anat / "sub-001_desc-preproc_T1w.nii.gz")
            _touch(anat / "sub-001_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz")
            _touch(anat / "sub-001_desc-brain_mask.nii.gz")
            _touch(anat / "sub-001_desc-ribbon_mask.nii.gz")
            _touch(anat / "sub-001_dseg.nii.gz")
            _touch(func / "sub-001_task-main_run-1_space-T1w_boldref.nii.gz")
            _touch(func / "sub-001_task-main_run-1_space-T1w_desc-brain_mask.nii.gz")
            _touch(func / "sub-001_task-main_run-1_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt")

            out_csv = root / "manifest.csv"
            summary = build_fmriprep_boldref_t1w_manifest(
                FmriprepRegistrationManifestConfig(fmriprep_root=root, out_csv=out_csv)
            )
            self.assertEqual(summary["n_manifest_rows"], 1)
            self.assertEqual(summary["segmentation_counts_by_subject"]["ribbon"], 1)
            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["subject_id"], "sub-001")
            self.assertEqual(rows[0]["task"], "main")
            self.assertEqual(rows[0]["run_label"], "run-1")
            self.assertTrue(rows[0]["segmentation_mask_path"].endswith("sub-001_desc-ribbon_mask.nii.gz"))

    def test_falls_back_to_dseg_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sid in ("sub-001", "sub-002"):
                anat = root / sid / "anat"
                func = root / sid / "func"
                _touch(anat / f"{sid}_desc-preproc_T1w.nii.gz")
                _touch(anat / f"{sid}_desc-brain_mask.nii.gz")
                _touch(anat / f"{sid}_dseg.nii.gz")
                _touch(func / f"{sid}_task-main_run-1_space-T1w_boldref.nii.gz")
                _touch(func / f"{sid}_task-main_run-1_space-T1w_desc-brain_mask.nii.gz")
                _touch(func / f"{sid}_task-other_run-2_space-T1w_boldref.nii.gz")
                _touch(func / f"{sid}_task-other_run-2_space-T1w_desc-brain_mask.nii.gz")

            out_csv = root / "manifest.csv"
            summary = build_fmriprep_boldref_t1w_manifest(
                FmriprepRegistrationManifestConfig(
                    fmriprep_root=root,
                    out_csv=out_csv,
                    subjects=("sub-002",),
                    tasks=("other",),
                    run_labels=("run-2",),
                )
            )
            self.assertEqual(summary["n_manifest_rows"], 1)
            self.assertEqual(summary["segmentation_counts_by_subject"]["dseg"], 1)
            with out_csv.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["subject_id"], "sub-002")
            self.assertEqual(row["task"], "other")
            self.assertEqual(row["run_label"], "run-2")
            self.assertTrue(row["segmentation_mask_path"].endswith("sub-002_dseg.nii.gz"))


if __name__ == "__main__":
    unittest.main()
