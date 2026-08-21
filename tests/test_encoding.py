from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_utils.encoding import (
    EncodingConfig,
    EncodingRun,
    SubjectData,
    build_cv_plan,
    discover_fmriprep_runs,
    fit_encoding,
    load_run_manifest,
    load_subject_data,
    save_encoding_result,
    write_run_manifest,
)
from fmri_utils.encoding.features import build_lagged_features, load_feature_matrix
from fmri_utils.encoding.preprocessing import residualize_run, voxelwise_r2


def _save_nifti(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values, np.eye(4)), str(path))


class EncodingFeatureTests(unittest.TestCase):
    def test_lags_use_past_feature_rows(self) -> None:
        values = np.arange(6, dtype=np.float32)[:, None]
        lagged, valid = build_lagged_features(values, (1, 2), mode="concat")
        np.testing.assert_array_equal(lagged[2], [1.0, 0.0])
        np.testing.assert_array_equal(valid, [False, False, True, True, True, True])

    def test_runwise_intercept_centers_each_column(self) -> None:
        values = np.column_stack([np.arange(8), np.arange(8) + 100]).astype(np.float32)
        residual = residualize_run(values, np.empty((8, 0)), add_intercept=True)
        np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=1.0e-6)

    def test_r2_definition(self) -> None:
        observed = np.arange(10, dtype=np.float32)[:, None]
        np.testing.assert_allclose(voxelwise_r2(observed, observed), [1.0])

    def test_ambiguous_npz_requires_named_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            np.savez(path, first=np.zeros((4, 2)), second=np.zeros((4, 3)))
            with self.assertRaisesRegex(ValueError, "exactly one 2D array"):
                load_feature_matrix(path)


class EncodingDataTests(unittest.TestCase):
    def test_manifest_rejects_blank_feature_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.csv"
            write_run_manifest(
                [
                    {
                        "subject_id": "sub-001",
                        "run_id": "run-1",
                        "bold_path": "bold.nii.gz",
                        "mask_path": "mask.nii.gz",
                        "features_path": "",
                    }
                ],
                path,
            )
            with self.assertRaisesRegex(ValueError, "empty required values"):
                load_run_manifest(path, "sub-001")

    def test_discovers_fmriprep_bold_mask_and_confounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            func = root / "sub-001" / "func"
            bold = func / "sub-001_task-story_run-1_space-T1w_desc-preproc_bold.nii.gz"
            mask = func / "sub-001_task-story_run-1_space-T1w_desc-brain_mask.nii.gz"
            confounds = func / "sub-001_task-story_run-1_desc-confounds_timeseries.tsv"
            _save_nifti(bold, np.zeros((2, 2, 2, 5), dtype=np.float32))
            _save_nifti(mask, np.ones((2, 2, 2), dtype=np.uint8))
            pd.DataFrame({"trans_x": np.zeros(5)}).to_csv(confounds, sep="\t", index=False)
            rows = discover_fmriprep_runs(
                root,
                task="story",
                feature_template=str(root / "features" / "{subject_id}_{run}.npy"),
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "task-story_run-1")
            self.assertEqual(Path(rows[0]["mask_path"]), mask)
            self.assertEqual(Path(rows[0]["confounds_path"]), confounds)

    def test_generic_manifest_aligns_trimmed_bold_and_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for run in range(1, 4):
                bold = root / "run-{}_bold.nii.gz".format(run)
                mask = root / "run-{}_mask.nii.gz".format(run)
                features = root / "run-{}_features.npy".format(run)
                data = np.zeros((2, 1, 1, 8), dtype=np.float32)
                data[0, 0, 0] = np.arange(8)
                data[1, 0, 0] = np.arange(8) * 2
                _save_nifti(bold, data)
                _save_nifti(mask, np.ones((2, 1, 1), dtype=np.uint8))
                np.save(features, np.arange(10, dtype=np.float32)[:, None])
                rows.append(
                    {
                        "subject_id": "sub-001",
                        "run_id": "run-{}".format(run),
                        "bold_path": str(bold),
                        "mask_path": str(mask),
                        "features_path": str(features),
                        "confounds_path": "",
                        "bold_start": "0",
                        "feature_start": "2",
                        "n_samples": "8",
                        "preprocessing_source": "custom_test",
                    }
                )
            manifest = write_run_manifest(rows, root / "runs.csv")
            specs = load_run_manifest(manifest, "sub-001")
            subject = load_subject_data(
                specs,
                EncodingConfig(lags=(0,), ridge_alphas=(1.0,)),
            )
            self.assertEqual(len(subject.runs), 3)
            np.testing.assert_array_equal(subject.runs[0].source_rows, np.arange(8))
            np.testing.assert_array_equal(subject.runs[0].features[:, 0], np.arange(2, 10))
            self.assertEqual(
                subject.input_provenance[0]["preprocessing_source"], "custom_test"
            )


class EncodingModelTests(unittest.TestCase):
    def _subject(self) -> SubjectData:
        rng = np.random.default_rng(5)
        runs = []
        for index in range(4):
            features = rng.normal(size=(36, 4)).astype(np.float32)
            weights = np.asarray(
                [[1.2, 0.0, 0.3], [0.0, -1.0, 0.2], [0.4, 0.2, 0.0], [0.0, 0.5, 1.0]],
                dtype=np.float32,
            )
            bold = features @ weights + rng.normal(scale=0.08, size=(36, 3))
            runs.append(
                EncodingRun(
                    run_id="run-{}".format(index + 1),
                    features=features,
                    bold=bold.astype(np.float32),
                    nuisance=np.empty((36, 0), dtype=np.float32),
                    source_rows=np.arange(36),
                )
            )
        reference = nib.Nifti1Image(np.zeros((3, 1, 1), dtype=np.float32), np.eye(4))
        return SubjectData(
            subject_id="sub-001",
            runs=runs,
            mask=np.ones((3, 1, 1), dtype=bool),
            reference_image=reference,
            voxel_indices=np.arange(3),
        )

    def test_nested_loro_recovers_synthetic_signal_and_writes_maps(self) -> None:
        subject = self._subject()
        config = EncodingConfig(
            lags=(0,),
            ridge_alphas=(0.01, 1.0, 100.0),
            pca_components=(None, 2),
        )
        result = fit_encoding(subject, config)
        self.assertEqual(result.outer_fold_correlation.shape, (4, 3))
        self.assertTrue(np.all(result.mean_correlation > 0.9))
        self.assertEqual(result.outer_selected_alpha.shape, (4, 3))
        self.assertEqual(result.mode_selected_pca_components.shape, (3,))
        with tempfile.TemporaryDirectory() as tmp:
            outputs = save_encoding_result(result, subject, Path(tmp))
            image = nib.load(outputs["mean_correlation"])
            self.assertEqual(image.shape, (3, 1, 1))
            outer = nib.load(outputs["outer_fold_correlation"])
            self.assertEqual(outer.shape, (3, 1, 1, 4))
            with np.load(outputs["outer_fold_traces"], allow_pickle=False) as traces:
                np.testing.assert_array_equal(traces["source_rows_00"], np.arange(36))
            metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))
            self.assertIn("input_provenance", metadata)
            self.assertEqual(metadata["cv_scheme"], "leave_one_run_out")
            self.assertEqual(len(metadata["folds"]), 4)

    def test_grouped_run_kfold_keeps_runs_whole(self) -> None:
        subject = self._subject()
        config = EncodingConfig(
            lags=(0,),
            ridge_alphas=(1.0,),
            cv_scheme="grouped_run_kfold",
            outer_splits=2,
            inner_splits=2,
        )
        plan = build_cv_plan(subject.runs, config)
        self.assertEqual(len(plan.outer_folds), 2)
        for fold in plan.outer_folds:
            for train_rows, test_rows, run in zip(fold.train, fold.test, subject.runs):
                self.assertIn(train_rows.size, {0, run.bold.shape[0]})
                self.assertIn(test_rows.size, {0, run.bold.shape[0]})
                self.assertNotEqual(train_rows.size > 0, test_rows.size > 0)

    def test_blocked_kfold_covers_rows_once_and_applies_embargo(self) -> None:
        subject = self._subject()
        config = EncodingConfig(
            lags=(0,),
            ridge_alphas=(1.0,),
            cv_scheme="blocked_kfold",
            outer_splits=4,
            inner_splits=3,
            embargo_rows=2,
        )
        plan = build_cv_plan(subject.runs, config)
        self.assertEqual(len(plan.outer_folds), 4)
        for run_index, run in enumerate(subject.runs):
            held_out = np.concatenate(
                [fold.test[run_index] for fold in plan.outer_folds]
            )
            np.testing.assert_array_equal(np.sort(held_out), np.arange(run.bold.shape[0]))
        for fold in plan.outer_folds:
            self.assertEqual(len(fold.inner_folds), 3)
            for run_index, run in enumerate(subject.runs):
                train_source = run.source_rows[fold.train[run_index]]
                test_source = run.source_rows[fold.test[run_index]]
                if train_source.size and test_source.size:
                    distance = np.min(
                        np.abs(train_source[:, None] - test_source[None, :])
                    )
                    self.assertGreater(distance, config.embargo_rows)

    def test_blocked_kfold_fits_one_run(self) -> None:
        subject = self._subject()
        subject.runs = subject.runs[:1]
        config = EncodingConfig(
            lags=(0,),
            ridge_alphas=(0.01, 1.0),
            cv_scheme="blocked_kfold",
            outer_splits=4,
            inner_splits=3,
        )
        result = fit_encoding(subject, config)
        self.assertEqual(result.outer_fold_correlation.shape, (4, 3))
        self.assertTrue(np.all(result.mean_correlation > 0.8))


if __name__ == "__main__":
    unittest.main()
