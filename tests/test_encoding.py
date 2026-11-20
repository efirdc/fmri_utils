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
    make_chunk_manifest,
    result_metric_arrays,
    save_encoding_chunk,
    save_encoding_result,
    stitch_encoding_chunks,
    write_run_manifest,
)
from fmri_utils.encoding.features import build_lagged_features, load_feature_matrix
from fmri_utils.encoding.preprocessing import residualize_run, voxelwise_r2
from fmri_utils.encoding.model import _fit_predict_ridge


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

    def test_loader_selects_a_stable_voxel_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for run in range(1, 4):
                bold = root / "run-{}_bold.nii.gz".format(run)
                mask = root / "run-{}_mask.nii.gz".format(run)
                features = root / "run-{}_features.npy".format(run)
                values = np.arange(48, dtype=np.float32).reshape(4, 12)
                _save_nifti(bold, values.reshape(4, 1, 1, 12))
                _save_nifti(mask, np.ones((4, 1, 1), dtype=np.uint8))
                np.save(features, np.arange(12, dtype=np.float32)[:, None])
                rows.append(
                    {
                        "subject_id": "sub-001",
                        "run_id": "run-{}".format(run),
                        "bold_path": str(bold),
                        "mask_path": str(mask),
                        "features_path": str(features),
                    }
                )
            manifest = write_run_manifest(rows, root / "runs.csv")
            specs = load_run_manifest(manifest, "sub-001")
            data = load_subject_data(
                specs,
                EncodingConfig(lags=(0,), ridge_alphas=(1.0,)),
                voxel_start=1,
                voxel_stop=3,
            )
            self.assertEqual(data.total_voxels, 4)
            self.assertEqual(data.voxel_start, 1)
            self.assertEqual(data.voxel_stop, 3)
            np.testing.assert_array_equal(data.voxel_indices, [1, 2])
            self.assertEqual(data.runs[0].bold.shape, (12, 2))


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

    def _strong_plus_weak(self) -> SubjectData:
        """One strong regressor (column 0) among many weak or irrelevant ones."""
        rng = np.random.default_rng(11)
        runs = []
        for index in range(4):
            features = rng.normal(size=(60, 40)).astype(np.float32)
            bold = 3.0 * features[:, :1] + 0.3 * features[:, 1:2] + rng.normal(scale=0.5, size=(60, 1))
            runs.append(EncodingRun(run_id="run-{}".format(index + 1), features=features,
                                    bold=bold.astype(np.float32),
                                    nuisance=np.empty((60, 0), dtype=np.float32),
                                    source_rows=np.arange(60)))
        reference = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.float32), np.eye(4))
        return SubjectData(subject_id="sub-001", runs=runs, mask=np.ones((1, 1, 1), dtype=bool),
                           reference_image=reference, voxel_indices=np.arange(1))

    def test_unpenalized_base_escapes_shrinkage(self) -> None:
        subject = self._strong_plus_weak()
        # A penalty strong enough to flatten every column when all are penalised.
        shrunk = EncodingConfig(lags=(0,), ridge_alphas=(1.0e5,))
        based = EncodingConfig(lags=(0,), ridge_alphas=(1.0e5,), unpenalized_features=1)
        plain = fit_encoding(subject, shrunk)
        with_base = fit_encoding(subject, based)
        self.assertGreater(float(with_base.mean_correlation[0]), 0.95)
        # Correlation is scale-free, so compare fitted amplitude via R2.
        self.assertGreater(float(with_base.mean_r2[0]), float(plain.mean_r2[0]) + 0.5)
        self.assertEqual(with_base.metadata["unpenalized_features"], 1)

    def test_unpenalized_base_is_taken_from_every_lag(self) -> None:
        from fmri_utils.encoding.model import _base_columns

        config = EncodingConfig(lags=(1, 2, 3), unpenalized_features=2)
        np.testing.assert_array_equal(_base_columns(15, config), [0, 1, 5, 6, 10, 11])
        with self.assertRaises(ValueError):
            _base_columns(6, EncodingConfig(lags=(1, 2, 3), unpenalized_features=2))

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

    def test_reusable_svd_ridge_matches_closed_form(self) -> None:
        rng = np.random.default_rng(41)
        x_train = rng.normal(size=(30, 7)).astype(np.float32)
        y_train = rng.normal(size=(30, 5)).astype(np.float32)
        x_test = rng.normal(size=(9, 7)).astype(np.float32)
        alpha = 3.5
        actual = _fit_predict_ridge(x_train, y_train, x_test, alpha)
        expected = x_test @ np.linalg.solve(
            x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


class EncodingChunkTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        rng = np.random.default_rng(12)
        weights = rng.normal(size=(3, 5)).astype(np.float32)
        rows = []
        for run in range(1, 4):
            features_array = rng.normal(size=(24, 3)).astype(np.float32)
            bold_array = features_array @ weights + rng.normal(
                scale=0.05, size=(24, 5)
            )
            bold = root / "run-{}_bold.nii.gz".format(run)
            mask = root / "run-{}_mask.nii.gz".format(run)
            features = root / "run-{}_features.npy".format(run)
            _save_nifti(
                bold,
                np.asarray(bold_array.T.reshape(5, 1, 1, 24), dtype=np.float32),
            )
            _save_nifti(mask, np.ones((5, 1, 1), dtype=np.uint8))
            np.save(features, features_array)
            rows.append(
                {
                    "subject_id": "sub-001",
                    "run_id": "run-{}".format(run),
                    "bold_path": str(bold),
                    "mask_path": str(mask),
                    "features_path": str(features),
                }
            )
        return write_run_manifest(rows, root / "runs.csv")

    def test_chunked_fit_matches_full_fit_and_stitches_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._dataset(root)
            config = EncodingConfig(
                lags=(0,),
                ridge_alphas=(0.1, 1.0),
                n_permutations=2,
                permutation_block_length=4,
                random_seed=17,
            )
            specs = load_run_manifest(manifest, "sub-001")
            full_data = load_subject_data(specs, config)
            full_result = fit_encoding(full_data, config)

            chunk_manifest = root / "chunks.csv"
            rows = make_chunk_manifest(manifest, chunk_manifest, chunk_size=2)
            self.assertEqual(len(rows), 3)
            output_root = root / "results"
            for row in rows:
                chunk_data = load_subject_data(
                    specs,
                    config,
                    voxel_start=int(row["voxel_start"]),
                    voxel_stop=int(row["voxel_stop"]),
                )
                chunk_result = fit_encoding(chunk_data, config)
                save_encoding_chunk(
                    chunk_result,
                    chunk_data,
                    output_root / row["chunk_relpath"],
                    save_traces=True,
                )

            outputs = stitch_encoding_chunks(chunk_manifest, output_root)["sub-001"]
            for metric, expected in result_metric_arrays(full_result).items():
                image = nib.load(outputs[metric])
                actual = np.asanyarray(image.dataobj)
                if expected.ndim == 1:
                    actual = actual.reshape(-1)
                else:
                    actual = actual.reshape((-1, expected.shape[0])).T
                np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5)
            with np.load(outputs["outer_fold_traces"], allow_pickle=False) as traces:
                np.testing.assert_allclose(
                    traces["predicted_00"], full_result.predicted_by_fold[0], atol=1.0e-5
                )

    def test_stitch_rejects_a_missing_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._dataset(root)
            chunk_manifest = root / "chunks.csv"
            make_chunk_manifest(manifest, chunk_manifest, chunk_size=2)
            with self.assertRaisesRegex(FileNotFoundError, "Missing encoding chunk"):
                stitch_encoding_chunks(chunk_manifest, root / "results")

    def test_stitch_rejects_mixed_scientific_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._dataset(root)
            chunk_manifest = root / "chunks.csv"
            rows = make_chunk_manifest(manifest, chunk_manifest, chunk_size=2)
            specs = load_run_manifest(manifest, "sub-001")
            output_root = root / "results"
            for index, row in enumerate(rows):
                config = EncodingConfig(
                    lags=(0,),
                    ridge_alphas=(1.0 if index < 2 else 10.0,),
                )
                data = load_subject_data(
                    specs,
                    config,
                    voxel_start=int(row["voxel_start"]),
                    voxel_stop=int(row["voxel_stop"]),
                )
                save_encoding_chunk(
                    fit_encoding(data, config),
                    data,
                    output_root / row["chunk_relpath"],
                )
            with self.assertRaisesRegex(ValueError, "different inputs or settings"):
                stitch_encoding_chunks(chunk_manifest, output_root)



class RidgeKernelRouteTest(unittest.TestCase):
    def test_kernel_route_matches_svd(self):
        from fmri_utils.encoding.model import _RidgeSVD
        rng = np.random.default_rng(0)
        for rows, columns in ((40, 300), (60, 25)):
            x_train = rng.standard_normal((rows, columns)).astype(np.float32)
            x_test = rng.standard_normal((15, columns)).astype(np.float32)
            y_train = rng.standard_normal((rows, 7)).astype(np.float32)
            svd = _RidgeSVD(x_train, x_test, method="svd")
            kernel = _RidgeSVD(x_train, x_test, method="kernel")
            self.assertEqual(_RidgeSVD(x_train, x_test).method, "kernel" if columns > rows else "svd")
            for alpha in (0.1, 10.0, 1e4):
                x, y = x_train.astype(np.float64), y_train.astype(np.float64)
                expected = x_test.astype(np.float64) @ np.linalg.solve(x.T @ x + alpha * np.eye(columns), x.T @ y)
                np.testing.assert_allclose(svd.predict(y_train, alpha), expected, rtol=1e-4, atol=1e-5)
                np.testing.assert_allclose(kernel.predict(y_train, alpha), expected, rtol=1e-4, atol=1e-5)


class BandedRidgeTest(unittest.TestCase):
    def test_band_scale_is_a_separate_penalty(self):
        from fmri_utils.encoding.model import _BandedSolvers
        rng = np.random.default_rng(1)
        band = [3, 11]
        for rows, columns in ((40, 300), (60, 25)):  # kernel route, then SVD route
            x_train = rng.standard_normal((rows, columns)).astype(np.float32)
            x_test = rng.standard_normal((15, columns)).astype(np.float32)
            y_train = rng.standard_normal((rows, 5)).astype(np.float32)
            from fmri_utils.encoding.config import EncodingConfig
            solvers = _BandedSolvers(x_train, x_test, EncodingConfig(pca_components=(None,), band_columns=tuple(band)))
            self.assertEqual(solvers.kernel, columns > rows)
            x, y, xt = x_train.astype(np.float64), y_train.astype(np.float64), x_test.astype(np.float64)
            rest = np.setdiff1d(np.arange(columns), band)
            for scale in (0.0, 0.5, 3.0):
                solver = solvers.solver(scale)
                for alpha in (0.1, 10.0, 1e3):
                    if scale == 0.0:  # the band is dropped: ridge on the other columns
                        a = x[:, rest]
                        expected = xt[:, rest] @ np.linalg.solve(a.T @ a + alpha * np.eye(rest.size), a.T @ y)
                    else:  # penalty alpha on the rest, alpha / scale^2 on the band
                        penalty = np.full(columns, alpha)
                        penalty[band] = alpha / scale**2
                        expected = xt @ np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
                    np.testing.assert_allclose(solver.predict(y_train, alpha), expected, rtol=1e-4, atol=1e-5)

    def test_fit_encoding_selects_band_scale(self):
        from fmri_utils.encoding.config import EncodingConfig
        rng = np.random.default_rng(2)
        n_runs, rows, columns = 4, 30, 12
        signal = rng.standard_normal((n_runs * rows, 1))
        features = rng.standard_normal((n_runs * rows, columns)).astype(np.float32)
        features[:, -1] = signal[:, 0]  # the band column carries the signal
        bold = np.hstack([signal + 0.3 * rng.standard_normal((n_runs * rows, 1)),
                          rng.standard_normal((n_runs * rows, 2))]).astype(np.float32)
        runs = [EncodingRun(run_id=f"r{i}", features=features[i * rows:(i + 1) * rows],
                            bold=bold[i * rows:(i + 1) * rows], nuisance=np.empty((rows, 0), np.float32),
                            source_rows=np.arange(rows, dtype=np.int32)) for i in range(n_runs)]
        data = SubjectData(subject_id="s", runs=runs, mask=np.ones((1, 1, 3), bool),
                           reference_image=nib.Nifti1Image(np.zeros((1, 1, 3), np.float32), np.eye(4)),
                           voxel_indices=np.arange(3))
        base = dict(lags=(0,), ridge_alphas=(1.0, 100.0, 1e4), pca_components=(None,),
                    add_run_intercept=False, cv_scheme="leave_one_run_out", outer_splits=4, inner_splits=3)
        plain = fit_encoding(data, EncodingConfig(**base))
        same = fit_encoding(data, EncodingConfig(**base, band_columns=(columns - 1,), band_scales=(1.0,)))
        np.testing.assert_allclose(same.mean_correlation, plain.mean_correlation, rtol=1e-5, atol=1e-6)
        self.assertIsNone(plain.outer_selected_scale)
        banded = fit_encoding(data, EncodingConfig(**base, band_columns=(columns - 1,),
                                                   band_scales=(0.0, 1.0, 10.0)))
        self.assertEqual(banded.outer_selected_scale.shape, banded.outer_selected_alpha.shape)
        self.assertTrue(np.all(np.isin(banded.outer_selected_scale, (0.0, 1.0, 10.0))))
        self.assertGreater(float(np.mean(banded.outer_selected_scale[:, 0])), 1.0)
        self.assertGreaterEqual(banded.mean_correlation[0], plain.mean_correlation[0] - 0.02)
        with self.assertRaises(ValueError):
            EncodingConfig(**{**base, "pca_components": (5,)}, band_columns=(0,)).validate()


class GatedBandTest(unittest.TestCase):
    def test_gated_band_matches_explicit_products(self):
        from fmri_utils.encoding.config import EncodingConfig
        from fmri_utils.encoding.model import _BandedSolvers
        rng = np.random.default_rng(3)
        # Two blocks (one per lag), one extra column per lag, and one gate per lag at the end.
        blocks, gates = ((0, 6), (7, 13)), (14, 15)
        for rows in (40, 200):  # narrow designs: the explicit SVD route
            columns = 16
            x_fit = rng.standard_normal((rows, columns)).astype(np.float32)
            x_predict = rng.standard_normal((12, columns)).astype(np.float32)
            raw_fit, raw_predict = x_fit.copy(), x_predict.copy()
            raw_fit[:, gates] = rng.uniform(0, 1, (rows, 2)) * (rng.uniform(size=(rows, 2)) > 0.5)
            raw_predict[:, gates] = rng.uniform(0, 1, (12, 2))
            y = rng.standard_normal((rows, 4)).astype(np.float32)
            config = EncodingConfig(lags=(0,), pca_components=(None,), gate_columns=gates, gated_blocks=blocks,
                                    band_scales=(0.0, 0.7, 3.0))
            solvers = _BandedSolvers(x_fit, x_predict, config, raw_fit, raw_predict)
            keep = [c for c in range(columns) if c not in gates]
            def explicit(x, raw, scale):
                g = [x[:, a:b].astype(np.float64) * raw[:, [gates[k]]].astype(np.float64) for k, (a, b) in enumerate(blocks)]
                return np.hstack([x[:, keep].astype(np.float64), *(scale * m for m in g)])
            for scale in (0.0, 0.7, 3.0):
                a, at = explicit(x_fit, raw_fit, scale), explicit(x_predict, raw_predict, scale)
                for alpha in (0.5, 50.0):
                    expected = at @ np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ y.astype(np.float64))
                    np.testing.assert_allclose(solvers.solver(scale).predict(y, alpha), expected, rtol=1e-4, atol=1e-5)
        # The kernel route (a design wider than its rows: zero columns pad it) against the explicit route.
        rows, columns = 20, 16
        x_fit = rng.standard_normal((rows, columns)).astype(np.float32)
        x_predict = rng.standard_normal((5, columns)).astype(np.float32)
        raw_fit, raw_predict = x_fit.copy(), x_predict.copy()
        raw_fit[:, gates] = rng.uniform(0, 1, (rows, 2))
        raw_predict[:, gates] = rng.uniform(0, 1, (5, 2))
        y = rng.standard_normal((rows, 3)).astype(np.float32)
        config = EncodingConfig(lags=(0,), pca_components=(None,), gate_columns=gates, gated_blocks=blocks)
        solvers = _BandedSolvers(x_fit, x_predict, config, raw_fit, raw_predict)
        self.assertFalse(solvers.kernel)
        explicit_pred = solvers.solver(2.0).predict(y, 5.0)
        kernel = _BandedSolvers(np.hstack([x_fit, np.zeros((rows, 10), np.float32)]),
                                np.hstack([x_predict, np.zeros((5, 10), np.float32)]),
                                config, np.hstack([raw_fit, np.zeros((rows, 10), np.float32)]),
                                np.hstack([raw_predict, np.zeros((5, 10), np.float32)]))
        self.assertTrue(kernel.kernel)
        np.testing.assert_allclose(kernel.solver(2.0).predict(y, 5.0), explicit_pred, rtol=1e-4, atol=1e-5)

    def test_fit_encoding_with_gated_band(self):
        from fmri_utils.encoding.config import EncodingConfig
        rng = np.random.default_rng(4)
        n_runs, rows, width = 4, 40, 6
        x = rng.standard_normal((n_runs * rows, width))
        gate = (rng.uniform(size=n_runs * rows) > 0.5).astype(float)
        signal = (x[:, 0] * gate)  # the response depends on feature 0 only while the gate is on
        features = np.hstack([x, gate[:, None]]).astype(np.float32)
        bold = np.column_stack([signal + 0.3 * rng.standard_normal(n_runs * rows),
                                rng.standard_normal(n_runs * rows)]).astype(np.float32)
        runs = [EncodingRun(run_id=f"r{i}", features=features[i * rows:(i + 1) * rows],
                            bold=bold[i * rows:(i + 1) * rows], nuisance=np.empty((rows, 0), np.float32),
                            source_rows=np.arange(rows, dtype=np.int32)) for i in range(n_runs)]
        data = SubjectData(subject_id="s", runs=runs, mask=np.ones((1, 1, 2), bool),
                           reference_image=nib.Nifti1Image(np.zeros((1, 1, 2), np.float32), np.eye(4)),
                           voxel_indices=np.arange(2))
        base = dict(lags=(0,), ridge_alphas=(1.0, 100.0), pca_components=(None,), add_run_intercept=False,
                    cv_scheme="leave_one_run_out", outer_splits=4, inner_splits=3)
        plain = fit_encoding(data, EncodingConfig(**base))
        gated = fit_encoding(data, EncodingConfig(**base, gate_columns=(width,), gated_blocks=((0, width),),
                                                  band_scales=(0.0, 1.0, 3.0)))
        self.assertGreater(gated.mean_correlation[0], plain.mean_correlation[0] + 0.1)
        self.assertGreater(float(np.mean(gated.outer_selected_scale[:, 0])), 0.0)
        with self.assertRaises(ValueError):
            EncodingConfig(**base, gate_columns=(1,), gated_blocks=((0, 3),)).validate()


if __name__ == "__main__":
    unittest.main()
