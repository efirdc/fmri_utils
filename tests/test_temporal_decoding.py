from __future__ import annotations

import unittest

import numpy as np

from fmri_utils.temporal_decoding import (
    TemporalDecoderConfig,
    TemporalRun,
    build_loso_plan,
    build_pooled_4x3_plan,
    fit_temporal_decoder,
)


class TemporalSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.runs = tuple(
            TemporalRun(
                run_id=f"story-{index + 1}",
                features=rng.normal(size=(80, 12)).astype(np.float32),
                source_rows=np.arange(7, 87, dtype=np.int32),
            )
            for index in range(4)
        )

    def test_loso_holds_each_story_out_once(self) -> None:
        plan = build_loso_plan(self.runs)
        self.assertEqual(len(plan.outer_folds), 4)
        for run_index in range(4):
            heldout = np.concatenate([fold.test[run_index] for fold in plan.outer_folds])
            np.testing.assert_array_equal(np.sort(heldout), np.arange(80))

    def test_pooled_plan_covers_rows_and_purges_training_only(self) -> None:
        plan = build_pooled_4x3_plan(self.runs, purge_rows=6)
        for run_index, run in enumerate(self.runs):
            heldout = np.concatenate([fold.test[run_index] for fold in plan.outer_folds])
            np.testing.assert_array_equal(np.sort(heldout), np.arange(80))
            for fold in plan.outer_folds:
                train_source = run.source_rows[fold.train[run_index]]
                test_source = run.source_rows[fold.test[run_index]]
                if train_source.size:
                    self.assertGreater(
                        np.min(np.abs(train_source[:, None] - test_source[None, :])),
                        6,
                    )


class TemporalDecoderTests(unittest.TestCase):
    def test_nested_decoder_recovers_signal_and_random_null_is_centered(self) -> None:
        rng = np.random.default_rng(91)
        weights = rng.normal(size=18)
        runs = []
        targets = []
        for index in range(4):
            features = rng.normal(size=(90, 18)).astype(np.float32)
            observed = features @ weights + rng.normal(scale=0.5, size=90)
            variants = np.column_stack(
                [observed] + [rng.normal(size=90) for _ in range(20)]
            )
            runs.append(
                TemporalRun(
                    run_id=f"story-{index + 1}",
                    features=features,
                    source_rows=np.arange(90, dtype=np.int32),
                )
            )
            targets.append(variants)
        result = fit_temporal_decoder(
            runs,
            targets,
            build_loso_plan(runs),
            TemporalDecoderConfig(
                pca_components=(4, 12),
                ridge_alphas=(0.1, 10.0),
                random_seed=3,
            ),
        )
        self.assertGreater(result.correlation[0], 0.7)
        self.assertLess(abs(np.nanmean(np.arctanh(result.correlation[1:]))), 0.2)
        self.assertEqual(result.outer_fold_correlation.shape, (4, 21))
        self.assertTrue(all(np.isfinite(trace).all() for trace in result.observed_predictions))

    def test_separate_evaluation_patterns_use_training_fold_transforms(self) -> None:
        rng = np.random.default_rng(123)
        weights = rng.normal(size=10)
        source_runs = []
        evaluation_runs = []
        targets = []
        for index in range(4):
            latent = rng.normal(size=(72, 10)).astype(np.float32)
            target = latent @ weights
            source_runs.append(TemporalRun(f"story-{index}", latent, np.arange(72)))
            evaluation_runs.append(
                TemporalRun(f"story-{index}", latent + rng.normal(scale=0.05, size=latent.shape), np.arange(72))
            )
            targets.append(target[:, None])
        result = fit_temporal_decoder(
            source_runs,
            targets,
            build_loso_plan(source_runs),
            TemporalDecoderConfig(pca_components=(4, 8), ridge_alphas=(0.1, 1.0)),
            evaluation_runs=evaluation_runs,
        )
        self.assertGreater(result.correlation[0], 0.7)
        self.assertEqual(result.metadata["evaluation_features"], "separate_evaluation_runs")

    def test_independent_temporally_smooth_data_has_zero_centered_null(self) -> None:
        rng = np.random.default_rng(812)
        runs = []
        targets = []
        for index in range(4):
            feature_noise = rng.normal(size=(84, 14))
            features = np.empty_like(feature_noise)
            features[0] = feature_noise[0]
            for row in range(1, 84):
                features[row] = 0.8 * features[row - 1] + feature_noise[row]
            raw_targets = rng.normal(size=(84, 41))
            smooth_targets = np.empty_like(raw_targets)
            smooth_targets[0] = raw_targets[0]
            for row in range(1, 84):
                smooth_targets[row] = 0.8 * smooth_targets[row - 1] + raw_targets[row]
            runs.append(TemporalRun(f"story-{index}", features.astype(np.float32), np.arange(84)))
            targets.append(smooth_targets)
        result = fit_temporal_decoder(
            runs,
            targets,
            build_loso_plan(runs),
            TemporalDecoderConfig(pca_components=(4, 8), ridge_alphas=(0.1, 10.0)),
        )
        null_z = np.arctanh(np.clip(result.correlation[1:], -0.999999, 0.999999))
        self.assertLess(abs(float(np.mean(null_z))), 0.15)


if __name__ == "__main__":
    unittest.main()
