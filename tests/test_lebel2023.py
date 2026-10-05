"""fmri_utils.lebel2023 on synthetic data (the real-data equivalence with the original scripts was
checked separately: features, regressors, removal and a fit chunk are bit-identical)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fmri_utils.lebel2023 import ablation, dataset, encoding, group, nulls, regressors
from fmri_utils.lebel2023.cross_participant import nuisance_set, residualise
from fmri_utils.lebel2023.registration import MNIWarp

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 3
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phone"
        xmin = 0
        xmax = 3
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 3
            text = "AH"
    item [2]:
        class = "IntervalTier"
        name = "word"
        xmin = 0
        xmax = 3
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1
            text = "Hello"
        intervals [2]:
            xmin = 1
            xmax = 2
            text = "sp"
        intervals [3]:
            xmin = 2
            xmax = 3
            text = "world"
"""


class Timing(unittest.TestCase):
    def test_tr_times(self):
        times = dataset.tr_times(4)
        np.testing.assert_allclose(times, [11.0, 13.0, 15.0, 17.0])

    def test_word_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "story.TextGrid"
            path.write_text(TEXTGRID, encoding="utf-8")
            words = dataset.parse_word_tier(path)
        self.assertEqual(words, [(0.0, 1.0, "hello"), (2.0, 3.0, "world")])

    def test_lanczos_identity_on_grid(self):
        times = np.arange(10, dtype=float)
        np.testing.assert_allclose(dataset.lanczos_weights(times, times), np.eye(10), atol=1e-12)

    def test_alphas(self):
        self.assertEqual(len(dataset.RIDGE_ALPHAS), 9)
        self.assertAlmostEqual(dataset.RIDGE_ALPHAS[0], 10.0)
        self.assertAlmostEqual(dataset.RIDGE_ALPHAS[-1], 1e5)


class Regressors(unittest.TestCase):
    def test_word_ratings(self):
        table = pd.DataFrame({"first_word": [0, 2], "n_words": [2, 3], "embodiment_mean": [1.0, 2.5]})
        np.testing.assert_allclose(regressors.word_ratings(table, 5, "embodiment"), [1, 1, 2.5, 2.5, 2.5])
        with self.assertRaises(ValueError):
            regressors.word_ratings(table, 6, "embodiment")


class Ablation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.stories = [f"s{i}" for i in range(6)] + [dataset.TEST_STORY]
        self.fit = self.stories[:-1]
        self.reg = {s: rng.normal(size=(80, 2)) for s in self.stories}
        self.features = {s: rng.normal(size=(80, 12)).astype(np.float32) for s in self.stories}

    def test_orthogonal_to_rate(self):
        series = ablation.orthogonalise(self.reg, self.fit)
        design = np.vstack([ablation.lagged_design(self.reg[s][:, 1]) for s in self.fit])
        target = np.concatenate([series[s] for s in self.fit])
        np.testing.assert_allclose(design.T @ target, 0, atol=1e-6)

    def test_removed_features_orthogonal(self):
        series = ablation.orthogonalise(self.reg, self.fit)
        removed, fraction = ablation.remove(self.features, series, self.fit)
        design = np.vstack([ablation.lagged_design(series[s]) for s in self.fit])
        stacked = np.vstack([removed[s] for s in self.fit]).astype(np.float64)
        np.testing.assert_allclose(design.T @ stacked, 0, atol=1e-3)
        self.assertGreater(fraction, 0)

    def test_null_score(self):
        d_null = np.random.default_rng(1).normal(size=(99, 5))
        result = nulls.score(np.full(5, 10.0), d_null)
        np.testing.assert_allclose(result["p"], 1 / 100)


class Encoding(unittest.TestCase):
    def test_design_shapes(self):
        rng = np.random.default_rng(0)
        train = [rng.normal(size=(50, 30)) for _ in range(4)]
        design = encoding.prepare_design(train, rng.normal(size=(40, 30)), pca_components=8,
                                         extra_train=[rng.normal(size=(50, 1)) for _ in range(4)],
                                         extra_test=rng.normal(size=(40, 1)))
        self.assertEqual(design.train.shape, (200, 4 * 9))
        self.assertEqual(design.test.shape, (40, 36))
        np.testing.assert_allclose(design.train.mean(axis=0), 0, atol=1e-5)

    def test_fit_recovers_signal(self):
        rng = np.random.default_rng(0)
        stories = [f"s{i}" for i in range(5)]
        x_train = [rng.normal(size=(120, 6)).astype(np.float32) for _ in stories]
        x_test = rng.normal(size=(100, 6)).astype(np.float32)
        weights = rng.normal(size=(6, 3))
        y_train = [(x @ weights + 0.1 * rng.normal(size=(120, 3))).astype(np.float32) for x in x_train]
        y_test = (x_test @ weights + 0.1 * rng.normal(size=(100, 3))).astype(np.float32)
        design = encoding.design_from_blocks(x_train, x_test)
        r, alpha = encoding.fit_voxelwise(design, stories, y_train, y_test)
        self.assertTrue((r > 0.95).all(), r)
        self.assertEqual(alpha.shape, (3,))

    def test_noise_ceiling(self):
        signal = np.random.default_rng(0).normal(size=(100, 4))
        self.assertTrue(np.allclose(encoding.noise_ceiling(np.stack([signal] * 5)), 1.0))
        self.assertAlmostEqual(float(encoding.normalise(np.array([0.2]), np.array([0.1]))[0]), 0.2 / 0.3, places=6)

    def test_chunks_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "model" / "UTS01" / "chunks"
            for chunk in range(2):
                start, stop = encoding.chunk_bounds(10, chunk, 2)
                index = np.arange(start, stop)[::2]
                encoding.save_chunk(folder, chunk, 2, index, {"correlation_raw": index.astype(np.float32)},
                                    {"n_columns": 10, "voxel_stop": stop})
            maps = encoding.read_chunks(folder.parent)
            np.testing.assert_allclose(maps["correlation_raw"][[0, 2, 4, 5, 7, 9]], [0, 2, 4, 5, 7, 9])
            self.assertTrue(np.isnan(maps["correlation_raw"][1]))
            (folder / "chunk_001_of_002.npz").unlink()
            with self.assertRaises(ValueError):
                encoding.read_chunks(folder.parent)


class CrossParticipant(unittest.TestCase):
    def test_nuisance_and_residual(self):
        rng = np.random.default_rng(0)
        response = rng.normal(size=(60, 50)).astype(np.float32)
        source = np.zeros(50, dtype=bool)
        source[:20] = True
        nuisance = nuisance_set(response, source, 5)
        self.assertEqual(nuisance.shape, (60, 5))
        cleaned = residualise(response, nuisance)
        np.testing.assert_allclose(nuisance.T.astype(np.float64) @ cleaned, 0, atol=1e-3)


class Group(unittest.TestCase):
    def test_sign_flip_and_bh(self):
        data = np.random.default_rng(0).normal(0.5, 1.0, size=(6, 40))
        result = group.sign_flip(data)
        self.assertTrue(((result["fwe"] >= 1 / 64) & (result["fwe"] <= 1)).all())
        q = group.bh_adjusted(result["p"])
        order = np.argsort(result["p"])
        self.assertTrue((np.diff(q[order]) >= -1e-12).all())

    def test_region_test(self):
        stack = np.vstack([np.full(6, 1.0) + 0.1 * k for k in range(8)])
        labels = np.zeros(np.prod((91, 109, 91)), dtype=int)
        voxels = np.arange(6)
        labels[voxels] = 1
        rows = group.region_test(stack, voxels, {"toy": (labels, {1: "region"})}, min_voxels=3)
        self.assertEqual(rows[0]["region"], "region")
        self.assertGreater(rows[0]["t"], 5)

    def test_warp_apply(self):
        columns = np.array([[0, 1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1, -1]])
        weights = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        warp = MNIWarp(np.array([0, 1]), columns, weights, np.array([0, 2]), 3)
        np.testing.assert_allclose(warp.apply(np.array([2.0, 4.0, 7.0])), [3.0, 7.0])
        out = warp.apply(np.array([np.nan, 4.0, 7.0]))
        self.assertTrue(np.isnan(out[0]))
        with tempfile.TemporaryDirectory() as tmp:
            again = MNIWarp.load(warp.save(Path(tmp) / "w.npz"))
            np.testing.assert_allclose(again.apply(np.array([2.0, 4.0, 7.0])), [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
