"""The column-data pipeline on synthetic data: run tables, column fits, ablation, cross-participant
encoding, group inference, stimulus tables and word-vector features, rating regressors."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from fmri_utils import ablation, cross_participant, group_inference
from fmri_utils.encoding import columns, design, noise_ceiling
from fmri_utils.encoding.run_table import RunTable
from fmri_utils.features import batch, static
from fmri_utils.features.stimuli import read_stimuli, write_stimuli
from fmri_utils.fsl_transforms import ColumnMNIWarp
from fmri_utils.story_ratings.timeseries import rating_regressors, word_values
from fmri_utils.story_ratings.transcripts import Word

RUNS = ["s0", "s1", "s2", "s3", "s4", "test"]


def make_dataset(root: Path, n_subjects: int = 3, rows: int = 60, shared_signal=True):
    """Subjects whose responses are a shared latent signal plus noise; npy runs; a 4x4x2 mask (32 columns)."""
    rng = np.random.default_rng(0)
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[:, :, :2] = 1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(mask, affine), root / "mask.nii.gz")
    latent = {r: rng.normal(size=(rows, 3)) for r in RUNS}
    features = {r: np.column_stack([latent[r], rng.normal(size=(rows, 5))]).astype(np.float32) for r in RUNS}
    (root / "features" / "f").mkdir(parents=True)
    for r in RUNS:
        np.save(root / "features" / "f" / f"{r}.npy", features[r])
    rows_out = []
    for s in range(n_subjects):
        weights = rng.normal(size=(3, 32))
        for r in RUNS:
            signal = latent[r] @ weights
            if r == "test":
                repeats = np.stack([signal + 0.5 * rng.normal(size=signal.shape) for _ in range(3)]).astype(np.float32)
                np.savez(root / f"sub{s}_{r}.npz", data=repeats.mean(axis=0), repeats=repeats)
                rows_out.append({"subject": f"sub{s}", "run": r, "role": "test", "response": f"sub{s}_{r}.npz",
                                 "response_key": "data", "repeats_key": "repeats", "mask": "mask.nii.gz"})
            else:
                np.save(root / f"sub{s}_{r}.npy", (signal + 0.5 * rng.normal(size=signal.shape)).astype(np.float32))
                rows_out.append({"subject": f"sub{s}", "run": r, "role": "train", "response": f"sub{s}_{r}.npy", "mask": "mask.nii.gz"})
    RunTable.write(rows_out, root / "runs.csv")
    return RunTable(root / "runs.csv"), features


class RunTables(unittest.TestCase):
    def test_table_and_unmask(self):
        with tempfile.TemporaryDirectory() as tmp:
            table, _ = make_dataset(Path(tmp))
            self.assertEqual(table.subjects(), ["sub0", "sub1", "sub2"])
            self.assertEqual(table.runs("sub0"), RUNS[:-1])
            self.assertEqual(table.test_run("sub0"), "test")
            self.assertEqual(table.n_columns("sub0"), 32)
            self.assertEqual(table.response("sub0", "s0", 4, 10).shape, (60, 6))
            self.assertEqual(table.repeats("sub0", "test").shape, (3, 60, 32))
            volume = table.unmask("sub0", np.arange(32, dtype=float))
            self.assertTrue(np.isnan(volume[:, :, 2]).all())
            self.assertEqual(int(table.column_volume("sub0").max()), 32)

    def test_pycortex_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            table, _ = make_dataset(Path(tmp))
            for row in table.rows:
                row["column_order"] = "pycortex"
            table._mask.cache_clear()
            volume = table.unmask("sub0", np.arange(32, dtype=float))
            # pycortex order runs over x fastest: the second column is x = 1
            self.assertEqual(volume[1, 0, 0], 1.0)


class ColumnFits(unittest.TestCase):
    def test_fit_feature_space_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table, _ = make_dataset(root)
            for chunk in range(2):
                columns.fit_feature_space(table, "sub0", root / "features", "f", root / "fits", chunk=chunk, n_chunks=2,
                                          pca_components=6, delays=(0,))
            maps = columns.read_chunks(root / "fits" / "f" / "sub0")
            self.assertGreater(np.nanmean(maps["correlation_raw"]), 0.5)
            self.assertIn("noise_ceiling", maps)
            stitched = columns.stitch(table, root / "fits" / "f" / "sub0", "sub0")
            self.assertTrue((root / "fits" / "f" / "sub0" / "correlation_raw.nii.gz").exists())
            self.assertEqual(stitched["correlation_raw"].shape, (32,))

    def test_design_and_ceiling(self):
        rng = np.random.default_rng(0)
        d = design.prepare_design([rng.normal(size=(50, 30)) for _ in range(4)], rng.normal(size=(40, 30)), pca_components=8,
                                  extra_train=[rng.normal(size=(50, 1)) for _ in range(4)], extra_test=rng.normal(size=(40, 1)))
        self.assertEqual(d.train.shape, (200, 36))
        signal = rng.normal(size=(100, 4))
        self.assertTrue(np.allclose(noise_ceiling.noise_ceiling(np.stack([signal] * 5)), 1.0))


class Ablation(unittest.TestCase):
    def test_removal_and_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table, features = make_dataset(root)
            rng = np.random.default_rng(1)
            (root / "regressors").mkdir()
            for r in RUNS:
                np.save(root / "regressors" / f"{r}.npy", np.column_stack([features[r][:, 0] + 0.1 * rng.normal(size=60),
                                                                            rng.normal(size=60)]))
            summary = ablation.build_removed_features(root / "features", "f", root / "regressors", root / "features", ["test"])
            self.assertGreater(summary["mean_fraction_variance_removed"], 0.02)
            removed = {r: np.load(root / "features" / "f_removed" / f"{r}.npy") for r in RUNS}
            regs = {r: np.load(root / "regressors" / f"{r}.npy") for r in RUNS}
            series = ablation.orthogonalise(regs, RUNS[:-1])
            D = np.vstack([ablation.lagged_design(series[r]) for r in RUNS[:-1]])
            np.testing.assert_allclose(D.T @ np.vstack([removed[r] for r in RUNS[:-1]]).astype(np.float64), 0, atol=1e-2)
            space = ablation.NullSpace(root / "features", "f", root / "regressors", ["test"])
            out = ablation.select_directions(space, root / "nulls.npz", n=3, tolerance=0.9, max_candidates=500)
            self.assertGreater(out["accepted"], 0)
            path = ablation.fit_nulls(table, "sub0", space, root / "nulls.npz", root / "nullfits", 0, 1, 0, 2)
            self.assertEqual(np.load(path)["correlation"].shape[0], min(2, out["accepted"]))


class CrossParticipant(unittest.TestCase):
    def test_prep_fit_and_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table, features = make_dataset(root)
            predictor = np.zeros(32, dtype=bool)
            predictor[:20] = True
            for s in table.subjects():
                cross_participant.prep(table, s, root / "xsub", predictor, ~predictor, k=4, n_nuisance=2)
            path = cross_participant.fit(table, "sub0", root / "xsub", 0, 1)
            self.assertGreater(float(np.nanmean(np.load(path)["correlation_raw"])), 0.3)
            shifted = cross_participant.fit(table, "sub0", root / "xsub", 0, 1, control="shifted")
            self.assertLess(float(np.nanmean(np.load(shifted)["correlation_raw"])), 0.3)
            (root / "regressors").mkdir()
            rng = np.random.default_rng(2)
            for r in RUNS:
                np.save(root / "regressors" / f"{r}.npy", np.column_stack([features[r][:, 0], rng.normal(size=60)]))
            for s in table.subjects():
                cross_participant.ablation_components(table, s, root / "xsub", "a", root / "regressors")
            target = cross_participant.fit_ablation(table, "sub0", root / "xsub", "a", 0, 1)
            removed = cross_participant.read_ablation(target.parent.parent, 32, [-1])[-1]
            self.assertEqual(np.isfinite(removed).sum(), 32)


class GroupInference(unittest.TestCase):
    def test_sign_flip_bh_regions_summary(self):
        data = np.random.default_rng(0).normal(0.5, 1.0, size=(6, 40))
        result = group_inference.sign_flip(data)
        self.assertTrue(((result["fwe"] >= 1 / 64) & (result["fwe"] <= 1)).all())
        stack = np.vstack([np.full(6, 1.0) + 0.1 * k for k in range(8)])
        labels = np.zeros(np.prod((91, 109, 91)), dtype=int)
        labels[:6] = 1
        rows = group_inference.region_test(stack, np.arange(6), {"toy": (labels, {1: "region"})}, min_voxels=3)
        self.assertGreater(rows[0]["t"], 5)
        full = np.linspace(0, 1, 100)
        summary = group_inference.top_voxel_summary(full, full - 0.1)
        self.assertAlmostEqual(summary["top_mean_delta_r"], 0.1, places=6)

    def test_warp_apply(self):
        cols = np.array([[0, 1, -1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1, -1]])
        weights = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        warp = ColumnMNIWarp(np.array([0, 1]), cols, weights, np.array([0, 2]), 3)
        np.testing.assert_allclose(warp.apply(np.array([2.0, 4.0, 7.0])), [3.0, 7.0])
        self.assertTrue(np.isnan(warp.apply(np.array([np.nan, 4.0, 7.0]))[0]))
        with tempfile.TemporaryDirectory() as tmp:
            again = ColumnMNIWarp.load(warp.save(Path(tmp) / "w.npz"))
            np.testing.assert_allclose(again.apply(np.array([2.0, 4.0, 7.0])), [3.0, 7.0])


class StimuliAndRatings(unittest.TestCase):
    def test_static_and_rate_from_a_stimulus_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            words = [{"word": w, "onset": 0.5 * i, "offset": 0.5 * i + 0.4} for i, w in enumerate(["the", "cat", "sat", "zzz"] * 10)]
            (root / "story.json").write_text(json.dumps({"words": words}))
            write_stimuli([{"stimulus": "story", "words": "story.json", "n_samples": 12, "tr": 2.0, "first_time": 1.0}], root / "stimuli.csv")
            np.savez(root / "table.npz", vectors=np.eye(3, dtype=np.float32), vocab=np.array(["The", "cat", "sat"]))
            stimuli = read_stimuli(root / "stimuli.csv")
            np.testing.assert_allclose(stimuli[0].sample_times[:2], [1.0, 3.0])
            vectors, found = static.embed(["the", "zzz"], static.load_table(root / "table.npz"))
            self.assertEqual(found.tolist(), [True, False])
            batch.static(stimuli, root / "table.npz", root / "out" / "static")
            batch.rate(stimuli, root / "out" / "rate")
            self.assertEqual(np.load(root / "out" / "static" / "story.npy").shape, (12, 3))
            rate = np.load(root / "out" / "rate" / "story.npy")
            self.assertEqual(rate.shape, (12, 1))
            self.assertAlmostEqual(float(rate[5, 0]), 2.0, places=2)   # two words per second mid-story

    def test_rating_regressors(self):
        table = pd.DataFrame({"first_word": [0, 2], "n_words": [2, 3], "embodiment_mean": [1.0, 2.5]})
        np.testing.assert_allclose(word_values(table, 5, "embodiment"), [1, 1, 2.5, 2.5, 2.5])
        words = [Word(text=str(i), onset=0.5 * i, offset=0.5 * i + 0.2) for i in range(5)]
        out = rating_regressors(words, table, np.arange(0, 4, 2.0), "embodiment")
        self.assertEqual(out.shape, (2, 2))
        with self.assertRaises(ValueError):
            word_values(table, 6, "embodiment")


if __name__ == "__main__":
    unittest.main()
