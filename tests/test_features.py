from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fmri_utils.features import (
    ContextSpec,
    FeatureSpec,
    ModelSpec,
    Unit,
    build_payloads,
    build_stack_payloads,
    cache,
    resample,
    units_from_texts,
)
from fmri_utils.features.pooling import pool_unit, tokens_in_span


def _words(texts, start=0.0, step=0.4, run=""):
    return [Unit(text=t, onset=start + i * step, offset=start + i * step + 0.3, run=run)
            for i, t in enumerate(texts)]


def _spec(**kwargs):
    model = kwargs.pop("model", ModelSpec(id="m", huggingface_id="hf"))
    return FeatureSpec(model=model, **kwargs)


class SpecTests(unittest.TestCase):
    def test_id_distinguishes_granularity_and_context(self) -> None:
        model = ModelSpec(id="bert", huggingface_id="bert-base-uncased")
        word = FeatureSpec(model=model, context=ContextSpec(previous=10), granularity="word")
        tr = FeatureSpec(model=model, context=ContextSpec(previous=10), granularity="tr")
        self.assertNotEqual(word.id, tr.id)
        self.assertEqual(word.id, "bert_wordctx10")
        self.assertEqual(tr.id, "bert_trctx10")

    def test_layer_appears_in_the_id(self) -> None:
        model = ModelSpec(id="gpt2xl", huggingface_id="gpt2-xl", layer=24)
        self.assertEqual(FeatureSpec(model=model).id, "gpt2xl_l24_wordctx0")

    def test_a_model_that_needs_an_id_says_so(self) -> None:
        with self.assertRaises(ValueError):
            ModelSpec(id="x", family="transformer", huggingface_id="")

    def test_layers_are_one_based(self) -> None:
        with self.assertRaises(ValueError):
            ModelSpec(id="x", huggingface_id="hf", layer=0)


class PayloadTests(unittest.TestCase):
    def test_context_is_prepended_and_the_unit_is_locatable(self) -> None:
        units = _words(["the", "cat", "sat"])
        payloads = build_payloads(units, previous=2)
        self.assertEqual(payloads[0].context_text, "the")
        self.assertEqual(payloads[2].context_text, "the cat sat")
        # the span must actually select the unit out of the context
        for payload in payloads:
            self.assertEqual(payload.context_text[payload.start:payload.end],
                             payload.unit_text)

    def test_context_does_not_cross_a_run_boundary(self) -> None:
        units = _words(["a", "b"], run="r1") + _words(["c", "d"], run="r2")
        payloads = build_payloads(units, previous="max")
        self.assertEqual(payloads[1].context_text, "a b")
        self.assertEqual(payloads[2].context_text, "c")
        self.assertEqual(payloads[3].context_text, "c d")

    def test_max_context_takes_everything_so_far(self) -> None:
        units = _words(["a", "b", "c", "d"])
        payloads = build_payloads(units, previous="max")
        self.assertEqual(payloads[3].context_text, "a b c d")

    def test_stacked_payloads_keep_a_fixed_width(self) -> None:
        units = _words(["a", "b", "c"])
        payloads = build_stack_payloads(units, slots=3, previous=2)
        for payload in payloads:
            self.assertEqual(len(payload.slot_spans), 3)
            # the unit is always the last slot
            start, end = payload.slot_spans[-1]
            self.assertEqual(payload.context_text[start:end], payload.unit_text)

    def test_word_spans_cover_a_multiword_unit(self) -> None:
        units = units_from_texts(["the cat sat", "on the mat"], times=[0.0, 2.0], duration=2.0)
        payload = build_payloads(units, previous=1)[1]
        spans = payload.word_spans
        self.assertEqual(len(spans), 3)
        self.assertEqual([payload.context_text[a:b] for a, b in spans],
                         ["on", "the", "mat"])


class PoolingTests(unittest.TestCase):
    def test_token_selection_is_by_overlap(self) -> None:
        offsets = np.array([[0, 3], [3, 7], [7, 10]])
        usable = np.ones(3, dtype=bool)
        # a unit at 5..9 overlaps the second and third tokens
        self.assertEqual(list(tokens_in_span(offsets, usable, 5, 9)), [False, True, True])

    def test_unused_tokens_never_contribute(self) -> None:
        offsets = np.array([[0, 2], [2, 4]])
        usable = np.array([True, False])
        hidden = np.array([[1.0, 1.0], [9.0, 9.0]])
        vector, backed = pool_unit(hidden, offsets, usable, 0, 4, "unit_token_mean")
        self.assertTrue(backed)
        np.testing.assert_allclose(vector, [1.0, 1.0])

    def test_a_unit_with_no_tokens_is_flagged_not_silently_zero(self) -> None:
        offsets = np.array([[0, 2]])
        usable = np.array([True])
        hidden = np.array([[5.0, 5.0]])
        vector, backed = pool_unit(hidden, offsets, usable, 40, 44, "unit_token_mean")
        self.assertFalse(backed)
        np.testing.assert_allclose(vector, [0.0, 0.0])

    def test_word_last_mean_averages_one_token_per_word(self) -> None:
        # two words, two tokens each; the last token of each should be used
        offsets = np.array([[0, 1], [1, 3], [4, 5], [5, 7]])
        usable = np.ones(4, dtype=bool)
        hidden = np.array([[0.0], [2.0], [0.0], [4.0]])
        vector, backed = pool_unit(hidden, offsets, usable, 0, 7, "unit_word_last_mean",
                                   word_spans=[(0, 3), (4, 7)])
        self.assertTrue(backed)
        np.testing.assert_allclose(vector, [3.0])


class ResampleTests(unittest.TestCase):
    def test_lanczos_sum_reproduces_the_published_resampler(self) -> None:
        # The reference implementation, transcribed: unnormalised, cutoff from
        # the output spacing. This is the behaviour prior work depends on.
        rng = np.random.default_rng(0)
        unit_times = np.sort(rng.uniform(0, 100, 260))
        sample_times = np.arange(0, 100, 2.0) + 1.0
        values = rng.normal(size=(260, 3))

        cutoff = 1.0 / np.mean(np.diff(sample_times))
        delta = (sample_times[:, None] - unit_times[None, :]) * cutoff
        expected = np.zeros_like(delta)
        inside = (np.abs(delta) <= 3) & (delta != 0)
        v = delta[inside]
        expected[inside] = 3 * np.sin(np.pi * v) * np.sin(np.pi * v / 3) / (np.pi ** 2 * v ** 2)
        expected[delta == 0] = 1.0

        got = resample.lanczos_sum(values, unit_times, sample_times)
        np.testing.assert_allclose(got, (expected @ values).astype(np.float32), rtol=1e-5)

    def test_normalised_kernels_preserve_a_constant(self) -> None:
        # The property the published resampler does not have: resampling a
        # constant signal should return that constant, not the event rate.
        unit_times = np.linspace(0.5, 99.5, 300)
        sample_times = np.arange(0, 100, 2.0) + 1.0
        values = np.full((300, 1), 2.5)
        for kernel in ("hann", "gaussian", "boxcar"):
            out = resample.resample(values, unit_times, sample_times, kernel=kernel)
            np.testing.assert_allclose(out[2:-2], 2.5, rtol=1e-5,
                                       err_msg=f"{kernel} did not preserve a constant")

    def test_lanczos_sum_does_not_preserve_a_constant(self) -> None:
        unit_times = np.linspace(0.5, 99.5, 300)
        sample_times = np.arange(0, 100, 2.0) + 1.0
        out = resample.lanczos_sum(np.full((300, 1), 2.5), unit_times, sample_times)
        # it scales with the oversampling factor instead, which is the whole point
        self.assertGreater(float(np.median(out)), 5.0)

    def test_hann_weights_are_non_negative_so_the_denominator_is_safe(self) -> None:
        delta = np.linspace(-4, 4, 401)
        self.assertGreaterEqual(float(resample.hann_kernel(delta).min()), 0.0)
        self.assertLess(float(resample.lanczos_kernel(delta).min()), 0.0)

    def test_event_rate_recovers_a_known_rate(self) -> None:
        # three units per second, evenly spaced
        unit_times = np.arange(0, 100, 1 / 3)
        sample_times = np.arange(0, 100, 2.0) + 1.0
        rate = resample.event_rate(unit_times, sample_times)
        self.assertAlmostEqual(float(np.median(rate[3:-3])), 3.0, delta=0.3)

    def test_support_is_reported_for_samples_nothing_reaches(self) -> None:
        unit_times = np.array([1.0, 2.0, 3.0])
        sample_times = np.arange(0, 60, 2.0) + 1.0
        _, support = resample.weights(unit_times, sample_times, kernel="boxcar")
        self.assertTrue(support[0])
        self.assertFalse(support[-1])


class CacheTests(unittest.TestCase):
    def test_a_round_trip_keeps_the_times_that_make_it_resamplable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            units = _words(["alpha", "beta", "gamma"])
            spec = _spec(context=ContextSpec(previous=2))
            embeddings = np.arange(9, dtype=np.float32).reshape(3, 3)
            valid = np.array([True, True, False])

            cache.write(root, spec, "story", units, embeddings, valid)
            entry = cache.read(root, spec, "story")

            np.testing.assert_allclose(entry["embeddings"], embeddings)
            np.testing.assert_array_equal(entry["valid"], valid)
            np.testing.assert_allclose(entry["midpoints"],
                                       [u.midpoint for u in units])
            self.assertEqual(entry["texts"], ["alpha", "beta", "gamma"])
            self.assertEqual(entry["meta"]["granularity"], "word")
            self.assertEqual(entry["meta"]["n_valid"], 2)

    def test_changed_timings_miss_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            units = _words(["alpha", "beta"])
            spec = _spec()
            cache.write(root, spec, "story", units, np.zeros((2, 4), np.float32),
                        np.ones(2, bool))
            self.assertTrue(cache.is_current(root, spec, "story", units))

            shifted = [Unit(text=u.text, onset=u.onset + 0.01, offset=u.offset)
                       for u in units]
            self.assertFalse(cache.is_current(root, spec, "story", shifted))

    def test_a_different_context_is_a_different_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            units = _words(["alpha", "beta"])
            near = _spec(context=ContextSpec(previous=2))
            far = _spec(context=ContextSpec(previous=10))
            cache.write(root, near, "story", units, np.zeros((2, 4), np.float32),
                        np.ones(2, bool))
            self.assertTrue(cache.is_current(root, near, "story", units))
            self.assertFalse(cache.is_current(root, far, "story", units))

    def test_mismatched_lengths_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                cache.write(Path(tmp), _spec(), "story", _words(["a", "b"]),
                            np.zeros((3, 4), np.float32), np.ones(3, bool))

    def test_cached_embeddings_resample_without_the_model(self) -> None:
        # the point of the package: one extraction, many kernels
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            units = _words([f"w{i}" for i in range(300)], step=1 / 3)
            spec = _spec()
            rng = np.random.default_rng(1)
            cache.write(root, spec, "story", units,
                        rng.normal(size=(300, 5)).astype(np.float32),
                        np.ones(300, bool))

            entry = cache.read(root, spec, "story")
            sample_times = np.arange(0, 90, 2.0) + 1.0
            hann = resample.resample(entry["embeddings"], entry["midpoints"],
                                     sample_times, kernel="hann")
            lanczos = resample.lanczos_sum(entry["embeddings"], entry["midpoints"],
                                           sample_times)
            self.assertEqual(hann.shape, (sample_times.size, 5))
            self.assertEqual(lanczos.shape, (sample_times.size, 5))
            self.assertGreater(np.abs(lanczos).mean(), np.abs(hann).mean())


class CommandLineTests(unittest.TestCase):
    def test_resample_reads_a_cache_entry_and_writes_samples_and_rate(self) -> None:
        import tempfile
        from fmri_utils.features import ContextSpec, FeatureSpec, ModelSpec, Unit, cache
        from fmri_utils.features.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = FeatureSpec(model=ModelSpec(id="m", huggingface_id="some/model"),
                               context=ContextSpec(previous=3))
            units = [Unit(text=f"w{i}", onset=0.5 * i, offset=0.5 * i + 0.3) for i in range(40)]
            embeddings = np.ones((40, 4), dtype=np.float32)
            entry = cache.write(tmp, spec, "story", units, embeddings, np.ones(40, dtype=bool))
            main(["resample", "--entry", str(entry), "--tr", "2.0", "--n-samples", "10",
                  "--kernel", "hann", "--out", str(tmp / "hann.npy"),
                  "--rate-out", str(tmp / "rate.npy")])
            hann = np.load(tmp / "hann.npy")
            self.assertEqual(hann.shape, (10, 4))
            # A normalised kernel keeps a constant where words reach.
            self.assertTrue(np.allclose(hann[1:8], 1.0, atol=1e-5))
            self.assertEqual(np.load(tmp / "rate.npy").shape, (10,))
            # The same entry found from the spec flags, through the published resampler.
            main(["resample", "--cache-root", str(tmp), "--stimulus", "story",
                  "--model", "some/model", "--model-id", "m", "--context", "3",
                  "--tr", "2.0", "--n-samples", "10", "--kernel", "lanczos-sum",
                  "--out", str(tmp / "sum.npy")])
            self.assertEqual(np.load(tmp / "sum.npy").shape, (10, 4))


if __name__ == "__main__":
    unittest.main()
