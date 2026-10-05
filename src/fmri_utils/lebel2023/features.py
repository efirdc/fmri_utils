"""Stimulus features on the response clock: English1000, BERT, GPT-2 XL, and word rate.

Every feature is a value per word, placed at the word's midpoint and summed onto the response rows
with the three-lobe Lanczos filter of the dataset's public code (unnormalised, so the features carry
speech density, as in LeBel et al.). One array per story: ``<root>/<feature>/<story>.npy``, rows =
response rows.

``english1000``           the dataset's 985-dimensional English1000 word embedding (zero for words
                          outside its vocabulary)
``bert_wordctx10``        bert-base-uncased, last layer, mean over the word's tokens, each word
                          embedded after its 10 preceding words
``gpt2xl_l24_wordctx10``  gpt2-xl block 24, the word's last token, same context
``word_rate``             words per second through a Hann window of half-width 2 TRs: the extra
                          column the encoding model appends after its PCA ("Lanczos + rate")

The language-model features are computed with ``fmri_utils.features`` (checked against the original
extraction: per word identical, on the TR clock within 1e-6). They need torch and transformers and,
for GPT-2 XL, a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import Dataset, hann_weights, lanczos_weights, tr_times

LLM_FEATURES = {
    # id: (Hugging Face model, layer (None: last), pooling)
    "bert_wordctx10": ("bert-base-uncased", None, "unit_token_mean"),
    "gpt2xl_l24_wordctx10": ("gpt2-xl", 24, "unit_word_last_mean"),
}
FEATURES = ("english1000", *LLM_FEATURES)
CONTEXT_WORDS = 10


def word_midpoints(dataset: Dataset, story: str) -> tuple[list[str], np.ndarray]:
    words = dataset.words(story)
    return [w for _, _, w in words], np.asarray([(a + b) / 2.0 for a, b, _ in words])


def english1000(dataset: Dataset, story: str) -> np.ndarray:
    import h5py
    with h5py.File(dataset.english1000_path(), "r") as handle:
        vectors = np.asarray(handle["data"], dtype=np.float32)
        vocab = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(handle["vocab"])]
    lookup = {word.lower(): index for index, word in enumerate(vocab)}
    words, midpoints = word_midpoints(dataset, story)
    per_word = np.zeros((len(words), vectors.shape[0]), dtype=np.float32)
    for row, word in enumerate(words):
        index = lookup.get(word.lower())
        if index is not None:
            per_word[row] = vectors[:, index]
    times = tr_times(dataset.response_rows(story))
    return (lanczos_weights(midpoints, times) @ per_word).astype(np.float32)


def llm_feature(dataset: Dataset, story: str, feature: str, device: str = "", batch_size: int = 32,
                max_length: int = 512) -> np.ndarray:
    from fmri_utils.features import ContextSpec, FeatureSpec, ModelSpec, Unit, embed_units
    model, layer, pooling = LLM_FEATURES[feature]
    spec = FeatureSpec(model=ModelSpec(id=feature.split("_")[0], huggingface_id=model, layer=layer, pooling=pooling,
                                       batch_size=batch_size, max_length=max_length),
                       context=ContextSpec(previous=CONTEXT_WORDS), granularity="word")
    words, midpoints = word_midpoints(dataset, story)
    units = [Unit(text=w, onset=float(t), offset=float(t)) for w, t in zip(words, midpoints)]
    vectors, _ = embed_units(units, spec, device=device)
    times = tr_times(dataset.response_rows(story))
    return (lanczos_weights(midpoints, times) @ np.asarray(vectors, dtype=np.float64)).astype(np.float32)


def word_rate(dataset: Dataset, story: str, half_width: float = 2.0) -> np.ndarray:
    """Words per second through a Hann window (its time integral is half_width * TR seconds)."""
    _, midpoints = word_midpoints(dataset, story)
    times = tr_times(dataset.response_rows(story))
    step = float(np.mean(np.diff(times)))
    rate = hann_weights(midpoints, times, half_width) @ np.ones(midpoints.size) / (half_width * step)
    return rate.astype(np.float32)[:, None]


def build(dataset: Dataset, feature: str, stories: list[str], output_root: Path, device: str = "",
          overwrite: bool = False) -> Path:
    """Writes ``<output_root>/<feature>/<story>.npy`` for each story (existing files are kept)."""
    folder = Path(output_root) / feature
    folder.mkdir(parents=True, exist_ok=True)
    built = {}
    for story in stories:
        target = folder / f"{story}.npy"
        if target.exists() and not overwrite:
            continue
        if feature == "english1000":
            values = english1000(dataset, story)
        elif feature == "word_rate":
            values = word_rate(dataset, story)
        elif feature in LLM_FEATURES:
            values = llm_feature(dataset, story, feature, device=device)
        else:
            raise ValueError(f"unknown feature {feature}; one of {(*FEATURES, 'word_rate')}")
        np.save(target, values)
        built[story] = list(values.shape)
        print(f"{feature} {story}: {values.shape}", flush=True)
    (folder / "build.json").write_text(json.dumps({"feature": feature, "stories": built,
                                                   "resampling": "word midpoint to response-row centre, "
                                                                 "Lanczos window 3 (word_rate: Hann, half-width 2)"},
                                                  indent=1), encoding="utf-8")
    return folder


def all_stories(dataset: Dataset) -> list[str]:
    """Every story with a TextGrid and a released response."""
    stories = sorted(p.stem for p in (dataset.derivatives / "TextGrids").glob("*.TextGrid"))
    out = []
    for story in stories:
        try:
            dataset.response_rows(story)
            out.append(story)
        except FileNotFoundError:
            pass
    return out


def load(root: Path, feature: str, story: str) -> np.ndarray:
    path = Path(root) / feature / f"{story}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path), dtype=np.float32)
