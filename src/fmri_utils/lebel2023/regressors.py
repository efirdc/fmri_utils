"""A story rating as a regressor on the response clock, with the word rate it is orthogonalised on.

The rating comes from a ``fmri_utils.story_ratings`` run (``<run>/<story>/segment_ratings.csv``, or
the older ``utterance_ratings.csv``): one row per rated segment, with ``first_word``, ``n_words`` and
``<field>_mean`` (the mean over replicate raters). Every word takes its segment's rating; the words
come from the punctuated word tables the segments were rated on (``transcripts/<story>.json``, the
dataset's TextGrid words with display punctuation), so ``first_word`` indexes them directly.

Each word's rating sits at the word's midpoint and is summed onto the response rows with the same
Lanczos filter as the features, so a regressor row and a feature row describe the same moment. No
HRF: the encoding model learns the response shape with its FIR delays. Two columns per story:

``load``       the Lanczos sum of the word ratings (scales with speech rate, like the features)
``word_rate``  the Lanczos sum of ones (the same words weighted by one)

Written as ``<output>/<story>.npy`` (rows x 2). The ablation regresses ``load`` on ``word_rate``
(lags 0-4) before removing it, so what is removed is the rating beyond speech rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Dataset, lanczos_weights, tr_times

COLUMNS = ("load", "word_rate")


def read_words(transcripts: Path, story: str) -> list[dict]:
    for name in (f"{story}.json", f"{story}_punctuated.json"):
        path = Path(transcripts) / name
        if path.exists():
            words = json.loads(path.read_text(encoding="utf-8"))["words"]
            if words:
                return words
    raise FileNotFoundError(f"no word table for {story} under {transcripts}")


def segment_table(run: Path, story: str) -> pd.DataFrame:
    for name in ("segment_ratings.csv", "utterance_ratings.csv"):
        path = Path(run) / story / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"no segment ratings for {story} under {run}")


def word_ratings(table: pd.DataFrame, n_words: int, field: str) -> np.ndarray:
    """Each segment's ``field`` spread over the words it covers."""
    column = field if field in table.columns else f"{field}_mean"
    if column not in table.columns:
        raise KeyError(f"no column {field} or {field}_mean; have {list(table.columns)}")
    values = np.full(n_words, np.nan)
    for start, count, value in zip(table["first_word"], table["n_words"], table[column]):
        start, stop = int(start), int(start) + int(count)
        if start < 0 or stop > n_words:
            raise ValueError(f"segment covers words {start}:{stop} of {n_words}")
        values[start:stop] = float(value)
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} words are not in any segment")
    return values


def story_regressor(dataset: Dataset, run: Path, transcripts: Path, story: str, field: str) -> np.ndarray:
    words = read_words(transcripts, story)
    ratings = word_ratings(segment_table(run, story), len(words), field)
    midpoints = np.asarray([(float(w.get("onset_s", w.get("onset"))) + float(w.get("offset_s", w.get("offset")))) / 2
                            for w in words])
    weights = lanczos_weights(midpoints, tr_times(dataset.response_rows(story)))
    return np.column_stack([weights @ ratings, weights @ np.ones_like(ratings)]).astype(np.float32)


def build(dataset: Dataset, run: Path, transcripts: Path, output: Path, field: str,
          stories: list[str] | None = None) -> Path:
    """Writes ``<output>/<story>.npy`` (columns: load, word_rate) for every rated story."""
    stories = stories or sorted(p.name for p in Path(run).iterdir()
                                if (p / "segment_ratings.csv").exists() or (p / "utterance_ratings.csv").exists())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    built = {}
    for story in stories:
        values = story_regressor(dataset, run, transcripts, story, field)
        np.save(output / f"{story}.npy", values)
        built[story] = {"rows": int(values.shape[0]), "mean": values.mean(axis=0).tolist()}
    (output / "regressors.json").write_text(json.dumps({
        "columns": list(COLUMNS), "field": field, "run": str(run), "transcripts": str(transcripts),
        "timing": "word midpoint to response-row centre, Lanczos window 3, no HRF", "stories": built}, indent=1),
        encoding="utf-8")
    print(f"wrote {len(built)} stories to {output}", flush=True)
    return output


def load(folder: Path, stories: list[str]) -> dict[str, np.ndarray]:
    return {s: np.load(Path(folder) / f"{s}.npy").astype(np.float64) for s in stories}
