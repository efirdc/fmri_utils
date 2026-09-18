from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import RatingSpec
from .segmentation import Segment
from .transcripts import Word


def lanczos_weights(old_times: np.ndarray, new_times: np.ndarray, window: int = 3) -> np.ndarray:
    """Three-lobe Lanczos resampling weights from word times to sample times.

    This is the filter LeBel et al. (2023) use to move word-rate features onto
    the scanner clock, so rating regressors built here share the timing
    treatment of the semantic features they are usually compared against.
    """
    if len(new_times) < 2:
        raise ValueError("need at least two sample times")
    cutoff = 1.0 / float(np.mean(np.diff(new_times)))
    delta = (np.asarray(new_times)[:, None] - np.asarray(old_times)[None, :]) * cutoff
    weights = np.zeros_like(delta, dtype=np.float64)
    nonzero = (np.abs(delta) <= window) & (delta != 0)
    weights[delta == 0] = 1.0
    values = delta[nonzero]
    weights[nonzero] = window * np.sin(np.pi * values) * np.sin(np.pi * values / window) / (np.pi**2 * values**2)
    return weights


def ratings_to_timeseries(
    words: Sequence[Word],
    segments: Sequence[Segment],
    consensus: pd.DataFrame,
    sample_times: Sequence[float],
    spec: RatingSpec,
    *,
    lanczos_window: int = 3,
    speaking_threshold: float = 0.5,
) -> pd.DataFrame:
    """Project segment ratings onto an acquisition time grid.

    Every word inherits its segment's mean rating, and three series are
    returned per sample time (for fMRI, one row per TR):

    - ``<field>_load``: Lanczos-resampled sum, which scales with speech rate
      the way word-rate features do, suitable as a regressor.
    - ``<field>_per_word``: that sum divided by the resampled word rate, so it
      stays on the rating scale; ``NaN`` where nobody is speaking.
    - ``<field>_held``: the most recent segment's rating carried forward
      through pauses, which treats a rating as a standing state.

    Silence is not the same as a zero rating, which is why ``per_word`` and
    ``held`` are kept separate from ``load``.
    """
    if not any(word.is_timed for word in words):
        raise ValueError("word timings are required to build a time series")
    counts = consensus["n_words"].to_numpy()
    if int(counts.sum()) != len(words):
        raise ValueError("segment word counts do not cover the word list")
    midpoints = np.asarray([(float(word.onset) + float(word.offset)) / 2.0 for word in words])
    sample_times = np.asarray(sample_times, dtype=float)
    weights = lanczos_weights(midpoints, sample_times, window=lanczos_window)
    word_rate = weights.sum(axis=1)
    speaking = word_rate > speaking_threshold

    frame = pd.DataFrame({"sample_index": np.arange(len(sample_times)), "sample_time_s": sample_times})
    frame["word_rate"] = word_rate
    fields = [spec.scale.name, *spec.flags]
    for field in fields:
        per_word_values = np.repeat(consensus[f"{field}_mean"].to_numpy(), counts)
        load = weights @ per_word_values
        frame[f"{field}_load"] = load
        frame[f"{field}_per_word"] = np.where(speaking, load / np.where(speaking, word_rate, 1.0), np.nan)

    onsets = consensus["onset_s"].to_numpy(dtype=float)
    active = np.searchsorted(onsets, sample_times, side="right") - 1
    last_offset = float(consensus["offset_s"].iloc[-1])
    for field in fields:
        means = consensus[f"{field}_mean"].to_numpy(dtype=float)
        held = np.where(active >= 0, means[np.clip(active, 0, None)], np.nan)
        held[sample_times > last_offset] = np.nan
        frame[f"{field}_held"] = held
    frame["active_segment"] = np.where(active >= 0, active, -1)
    return frame


def scanner_sample_times(n_samples: int, tr_seconds: float, start_time: float = 0.0) -> np.ndarray:
    """Sample times at TR centres, the usual grid for an fMRI regressor."""
    return start_time + (np.arange(n_samples, dtype=float) + 0.5) * tr_seconds
