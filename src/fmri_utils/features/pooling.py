"""Turning token states into one vector per unit.

The model sees a context string; we want a vector for the unit inside it. Every
strategy here is some reduction over the tokens whose character offsets fall in
the unit's span, which is why the extractor works the same for a word and for a
TR's worth of text.

Offsets are the load-bearing part. A token belongs to the unit when its
character span overlaps the unit's, not when it is merely nearby, and special
and padding tokens never belong to anything.
"""

from __future__ import annotations

import numpy as np


def tokens_in_span(offsets: np.ndarray, usable: np.ndarray,
                   start: int, end: int) -> np.ndarray:
    """Which tokens overlap ``[start, end)``.

    Overlap rather than containment, because a tokenizer is free to merge the
    end of one word with the start of the next, and dropping those tokens would
    quietly shorten the unit.
    """
    if end <= start:
        return np.zeros(offsets.shape[0], dtype=bool)
    begins, ends = offsets[:, 0], offsets[:, 1]
    return usable & (ends > start) & (begins < end) & (ends > begins)


def pool_unit(hidden: np.ndarray, offsets: np.ndarray, usable: np.ndarray,
              start: int, end: int, mode: str,
              word_spans: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, bool]:
    """One vector for one unit, plus whether any token backed it.

    The flag matters: a unit whose tokens were all truncated away returns
    zeros, and a caller that cannot tell that from a genuine zero will average
    padding into its features.
    """
    selected = tokens_in_span(offsets, usable, start, end)
    if mode == "unit_token_mean":
        if not selected.any():
            return np.zeros(hidden.shape[1], dtype=np.float32), False
        return hidden[selected].mean(axis=0).astype(np.float32), True

    if mode == "unit_last_token":
        if not selected.any():
            return np.zeros(hidden.shape[1], dtype=np.float32), False
        last = np.nonzero(selected)[0][-1]
        return hidden[last].astype(np.float32), True

    if mode == "unit_word_last_mean":
        # One vector per word -- its final token, which for a causal model is
        # the only position that has seen the whole word -- then the mean.
        vectors = []
        for word_start, word_end in (word_spans or []):
            chosen = tokens_in_span(offsets, usable, word_start, word_end)
            if chosen.any():
                vectors.append(hidden[np.nonzero(chosen)[0][-1]])
        if not vectors:
            return np.zeros(hidden.shape[1], dtype=np.float32), False
        return np.mean(vectors, axis=0).astype(np.float32), True

    raise ValueError(f"unknown pooling mode {mode!r}")


def pool_slots(hidden: np.ndarray, offsets: np.ndarray, usable: np.ndarray,
               slot_spans: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, bool]:
    """One pooled vector per context slot, concatenated oldest to newest.

    Empty slots stay zero so the width is the same for every unit, including
    the first few of a run where there is no history yet.
    """
    pieces = []
    unit_backed = False
    for index, (start, end) in enumerate(slot_spans):
        selected = tokens_in_span(offsets, usable, start, end)
        if selected.any():
            pieces.append(hidden[selected].mean(axis=0))
            if index == len(slot_spans) - 1:
                unit_backed = True
        else:
            pieces.append(np.zeros(hidden.shape[1], dtype=hidden.dtype))
    return np.concatenate(pieces).astype(np.float32), unit_backed
