"""A stimulus table: each run's words and its sample clock, for building features in batch.

A CSV with one row per stimulus (a story, a movie part):

``stimulus``    its id; output files are ``<stimulus>.npy``
``words``       its word timings: a Praat TextGrid (word tier), or a word table (.json/.csv/.tsv
                with word/onset/offset), relative paths against the table
``n_samples``   samples (TRs) on its response clock
``tr``          seconds between samples
``first_time``  time of the first sample in the words' time base (seconds; default tr / 2)

so sample i is at ``first_time + i * tr``. Word times are midpoints (onset + offset) / 2.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Stimulus:
    name: str
    words: Path
    n_samples: int
    tr: float
    first_time: float

    @property
    def sample_times(self) -> np.ndarray:
        return self.first_time + np.arange(self.n_samples, dtype=np.float64) * self.tr

    def read_words(self):
        """The stimulus's words (``story_ratings.Word``), in order."""
        from fmri_utils.story_ratings.transcripts import read_textgrid_words, read_word_table
        if self.words.suffix.lower() == ".textgrid":
            return read_textgrid_words(self.words)
        return read_word_table(self.words)

    def word_midpoints(self) -> tuple[list[str], np.ndarray]:
        words = self.read_words()
        return [w.text for w in words], np.asarray([(float(w.onset) + float(w.offset)) / 2.0 for w in words])


def read_stimuli(path: Path | str) -> list[Stimulus]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        words = Path(row["words"])
        tr = float(row["tr"])
        first = row.get("first_time")
        out.append(Stimulus(row["stimulus"], words if words.is_absolute() else path.parent / words,
                            int(row["n_samples"]), tr, float(first) if first not in (None, "") else tr / 2.0))
    return out


def write_stimuli(stimuli: list[dict], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stimulus", "words", "n_samples", "tr", "first_time"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in stimuli:
            writer.writerow({f: row.get(f, "") for f in fields})
    return path
