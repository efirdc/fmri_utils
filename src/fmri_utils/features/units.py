"""Units of stimulus, and the context strings a model actually sees.

A unit is one thing you want a vector for: a word, a TR's worth of text, a
clause. It carries its own timing, because the whole reason to keep units
rather than a resampled matrix is that timing is what lets you resample later,
under any kernel, without running the model again.

Building a context payload is the same operation at every granularity: glue
some preceding units in front of the one you want, and remember where in that
string the unit itself sits. The extractor then pools model states over those
character offsets. Nothing here knows whether a unit is a word or a TR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Unit:
    """One unit of stimulus, with the timing that makes it resamplable."""

    text: str
    onset: Optional[float] = None
    offset: Optional[float] = None
    run: str = ""

    @property
    def is_timed(self) -> bool:
        return self.onset is not None and self.offset is not None

    @property
    def midpoint(self) -> Optional[float]:
        if not self.is_timed:
            return None
        return (float(self.onset) + float(self.offset)) / 2.0


@dataclass(frozen=True)
class Payload:
    """One model input: the context string, and where the unit sits in it."""

    index: int
    context_text: str
    unit_text: str
    start: int
    end: int
    # For ``context_slot_stack``: one (start, end) per slot, the unit's last.
    slot_spans: tuple[tuple[int, int], ...] = ()

    @property
    def word_spans(self) -> list[tuple[int, int]]:
        """Character spans of each word of the unit, in context coordinates."""
        return [(self.start + m.start(), self.start + m.end())
                for m in WORD.finditer(self.unit_text)]


def build_payloads(units: Sequence[Unit], previous: int | str = 0) -> list[Payload]:
    """Glue ``previous`` units of context in front of each unit.

    Context never crosses a run boundary: a unit at the start of a run gets
    whatever context that run has so far and nothing from the run before, which
    is the only defensible reading when runs are separate acquisitions.
    """
    payloads: list[Payload] = []
    for start, stop in _run_slices(units):
        texts = [unit.text.strip() for unit in units[start:stop]]
        for offset, text in enumerate(texts):
            first = 0 if previous == "max" else max(0, offset - int(previous))
            before = " ".join(piece for piece in texts[first:offset] if piece)
            if before and text:
                context = f"{before} {text}"
                begin = len(before) + 1
            elif before:
                context, begin = before, len(before)
            else:
                context, begin = text, 0
            payloads.append(Payload(
                index=start + offset,
                context_text=context,
                unit_text=text,
                start=begin,
                end=begin + len(text),
            ))
    return payloads


def build_stack_payloads(units: Sequence[Unit], slots: int,
                         previous: int | str = 0) -> list[Payload]:
    """Context payloads that also record where each preceding slot sits.

    Used by ``context_slot_stack``, which returns one pooled vector per slot
    rather than one per unit, so a model can express the recent past as
    separate dimensions instead of averaging it away.
    """
    if slots <= 0:
        raise ValueError("slots must be > 0 for a stacked payload")
    payloads: list[Payload] = []
    for start, stop in _run_slices(units):
        texts = [unit.text.strip() for unit in units[start:stop]]
        for offset, text in enumerate(texts):
            first = 0 if previous == "max" else max(0, offset - int(previous))
            window = list(range(first, offset + 1))[-slots:]
            pieces, spans, cursor = [], [], 0
            for position in window:
                piece = texts[position]
                if pieces:
                    cursor += 1  # the joining space
                spans.append((cursor, cursor + len(piece)))
                cursor += len(piece)
                pieces.append(piece)
            # Empty leading slots keep the output width fixed.
            spans = [(0, 0)] * (slots - len(spans)) + spans
            payloads.append(Payload(
                index=start + offset,
                context_text=" ".join(pieces),
                unit_text=text,
                start=spans[-1][0],
                end=spans[-1][1],
                slot_spans=tuple(spans),
            ))
    return payloads


def _run_slices(units: Sequence[Unit]) -> Iterable[tuple[int, int]]:
    """Contiguous spans of units sharing a run label."""
    if not units:
        return []
    bounds, start = [], 0
    for index in range(1, len(units) + 1):
        if index == len(units) or units[index].run != units[start].run:
            bounds.append((start, index))
            start = index
    return bounds


def words_from_table(rows: Iterable[dict], text_key: str = "word",
                     onset_key: str = "onset", offset_key: str = "offset",
                     run_key: str = "") -> list[Unit]:
    """Units from any table of words with timings."""
    units = []
    for row in rows:
        units.append(Unit(
            text=str(row[text_key]),
            onset=None if row.get(onset_key) is None else float(row[onset_key]),
            offset=None if row.get(offset_key) is None else float(row[offset_key]),
            run=str(row.get(run_key, "")) if run_key else "",
        ))
    return units


def units_from_texts(texts: Sequence[str], times: Optional[Sequence[float]] = None,
                     duration: float = 0.0, run: str = "") -> list[Unit]:
    """Units from a list of strings, optionally on a regular clock.

    This is the TR case: ``times`` are the onsets and ``duration`` is the TR,
    so each unit spans its own sample.
    """
    units = []
    for index, text in enumerate(texts):
        onset = None if times is None else float(times[index])
        offset = None if onset is None else onset + float(duration)
        units.append(Unit(text=text, onset=onset, offset=offset, run=run))
    return units
