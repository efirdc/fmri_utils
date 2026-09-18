from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import SegmentationConfig
from .transcripts import CLAUSE_END, SENTENCE_END, Word


@dataclass(frozen=True)
class Segment:
    """One unit of text to rate, and where it sits in the story."""

    index: int
    text: str
    first_word: int
    n_words: int
    onset: Optional[float] = None
    offset: Optional[float] = None

    @property
    def last_word(self) -> int:
        return self.first_word + self.n_words - 1


def _build(words: Sequence[Word], runs: Sequence[Sequence[int]]) -> List[Segment]:
    segments: List[Segment] = []
    for part in runs:
        if not part:
            continue
        first, last = part[0], part[-1]
        segments.append(
            Segment(
                index=len(segments),
                text=" ".join(words[i].display for i in part),
                first_word=first,
                n_words=len(part),
                onset=words[first].onset,
                offset=words[last].offset,
            )
        )
    return segments


def _pause_after(words: Sequence[Word], index: int) -> float:
    """Silence between a word and the next one; 0 when the text has no timing."""
    if index + 1 >= len(words) or not words[index].is_timed or not words[index + 1].is_timed:
        return 0.0
    return max(0.0, float(words[index + 1].onset) - float(words[index].offset))


def _split_long(words: Sequence[Word], indices: List[int], max_words: int, min_words: int = 1) -> List[List[int]]:
    """Split an over-long run at its longest internal pause, recursively."""
    if len(indices) <= max_words:
        return [indices]
    margin = min(min_words, len(indices) // 2)
    interior = list(range(margin, len(indices) - margin)) or list(range(1, len(indices)))
    cut = max(interior, key=lambda k: _pause_after(words, indices[k - 1]))
    return _split_long(words, indices[:cut], max_words, min_words) + _split_long(words, indices[cut:], max_words, min_words)


def segment_words(words: Sequence[Word], config: SegmentationConfig) -> List[Segment]:
    """Split a story's words into rating units according to ``config``.

    ``pause`` needs word timings; ``sentence`` and ``clause`` need display
    tokens carrying punctuation (see
    :func:`fmri_utils.story_ratings.transcripts.transfer_punctuation`);
    ``whole`` needs neither.
    """
    config.validate()
    if not words:
        raise ValueError("no words to segment")
    if config.mode in {"pause", "clause"} and not any(word.is_timed for word in words):
        raise ValueError(f"{config.mode} segmentation needs word timings")

    if config.mode == "whole":
        runs = [list(range(len(words)))]
    elif config.mode == "pause":
        runs = [[0]]
        for index in range(1, len(words)):
            if _pause_after(words, index - 1) >= config.pause_seconds:
                runs.append([index])
            else:
                runs[-1].append(index)
    elif config.mode == "sentence":
        runs = [[]]
        for index, word in enumerate(words):
            runs[-1].append(index)
            if SENTENCE_END.search(word.display) and index < len(words) - 1:
                runs.append([])
    else:  # clause
        runs = [[]]
        since_break = 0
        for index, word in enumerate(words):
            runs[-1].append(index)
            if index == len(words) - 1:
                break
            since_break += 1
            pause = _pause_after(words, index)
            sentence_end = bool(SENTENCE_END.search(word.display))
            clause_end = bool(CLAUSE_END.search(word.display))
            cut = (
                sentence_end
                or (clause_end and (pause >= config.clause_pause_seconds or since_break >= config.max_words // 2))
                or pause >= config.hard_pause_seconds
            )
            if cut and (since_break >= config.min_words or sentence_end):
                runs.append([])
                since_break = 0

    bounded: List[List[int]] = []
    for run in runs:
        bounded.extend(_split_long(words, list(run), config.max_words, config.min_words))
    segments = _build(words, bounded)
    covered = sum(segment.n_words for segment in segments)
    if covered != len(words):
        raise ValueError(f"segmentation covered {covered} of {len(words)} words")
    return segments


def segments_to_records(segments: Sequence[Segment]) -> List[dict]:
    """Plain dictionaries, for writing to CSV or JSON."""
    return [
        {
            "index": segment.index,
            "onset_s": segment.onset,
            "offset_s": segment.offset,
            "first_word": segment.first_word,
            "n_words": segment.n_words,
            "text": segment.text,
        }
        for segment in segments
    ]
