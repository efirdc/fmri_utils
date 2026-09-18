from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

SENTENCE_END = re.compile(r"[.?!]+[\"')\]]*$")
CLAUSE_END = re.compile(r"[,;:–—-]+[\"')\]]*$")
BAD_WORDS = frozenset({"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", ""})


@dataclass(frozen=True)
class Word:
    """One spoken or written word.

    ``text`` is the canonical token (from forced alignment, if any) and
    ``display`` is what a reader sees, which may add capitalization and
    punctuation. Times are seconds and may be ``None`` for written text that
    has no timing, in which case only the ``sentence`` and ``whole``
    segmentation modes are available.
    """

    text: str
    display: str = ""
    onset: Optional[float] = None
    offset: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.display:
            object.__setattr__(self, "display", self.text)

    @property
    def is_timed(self) -> bool:
        return self.onset is not None and self.offset is not None


def normalize(token: str) -> str:
    """Lowercase a token and drop everything but letters, digits and apostrophes."""
    return re.sub(r"[^a-z0-9']", "", token.lower().replace("’", "'"))


def words_from_text(text: str) -> List[Word]:
    """Split written text into untimed words, keeping punctuation for display."""
    return [Word(text=normalize(token) or token, display=token) for token in text.split() if token]


def read_textgrid_words(path: Path) -> List[Word]:
    """Read the word tier of a Praat TextGrid, in any of its serializations."""
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    if content.startswith('"Praat chronological TextGrid text file"'):
        matches = re.findall(r'(?m)^2\s+([^\s]+)\s+([^\s]+)\s*\r?\n"(.*)"\s*$', content)
    elif "item [" not in content:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        starts = [index for index, line in enumerate(lines) if line == '"IntervalTier"']
        if len(starts) < 2:
            raise ValueError(f"expected a phone tier and a word tier in {path}")
        start = starts[1]
        count = int(lines[start + 4])
        entries = lines[start + 5 : start + 5 + 3 * count]
        matches = [tuple(entries[index : index + 3]) for index in range(0, len(entries), 3)]
    else:
        tiers = [tier for tier in re.split(r"(?=\s*item \[\d+\]:)", content) if 'class = "IntervalTier"' in tier]
        if len(tiers) < 2:
            raise ValueError(f"expected a phone tier and a word tier in {path}")
        matches = re.findall(
            r"intervals \[\d+\]:\s+xmin = ([^\r\n]+)\s+xmax = ([^\r\n]+)\s+text = \"(.*?)\"",
            tiers[1],
            flags=re.DOTALL,
        )
    words: List[Word] = []
    for start, stop, token in matches:
        word = token.replace('""', '"').strip().strip('"').strip("{}").lower()
        if word not in BAD_WORDS:
            words.append(Word(text=word, onset=float(start), offset=float(stop)))
    if not words:
        raise ValueError(f"no words parsed from {path}")
    return words


def read_word_table(path: Path) -> List[Word]:
    """Read words from a CSV/TSV/JSON table.

    Recognized columns: ``word`` or ``text`` (required), ``display``,
    ``onset``/``start`` and ``offset``/``end`` in seconds.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["words"] if isinstance(payload, dict) else payload
    else:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    words: List[Word] = []
    for row in rows:
        text = row.get("word", row.get("text"))
        if text is None:
            raise ValueError(f"{path}: rows need a 'word' or 'text' column")
        onset, offset = row.get("onset", row.get("start")), row.get("offset", row.get("end"))
        words.append(
            Word(
                text=str(text),
                display=str(row.get("display") or text),
                onset=float(onset) if onset not in (None, "") else None,
                offset=float(offset) if offset not in (None, "") else None,
            )
        )
    if not words:
        raise ValueError(f"no words read from {path}")
    return words


def write_word_table(words: Sequence[Word], path: Path, story: str = "", extra: Optional[dict] = None) -> Path:
    """Write words as JSON, the format `read_word_table` reads back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "story": story,
        **(extra or {}),
        "words": [
            {"i": index, "word": word.text, "display": word.display, "onset": word.onset, "offset": word.offset}
            for index, word in enumerate(words)
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def has_punctuation(words: Sequence[Word]) -> bool:
    """Whether display tokens carry the punctuation sentence/clause modes need."""
    return any(SENTENCE_END.search(word.display) or CLAUSE_END.search(word.display) for word in words)


