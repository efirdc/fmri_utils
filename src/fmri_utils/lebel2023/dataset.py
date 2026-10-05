"""The LeBel et al. (2023) natural-language fMRI dataset (OpenNeuro ds003020, snapshot 3.1.1).

Responses are the released preprocessed data: ``derivatives/preprocessed_data/<subject>/<story>.hf5``,
one matrix per story (TRs x voxels, the subject's pycortex ``mask_thick`` voxels in pycortex order).
The first 10 and last 5 TRs of each run are already trimmed. The test story ``wheretheressmoke`` is
the mean of its repeats, and its ``individual_repeats`` give the noise ceiling.

Everything here works on that column order ("response columns"); ``unmask`` puts a vector of column
values back into the subject's functional volume.

Timing follows the dataset's public code: the 2 s TR clock starts 10 s before the sound, so the
response row ``i`` is centred at ``2 i + 11`` seconds of story time.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

SUBJECTS = tuple(f"UTS{index:02d}" for index in range(1, 9))
TEST_STORY = "wheretheressmoke"
TR = 2.0
TRIM_START, TRIM_END = 10, 5
FIR_DELAYS = (1, 2, 3, 4)
RIDGE_ALPHAS = tuple(float(10 ** (1 + 0.5 * k)) for k in range(9))   # 10 ... 1e5, half-decade steps
BAD_WORDS = frozenset({"sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp", ""})


class Dataset:
    """Paths and loaders for one copy of ds003020."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        for name in ("derivatives", "derivative"):
            if (self.root / name).exists():
                self.derivatives = self.root / name
                break
        else:
            raise FileNotFoundError(f"no derivatives directory under {self.root}")

    # ---- stories ----------------------------------------------------------------------------
    def story_sessions(self) -> dict[str, list[str]]:
        for path in (self.root / "code/deep-fMRI-dataset/em_data/sess_to_story.json",
                     self.derivatives / "sess_to_story.json"):
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                return {str(key): list(value[0]) for key, value in raw.items()}
        raise FileNotFoundError("sess_to_story.json (code/deep-fMRI-dataset/em_data) is required")

    def training_stories(self, subject: str) -> list[str]:
        """The canonical training stories this subject has responses for, in session order."""
        sessions = self.story_sessions()
        stories = list(dict.fromkeys(s for key in sorted(sessions, key=int) for s in sessions[key]))
        return [s for s in stories if self.response_path(subject, s).exists()]

    def shared_training_stories(self, subjects=SUBJECTS) -> list[str]:
        """The training stories every one of ``subjects`` heard."""
        shared = None
        for subject in subjects:
            have = self.training_stories(subject)
            shared = have if shared is None else [s for s in shared if s in have]
        return shared or []

    def response_rows(self, story: str) -> int:
        """The number of response rows (TRs) of a story, from any subject that heard it."""
        import h5py
        for subject in SUBJECTS:
            path = self.response_path(subject, story)
            if path.exists():
                with h5py.File(path, "r") as handle:
                    return int(handle["data"].shape[0])
        raise FileNotFoundError(f"no released response for {story}")

    # ---- responses --------------------------------------------------------------------------
    def response_path(self, subject: str, story: str) -> Path:
        return self.derivatives / "preprocessed_data" / subject / f"{story}.hf5"

    def response(self, subject: str, story: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """TRs x columns ``start:stop`` of a story's response (h5py reads only that slab)."""
        import h5py
        with h5py.File(self.response_path(subject, story), "r") as handle:
            data = np.asarray(handle["data"][:, start:stop], dtype=np.float32)
        if data.ndim != 2 or np.isinf(data).any():
            raise ValueError(f"invalid response {subject} {story}: {data.shape}")
        return data

    def repeats(self, subject: str, story: str = TEST_STORY, start: int = 0, stop: int | None = None) -> np.ndarray:
        """The test story's single repeats: repeats x TRs x columns."""
        import h5py
        with h5py.File(self.response_path(subject, story), "r") as handle:
            data = np.asarray(handle["individual_repeats"][:, :, start:stop], dtype=np.float32)
        if data.ndim != 3 or np.isinf(data).any():
            raise ValueError(f"invalid repeats {subject} {story}: {data.shape}")
        return data

    # ---- the functional grid ----------------------------------------------------------------
    def _transform_file(self, subject: str, name: str) -> Path:
        found = sorted((self.derivatives / "pycortex-db" / subject / "transforms").glob(f"**/{name}"))
        if not found:
            raise FileNotFoundError(f"no {name} for {subject}")
        return found[0]

    def mask_path(self, subject: str) -> Path:
        return self._transform_file(subject, "mask_thick.nii.gz")

    def reference_path(self, subject: str) -> Path:
        return self._transform_file(subject, "reference.nii.gz")

    def anatomical(self, subject: str, name: str = "raw.nii.gz") -> Path:
        return self.derivatives / "pycortex-db" / subject / "anatomicals" / name

    @lru_cache(maxsize=16)
    def _mask(self, subject: str):
        import nibabel as nib
        image = nib.load(self.mask_path(subject))
        return image, np.asarray(image.dataobj) > 0

    def n_columns(self, subject: str) -> int:
        return int(self._mask(subject)[1].sum())

    def unmask(self, subject: str, values: np.ndarray) -> np.ndarray:
        """A vector over response columns as an (x, y, z) volume, NaN off the mask.

        Pycortex transposes NIfTI arrays on load and orders masked vectors (z, y, x)."""
        mask_zyx = self._mask(subject)[1].T
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != int(mask_zyx.sum()):
            raise ValueError(f"{values.size} values for {int(mask_zyx.sum())} columns")
        volume = np.full(mask_zyx.shape, np.nan, dtype=np.float32)
        volume[mask_zyx] = values
        return volume.T

    def column_volume(self, subject: str) -> np.ndarray:
        """Each functional voxel's response column, 1-based (0 off the mask)."""
        order = self.unmask(subject, np.arange(1, self.n_columns(subject) + 1, dtype=np.float32))
        return np.rint(np.nan_to_num(order)).astype(np.int64)

    def save_map(self, subject: str, values: np.ndarray, path: Path) -> Path:
        """Write a column vector as a NIfTI on the subject's functional grid."""
        import nibabel as nib
        image = self._mask(subject)[0]
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.unmask(subject, values), image.affine, header), path)
        return path

    # ---- words and timing -------------------------------------------------------------------
    def textgrid_path(self, story: str) -> Path:
        return self.derivatives / "TextGrids" / f"{story}.TextGrid"

    def words(self, story: str) -> list[tuple[float, float, str]]:
        """(onset, offset, word) from the TextGrid word tier, lower case, silences dropped."""
        return parse_word_tier(self.textgrid_path(story))

    def english1000_path(self) -> Path:
        for path in (self.root / "code/deep-fMRI-dataset/em_data/english1000sm.hf5",
                     self.derivatives / "english1000sm.hf5"):
            if path.exists():
                return path
        raise FileNotFoundError("english1000sm.hf5 is required")


def tr_times(n_rows: int) -> np.ndarray:
    """Story time (s) of each response row's centre: 2 i + 11 for row i (the trimmed 2 s clock)."""
    full = np.arange(n_rows + TRIM_START + TRIM_END, dtype=np.float64) * TR - 10.0 + TR / 2.0
    return full[TRIM_START:-TRIM_END]


def parse_word_tier(path: Path) -> list[tuple[float, float, str]]:
    """The word tier of a Praat TextGrid, in any of the three serialisations the release uses."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if text.startswith('"Praat chronological TextGrid text file"'):
        matches = re.findall(r'(?m)^2\s+([^\s]+)\s+([^\s]+)\s*\r?\n"(.*)"\s*$', text)
    elif "item [" not in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        tiers = [i for i, line in enumerate(lines) if line == '"IntervalTier"']
        if len(tiers) < 2:
            raise ValueError(f"expected a phone and a word tier in {path}")
        start = tiers[1]
        count = int(lines[start + 4])
        entries = lines[start + 5:start + 5 + 3 * count]
        matches = [tuple(entries[i:i + 3]) for i in range(0, len(entries), 3)]
    else:
        tiers = [t for t in re.split(r"(?=\s*item \[\d+\]:)", text) if 'class = "IntervalTier"' in t]
        if len(tiers) < 2:
            raise ValueError(f"expected a phone and a word tier in {path}")
        matches = re.findall(r"intervals \[\d+\]:\s+xmin = ([^\r\n]+)\s+xmax = ([^\r\n]+)\s+text = \"(.*?)\"",
                             tiers[1], flags=re.DOTALL)
    words = []
    for start, stop, token in matches:
        word = token.replace('""', '"').strip().strip('"').strip("{}").lower()
        if word not in BAD_WORDS:
            words.append((float(start), float(stop), word))
    if not words:
        raise ValueError(f"no words in {path}")
    return words


def lanczos_weights(old_times: np.ndarray, new_times: np.ndarray, window: int = 3) -> np.ndarray:
    """Three-lobe Lanczos weights from event times to sample times (the LeBel feature resampling)."""
    cutoff = 1.0 / float(np.mean(np.diff(new_times)))
    delta = (np.asarray(new_times)[:, None] - np.asarray(old_times)[None, :]) * cutoff
    weights = np.zeros_like(delta, dtype=np.float64)
    nonzero = (np.abs(delta) <= window) & (delta != 0)
    weights[delta == 0] = 1.0
    values = delta[nonzero]
    weights[nonzero] = window * np.sin(np.pi * values) * np.sin(np.pi * values / window) / (np.pi ** 2 * values ** 2)
    return weights


def hann_weights(old_times: np.ndarray, new_times: np.ndarray, half_width: float = 2.0) -> np.ndarray:
    """A raised cosine of ``half_width`` samples (non-negative, compact support)."""
    step = float(np.mean(np.diff(new_times)))
    delta = (np.asarray(new_times)[:, None] - np.asarray(old_times)[None, :]) / step
    return np.where(np.abs(delta) <= half_width, 0.5 * (1.0 + np.cos(np.pi * delta / half_width)), 0.0)


def common_finite(matrices) -> np.ndarray:
    """Columns finite in every matrix (2-D or stacked 3-D, columns last)."""
    valid = None
    for matrix in matrices:
        ok = np.isfinite(matrix).reshape(-1, matrix.shape[-1]).all(axis=0)
        valid = ok if valid is None else valid & ok
    if valid is None or not valid.any():
        raise ValueError("no column is finite in every matrix")
    return valid
