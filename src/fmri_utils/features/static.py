"""Static word embeddings: one vector per word type from a lookup table (English1000, GloVe, ...).

``load_table`` reads a table from

- HDF5 with datasets ``data`` and ``vocab`` (``data`` as dims x vocab, or vocab x dims);
- ``.npz`` with ``vectors`` (vocab x dims) and ``vocab``;
- text, one word per line followed by its values (the GloVe format).

``embed`` gives each word its vector (zeros for words outside the vocabulary, looked up in lower
case by default), to be resampled like any unit embedding (``resample.lanczos_sum`` and friends).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_table(path: Path | str) -> tuple[np.ndarray, list[str]]:
    """(vectors: vocab x dims, vocab)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".hf5", ".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as handle:
            data = np.asarray(handle["data"], dtype=np.float32)
            vocab = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(handle["vocab"])]
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        data, vocab = np.asarray(archive["vectors"], dtype=np.float32), [str(v) for v in archive["vocab"]]
    else:
        vocab, rows = [], []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip().split(" ")
                if len(parts) > 2:
                    vocab.append(parts[0])
                    rows.append(np.asarray(parts[1:], dtype=np.float32))
        data = np.stack(rows)
    if data.shape[0] != len(vocab) and data.shape[1] == len(vocab):
        data = data.T
    if data.shape[0] != len(vocab):
        raise ValueError(f"{path}: {data.shape} vectors for {len(vocab)} words")
    return data, vocab


def embed(words: Sequence[str], table: tuple[np.ndarray, list[str]], lowercase: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(vectors: words x dims, in_vocabulary: bool per word)."""
    vectors, vocab = table
    lookup = {(w.lower() if lowercase else w): i for i, w in enumerate(vocab)}
    out = np.zeros((len(words), vectors.shape[1]), dtype=np.float32)
    found = np.zeros(len(words), dtype=bool)
    for row, word in enumerate(words):
        index = lookup.get(word.lower() if lowercase else word)
        if index is not None:
            out[row] = vectors[index]
            found[row] = True
    return out, found
