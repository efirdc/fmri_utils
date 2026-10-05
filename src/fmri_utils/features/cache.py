"""Unit-level embeddings on disk, so the model runs once.

This is the point of the package. Extraction and resampling used to happen in
one call, with nothing kept in between, so asking "what would these features
look like under a different kernel?" meant running the language model over the
whole stimulus again. That made a question worth asking into one nobody asked.

A cache entry is the embeddings *and* the unit times, because the times are
what make the embeddings resamplable. Without them the array is just as stuck
on whatever clock it was written for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .spec import FeatureSpec
from .units import Unit

SCHEMA = "fmri_utils.features.cache/1"


def fingerprint(spec: FeatureSpec, units: list[Unit]) -> str:
    """Identify a cache entry by what would change its contents.

    The unit text and timing are hashed, not just counted: a re-run of a
    transcript that shifts one word should miss the cache rather than silently
    reuse vectors aligned to the old timing.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "model": asdict(spec.model),
        "context": asdict(spec.context),
        "granularity": spec.granularity,
    }, sort_keys=True, default=str).encode("utf-8"))
    for unit in units:
        digest.update(unit.text.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(f"{unit.onset}:{unit.offset}:{unit.run}".encode("utf-8"))
    return digest.hexdigest()[:16]


def entry_path(root: Path, spec: FeatureSpec, stimulus: str) -> Path:
    return Path(root) / spec.id / f"{stimulus}.npz"


def write(root: Path, spec: FeatureSpec, stimulus: str, units: list[Unit],
          embeddings: np.ndarray, valid: np.ndarray, extra: dict | None = None) -> Path:
    """Store one stimulus worth of unit embeddings."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(units):
        raise ValueError(f"{embeddings.shape[0]} embeddings for {len(units)} units")
    target = entry_path(root, spec, stimulus)
    target.parent.mkdir(parents=True, exist_ok=True)
    onsets = np.array([np.nan if u.onset is None else u.onset for u in units], dtype=np.float64)
    offsets = np.array([np.nan if u.offset is None else u.offset for u in units], dtype=np.float64)
    np.savez_compressed(
        target,
        embeddings=embeddings,
        valid=np.asarray(valid, dtype=bool),
        onsets=onsets,
        offsets=offsets,
        texts=np.array([u.text for u in units], dtype=object),
        runs=np.array([u.run for u in units], dtype=object),
        meta=json.dumps({
            "schema": SCHEMA,
            "feature_id": spec.id,
            "model": asdict(spec.model),
            "context": asdict(spec.context),
            "granularity": spec.granularity,
            "stimulus": stimulus,
            "n_units": len(units),
            "dimensions": int(embeddings.shape[1]),
            "n_valid": int(np.count_nonzero(valid)),
            "fingerprint": fingerprint(spec, units),
            **(extra or {}),
        }),
    )
    return target


def read(root: Path, spec: FeatureSpec, stimulus: str) -> dict:
    """Load one entry: embeddings, validity, unit times, and the metadata."""
    path = entry_path(root, spec, stimulus)
    with np.load(path, allow_pickle=True) as handle:
        meta = json.loads(str(handle["meta"]))
        return {
            "embeddings": handle["embeddings"],
            "valid": handle["valid"],
            "onsets": handle["onsets"],
            "offsets": handle["offsets"],
            "texts": list(handle["texts"]),
            "runs": list(handle["runs"]),
            "midpoints": (handle["onsets"] + handle["offsets"]) / 2.0,
            "meta": meta,
        }


def is_current(root: Path, spec: FeatureSpec, stimulus: str, units: list[Unit]) -> bool:
    """Whether a usable entry already exists for exactly these units."""
    path = entry_path(root, spec, stimulus)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as handle:
            meta = json.loads(str(handle["meta"]))
    except Exception:
        return False
    return meta.get("fingerprint") == fingerprint(spec, units)
