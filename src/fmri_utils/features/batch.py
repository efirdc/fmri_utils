"""Build a feature space for every stimulus of a stimulus table: ``<output>/<stimulus>.npy``.

``language_model``  a ``FeatureSpec`` (any Hugging Face model, layer, pooling, context), each word
                    embedded once (cached when ``cache_root`` is given)
``static``          a word-vector table (``static.load_table``)
``rate``            words per second through a kernel (Hann, half-width 2 samples, by default)

Words sit at their midpoints. ``kernel="lanczos-sum"`` is the unnormalised three-lobe Lanczos
resampler of the Huth/LeBel features (each sample a rate-weighted sum); any ``resample`` kernel
(``hann``, ``lanczos``, ...) gives a normalised average instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import resample
from .stimuli import Stimulus


def _onto_clock(values: np.ndarray, times: np.ndarray, stimulus: Stimulus, kernel: str, **kernel_args) -> np.ndarray:
    if kernel == "lanczos-sum":
        return resample.lanczos_sum(values, times, stimulus.sample_times, **kernel_args)
    return resample.resample(values, times, stimulus.sample_times, kernel=kernel, **kernel_args)


def _save(output: Path, stimulus: Stimulus, values: np.ndarray, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{stimulus.name}.npy", np.asarray(values, dtype=np.float32))
    report[stimulus.name] = list(values.shape)
    print(f"{output.name} {stimulus.name}: {values.shape}", flush=True)


def language_model(stimuli: Sequence[Stimulus], spec, output: Path, cache_root: Path | None = None,
                   kernel: str = "lanczos-sum", device: str = "", overwrite: bool = False) -> Path:
    from . import cache
    from .extract import embed_units
    from .units import Unit
    output, report = Path(output), {}
    for stimulus in stimuli:
        if (output / f"{stimulus.name}.npy").exists() and not overwrite:
            continue
        words, midpoints = stimulus.word_midpoints()
        units = [Unit(text=w, onset=float(t), offset=float(t)) for w, t in zip(words, midpoints)]
        if cache_root is not None and cache.is_current(cache_root, spec, stimulus.name, units):
            vectors = cache.read(cache_root, spec, stimulus.name)["embeddings"]
        else:
            vectors, valid = embed_units(units, spec, device=device)
            if cache_root is not None:
                cache.write(cache_root, spec, stimulus.name, units, vectors, valid)
        _save(output, stimulus, _onto_clock(np.asarray(vectors, dtype=np.float64), midpoints, stimulus, kernel), report)
    (output / "build.json").write_text(json.dumps({"source": spec.id, "kernel": kernel, "stimuli": report}, indent=1))
    return output


def static(stimuli: Sequence[Stimulus], table_path: Path, output: Path, kernel: str = "lanczos-sum",
           lowercase: bool = True) -> Path:
    from .static import embed, load_table
    table = load_table(table_path)
    output, report = Path(output), {}
    for stimulus in stimuli:
        words, midpoints = stimulus.word_midpoints()
        vectors, _ = embed(words, table, lowercase=lowercase)
        _save(output, stimulus, _onto_clock(vectors, midpoints, stimulus, kernel), report)
    (output / "build.json").write_text(json.dumps({"source": str(table_path), "kernel": kernel, "stimuli": report}, indent=1))
    return output


def rate(stimuli: Sequence[Stimulus], output: Path, kernel: str = "hann", width: float = 2.0) -> Path:
    """Words per second (one column), through ``kernel`` of ``width`` samples."""
    output, report = Path(output), {}
    args = {"hann": {"half_width": width}, "boxcar": {"half_width": width}, "gaussian": {"sigma": width},
            "lanczos": {"window": int(width)}}[kernel]
    for stimulus in stimuli:
        _, midpoints = stimulus.word_midpoints()
        _save(output, stimulus, resample.event_rate(midpoints, stimulus.sample_times, kernel=kernel, **args)[:, None], report)
    (output / "build.json").write_text(json.dumps({"source": "word rate", "kernel": kernel, "width": width, "stimuli": report}, indent=1))
    return output
