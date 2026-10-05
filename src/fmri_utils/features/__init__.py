"""Text features for encoding models: embed once, resample as often as you like.

The two halves of this package are deliberately separable. Running a language
model over a stimulus is expensive; deciding how to put the result on a scanner
clock is not. Keeping them in one call, which is what the scripts this replaces
did, makes the cheap decision as expensive as the dear one, and a question
nobody asks.

    from fmri_utils.features import (
        ContextSpec, FeatureSpec, ModelSpec, Unit, cache, embed_units, resample,
    )

    units = [Unit(text=w, onset=a, offset=b) for w, a, b in words]
    spec = FeatureSpec(
        model=ModelSpec(id="bert", huggingface_id="bert-base-uncased",
                        pooling="unit_word_last_mean"),
        context=ContextSpec(previous=10),
        granularity="word",
    )

    embeddings, valid = embed_units(units, spec)
    cache.write("features/", spec, "mystory", units, embeddings, valid)

and then, as many times as you like, with no model in sight:

    entry = cache.read("features/", spec, "mystory")
    tr = resample.resample(entry["embeddings"], entry["midpoints"], tr_times,
                           kernel="hann")

``resample`` also carries ``lanczos_sum``, the Huth lab's published resampler,
for reproducing prior work. It is unnormalised: its weights sum to the local
event rate, so it returns a rate-weighted sum rather than an average and the
features carry stimulus density. Prefer a normalised kernel for new work, and
``resample.event_rate`` if you want that density as its own regressor.

Video models have their own module, ``fmri_utils.features.video``: clip
embeddings cached per video, then placed on a run's clock for every
presentation.
"""

from __future__ import annotations

from . import cache, pooling, resample, video
from .extract import embed_units, select_device
from .spec import ContextSpec, FeatureSpec, ModelSpec, specs_from_registry
from .units import Payload, Unit, build_payloads, build_stack_payloads, units_from_texts, words_from_table

__all__ = [
    "ContextSpec",
    "FeatureSpec",
    "ModelSpec",
    "Payload",
    "Unit",
    "build_payloads",
    "build_stack_payloads",
    "cache",
    "embed_units",
    "pooling",
    "resample",
    "select_device",
    "video",
    "specs_from_registry",
    "units_from_texts",
    "words_from_table",
]
