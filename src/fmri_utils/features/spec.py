"""What to embed, and with what.

A feature space is a model plus a decision about what counts as one unit of
stimulus and how much context that unit is shown. Keeping those three apart is
the point of this module: the same model produces quite different features at
word and at TR granularity, and the difference has bitten this project before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence

Family = Literal["transformer", "sentence_embedding", "clip_text", "llm2vec", "lookup"]
Pooling = Literal[
    "unit_token_mean",      # mean over the unit's own tokens
    "unit_last_token",      # the unit's final token
    "unit_word_last_mean",  # last token of each word in the unit, then mean
    "context_slot_stack",   # one pooled vector per context slot, concatenated
]


@dataclass(frozen=True)
class ModelSpec:
    """A model and how to read a vector out of it.

    ``layer`` is a 1-based transformer block, or ``None`` for the final hidden
    state. ``base_id`` is only for adapters such as LLM2Vec's MNTP weights,
    which need the decoder they were trained on.
    """

    id: str
    family: Family = "transformer"
    huggingface_id: str = ""
    base_id: str = ""
    layer: Optional[int] = None
    pooling: Pooling = "unit_token_mean"
    max_length: int = 512
    batch_size: int = 32
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family != "lookup" and not self.huggingface_id:
            raise ValueError(f"{self.id}: {self.family} needs a huggingface_id")
        if self.layer is not None and self.layer < 1:
            raise ValueError(f"{self.id}: layer is 1-based, got {self.layer}")


@dataclass(frozen=True)
class ContextSpec:
    """How much of the surrounding stimulus a unit is shown.

    ``previous`` units of context are prepended to the unit being embedded.
    ``"max"`` means everything since the start of the run, which is what a
    causal model can legitimately see. Context changes the vector for a unit
    without changing which unit it is, so it belongs here rather than in the
    model spec.
    """

    previous: int | Literal["max"] = 0
    # For ``context_slot_stack``: how many slots the stacked output has. The
    # unit's own slot is always the last one.
    slots: int = 0

    def __post_init__(self) -> None:
        if self.previous != "max" and int(self.previous) < 0:
            raise ValueError("context.previous must be >= 0 or 'max'")

    @property
    def label(self) -> str:
        return f"ctx{self.previous}"


@dataclass(frozen=True)
class FeatureSpec:
    """A model, a context, and the granularity the units were cut at.

    ``granularity`` is not used by the extractor -- the units carry their own
    meaning -- but it is recorded with the cache so that two feature sets from
    the same model cannot be confused later. `bert_wordctx10` and `bert_ctx10`
    in this project differ only in this field and are not comparable.
    """

    model: ModelSpec
    context: ContextSpec = field(default_factory=ContextSpec)
    granularity: Literal["word", "tr", "other"] = "word"

    @property
    def id(self) -> str:
        pieces = [self.model.id, f"{self.granularity}{self.context.label}"]
        if self.model.layer is not None:
            pieces.insert(1, f"l{self.model.layer}")
        return "_".join(pieces)


def specs_from_registry(entries: Sequence[Mapping[str, object]]) -> dict[str, ModelSpec]:
    """Read a list of plain dicts, as a JSON or CSV model registry gives them."""
    out: dict[str, ModelSpec] = {}
    for entry in entries:
        layer = entry.get("layer", entry.get("transformer_layer"))
        if layer in (None, "", "final"):
            layer_value = None
        else:
            layer_value = int(layer)
        spec = ModelSpec(
            id=str(entry["id"] if "id" in entry else entry["model_id"]),
            family=str(entry.get("family", entry.get("model_family", "transformer"))),
            huggingface_id=str(entry.get("huggingface_id", "")),
            base_id=str(entry.get("base_id", entry.get("base_huggingface_id", ""))),
            layer=layer_value,
            pooling=str(entry.get("pooling", "unit_token_mean")),
            max_length=int(entry.get("max_length", 512)),
            batch_size=int(entry.get("batch_size", 32)),
        )
        out[spec.id] = spec
    return out
