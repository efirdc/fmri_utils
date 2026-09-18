from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ScaleField:
    """An ordinal rating, for example a 0-3 mental-state scale."""

    name: str = "rating"
    minimum: int = 0
    maximum: int = 3
    description: str = ""
    level_labels: Tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"scale field name must be an identifier: {self.name!r}")
        if self.maximum <= self.minimum:
            raise ValueError("scale maximum must exceed minimum")
        if self.level_labels and len(self.level_labels) != self.maximum - self.minimum + 1:
            raise ValueError("level_labels must cover every level from minimum to maximum")

    @property
    def levels(self) -> Tuple[int, ...]:
        return tuple(range(self.minimum, self.maximum + 1))


@dataclass(frozen=True)
class RatingSpec:
    """What the annotator returns for each segment.

    A spec is a scale plus any number of binary flags and single-choice
    categorical fields. The theory-of-mind example in the docs uses a 0-3
    scale, four flags, and one categorical field, but nothing here is specific
    to that construct: an arousal rating, a topic label, or a dialogue-act
    code fits the same shape.
    """

    scale: ScaleField = field(default_factory=ScaleField)
    flags: Tuple[str, ...] = ()
    categoricals: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    include_confidence: bool = True
    include_reason: bool = True

    def validate(self) -> None:
        self.scale.validate()
        names = [self.scale.name, *self.flags, *self.categoricals]
        if self.include_confidence:
            names.append("confidence")
        if self.include_reason:
            names.append("reason")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in rating spec: {names}")
        for name in [*self.flags, *self.categoricals]:
            if not name.isidentifier():
                raise ValueError(f"field name must be an identifier: {name!r}")
        for name, choices in self.categoricals.items():
            if len(choices) < 2:
                raise ValueError(f"categorical {name!r} needs at least two choices")

    @property
    def field_names(self) -> Tuple[str, ...]:
        names = ["index", self.scale.name, *self.flags, *self.categoricals]
        if self.include_confidence:
            names.append("confidence")
        if self.include_reason:
            names.append("reason")
        return tuple(names)


@dataclass(frozen=True)
class SegmentationConfig:
    """How a story's words become the units that get rated.

    ``mode`` is one of:

    - ``"pause"``: split wherever the speaker paused for ``pause_seconds``.
      Works on bare forced-alignment output with no punctuation.
    - ``"sentence"``: split at sentence-final punctuation. Needs display
      tokens that carry punctuation.
    - ``"clause"``: split at sentence ends, at clause marks the speaker
      paused on, and at long pauses. Keeps clauses intact while following
      speech timing, and is the recommended default for spoken stories.
    - ``"whole"``: one segment per story, for short texts.

    ``max_words`` bounds the longest segment in every mode; ``min_words``
    keeps clause mode from splitting off scraps.
    """

    mode: str = "clause"
    pause_seconds: float = 0.3
    clause_pause_seconds: float = 0.12
    hard_pause_seconds: float = 0.45
    min_words: int = 4
    max_words: int = 25

    def validate(self) -> None:
        if self.mode not in {"pause", "sentence", "clause", "whole"}:
            raise ValueError("mode must be pause, sentence, clause, or whole")
        if self.max_words < 1:
            raise ValueError("max_words must be >= 1")
        if self.min_words < 1 or self.min_words > self.max_words:
            raise ValueError("min_words must be between 1 and max_words")
        for name in ("pause_seconds", "clause_pause_seconds", "hard_pause_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class ProviderConfig:
    """Which model rates the segments, and how the request is shaped.

    ``backend`` selects where the request goes:

    - ``"anthropic"`` / ``"openai"``: metered API keys, no usage windows.
    - ``"claude-cli"`` / ``"codex-cli"``: the local ``claude`` or ``codex``
      command, billed to a Claude or ChatGPT subscription instead of an API
      key. Useful when you have subscription headroom, but subject to that
      plan's usage limits, which stop a run until the window resets.

    ``thinking`` applies to Anthropic models: the API accepts
    ``thinking={"type": "disabled"}``, while the CLI can only zero the
    thinking budget, which silences budget-based models such as Claude Haiku
    4.5 but not adaptive ones such as Claude Sonnet 5. ``reasoning_effort``
    applies to OpenAI reasoning models. ``chunk_size`` splits one story into
    several calls: some models silently drop items from long lists, and 50-80
    segments per call avoids that. ``max_attempts`` retries a chunk whose
    ratings do not cover every index.
    """

    backend: str = "anthropic"
    model: str = "claude-sonnet-5"
    thinking: bool = True
    reasoning_effort: str = "low"
    max_output_tokens: int = 32000
    chunk_size: int = 60
    max_attempts: int = 3
    timeout_seconds: int = 900
    compact_schema: bool = False
    system_prompt: str = (
        "You are a careful, independent research annotator. Follow the rubric exactly."
    )

    def validate(self) -> None:
        if self.backend not in {"anthropic", "openai", "claude-cli", "codex-cli"}:
            raise ValueError("backend must be anthropic, openai, claude-cli, or codex-cli")
        if not self.model:
            raise ValueError("model is required")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be >= 0 (0 rates a whole story per call)")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_output_tokens < 1000:
            raise ValueError("max_output_tokens must be >= 1000")

    @property
    def api_key_env(self) -> Optional[str]:
        """Which key this backend needs, or ``None`` for the subscription CLIs."""
        return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(self.backend)


@dataclass(frozen=True)
class RatingRunConfig:
    """A complete rating run: rubric, fields, segmentation, provider, replicates."""

    prompt_template: str
    spec: RatingSpec = field(default_factory=RatingSpec)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    replicates: int = 3
    utterance_placeholder: str = "{utterances}"

    def validate(self) -> None:
        self.spec.validate()
        self.segmentation.validate()
        self.provider.validate()
        if self.replicates < 1:
            raise ValueError("replicates must be >= 1")
        if self.utterance_placeholder not in self.prompt_template:
            raise ValueError(
                f"prompt_template must contain {self.utterance_placeholder!r} where the "
                "numbered segments are inserted"
            )


def tom_rating_spec() -> RatingSpec:
    """The theory-of-mind spec used in the documented example."""
    return RatingSpec(
        scale=ScaleField(
            name="tom",
            minimum=0,
            maximum=3,
            description="strength of mental-state attribution",
            level_labels=(
                "no mental state",
                "purposeful action",
                "implied state",
                "named state",
            ),
        ),
        flags=("emotion", "intention", "belief", "social_relationship"),
        categoricals={"target": ("none", "self", "other", "both")},
    )
