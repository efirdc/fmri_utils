"""Rate story segments with an LLM, and turn the ratings into time series.

The package takes written or spoken stories, splits them into segments, sends
each segment list to an LLM with a rubric you supply, and returns per-segment
ratings with replicate agreement. For spoken stories with word timings it also
projects ratings onto an acquisition grid, and it can render a self-contained
HTML reader for checking ratings by hand.

Nothing here is specific to one construct: the rubric is a text file and the
returned fields are described by a :class:`RatingSpec`, so the same code rates
theory of mind, emotional intensity, topic labels, or anything else that fits
a scale plus flags and categories.
"""

from .config import (
    ProviderConfig,
    RatingRunConfig,
    RatingSpec,
    ScaleField,
    SegmentationConfig,
    tom_rating_spec,
)
from .consensus import consensus_table, rater_agreement, rating_distribution, weighted_kappa
from .providers import MODEL_PRICES_USD_PER_MTOK, Usage, UsageLimitError, call_model, read_api_key
from .rate import StoryRatingResult, rate_segments, rate_stories, rate_story
from .schema import build_prompt, build_schema, expand_compact, validate_ratings
from .segmentation import Segment, segment_words, segments_to_records
from .timeseries import lanczos_weights, ratings_to_timeseries, scanner_sample_times
from .transcripts import (
    Word,
    has_punctuation,
    read_textgrid_words,
    read_word_table,
    words_from_text,
    write_word_table,
)
from .viewer import collect_runs, render_viewer

__all__ = [
    "MODEL_PRICES_USD_PER_MTOK",
    "ProviderConfig",
    "RatingRunConfig",
    "RatingSpec",
    "ScaleField",
    "Segment",
    "SegmentationConfig",
    "StoryRatingResult",
    "Usage",
    "UsageLimitError",
    "Word",
    "build_prompt",
    "build_schema",
    "call_model",
    "collect_runs",
    "consensus_table",
    "expand_compact",
    "has_punctuation",
    "lanczos_weights",
    "rate_segments",
    "rate_stories",
    "rate_story",
    "rater_agreement",
    "rating_distribution",
    "ratings_to_timeseries",
    "read_api_key",
    "read_textgrid_words",
    "read_word_table",
    "render_viewer",
    "scanner_sample_times",
    "segment_words",
    "segments_to_records",
    "tom_rating_spec",
    "validate_ratings",
    "weighted_kappa",
    "words_from_text",
    "write_word_table",
]
