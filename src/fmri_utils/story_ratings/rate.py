from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from .config import RatingRunConfig
from .consensus import consensus_table, rater_agreement, rating_distribution
from .providers import Usage, UsageLimitError, call_model
from .schema import build_prompt, build_schema, expand_compact, validate_ratings
from .segmentation import Segment, segment_words, segments_to_records
from .transcripts import Word

SCHEMA_VERSION = 1


@dataclass
class StoryRatingResult:
    """Everything one rated story produces."""

    story: str
    segments: List[Segment]
    consensus: pd.DataFrame
    replicates: List[List[Dict[str, Any]]]
    usage: Usage
    summary: Dict[str, Any]

    def write(self, story_dir: Path) -> Path:
        """Write the consensus table and run summary next to the raw replies."""
        story_dir = Path(story_dir)
        story_dir.mkdir(parents=True, exist_ok=True)
        self.consensus.to_csv(story_dir / "segment_ratings.csv", index=False)
        (story_dir / "run_summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        return story_dir


def _prompt_fingerprint(config: RatingRunConfig, texts: Sequence[str]) -> str:
    """Identify the exact request, so cached replies are never mixed."""
    provider = config.provider
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt": build_prompt(config.prompt_template, texts, config.utterance_placeholder),
        "spec": [config.spec.scale.name, *config.spec.flags, *sorted(config.spec.categoricals)],
        "backend": provider.backend,
        "model": provider.model,
        "thinking": provider.thinking,
        "reasoning_effort": provider.reasoning_effort if provider.backend == "openai" else None,
        "compact_schema": provider.compact_schema,
        "chunk_size": provider.chunk_size,
        "system_prompt": provider.system_prompt,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def rate_segments(
    texts: Sequence[str],
    config: RatingRunConfig,
    *,
    on_progress: Optional[Callable[[str], None]] = None,
) -> tuple[List[Dict[str, Any]], Usage]:
    """Rate a list of texts once, in chunks, retrying incomplete replies."""
    config.validate()
    provider = config.provider
    schema = build_schema(config.spec, compact=provider.compact_schema)
    size = provider.chunk_size or len(texts)
    ratings: List[Dict[str, Any]] = []
    total = Usage(calls=0)
    for start in range(0, len(texts), size):
        chunk = list(texts[start : start + size])
        prompt = build_prompt(config.prompt_template, chunk, config.utterance_placeholder)
        last_error: Optional[Exception] = None
        for attempt in range(1, provider.max_attempts + 1):
            payload, usage = call_model(prompt, schema, provider)
            total = total + usage
            if provider.compact_schema:
                payload = expand_compact(payload, config.spec)
            try:
                ordered = validate_ratings(payload, len(chunk), config.spec)
            except ValueError as error:
                last_error = error
                if on_progress:
                    on_progress(f"  incomplete reply (attempt {attempt}/{provider.max_attempts}): {error}")
                continue
            ratings.extend({**item, "index": int(item["index"]) + start} for item in ordered)
            break
        else:
            raise last_error or RuntimeError("chunk failed")
    return ratings, total


def rate_story(
    story: str,
    words: Sequence[Word],
    config: RatingRunConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> StoryRatingResult:
    """Rate one story with independent replicate raters.

    Each replicate is cached as a raw reply under ``<output_dir>/<story>/raw``
    and reused when the rubric, fields, model and segmentation are unchanged,
    so an interrupted run resumes without paying for finished work.
    """
    config.validate()
    log = on_progress or (lambda message: None)
    segments = segment_words(words, config.segmentation)
    texts = [segment.text for segment in segments]
    fingerprint = _prompt_fingerprint(config, texts)
    story_dir = Path(output_dir) / story
    raw_dir = story_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log(f"{story}: {len(words)} words -> {len(segments)} segments")

    def one(replicate: int) -> tuple[List[Dict[str, Any]], Usage]:
        path = raw_dir / f"replicate-{replicate}.json"
        if path.exists() and not overwrite:
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"{path} was produced with different settings; pass overwrite=True or use a new output_dir"
                )
            log(f"{story} r{replicate}: cached")
            usage = stored.get("usage", {})
            return stored["ratings"], Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                cached_input_tokens=usage.get("cached_input_tokens", 0),
                duration_ms=usage.get("duration_ms", 0),
                calls=usage.get("calls", 1),
            )
        ratings, usage = rate_segments(texts, config, on_progress=log)
        record = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "story": story,
            "replicate": replicate,
            "fingerprint": fingerprint,
            "backend": config.provider.backend,
            "model": config.provider.model,
            "thinking": config.provider.thinking,
            "reasoning_effort": config.provider.reasoning_effort if config.provider.backend == "openai" else None,
            "segmentation": vars(config.segmentation),
            "n_segments": len(segments),
            "usage": usage.to_dict(config.provider.model),
            "ratings": ratings,
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        log(
            f"{story} r{replicate}: ok in {usage.duration_ms / 1000:.0f}s, "
            f"{usage.input_tokens} in / {usage.output_tokens} out tokens"
        )
        return ratings, usage

    with ThreadPoolExecutor(max_workers=config.replicates) as pool:
        results = list(pool.map(one, range(1, config.replicates + 1)))
    replicates = [ratings for ratings, _ in results]
    usage = results[0][1]
    for _, extra in results[1:]:
        usage = usage + extra

    consensus = consensus_table(segments, replicates, config.spec)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "story": story,
        "backend": config.provider.backend,
        "model": config.provider.model,
        "thinking": config.provider.thinking,
        "reasoning_effort": config.provider.reasoning_effort if config.provider.backend == "openai" else None,
        "replicates": config.replicates,
        "fingerprint": fingerprint,
        "segmentation": vars(config.segmentation),
        "scale": vars(config.spec.scale),
        "flags": list(config.spec.flags),
        "categoricals": {name: list(choices) for name, choices in config.spec.categoricals.items()},
        "n_words": len(words),
        "n_segments": len(segments),
        "mean_words_per_segment": float(consensus["n_words"].mean()),
        "agreement": rater_agreement(replicates, config.spec),
        "distribution": rating_distribution(replicates, config.spec),
        f"mean_{config.spec.scale.name}": float(consensus[f"{config.spec.scale.name}_mean"].mean()),
        "usage_total": usage.to_dict(config.provider.model),
        "segments": segments_to_records(segments),
    }
    result = StoryRatingResult(story, segments, consensus, replicates, usage, summary)
    result.write(story_dir)
    return result


def rate_stories(
    stories: Dict[str, Sequence[Word]],
    config: RatingRunConfig,
    output_dir: Path,
    *,
    overwrite: bool = False,
    on_progress: Optional[Callable[[str], None]] = print,
) -> Dict[str, StoryRatingResult]:
    """Rate several stories in sequence.

    Stops early on a usage limit rather than burning through every remaining
    story: finished replicates are cached, so rerunning resumes.
    """
    results: Dict[str, StoryRatingResult] = {}
    log = on_progress or (lambda message: None)
    for story, words in stories.items():
        try:
            results[story] = rate_story(story, words, config, output_dir, overwrite=overwrite, on_progress=log)
        except UsageLimitError as error:
            log(f"{story}: usage limit reached, stopping. Cached work is kept; rerun to resume.\n{error}")
            break
        except Exception as error:  # noqa: BLE001 - report and continue to the next story
            log(f"{story}: FAILED {type(error).__name__}: {error}")
    return results
