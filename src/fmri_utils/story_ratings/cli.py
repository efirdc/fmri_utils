from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import ProviderConfig, RatingRunConfig, RatingSpec, ScaleField, SegmentationConfig
from .rate import rate_stories
from .timeseries import ratings_to_timeseries, scanner_sample_times
from .transcripts import Word, read_textgrid_words, read_word_table, words_from_text
from .viewer import render_viewer


def load_spec(path: Optional[Path]) -> RatingSpec:
    """Read a rating spec from JSON, or return the default 0-3 scale.

    Example file::

        {
          "scale": {"name": "tom", "minimum": 0, "maximum": 3,
                    "level_labels": ["none", "action", "implied", "named"]},
          "flags": ["emotion", "intention"],
          "categoricals": {"target": ["none", "self", "other", "both"]}
        }
    """
    if path is None:
        return RatingSpec()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scale = payload.get("scale", {})
    return RatingSpec(
        scale=ScaleField(
            name=scale.get("name", "rating"),
            minimum=int(scale.get("minimum", 0)),
            maximum=int(scale.get("maximum", 3)),
            description=scale.get("description", ""),
            level_labels=tuple(scale.get("level_labels", ())),
        ),
        flags=tuple(payload.get("flags", ())),
        categoricals={name: tuple(choices) for name, choices in payload.get("categoricals", {}).items()},
        include_confidence=bool(payload.get("include_confidence", True)),
        include_reason=bool(payload.get("include_reason", True)),
    )


def load_story_words(path: Path) -> List[Word]:
    """Read one story from a TextGrid, a word table, or plain text."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".textgrid":
        return read_textgrid_words(path)
    if suffix in {".json", ".csv", ".tsv", ".tab"}:
        return read_word_table(path)
    return words_from_text(path.read_text(encoding="utf-8"))


def discover_stories(inputs: Sequence[Path], names: Optional[Sequence[str]] = None) -> Dict[str, List[Word]]:
    """Collect stories from files and directories, keyed by file stem."""
    paths: List[Path] = []
    for item in inputs:
        item = Path(item)
        paths.extend(sorted(p for p in item.iterdir() if p.is_file()) if item.is_dir() else [item])
    stories: Dict[str, List[Word]] = {}
    for path in paths:
        name = path.stem.split("_")[0] if path.suffix.lower() == ".json" and "_" in path.stem else path.stem
        if names and name not in names:
            continue
        stories[name] = load_story_words(path)
    if not stories:
        raise FileNotFoundError(f"no stories found in {[str(item) for item in inputs]}")
    return stories


def _add_rate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="story files or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True, help="rubric containing {utterances}")
    parser.add_argument("--spec", type=Path, help="rating spec JSON (default: a 0-3 scale)")
    parser.add_argument("--stories", nargs="*", help="limit to these story names")
    parser.add_argument(
        "--backend", choices=("anthropic", "openai", "claude-cli", "codex-cli"), default="anthropic",
        help="anthropic/openai use API keys; claude-cli/codex-cli use a local CLI and your subscription",
    )
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--no-thinking", action="store_true", help="anthropic: disable thinking")
    parser.add_argument("--reasoning-effort", default="low", choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--segmentation", choices=("pause", "sentence", "clause", "whole"), default="clause")
    parser.add_argument("--pause-seconds", type=float, default=0.3)
    parser.add_argument("--clause-pause-seconds", type=float, default=0.12)
    parser.add_argument("--hard-pause-seconds", type=float, default=0.45)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=32000)
    parser.add_argument("--compact-schema", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tr-seconds", type=float, help="also write timeseries.csv on this TR grid")
    parser.add_argument("--tr-count", type=int, help="number of TRs (default: covers the story)")


def run_rate(args: argparse.Namespace) -> None:
    config = RatingRunConfig(
        prompt_template=Path(args.prompt).read_text(encoding="utf-8"),
        spec=load_spec(args.spec),
        segmentation=SegmentationConfig(
            mode=args.segmentation,
            pause_seconds=args.pause_seconds,
            clause_pause_seconds=args.clause_pause_seconds,
            hard_pause_seconds=args.hard_pause_seconds,
            min_words=args.min_words,
            max_words=args.max_words,
        ),
        provider=ProviderConfig(
            backend=args.backend,
            model=args.model,
            thinking=not args.no_thinking,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            chunk_size=args.chunk_size,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout,
            compact_schema=args.compact_schema,
        ),
        replicates=args.replicates,
    )
    stories = discover_stories(args.inputs, args.stories)
    results = rate_stories(stories, config, args.output_dir, overwrite=args.overwrite)
    for story, result in results.items():
        if args.tr_seconds:
            words = stories[story]
            end = max(float(word.offset) for word in words if word.is_timed)
            count = args.tr_count or int(end // args.tr_seconds) + 1
            series = ratings_to_timeseries(
                words, result.segments, result.consensus,
                scanner_sample_times(count, args.tr_seconds), config.spec,
            )
            series.to_csv(Path(args.output_dir) / story / "timeseries.csv", index=False)
    print(f"\nrated {len(results)} of {len(stories)} stories into {args.output_dir}")


def run_render(args: argparse.Namespace) -> None:
    path = render_viewer(args.runs, args.output, stories=args.stories, title=args.title, eyebrow=args.eyebrow)
    size_kb = path.stat().st_size / 1024
    print(f"{path} ({size_kb:.0f} KB)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="fmri-story-ratings", description="Rate story segments with an LLM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rate_parser = subparsers.add_parser("rate", help="rate stories and write ratings, summaries, and raw replies")
    _add_rate_arguments(rate_parser)
    rate_parser.set_defaults(func=run_rate)

    render_parser = subparsers.add_parser("render", help="build the HTML reader from one or more run directories")
    render_parser.add_argument("--runs", type=Path, nargs="+", required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--stories", nargs="*")
    render_parser.add_argument("--title", default="Story Ratings")
    render_parser.add_argument("--eyebrow", default="LLM segment ratings")
    render_parser.set_defaults(func=run_render)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
