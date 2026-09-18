from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmri_utils.story_ratings import (
    ProviderConfig,
    RatingRunConfig,
    RatingSpec,
    ScaleField,
    SegmentationConfig,
    Word,
    build_prompt,
    build_schema,
    consensus_table,
    expand_compact,
    lanczos_weights,
    rate_story,
    rater_agreement,
    ratings_to_timeseries,
    read_word_table,
    render_viewer,
    scanner_sample_times,
    has_punctuation,
    segment_words,
    tom_rating_spec,
    validate_ratings,
    weighted_kappa,
    words_from_text,
    write_word_table,
)
from fmri_utils.story_ratings import providers, rate as rate_module
from fmri_utils.story_ratings.schema import compact_key_map

# A spoken story: the aligner's bare tokens, plus the punctuated display
# tokens a caller supplies (from their own transcript, or by hand).
SPOKEN = [
    ("i", "I", 0.0, 0.2), ("reached", "reached", 0.2, 0.5), ("over", "over", 0.5, 0.8),
    ("and", "and", 1.6, 1.8), ("secretly", "secretly", 1.8, 2.3), ("undid", "undid", 2.3, 2.7),
    ("my", "my", 2.7, 2.9), ("seatbelt", "seatbelt.", 2.9, 3.4),
    ("i", "I", 4.2, 4.4), ("was", "was", 4.4, 4.6), ("crying", "crying.", 4.6, 5.1),
]


def spoken_words():
    """Timed words with no punctuation, as forced alignment produces."""
    return [Word(text=text, onset=onset, offset=offset) for text, _, onset, offset in SPOKEN]


def punctuated_words():
    """The same words, with display tokens that carry punctuation."""
    return [Word(text=text, display=display, onset=onset, offset=offset) for text, display, onset, offset in SPOKEN]


def test_word_defaults_display_to_text_and_reports_timing():
    bare = Word(text="hello")
    assert bare.display == "hello" and not bare.is_timed
    assert spoken_words()[0].is_timed


def test_has_punctuation_distinguishes_display_tokens():
    assert not has_punctuation(spoken_words())
    assert has_punctuation(punctuated_words())


def test_segment_pause_mode_uses_timing():
    segments = segment_words(spoken_words(), SegmentationConfig(mode="pause", pause_seconds=0.5))
    assert [segment.n_words for segment in segments] == [3, 5, 3]
    assert segments[0].text == "i reached over"
    assert segments[1].onset == pytest.approx(1.6)


def test_segment_sentence_mode_follows_punctuation():
    segments = segment_words(punctuated_words(), SegmentationConfig(mode="sentence"))
    assert [segment.n_words for segment in segments] == [8, 3]
    assert segments[0].text.endswith("seatbelt.")


def test_segment_clause_mode_splits_on_pause_and_punctuation():
    segments = segment_words(punctuated_words(), SegmentationConfig(mode="clause", min_words=2, hard_pause_seconds=0.6))
    assert sum(segment.n_words for segment in segments) == len(SPOKEN)
    assert len(segments) >= 2


def test_segment_whole_mode_and_untimed_text():
    words = words_from_text("She knew it was over. He said nothing.")
    segments = segment_words(words, SegmentationConfig(mode="whole"))
    assert len(segments) == 1 and segments[0].n_words == len(words)
    sentences = segment_words(words, SegmentationConfig(mode="sentence"))
    assert len(sentences) == 2


def test_segment_max_words_bounds_length():
    words = words_from_text(" ".join(f"word{i}" for i in range(40)))
    segments = segment_words(words, SegmentationConfig(mode="whole", max_words=12, min_words=2))
    assert max(segment.n_words for segment in segments) <= 12
    assert sum(segment.n_words for segment in segments) == 40


def test_pause_mode_requires_timing():
    with pytest.raises(ValueError, match="timings"):
        segment_words(words_from_text("no timing here"), SegmentationConfig(mode="pause"))


def test_schema_matches_spec_and_compact_round_trip():
    spec = tom_rating_spec()
    schema = build_schema(spec)
    properties = schema["properties"]["ratings"]["items"]["properties"]
    assert set(properties) == set(spec.field_names)
    assert properties["tom"]["maximum"] == 3
    assert properties["target"]["enum"] == ["none", "self", "other", "both"]

    compact = build_schema(spec, compact=True)
    compact_properties = set(compact["properties"]["ratings"]["items"]["properties"])
    assert "tom" not in compact_properties and len(compact_properties) == len(spec.field_names)

    # Keys are assigned from first letters, with a suffix when one is taken
    # ("index" claims "i", so "intention" gets "i1").
    keys = compact_key_map(spec)
    assert keys["i"] == "index" and keys["t"] == "tom"
    short = {short_key: 0 for short_key in keys}
    short[next(k for k, v in keys.items() if v == "target")] = "self"
    short[next(k for k, v in keys.items() if v == "reason")] = "cue"
    short[next(k for k, v in keys.items() if v == "tom")] = 2
    expanded = expand_compact({"ratings": [short]}, spec)
    assert expanded["ratings"][0]["tom"] == 2
    assert expanded["ratings"][0]["target"] == "self"
    assert expanded["ratings"][0]["intention"] == 0


def test_schema_without_bounds_uses_enum():
    schema = build_schema(RatingSpec(), bounded_integers=False)
    assert schema["properties"]["ratings"]["items"]["properties"]["rating"]["enum"] == [0, 1, 2, 3]


def test_validate_ratings_rejects_missing_and_out_of_range():
    spec = RatingSpec(flags=(), categoricals={}, include_reason=False, include_confidence=False)
    good = {"ratings": [{"index": 1, "rating": 3}, {"index": 0, "rating": 1}]}
    assert [item["index"] for item in validate_ratings(good, 2, spec)] == [0, 1]
    with pytest.raises(ValueError, match="expected 3 ratings"):
        validate_ratings(good, 3, spec)
    with pytest.raises(ValueError, match="outside the scale"):
        validate_ratings({"ratings": [{"index": 0, "rating": 9}]}, 1, spec)


def test_build_prompt_numbers_segments():
    prompt = build_prompt("Rate:\n{utterances}\nEnd.", ["first", "second"])
    assert "000 | first" in prompt and "001 | second" in prompt


def test_weighted_kappa_bounds():
    perfect = [0, 1, 2, 3, 3]
    assert weighted_kappa(perfect, perfect, 4) == pytest.approx(1.0)
    assert weighted_kappa([0, 0, 3, 3], [3, 3, 0, 0], 4) < 0


def test_consensus_and_agreement():
    spec = tom_rating_spec()
    segments = segment_words(punctuated_words(), SegmentationConfig(mode="sentence"))

    def replicate(values):
        return [
            {"index": i, "tom": v, "emotion": 1, "intention": 0, "belief": 0,
             "social_relationship": 0, "target": "self", "confidence": 2, "reason": "cue"}
            for i, v in enumerate(values)
        ]

    table = consensus_table(segments, [replicate([1, 3]), replicate([2, 3])], spec)
    assert list(table["tom_mean"]) == [1.5, 3.0]
    assert list(table["tom_by_replicate"]) == ["1;2", "3;3"]
    assert table["emotion_mean"].tolist() == [1.0, 1.0]
    assert table["target_mode"].tolist() == ["self", "self"]

    agreement = rater_agreement([replicate([1, 3]), replicate([2, 3])], spec)
    assert agreement["exact"] == pytest.approx(0.5)
    assert agreement["within_one"] == pytest.approx(1.0)
    assert agreement["all_raters_exact"] == pytest.approx(0.5)


def test_lanczos_weights_are_centred():
    weights = lanczos_weights(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert weights.shape == (3, 2)
    assert weights[0, 0] == pytest.approx(1.0)


def test_timeseries_projection():
    spec = tom_rating_spec()
    words = punctuated_words()
    segments = segment_words(words, SegmentationConfig(mode="sentence"))

    def replicate(values):
        return [
            {"index": i, "tom": v, "emotion": 0, "intention": 0, "belief": 0,
             "social_relationship": 0, "target": "none", "confidence": 2, "reason": "x"}
            for i, v in enumerate(values)
        ]

    table = consensus_table(segments, [replicate([0, 3])], spec)
    series = ratings_to_timeseries(words, segments, table, scanner_sample_times(6, 1.0), spec)
    assert len(series) == 6
    assert {"tom_load", "tom_per_word", "tom_held", "word_rate"} <= set(series.columns)
    early = series.loc[series.sample_time_s < 3, "tom_held"].dropna()
    late = series.loc[(series.sample_time_s > 4.6) & (series.sample_time_s < 5.5), "tom_held"].dropna()
    assert (early == 0).all()
    assert (late == 3).all()


def test_word_table_round_trip(tmp_path: Path):
    words = punctuated_words()
    path = write_word_table(words, tmp_path / "story.json", story="story")
    restored = read_word_table(path)
    assert [word.text for word in restored] == [word.text for word in words]
    assert [word.display for word in restored] == [word.display for word in words]
    assert restored[0].onset == pytest.approx(words[0].onset)


class FakeProvider:
    """Deterministic stand-in for an API, so tests never make network calls."""

    def __init__(self, spec, drop_first_call: bool = False):
        self.spec = spec
        self.calls = 0
        self.drop_first_call = drop_first_call

    def __call__(self, prompt, schema, config):
        self.calls += 1
        n_items = sum(1 for line in prompt.splitlines() if line[:3].isdigit() and " | " in line)
        if self.drop_first_call and self.calls == 1:
            n_items -= 1
        ratings = []
        for index in range(n_items):
            item = {"index": index, self.spec.scale.name: index % (self.spec.scale.maximum + 1)}
            item.update({flag: 0 for flag in self.spec.flags})
            item.update({name: choices[0] for name, choices in self.spec.categoricals.items()})
            if self.spec.include_confidence:
                item["confidence"] = 2
            if self.spec.include_reason:
                item["reason"] = "because"
            ratings.append(item)
        return {"ratings": ratings}, providers.Usage(input_tokens=10, output_tokens=20, duration_ms=5.0)


def run_config(**overrides) -> RatingRunConfig:
    defaults = dict(
        prompt_template="Rate each line.\n{utterances}",
        spec=tom_rating_spec(),
        segmentation=SegmentationConfig(mode="sentence"),
        provider=ProviderConfig(backend="anthropic", model="test-model", chunk_size=0),
        replicates=2,
    )
    defaults.update(overrides)
    return RatingRunConfig(**defaults)


def test_rate_story_end_to_end_with_fake_provider(monkeypatch, tmp_path: Path):
    config = run_config()
    fake = FakeProvider(config.spec)
    monkeypatch.setattr(rate_module, "call_model", fake)

    result = rate_story("story", punctuated_words(), config, tmp_path)
    assert len(result.segments) == 2
    assert result.consensus.shape[0] == 2
    assert result.summary["n_segments"] == 2
    assert result.summary["agreement"]["all_raters_exact"] == 1.0
    assert (tmp_path / "story" / "segment_ratings.csv").exists()
    assert len(list((tmp_path / "story" / "raw").glob("replicate-*.json"))) == 2
    assert fake.calls == 2

    # A rerun reuses the cached replies instead of calling the model again.
    again = rate_story("story", punctuated_words(), config, tmp_path)
    assert fake.calls == 2
    assert again.consensus.equals(result.consensus)


def test_rate_story_refuses_cache_from_different_settings(monkeypatch, tmp_path: Path):
    config = run_config()
    monkeypatch.setattr(rate_module, "call_model", FakeProvider(config.spec))
    rate_story("story", punctuated_words(), config, tmp_path)

    changed = run_config(prompt_template="A different rubric.\n{utterances}")
    monkeypatch.setattr(rate_module, "call_model", FakeProvider(changed.spec))
    with pytest.raises(ValueError, match="different settings"):
        rate_story("story", punctuated_words(), changed, tmp_path)


def test_rate_story_retries_incomplete_reply(monkeypatch, tmp_path: Path):
    config = run_config(replicates=1)
    fake = FakeProvider(config.spec, drop_first_call=True)
    monkeypatch.setattr(rate_module, "call_model", fake)
    result = rate_story("story", punctuated_words(), config, tmp_path)
    assert fake.calls == 2  # first reply was short, second succeeded
    assert len(result.consensus) == 2


def test_rate_story_chunking_covers_every_segment(monkeypatch, tmp_path: Path):
    words = words_from_text(". ".join(f"sentence number {i} here" for i in range(10)) + ".")
    config = run_config(
        segmentation=SegmentationConfig(mode="sentence"),
        provider=ProviderConfig(backend="anthropic", model="test-model", chunk_size=3),
        replicates=1,
    )
    monkeypatch.setattr(rate_module, "call_model", FakeProvider(config.spec))
    result = rate_story("story", words, config, tmp_path)
    assert list(result.consensus["index"]) == list(range(len(result.segments)))


def test_rate_stories_stops_on_usage_limit(monkeypatch, tmp_path: Path):
    config = run_config(replicates=1)

    def limited(prompt, schema, provider_config):
        raise providers.UsageLimitError("usage limit reached")

    monkeypatch.setattr(rate_module, "call_model", limited)
    messages = []
    results = rate_module.rate_stories(
        {"a": punctuated_words(), "b": punctuated_words()}, config, tmp_path, on_progress=messages.append
    )
    assert results == {}
    assert any("usage limit" in message for message in messages)


def test_render_viewer_writes_self_contained_html(monkeypatch, tmp_path: Path):
    config = run_config()
    monkeypatch.setattr(rate_module, "call_model", FakeProvider(config.spec))
    run_dir = tmp_path / "run-a"
    rate_story("story", punctuated_words(), config, run_dir)

    output = render_viewer([run_dir], tmp_path / "reader.html", title="Example")
    html = output.read_text(encoding="utf-8")
    assert "<title>Example</title>" in html
    assert "secretly undid my seatbelt" in html
    assert "story" in html and "run-a" in html
    assert html.count("</script>") == 1  # only the page's own closing tag


def test_render_viewer_escapes_markup_in_story_text(monkeypatch, tmp_path: Path):
    config = run_config(replicates=1)
    monkeypatch.setattr(rate_module, "call_model", FakeProvider(config.spec))
    run_dir = tmp_path / "run-a"
    rate_story("story", words_from_text("Text with </script> inside it."), config, run_dir)

    html = render_viewer([run_dir], tmp_path / "reader.html").read_text(encoding="utf-8")
    assert "<\\/script>" in html  # escaped so it cannot close the inline script
    assert html.count("</script>") == 1


def test_backends_report_their_key_requirement():
    assert ProviderConfig(backend="anthropic").api_key_env == "ANTHROPIC_API_KEY"
    assert ProviderConfig(backend="openai", model="gpt-5.6-luna").api_key_env == "OPENAI_API_KEY"
    # The subscription CLIs authenticate themselves, so they need no key.
    assert ProviderConfig(backend="claude-cli").api_key_env is None
    assert ProviderConfig(backend="codex-cli", model="gpt-5.6-sol").api_key_env is None
    with pytest.raises(ValueError, match="backend must be"):
        ProviderConfig(backend="not-a-backend").validate()


def test_claude_cli_backend_builds_request_and_reads_usage(monkeypatch):
    spec = RatingSpec(flags=(), categoricals={}, include_reason=False, include_confidence=False)
    captured = {}

    def fake_run(command, prompt, config, environment):
        captured["command"] = command
        captured["environment"] = environment
        reply = {
            "structured_output": {"ratings": [{"index": 0, "rating": 2}]},
            "usage": {"input_tokens": 11, "output_tokens": 22},
        }
        return json.dumps(reply), 1234.0

    monkeypatch.setattr(providers, "_run_cli", fake_run)
    payload, usage = providers.call_claude_cli(
        "prompt", build_schema(spec), ProviderConfig(backend="claude-cli", model="claude-sonnet-5", thinking=False)
    )
    assert payload["ratings"][0]["rating"] == 2
    assert (usage.input_tokens, usage.output_tokens) == (11, 22)
    assert "--json-schema" in captured["command"] and "--strict-mcp-config" in captured["command"]
    # Nested Claude Code sessions are refused, so those variables are dropped.
    assert "CLAUDECODE" not in captured["environment"]
    assert captured["environment"]["MAX_THINKING_TOKENS"] == "0"


def test_codex_cli_backend_reads_reply_file(monkeypatch):
    spec = RatingSpec(flags=(), categoricals={}, include_reason=False, include_confidence=False)

    def fake_run(command, prompt, config, environment):
        reply_path = Path(command[command.index("--output-last-message") + 1])
        reply_path.write_text(json.dumps({"ratings": [{"index": 0, "rating": 1}]}), encoding="utf-8")
        return "", 500.0

    monkeypatch.setattr(providers, "_run_cli", fake_run)
    payload, usage = providers.call_codex_cli(
        "prompt", build_schema(spec), ProviderConfig(backend="codex-cli", model="gpt-5.6-sol")
    )
    assert payload["ratings"][0]["rating"] == 1
    assert usage.output_tokens == 0  # the CLI reports no token counts
    assert usage.duration_ms == pytest.approx(500.0)


def test_cli_usage_limit_is_detected(monkeypatch):
    import subprocess

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "Claude AI usage limit reached; resets at 5pm"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(providers.UsageLimitError):
        providers._run_cli(["claude"], "prompt", ProviderConfig(backend="claude-cli"), {})


def test_usage_cost_lookup():
    usage = providers.Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_usd("gpt-5.6-luna") == pytest.approx(1.40)
    assert usage.cost_usd("unknown-model") is None
