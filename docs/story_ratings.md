# Story Ratings

`fmri_utils.story_ratings` rates segments of a story with an LLM and turns
those ratings into time series. It handles the parts that are the same for
every rating project: splitting a story into units, keeping replicate raters
independent, validating that every unit came back rated, caching raw replies,
and projecting ratings onto an acquisition grid.

The construct being rated is yours. The rubric is a text file you write, and
the fields the model returns are described by a `RatingSpec`. The worked
example below rates theory of mind in spoken stories, but an arousal scale, a
topic label, or a dialogue-act code fits the same shape.

## What Is Included

- word input from Praat TextGrids, CSV/TSV/JSON word tables, or plain text
- four segmentation modes: pause, sentence, clause, whole
- four backends: the Anthropic and OpenAI APIs, or the local `claude` and
  `codex` CLIs billed to a subscription
- structured output schemas built from the spec
- replicate raters, per-rater agreement, and quadratic-weighted kappa
- chunked requests with retries, because some models drop items from long lists
- cached raw replies, so an interrupted run resumes without re-paying
- early stop on a usage limit rather than failing every remaining story
- Lanczos-resampled and carried-forward time series on any sample grid
- a self-contained HTML reader for checking ratings by hand

The package does not transcribe audio, force-align it, or recover punctuation
from speech. Bring words from your own aligner or tokenizer, or rate written
text with no timings at all.

## Install

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git#egg=fmri-utils[ratings]"
```

The `ratings` extra adds the `anthropic` and `openai` SDKs, needed only for the
API backends. Set whichever key you use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY=sk-...
```

On Windows, `setx ANTHROPIC_API_KEY "sk-ant-..."` persists the key; the package
also reads it from the user environment store, because a process started before
`setx` ran inherits an environment without it.

## Backends: API Keys Or A Subscription

| backend | runs through | billed to | needs |
|---|---|---|---|
| `anthropic` | Messages API | API credit | `ANTHROPIC_API_KEY` |
| `openai` | Responses API | API credit | `OPENAI_API_KEY` |
| `claude-cli` | local `claude -p` | Claude subscription | the CLI, signed in |
| `codex-cli` | local `codex exec` | ChatGPT/Codex subscription | the CLI, signed in |

The CLI backends spend subscription allowance instead of API credit, which is
useful when a plan has headroom. Two things to know:

- **They are subject to the plan's usage limits.** When one is hit, the run
  stops rather than failing every remaining story, and finished replicates are
  cached, so rerunning after the window resets picks up where it left off.
- **Thinking control is weaker.** The Anthropic API accepts
  `thinking={"type": "disabled"}`; through the CLI, `thinking=False` only sets
  a zero thinking budget, which silences budget-based models (Claude Haiku 4.5)
  but not adaptive ones (Claude Sonnet 5). Use the `anthropic` backend when
  thinking must be off.

The CLI backends also strip the session's own context: a replaced system
prompt with MCP servers and project settings disabled keeps fixed overhead near
1K tokens instead of the ~44K a default `claude -p` session carries.

`codex-cli` reports no token counts, so its usage totals are zeros with only
wall time filled in.

## Quick Start

### 1. Write a rubric

A rubric is a text file containing `{utterances}`, where the numbered segments
are inserted. Everything else is yours: scale definitions, rules, examples.

```text
Rate how strongly each utterance attributes inner mental life to a person.

0 = no mental state        2 = a state is implied
1 = purposeful action      3 = a state is named outright

Return only the JSON object required by the supplied schema.

{utterances}
```

### 2. Describe the fields

```python
from fmri_utils.story_ratings import RatingSpec, ScaleField

spec = RatingSpec(
    scale=ScaleField(
        name="tom",
        minimum=0,
        maximum=3,
        level_labels=("no mental state", "purposeful action", "implied state", "named state"),
    ),
    flags=("emotion", "intention", "belief", "social_relationship"),
    categoricals={"target": ("none", "self", "other", "both")},
)
```

Every segment then comes back with `tom`, the four binary flags, a `target`
choice, a `confidence` (1-3) and a short `reason`. Confidence and reason can be
switched off with `include_confidence=False` / `include_reason=False`.

`tom_rating_spec()` returns exactly this spec, since it is the documented
example.

### 3. Rate a story

```python
from pathlib import Path
from fmri_utils.story_ratings import (
    ProviderConfig, RatingRunConfig, SegmentationConfig, rate_story, read_textgrid_words,
)

words = read_textgrid_words(Path("TextGrids/wheretheressmoke.TextGrid"))

config = RatingRunConfig(
    prompt_template=Path("rubrics/tom.md").read_text(encoding="utf-8"),
    spec=spec,
    segmentation=SegmentationConfig(mode="clause"),
    provider=ProviderConfig(backend="anthropic", model="claude-sonnet-5"),
    replicates=3,
)

result = rate_story("wheretheressmoke", words, config, Path("ratings/tom-sonnet"))
print(result.summary["agreement"])
print(result.consensus.head())
```

This writes, under `ratings/tom-sonnet/wheretheressmoke/`:

```text
segment_ratings.csv   one row per segment: mean, SD, per-replicate values, flags, reasons
run_summary.json      settings, agreement, rating distribution, token usage and cost
raw/replicate-1.json  the raw reply, its fingerprint, and its usage
```

Rerunning reuses `raw/*.json` when the rubric, spec, model and segmentation are
unchanged, and refuses the cache when any of them differ, so two settings can
never be mixed in one directory.

### 4. Build a time series

```python
from fmri_utils.story_ratings import ratings_to_timeseries, scanner_sample_times

series = ratings_to_timeseries(
    words, result.segments, result.consensus,
    scanner_sample_times(n_samples=291, tr_seconds=2.0), spec,
)
series.to_csv("ratings/tom-sonnet/wheretheressmoke/timeseries.csv", index=False)
```

Three series are produced per field:

| column | meaning |
|---|---|
| `tom_load` | Lanczos-resampled sum; scales with speech rate, like word-rate features |
| `tom_per_word` | that sum divided by the resampled word rate, so it stays on the 0-3 scale; `NaN` in silence |
| `tom_held` | the most recent segment's rating carried through pauses |

Silence is not a zero rating, which is why `per_word` and `held` are separate
from `load`. `lanczos_weights` is the same three-lobe filter LeBel et al. (2023)
use for word-rate features, so rating regressors and semantic features share
their timing treatment.

### 5. Render the reader

```python
from fmri_utils.story_ratings import render_viewer

render_viewer([Path("ratings/tom-sonnet")], Path("reader.html"), title="ToM ratings")
```

One self-contained HTML file: each segment tinted by its consensus rating, a
rating-over-time chart, and every replicate's rating and reason for the selected
segment. Passing several run directories adds a run selector, which is the
quickest way to compare two models or two rubrics on the same text.

## Segmentation

| mode | splits at | needs |
|---|---|---|
| `pause` | silences of `pause_seconds` | word timings |
| `sentence` | sentence-final punctuation | punctuated display tokens |
| `clause` | sentence ends, clause marks the speaker paused on, long pauses | both |
| `whole` | nothing; one segment per story | neither |

`max_words` bounds every mode by splitting over-long runs at their longest
internal pause. In `clause` mode, `min_words` stops short scraps being split
off.

Why `clause` is the default for spoken stories: pause-only segmentation cuts
mid-clause ("and when his foot" / "hit the brake at the red light"), which
forces the model to rate fragments. Sentence-only segmentation gives clean
units but spreads one rating across a long sentence, blurring timing. Clause
mode keeps clauses intact while following the speaker's own pauses.

## Words, Display Tokens, And Punctuation

Each `Word` carries a `text` token (what the aligner or tokenizer produced) and
a `display` token (what a reader sees). They differ only when you supply
punctuation and casing that the source lacked; by default `display` mirrors
`text`.

Sentence and clause segmentation read `display`, so they need punctuation.
Written text already has it. Forced-alignment output usually does not, and
`has_punctuation(words)` reports whether it does. When it doesn't, either use
`pause` segmentation, which needs only timings, or supply display tokens
yourself and save them with `write_word_table` for reuse:

```python
from fmri_utils.story_ratings import has_punctuation, write_word_table

if not has_punctuation(words):
    ...  # attach punctuated display tokens from your own source
write_word_table(words, Path("transcripts/story.json"), story="story")
```

Where those tokens come from is dataset-specific and outside this package. For
the example dataset, punctuation was recovered by transcribing the audio with
Whisper and aligning that transcript to the released word tier; the script that
does it lives with that project, not here, and the resulting word tables ship
in the shared bundle.

## Choosing A Model

Measured on one 10-minute story (236 clause segments, 3 replicates), rating
theory of mind:

| model | agreement between raters | weighted kappa | cost |
|---|---:|---:|---:|
| Claude Sonnet 5, thinking | 85% | 0.895 | $0.81 |
| Claude Sonnet 5, no thinking | 86% | 0.921 | $0.52 |
| gpt-5.6-luna, low reasoning | 85% | 0.899 | $0.06 |
| gpt-5-mini, low reasoning | 88% | 0.914 | $0.14 |
| Claude Haiku 4.5, no thinking | 56% | 0.65 | $0.28 |

Notes that transfer to other rubrics:

- **Self-consistency is not correctness.** Disabling thinking made Sonnet
  slightly *more* self-consistent while making it read more literally: it
  scored "we start to reminisce" as 0.67 where the thinking run said 3, though
  the rubric names remembering as a top-scale cue. Check disagreements by hand
  in the reader before trusting an agreement number.
- **Weak models fail on the same items**, so more replicates will not rescue
  them. Haiku 4.5 missed explicit cues ("I'm really sorry", "she's crying")
  that every other model caught.
- **Some models drop items** from long lists, silently. gpt-5-mini returned 234
  of 236, then 194, then 58 of 60. Chunking (`chunk_size=60`) plus retries
  (`max_attempts=3`) made it reliable; validation catches the rest.

## Worked Example: Theory Of Mind In Moth Stories

The example dataset is the 84 spoken stories from [LeBel et al.
(2023)](https://openneuro.org/datasets/ds003020), autobiographical stories from
The Moth Radio Hour told to a live audience, used as fMRI stimuli. The released
TextGrids give word-level timings but no punctuation.

The shared bundle (see the project's `share/` directory) contains punctuated
word tables for all 84 stories, the theory-of-mind rubric and spec, and one
complete set of ratings to compare against.

```bash
fmri-story-ratings rate \
  --inputs share/transcripts \
  --prompt share/rubrics/theory_of_mind.md \
  --spec share/rubrics/theory_of_mind_spec.json \
  --output-dir ratings/my-run \
  --backend openai --model gpt-5.6-luna --reasoning-effort low \
  --segmentation clause --replicates 3 --chunk-size 60 \
  --tr-seconds 2.0

fmri-story-ratings render \
  --runs ratings/my-run share/ratings/tom-luna \
  --output reader.html --title "ToM ratings"
```

The second command renders both your run and the shared one, so the run
selector compares them segment by segment on the same text.

Rating all 84 stories with three replicates took about an hour and cost $4.23
with `gpt-5.6-luna`, covering 16,517 segments.

## Reference

| Function | Purpose |
|---|---|
| `read_textgrid_words`, `read_word_table`, `words_from_text` | load words |
| `has_punctuation`, `write_word_table` | check display tokens, save word tables |
| `segment_words` | split words into rating units |
| `rate_story`, `rate_stories`, `rate_segments` | run the raters |
| `consensus_table`, `rater_agreement`, `weighted_kappa` | combine and score replicates |
| `ratings_to_timeseries`, `scanner_sample_times`, `lanczos_weights` | build regressors |
| `render_viewer`, `collect_runs` | build the HTML reader |
| `build_schema`, `validate_ratings`, `build_prompt` | request shaping, used internally |

Costs in run summaries come from a small published price table
(`MODEL_PRICES_USD_PER_MTOK`); unknown models report `null` rather than a guess.
