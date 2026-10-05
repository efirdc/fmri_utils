# Text Features

`fmri_utils.features` turns a timed transcript into encoding-model features in
two separate steps:

1. **Embed** each unit (usually a word) with a language model, once, into a
   cache.
2. **Resample** the cached unit vectors onto the scanner clock, as often and as
   many ways as you like, with no model loaded.

Running the model is the expensive part; choosing how to put the result on a
TR clock is cheap. Doing both in one call made every resampling question cost
a full re-extraction, which is why they are apart here.

## Quick Start

```python
from fmri_utils.features import (
    ContextSpec, FeatureSpec, ModelSpec, Unit, cache, embed_units, resample,
)

units = [Unit(text=word, onset=start, offset=stop) for word, start, stop in words]
spec = FeatureSpec(
    model=ModelSpec(id="bert", huggingface_id="bert-base-uncased",
                    pooling="unit_token_mean"),
    context=ContextSpec(previous=10),   # ten preceding words of context
    granularity="word",
)

embeddings, valid = embed_units(units, spec)            # (words, 768), per-word flag
cache.write("features/", spec, "mystory", units, embeddings, valid)

# later, and as many times as you like, with no model in sight
entry = cache.read("features/", spec, "mystory")
tr_times = 1.0 + 2.0 * np.arange(n_trs)                 # TR centres, seconds
hann = resample.resample(entry["embeddings"], entry["midpoints"], tr_times, kernel="hann")
rate = resample.event_rate(entry["midpoints"], tr_times, kernel="hann")   # words / s
```

From the shell:

```
fmri-features extract --words story.tsv --stimulus story --cache-root features/ \
    --model bert-base-uncased --model-id bert --context 10
fmri-features resample --cache-root features/ --stimulus story \
    --model bert-base-uncased --model-id bert --context 10 \
    --tr 2.0 --n-samples 300 --kernel hann --out story_bert_hann.npy --rate-out story_rate.npy
```

`--words` is a TSV or CSV with a header. The text, onset and offset columns
default to `word`, `onset` and `offset` (in seconds). `--run-column` names a
run column when context must not cross runs. `extract` skips a stimulus whose
cache entry is already current, and `--force` re-extracts anyway.

## What A Feature Space Is

A `FeatureSpec` is three decisions, kept apart because each changes the
features on its own:

- **`ModelSpec`** is the model and how a vector is read out of it:
  - `huggingface_id`;
  - `layer`: a 1-based transformer block, or `None` for the final hidden
    state;
  - `pooling`, one of:
    - `unit_token_mean`: mean over the unit's own tokens;
    - `unit_last_token`: the unit's final token;
    - `unit_word_last_mean`: the last token of each word in the unit, then
      the mean;
    - `context_slot_stack`: one pooled vector per context slot,
      concatenated.
- **`ContextSpec`** is how much preceding stimulus each unit is shown:
  `previous=10` units, or `"max"` for everything so far in the run. Context
  never crosses a run boundary.
- **`granularity`** is what a unit is (`word`, `tr`, `other`). The extractor
  doesn't use it, but it is recorded, so two spaces from one model at word and
  TR granularity cannot be confused. They are not comparable: "BERT, 10 words
  of context, one call per word" and "BERT, 10 TRs of context, one call per
  TR" are different features.

The spec's `id` (e.g. `bert_wordctx10`, `gpt2xl_l24_wordctx10`) names the
cache folder.

Families: `transformer` (any Hugging Face encoder or decoder, with token
offsets), and `sentence_embedding`, `clip_text` and `llm2vec`, which embed the
whole context string. torch and transformers are imported only when
extracting, so reading or resampling a cache needs neither.

A unit that no token backs gets a zero vector with `valid = False` rather than
passing silently as a genuine zero. This happens with an empty unit, or one
truncated out of its own context string.

## The Cache

One `.npz` per stimulus under `<cache-root>/<spec id>/`, holding:

- the embeddings;
- the per-unit valid flags;
- **each unit's onset and offset**, since the times are what make the vectors
  resamplable;
- the texts and runs;
- a fingerprint.

The fingerprint hashes the spec and every unit's text and timing. A
transcript that shifts one word then misses the cache instead of reusing
vectors aligned to the old timing (`cache.is_current`).

## Resampling

`resample.resample(values, unit_times, sample_times, kernel=...)` puts
(units, dims) onto the sample clock. Unit times are the word midpoints, and
kernel widths are in units of the output spacing:

| kernel | argument (default) | notes |
|---|---|---|
| `hann` | `half_width=2.0` | raised cosine, non-negative, compact |
| `gaussian` | `sigma=0.5` | |
| `boxcar` | `half_width=0.5` | the words inside each TR |
| `lanczos` | `window=3` | three lobes; has negative sidelobes |

By default the weights are normalised to sum to one per sample, so each TR is
a weighted **mean** of the words near it. A sample that no word reaches is
left at zero.

`resample.lanczos_sum` is the resampler of the published LeBel/Huth features,
reproduced exactly. It is unnormalised: its weights sum to the local word
rate, so a TR is a rate-weighted **sum**. The features then carry speech
density, and sparse stretches can even get negative totals from the
sidelobes. Use it to reproduce prior work. For new work, prefer a normalised
kernel plus `resample.event_rate`, which gives words per second through the
same kernel as an explicit regressor, so density is a column you can see
rather than something mixed into every dimension.

## Validation

Checked against an independent extraction of the same features (the original
analysis scripts of an encoding study on the LeBel et al. 2023 stories), per word
and on the TR clock.

**BERT** (`bert-base-uncased`, final layer, `unit_token_mean`, 10 words of
context), on *adollshouse* (1,654 words) and the held-out test story:

- **Per word**, against the original extraction code run on the same machine,
  the vectors are **bit-identical** (max difference 0.0), with the same
  validity flags.
- **On the TR clock**, `lanczos_sum` of the package's vectors against the
  cached features the campaign used (made on a GPU on another machine): max
  difference 8.6e-7 of the value range, row correlations ≥ 0.99999999999.
  That is device rounding.
- `lanczos_sum` against the campaign's own Lanczos weights: max relative
  difference 2.7e-8.

**GPT-2 XL** (`gpt2-xl`, block 24, `unit_word_last_mean`, 10 words of
context), on *adollshouse*:

- **Per word:** bit-identical to the original extraction (max difference
  0.0), with the same validity flags.
- **On the TR clock:** max difference 6.8e-8 of the value range against the
  cached campaign features.

Speed on a laptop CPU: BERT takes about 45 s per story of ~1,700 words, and
GPT-2 XL about 8 minutes. A GPU is worth having only for the larger models or
for many stories.

## Many Stimuli At Once: The Stimulus Table

`fmri-features build` builds a feature space for every stimulus listed in a
stimulus table (`features.stimuli`), writing `<output>/<stimulus>.npy` with one
row per sample. The table is a CSV with one row per stimulus:

| column | meaning |
|---|---|
| `stimulus` | its id, used as the output file name |
| `words` | its word timings: a Praat TextGrid (word tier) or a word table (`.json`, `.csv`, `.tsv` with word/onset/offset) |
| `n_samples` | samples on its response clock |
| `tr` | seconds between samples |
| `first_time` | time of the first sample, in the words' time base (default tr / 2) |

Sample i is at `first_time + i * tr`, and words sit at their midpoints.

```bash
fmri-features build --stimuli stimuli.csv --source lm --model gpt2-xl --model-id gpt2xl --layer 24 \
    --pooling unit_word_last_mean --context 10 --output features/gpt2xl_l24_wordctx10 --cache-root cache
fmri-features build --stimuli stimuli.csv --source static --table english1000sm.hf5 --output features/english1000
fmri-features build --stimuli stimuli.csv --source rate --kernel hann --width 2 --output features/word_rate
```

- `--source lm` embeds each word once with any model spec (cached with
  `--cache-root`).
- `--source static` looks words up in a word-vector table (`features.static`):
  - HDF5 with `data` and `vocab`, as in English1000;
  - `.npz` with `vectors` and `vocab`;
  - GloVe text.
- `--source rate` gives words per second through a kernel.
- The default kernel for `lm` and `static` is `lanczos-sum`: the unnormalised
  three-lobe Lanczos of the Huth/LeBel features, a rate-weighted sum. Pass a
  normalised kernel (`hann`, `lanczos`) for an average instead, and add the rate
  as its own column.
