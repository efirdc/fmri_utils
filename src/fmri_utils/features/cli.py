"""Command line for text features: embed once into a cache, resample from it.

    # one model call per word, ten words of context, cached per word
    fmri-features extract --words story.tsv --stimulus story --cache-root features/ \\
        --model bert-base-uncased --model-id bert --context 10

    # the same cache onto a 2 s TR clock, as often and as many ways as you like
    fmri-features resample --cache-root features/ --stimulus story \\
        --model bert-base-uncased --model-id bert --context 10 \\
        --tr 2.0 --n-samples 300 --kernel hann --out story_bert_hann.npy --rate-out story_rate.npy

``--words`` is a TSV or CSV with a header; the text, onset and offset columns
default to ``word``, ``onset`` and ``offset`` (seconds), and ``--run-column``
names a run column if context must not cross runs. ``resample`` finds the
cache entry from the same model and context flags ``extract`` was given, or
from ``--entry`` directly. Kernels: ``hann``, ``gaussian``, ``boxcar`` and
``lanczos`` (normalised), and ``lanczos-sum``, the unnormalised resampler of
the published LeBel/Huth features, for reproducing them.

    # every stimulus of a stimulus table (stimulus, words, n_samples, tr, first_time), in one go
    fmri-features build --stimuli stimuli.csv --source lm --model gpt2-xl --model-id gpt2xl         --layer 24 --pooling unit_word_last_mean --context 10 --output features/gpt2xl_l24_wordctx10
    fmri-features build --stimuli stimuli.csv --source static --table english1000sm.hf5 --output features/english1000
    fmri-features build --stimuli stimuli.csv --source rate --kernel hann --width 2 --output features/word_rate
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="", help="Hugging Face id, e.g. bert-base-uncased")
    parser.add_argument("--model-id", default="", help="short name used in the cache path")
    parser.add_argument("--family", default="transformer",
                        choices=["transformer", "sentence_embedding", "clip_text", "llm2vec"])
    parser.add_argument("--base-model", default="", help="decoder an adapter needs (LLM2Vec)")
    parser.add_argument("--layer", type=int, default=None, help="1-based block; default final")
    parser.add_argument("--pooling", default="unit_token_mean",
                        choices=["unit_token_mean", "unit_last_token", "unit_word_last_mean"])
    parser.add_argument("--context", default="0", help="preceding units shown, or 'max'")
    parser.add_argument("--granularity", default="word", choices=["word", "tr", "other"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)


def _spec(args):
    from .spec import ContextSpec, FeatureSpec, ModelSpec

    if not args.model:
        raise SystemExit("--model is required (or give resample an --entry)")
    context = "max" if args.context == "max" else int(args.context)
    return FeatureSpec(
        model=ModelSpec(id=args.model_id or args.model.split("/")[-1], family=args.family,
                        huggingface_id=args.model, base_id=args.base_model, layer=args.layer,
                        pooling=args.pooling, batch_size=args.batch_size,
                        max_length=args.max_length),
        context=ContextSpec(previous=context),
        granularity=args.granularity,
    )


def _read_words(path: Path, text: str, onset: str, offset: str, run: str):
    from .units import words_from_table

    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab", ".txt") else ","
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    missing = [c for c in (text, onset, offset) if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: no column(s) {missing}; have {list(rows[0]) if rows else []}")
    return words_from_table(rows, text_key=text, onset_key=onset, offset_key=offset, run_key=run)


def _sample_times(args) -> np.ndarray:
    if args.times:
        return np.loadtxt(args.times, dtype=np.float64).ravel()
    if not args.tr or not args.n_samples:
        raise SystemExit("give --times, or --tr and --n-samples")
    first = args.tr / 2.0 if args.first_time is None else args.first_time
    return first + np.arange(args.n_samples, dtype=np.float64) * args.tr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fmri-features", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="embed every unit of a stimulus into the cache")
    extract.add_argument("--words", type=Path, required=True, help="TSV/CSV of units with timings")
    extract.add_argument("--stimulus", required=True, help="name of the cache entry")
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--text-column", default="word")
    extract.add_argument("--onset-column", default="onset")
    extract.add_argument("--offset-column", default="offset")
    extract.add_argument("--run-column", default="")
    extract.add_argument("--device", default="", help="cuda, mps or cpu; default: best available")
    extract.add_argument("--force", action="store_true", help="re-extract even if the cache is current")
    _spec_arguments(extract)

    sample = sub.add_parser("resample", help="put cached unit embeddings on a sample clock")
    sample.add_argument("--cache-root", type=Path, default=None)
    sample.add_argument("--stimulus", default="")
    sample.add_argument("--entry", type=Path, default=None, help="a cache .npz, instead of the spec flags")
    sample.add_argument("--kernel", default="hann",
                        choices=["hann", "gaussian", "boxcar", "lanczos", "lanczos-sum"])
    sample.add_argument("--width", type=float, default=None,
                        help="kernel width in samples: Hann/boxcar half-width, Gaussian sigma, "
                             "Lanczos lobes")
    sample.add_argument("--times", type=Path, default=None, help="text file of sample times (s)")
    sample.add_argument("--tr", type=float, default=0.0)
    sample.add_argument("--n-samples", type=int, default=0)
    sample.add_argument("--first-time", type=float, default=None,
                        help="time of the first sample; default the centre of the first TR")
    sample.add_argument("--out", type=Path, required=True, help=".npy, (samples, dimensions)")
    sample.add_argument("--rate-out", type=Path, default=None,
                        help="also write units per second through the same kernel (.npy)")
    _spec_arguments(sample)

    build = sub.add_parser("build", help="a feature space for every stimulus of a stimulus table")
    build.add_argument("--stimuli", type=Path, required=True, help="CSV: stimulus, words, n_samples, tr, first_time")
    build.add_argument("--source", choices=["lm", "static", "rate"], required=True)
    build.add_argument("--output", type=Path, required=True, help="writes <output>/<stimulus>.npy")
    build.add_argument("--table", type=Path, default=None, help="word-vector table (--source static)")
    build.add_argument("--kernel", default=None, help="default: lanczos-sum (lm, static), hann (rate)")
    build.add_argument("--width", type=float, default=None, help="rate kernel width in samples (default 2)")
    build.add_argument("--cache-root", type=Path, default=None, help="cache the word embeddings (--source lm)")
    build.add_argument("--device", default="")
    build.add_argument("--overwrite", action="store_true")
    _spec_arguments(build)

    args = parser.parse_args(argv)

    if args.command == "build":
        from . import batch
        from .stimuli import read_stimuli
        stimuli = read_stimuli(args.stimuli)
        if args.source == "lm":
            batch.language_model(stimuli, _spec(args), args.output, args.cache_root, args.kernel or "lanczos-sum",
                                 args.device, args.overwrite)
        elif args.source == "static":
            if not args.table:
                raise SystemExit("--source static needs --table")
            batch.static(stimuli, args.table, args.output, args.kernel or "lanczos-sum")
        else:
            batch.rate(stimuli, args.output, args.kernel or "hann", 2.0 if args.width is None else args.width)
        return

    if args.command == "extract":
        from . import cache
        from .extract import embed_units

        spec = _spec(args)
        units = _read_words(args.words, args.text_column, args.onset_column,
                            args.offset_column, args.run_column)
        if not args.force and cache.is_current(args.cache_root, spec, args.stimulus, units):
            print(f"{cache.entry_path(args.cache_root, spec, args.stimulus)} is current", flush=True)
            return
        embeddings, valid = embed_units(
            units, spec, device=args.device,
            progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True))
        print(flush=True)
        path = cache.write(args.cache_root, spec, args.stimulus, units, embeddings, valid)
        print(f"{path}: {embeddings.shape[0]} units x {embeddings.shape[1]}, "
              f"{int(valid.sum())} backed by tokens", flush=True)

    elif args.command == "resample":
        from . import cache, resample

        if args.entry:
            with np.load(args.entry, allow_pickle=True) as handle:
                embeddings = handle["embeddings"]
                midpoints = (handle["onsets"] + handle["offsets"]) / 2.0
                meta = json.loads(str(handle["meta"]))
        else:
            if not args.cache_root or not args.stimulus:
                raise SystemExit("give --entry, or --cache-root, --stimulus and the spec flags")
            entry = cache.read(args.cache_root, _spec(args), args.stimulus)
            embeddings, midpoints, meta = entry["embeddings"], entry["midpoints"], entry["meta"]
        timed = np.isfinite(midpoints)
        if not timed.all():
            print(f"skipping {int((~timed).sum())} untimed units", flush=True)
        times = _sample_times(args)
        width = {} if args.width is None else {
            "hann": {"half_width": args.width}, "boxcar": {"half_width": args.width},
            "gaussian": {"sigma": args.width}, "lanczos": {"window": int(args.width)},
            "lanczos-sum": {"window": int(args.width)}}[args.kernel]
        if args.kernel == "lanczos-sum":
            out = resample.lanczos_sum(embeddings[timed], midpoints[timed], times, **width)
        else:
            out = resample.resample(embeddings[timed], midpoints[timed], times,
                                    kernel=args.kernel, **width)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, out.astype(np.float32))
        print(f"{args.out}: {out.shape[0]} samples x {out.shape[1]} "
              f"({meta.get('feature_id', '')}, {args.kernel})", flush=True)
        if args.rate_out:
            kernel = "lanczos" if args.kernel == "lanczos-sum" else args.kernel
            rate = resample.event_rate(midpoints[timed], times, kernel=kernel, **width)
            np.save(args.rate_out, rate)
            print(f"{args.rate_out}: units per second", flush=True)


if __name__ == "__main__":
    main()
