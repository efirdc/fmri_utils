"""fmri-lebel: the LeBel et al. (2023) encoding pipeline, step by step (see docs/lebel2023.md).

Stimulus encoding and the rating ablation
    features      build a feature space on the response clock (english1000, bert_wordctx10,
                  gpt2xl_l24_wordctx10, word_rate)
    regressors    a story-ratings run -> rating and word-rate regressors per story
    remove        project the (rate-orthogonalised) rating out of a feature space
    fit           fit one subject's voxel chunk on a feature space (optionally + the word-rate column)
    stitch        a fit's chunks -> native maps

Cross-participant encoding
    xsub-prep, xsub-fit, xsub-noise-ceiling
    xsub-components, xsub-ablation-fit   the rating removed from the predictor brains

Summaries and group statistics (MNI)
    summary       per subject, over its best-predicted 5% of voxels: r, delta r, delta r as a share of r
    warp-matrix   a subject's sparse column -> MNI152 2 mm warp (needs the registration and fslpy)
    group         sign-flip voxel t and region tests on r or on delta r = full - removed

Optional variance-matched null
    nulls-select, nulls-fit, group-nulls, xsub-select-directions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import SUBJECTS, Dataset


def _ids(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            out.extend(range(int(a), int(b)))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="fmri-lebel", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, dataset=True):
        p = sub.add_parser(name, help=help_text)
        if dataset:
            p.add_argument("--dataset", type=Path, required=True, help="the ds003020 root")
        return p

    p = command("features", "build a feature space")
    p.add_argument("--feature", required=True)
    p.add_argument("--output", type=Path, required=True, help="writes <output>/<feature>/<story>.npy")
    p.add_argument("--stories", default="", help="comma separated (default: every story)")
    p.add_argument("--device", default="")
    p.add_argument("--overwrite", action="store_true")

    p = command("regressors", "rating regressors from a story-ratings run")
    p.add_argument("--ratings", type=Path, required=True, help="the run folder (<run>/<story>/segment_ratings.csv)")
    p.add_argument("--transcripts", type=Path, required=True, help="the word tables the run rated")
    p.add_argument("--field", required=True, help="the rating field, e.g. tom or embodiment")
    p.add_argument("--output", type=Path, required=True)

    p = command("remove", "project a rating out of a feature space", dataset=False)
    p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--regressors", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--suffix", default="_removed")

    p = command("fit", "fit a subject's voxel chunk on a feature space")
    p.add_argument("--subject", required=True)
    p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=None, help="output folder name (default: the feature)")
    p.add_argument("--extra-root", type=Path, default=None, help="e.g. <features>/word_rate: appended after the PCA")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=5)

    p = command("stitch", "a fit's chunks -> native maps")
    p.add_argument("--folder", type=Path, required=True, help="<output>/<model>/<subject>")
    p.add_argument("--subject", required=True)

    for name in ("xsub-prep", "xsub-fit", "xsub-noise-ceiling", "xsub-components", "xsub-ablation-fit"):
        p = command(name, name.replace("-", " "))
        p.add_argument("--subject", required=True)
        p.add_argument("--work", type=Path, required=True)
        if name == "xsub-prep":
            p.add_argument("--atlases", type=Path, required=True, help="<atlases>/<subject>/HarvardOxford-*_space-T1.nii.gz")
            p.add_argument("--registration", type=Path, required=True)
        if name in ("xsub-fit", "xsub-ablation-fit"):
            p.add_argument("--chunk", type=int, default=0)
            p.add_argument("--n-chunks", type=int, default=5)
        if name == "xsub-fit":
            p.add_argument("--control", choices=("none", "shifted", "crossstory"), default="none")
        if name in ("xsub-components", "xsub-ablation-fit"):
            p.add_argument("--name", required=True, help="the ablation's name, e.g. embodiment")
            p.add_argument("--ids", default="-1", help="series ids: -1 the rating; 0:1000 the null directions")
        if name == "xsub-components":
            p.add_argument("--regressors", type=Path, required=True)
            p.add_argument("--directions", type=Path, default=None)
            p.add_argument("--null-features", type=Path, default=None, help="english1000 folder (for the null)")

    p = command("xsub-select-directions", "null directions for the cross-participant ablation")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--regressors", type=Path, required=True)
    p.add_argument("--null-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)

    p = command("warp-matrix", "a subject's column -> MNI warp matrix")
    p.add_argument("--subject", required=True)
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="writes <output>/<subject>.npz")

    p = command("group", "sign-flip voxel t and region tests")
    p.add_argument("--warps", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True, help="<fits>/<model> (holding <subject>/chunks)")
    p.add_argument("--removed", type=Path, default=None, help="<fits>/<model>: maps become delta r = full - removed")
    p.add_argument("--xsub-ablation", type=Path, default=None,
                   help="<work>/ablation/<name>/fits: delta r against the cross-participant ablation (series -1)")
    p.add_argument("--metric", default="correlation_raw")
    p.add_argument("--fsldir", type=Path, default=None, help="for the Harvard-Oxford region tests")
    p.add_argument("--atlas", action="append", default=[], metavar="NAME=IMAGE:NAMES_JSON",
                   help="more MNI label images for region tests (names: {value: name})")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--subjects", default=",".join(SUBJECTS))

    p = command("summary", "per-subject headline numbers over the top 5%% of voxels", dataset=False)
    p.add_argument("--full", type=Path, required=True, help="<fits>/<model> (holding <subject>/chunks)")
    p.add_argument("--removed", type=Path, default=None)
    p.add_argument("--xsub-ablation", type=Path, default=None, help="<work>/ablation/<name>/fits")
    p.add_argument("--subjects", default=",".join(SUBJECTS))
    p.add_argument("--output", type=Path, default=None, help="a JSON file")

    p = command("nulls-select", "variance-matched null directions for a feature space", dataset=False)
    p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--regressors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="the directions file (.npz)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)

    p = command("nulls-fit", "fit a block of null directions for one voxel chunk")
    p.add_argument("--subject", required=True)
    p.add_argument("--feature-root", type=Path, required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--regressors", type=Path, required=True)
    p.add_argument("--directions", type=Path, required=True)
    p.add_argument("--extra-root", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=5)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=0)

    p = command("group-nulls", "group null draws (the optional null)")
    p.add_argument("--warps", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--removed", type=Path, default=None)
    p.add_argument("--nulls", type=Path, default=None, help="the nulls-fit output folder (<nulls>/<subject>/chunk_*)")
    p.add_argument("--xsub-ablation", type=Path, default=None, help="<work>/ablation/<name>/fits (ids -1 and 0..n-1)")
    p.add_argument("--n-nulls", type=int, default=1000)
    p.add_argument("--n-draws", type=int, default=10000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--subjects", default=",".join(SUBJECTS))

    args = parser.parse_args(argv)
    dataset = Dataset(args.dataset) if getattr(args, "dataset", None) else None
    run(args, dataset)


def run(args, dataset) -> None:
    from . import ablation, cross_participant, encoding, features, group, nulls, regressors, registration
    c = args.command
    if c == "features":
        stories = [s for s in args.stories.split(",") if s] or features.all_stories(dataset)
        features.build(dataset, args.feature, stories, args.output, device=args.device, overwrite=args.overwrite)
    elif c == "regressors":
        regressors.build(dataset, args.ratings, args.transcripts, args.output, args.field)
    elif c == "remove":
        ablation.build_removed_features(args.feature_root, args.feature, args.regressors, args.output_root, args.suffix)
    elif c == "fit":
        encoding.fit_features(dataset, args.subject, args.feature_root, args.feature, args.output, model=args.model,
                              extra_root=args.extra_root, chunk=args.chunk, n_chunks=args.n_chunks)
    elif c == "stitch":
        encoding.stitch(dataset, args.folder, args.subject)
    elif c == "xsub-prep":
        cross_participant.prep(dataset, args.subject, args.atlases, args.registration, args.work)
    elif c == "xsub-fit":
        cross_participant.fit(dataset, args.subject, args.work, args.chunk, args.n_chunks, args.control)
    elif c == "xsub-noise-ceiling":
        cross_participant.cleaned_noise_ceiling(dataset, args.subject, args.work)
    elif c == "xsub-components":
        cross_participant.ablation_components(dataset, args.subject, args.work, args.name, args.regressors, _ids(args.ids),
                                              args.directions, args.null_features)
    elif c == "xsub-ablation-fit":
        cross_participant.fit_ablation(dataset, args.subject, args.work, args.name, args.chunk, args.n_chunks, _ids(args.ids))
    elif c == "xsub-select-directions":
        cross_participant.select_directions(dataset, args.work, args.regressors, args.null_features, args.output, n=args.n)
    elif c == "warp-matrix":
        registration.MNIWarp.build(dataset, args.subject, args.registration).save(Path(args.output) / f"{args.subject}.npz")
    elif c in ("group", "group-nulls"):
        subjects = [s for s in args.subjects.split(",") if s]
        warps = {s: registration.MNIWarp.load(Path(args.warps) / f"{s}.npz") for s in subjects}
        maps, null_maps = {}, {}
        for s in subjects:
            n_columns = warps[s].n_columns
            full = encoding.read_chunks(Path(args.full) / s, n_columns)[getattr(args, "metric", "correlation_raw")]
            removed = None
            if args.removed is not None:
                removed = encoding.read_chunks(Path(args.removed) / s, n_columns)["correlation_raw"]
            elif args.xsub_ablation is not None:
                ids = [-1] + (list(range(args.n_nulls)) if c == "group-nulls" else [])
                series = cross_participant.read_ablation(Path(args.xsub_ablation) / s, n_columns, ids)
                removed = series[-1]
                if c == "group-nulls":
                    null_maps[s] = full[None] - np.stack([series[i] for i in range(args.n_nulls)])
            maps[s] = full if removed is None else full - removed
            if c == "group-nulls" and args.nulls is not None:
                null_maps[s] = full[None] - nulls.read_nulls(Path(args.nulls) / s, n_columns, args.n_nulls)
        if c == "group":
            stack, voxels = group.to_mni(maps, warps)
            result = group.sign_flip(stack)
            group.save_results(result, voxels, args.output, args.prefix)
            atlases = group.harvard_oxford(args.fsldir) if args.fsldir else {}
            for entry in args.atlas:
                import nibabel as nib
                name, rest = entry.split("=", 1)
                image, names = rest.rsplit(":", 1)
                atlases[name] = (np.asarray(nib.load(image).dataobj).astype(int),
                                 {int(k): v for k, v in json.loads(Path(names).read_text(encoding="utf-8")).items()})
            if atlases:
                rows = group.region_test(stack, voxels, atlases)
                group.write_regions(rows, Path(args.output) / f"{args.prefix}_regions.json")
            print(f"{voxels.size} voxels; max t {np.nanmax(result['t']):.2f}; q < 0.05: {(result['q'] < 0.05).sum()}; "
                  f"FWE < 0.05: {(result['fwe'] < 0.05).sum()}", flush=True)
        else:
            result = group.null_draws(maps, null_maps, warps, n_draws=args.n_draws)
            group.save_results(result, result["voxels"], args.output, args.prefix)
            print(f"max z {np.nanmax(result['z']):.2f}; q < 0.05: {(result['q'] < 0.05).sum()}; "
                  f"FWE < 0.05: {(result['fwe'] < 0.05).sum()}", flush=True)
    elif c == "summary":
        rows = {}
        for s in [s for s in args.subjects.split(",") if s]:
            full = encoding.read_chunks(Path(args.full) / s)["correlation_raw"]
            removed = None
            if args.removed is not None:
                removed = encoding.read_chunks(Path(args.removed) / s, full.size)["correlation_raw"]
            elif args.xsub_ablation is not None:
                removed = cross_participant.read_ablation(Path(args.xsub_ablation) / s, full.size, [-1])[-1]
            rows[s] = group.top_voxel_summary(full, removed)
            print(s, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[s].items()}), flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    elif c == "nulls-select":
        nulls.select_directions(args.feature_root, args.feature, args.regressors, args.output, n=args.n, seed=args.seed)
    elif c == "nulls-fit":
        nulls.fit_nulls(dataset, args.subject, args.feature_root, args.feature, args.regressors, args.directions,
                        args.output, args.chunk, args.n_chunks, args.start, args.stop, args.extra_root)


if __name__ == "__main__":
    main()
