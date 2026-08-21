from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .chunks import (
    chunk_output_path,
    load_chunk_manifest,
    make_chunk_manifest,
    resolve_run_manifest,
    save_encoding_chunk,
    stitch_encoding_chunks,
)
from .config import EncodingConfig, fmriprep_nuisance_columns
from .data import (
    discover_fmriprep_runs,
    load_run_manifest,
    load_subject_data,
    write_run_manifest,
)
from .model import fit_encoding
from .outputs import save_encoding_result


def _csv_ints(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> Tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _csv_strings(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _pca_grid(value: str) -> Tuple[Optional[int], ...]:
    output = []
    for item in value.split(","):
        token = item.strip().lower()
        if token in {"none", "off", "0"}:
            output.append(None)
        elif token:
            output.append(int(token))
    return tuple(output)


def _nuisance_columns(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.nuisance_profile == "none":
        return ()
    if args.nuisance_profile == "fmriprep":
        return fmriprep_nuisance_columns(n_acompcor=args.n_acompcor)
    return _csv_strings(args.nuisance_columns)


def _config_from_args(
    args: argparse.Namespace,
    *,
    mask_strategy: Optional[str] = None,
    analysis_mask: Optional[str] = None,
) -> EncodingConfig:
    return EncodingConfig(
        lags=_csv_ints(args.lags),
        lag_mode=args.lag_mode,
        ridge_alphas=_csv_floats(args.ridge_alphas),
        pca_components=_pca_grid(args.pca_components),
        fisher_z_selection=not args.raw_correlation_selection,
        nuisance_columns=_nuisance_columns(args),
        add_run_intercept=not args.no_run_intercept,
        cv_scheme=args.cv_scheme,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        cv_shuffle=args.cv_shuffle,
        cv_seed=args.cv_seed,
        embargo_rows=args.embargo_rows,
        mask_strategy=mask_strategy or args.mask_strategy,
        analysis_mask_path=(
            analysis_mask if mask_strategy is not None else args.analysis_mask
        ),
        n_permutations=args.n_permutations,
        permutation_block_length=args.permutation_block_length,
        random_seed=args.seed,
    )


def _add_scientific_arguments(
    parser: argparse.ArgumentParser, *, include_mask: bool = True
) -> None:
    parser.add_argument("--lags", default="2,3,4,5")
    parser.add_argument("--lag-mode", choices=["concat", "mean"], default="concat")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--pca-components", default="none")
    parser.add_argument("--raw-correlation-selection", action="store_true")
    parser.add_argument(
        "--nuisance-profile",
        choices=["none", "fmriprep", "custom"],
        default="none",
    )
    parser.add_argument("--nuisance-columns", default="")
    parser.add_argument("--n-acompcor", type=int, default=4)
    parser.add_argument("--no-run-intercept", action="store_true")
    parser.add_argument(
        "--cv-scheme",
        choices=["leave_one_run_out", "grouped_run_kfold", "blocked_kfold"],
        default="leave_one_run_out",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--cv-shuffle", action="store_true")
    parser.add_argument("--cv-seed", type=int, default=0)
    parser.add_argument(
        "--embargo-rows",
        type=int,
        default=0,
        help="Training-row purge on each side of blocked validation/test intervals.",
    )
    if include_mask:
        parser.add_argument(
            "--mask-strategy",
            choices=["run_mask_union", "run_mask_intersection", "analysis_mask_only"],
            default="run_mask_union",
        )
        parser.add_argument("--analysis-mask")
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--permutation-block-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)


def _discover(args: argparse.Namespace) -> None:
    rows = discover_fmriprep_runs(
        Path(args.fmriprep_root),
        task=args.task,
        space=None if args.space.lower() == "none" else args.space,
        feature_template=args.feature_template,
    )
    output = write_run_manifest(rows, Path(args.output))
    missing_features = sum(not row["features_path"] for row in rows)
    print(
        "Wrote {} runs to {} ({} rows still need features_path)".format(
            len(rows), output, missing_features
        )
    )


def _fit(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    specs = load_run_manifest(Path(args.manifest), subject_id=args.subject)
    data = load_subject_data(specs, config)
    result = fit_encoding(data, config)
    outputs = save_encoding_result(
        result,
        data,
        Path(args.output_dir),
        prefix=args.prefix,
    )
    print("Wrote {} outputs to {}".format(len(outputs), args.output_dir))


def _make_chunks(args: argparse.Namespace) -> None:
    subjects = _csv_strings(args.subjects) if args.subjects else None
    rows = make_chunk_manifest(
        Path(args.manifest),
        Path(args.output),
        chunk_size=args.chunk_size,
        mask_strategy=args.mask_strategy,
        analysis_mask_path=args.analysis_mask,
        subjects=subjects,
    )
    print(
        "Wrote {} chunk tasks for {} subjects to {} (SLURM array 1-{})".format(
            len(rows), len({row["subject_id"] for row in rows}), args.output, len(rows)
        )
    )


def _fit_chunk(args: argparse.Namespace) -> None:
    chunk_manifest = Path(args.chunk_manifest).resolve()
    row = load_chunk_manifest(chunk_manifest, task_id=args.task_id)[0]
    analysis_mask = (
        str((chunk_manifest.parent / row["analysis_mask_path"]).resolve())
        if row["analysis_mask_path"]
        else None
    )
    config = _config_from_args(
        args,
        mask_strategy=row["mask_strategy"],
        analysis_mask=analysis_mask,
    )
    run_manifest = resolve_run_manifest(chunk_manifest, row)
    specs = load_run_manifest(run_manifest, subject_id=row["subject_id"])
    data = load_subject_data(
        specs,
        config,
        voxel_start=int(row["voxel_start"]),
        voxel_stop=int(row["voxel_stop"]),
    )
    if int(data.total_voxels or 0) != int(row["total_voxels"]):
        raise ValueError(
            "The current mask contains {} voxels but the chunk manifest expects {}. "
            "Regenerate the chunk manifest.".format(
                data.total_voxels, row["total_voxels"]
            )
        )
    result = fit_encoding(data, config)
    output = save_encoding_chunk(
        result,
        data,
        chunk_output_path(Path(args.output_root), row),
        save_traces=args.save_traces,
    )
    print(
        "Wrote task {}: {} voxels for {} to {}".format(
            row["task_id"], row["n_voxels"], row["subject_id"], output
        )
    )


def _stitch(args: argparse.Namespace) -> None:
    outputs = stitch_encoding_chunks(
        Path(args.chunk_manifest),
        Path(args.output_root),
        subject_id=args.subject,
    )
    print("Stitched {} subjects under {}".format(len(outputs), args.output_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmri-encoding",
        description="Manifest-driven voxelwise fMRI encoding analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover-fmriprep",
        help="Create a generic run manifest from fMRIPrep derivatives.",
    )
    discover.add_argument("--fmriprep-root", required=True)
    discover.add_argument("--output", required=True)
    discover.add_argument("--task")
    discover.add_argument("--space", default="T1w")
    discover.add_argument(
        "--feature-template",
        help=(
            "Feature path template with {subject_id}, {task}, {run}, and/or "
            "{run_id} placeholders."
        ),
    )
    discover.set_defaults(func=_discover)

    fit = subparsers.add_parser("fit", help="Fit one complete subject.")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--subject", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--prefix", default="")
    _add_scientific_arguments(fit)
    fit.set_defaults(func=_fit)

    make_chunks = subparsers.add_parser(
        "make-chunks", help="Create a voxel-chunk task manifest."
    )
    make_chunks.add_argument("--manifest", required=True)
    make_chunks.add_argument("--output", required=True)
    make_chunks.add_argument("--chunk-size", required=True, type=int)
    make_chunks.add_argument(
        "--mask-strategy",
        choices=["run_mask_union", "run_mask_intersection", "analysis_mask_only"],
        default="run_mask_union",
    )
    make_chunks.add_argument("--analysis-mask")
    make_chunks.add_argument(
        "--subjects", help="Optional comma-separated subject IDs; default is all subjects."
    )
    make_chunks.set_defaults(func=_make_chunks)

    fit_chunk = subparsers.add_parser(
        "fit-chunk", help="Fit one row from a voxel-chunk task manifest."
    )
    fit_chunk.add_argument("--chunk-manifest", required=True)
    fit_chunk.add_argument("--task-id", required=True, type=int)
    fit_chunk.add_argument("--output-root", required=True)
    fit_chunk.add_argument(
        "--save-traces",
        action="store_true",
        help="Store fold traces in every chunk so stitching can reconstruct them.",
    )
    _add_scientific_arguments(fit_chunk, include_mask=False)
    fit_chunk.set_defaults(func=_fit_chunk)

    stitch = subparsers.add_parser(
        "stitch", help="Validate and stitch completed voxel chunks."
    )
    stitch.add_argument("--chunk-manifest", required=True)
    stitch.add_argument("--output-root", required=True)
    stitch.add_argument("--subject", help="Stitch one subject; default is all subjects.")
    stitch.set_defaults(func=_stitch)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
