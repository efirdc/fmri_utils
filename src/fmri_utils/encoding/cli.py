from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

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
    return tuple(
        item.strip() for item in args.nuisance_columns.split(",") if item.strip()
    )


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
    config = EncodingConfig(
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
        mask_strategy=args.mask_strategy,
        analysis_mask_path=args.analysis_mask,
        n_permutations=args.n_permutations,
        permutation_block_length=args.permutation_block_length,
        random_seed=args.seed,
    )
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

    fit = subparsers.add_parser("fit", help="Fit one subject from a run manifest.")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--subject", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--prefix", default="")
    fit.add_argument("--lags", default="2,3,4,5")
    fit.add_argument("--lag-mode", choices=["concat", "mean"], default="concat")
    fit.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    fit.add_argument("--pca-components", default="none")
    fit.add_argument("--raw-correlation-selection", action="store_true")
    fit.add_argument(
        "--nuisance-profile",
        choices=["none", "fmriprep", "custom"],
        default="none",
    )
    fit.add_argument("--nuisance-columns", default="")
    fit.add_argument("--n-acompcor", type=int, default=4)
    fit.add_argument("--no-run-intercept", action="store_true")
    fit.add_argument(
        "--cv-scheme",
        choices=["leave_one_run_out", "grouped_run_kfold", "blocked_kfold"],
        default="leave_one_run_out",
    )
    fit.add_argument("--outer-splits", type=int, default=5)
    fit.add_argument("--inner-splits", type=int, default=4)
    fit.add_argument("--cv-shuffle", action="store_true")
    fit.add_argument("--cv-seed", type=int, default=0)
    fit.add_argument(
        "--embargo-rows",
        type=int,
        default=0,
        help="Training-row purge on each side of blocked validation/test intervals.",
    )
    fit.add_argument(
        "--mask-strategy",
        choices=["run_mask_union", "run_mask_intersection", "analysis_mask_only"],
        default="run_mask_union",
    )
    fit.add_argument("--analysis-mask")
    fit.add_argument("--n-permutations", type=int, default=0)
    fit.add_argument("--permutation-block-length", type=int, default=24)
    fit.add_argument("--seed", type=int, default=0)
    fit.set_defaults(func=_fit)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
