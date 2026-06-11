from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


MANIFEST_COLUMNS = [
    "subject_id",
    "task",
    "run_label",
    "reference_img_path",
    "coreg_img_path",
    "reference_mask_path",
    "coreg_mask_path",
    "affine_transform_path",
    "segmentation_mask_path",
    "source_label",
]


@dataclass(frozen=True)
class FmriprepRegistrationManifestConfig:
    fmriprep_root: str | Path
    out_csv: str | Path
    one_row_csv: str | Path | None = None
    summary_json: str | Path | None = None
    subjects: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    run_labels: tuple[str, ...] = ()
    segmentation_source: str = "ribbon_or_dseg"
    require_space_t1w: bool = True
    source_label: str = "fmriprep_T1w"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_filter(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    return tuple(str(x).strip() for x in raw if str(x).strip())


def _normalize_subject(value: str) -> str:
    text = str(value).strip()
    return text if text.startswith("sub-") else f"sub-{text}"


def _normalize_run_label(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text if text.startswith("run-") else f"run-{text}"


def _entity(name: str, key: str) -> str:
    match = re.search(rf"(?:^|_){re.escape(key)}-([^_]+)", name)
    return match.group(1) if match else ""


def _run_sort_value(run_label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", run_label)
    return (int(match.group(1)) if match else 0, run_label)


def _pick_first_existing(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(x) for x in value]
    if isinstance(value, list):
        return [_json_ready(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    return value


def _pick_segmentation(anat_dir: Path, subject_id: str, mode: str) -> Path | None:
    mode = str(mode).strip().lower()
    ribbon = anat_dir / f"{subject_id}_desc-ribbon_mask.nii.gz"
    dseg = anat_dir / f"{subject_id}_dseg.nii.gz"
    if mode == "none":
        return None
    if mode == "ribbon":
        return ribbon if ribbon.exists() else None
    if mode == "dseg":
        return dseg if dseg.exists() else None
    if mode == "ribbon_or_dseg":
        return _pick_first_existing([ribbon, dseg])
    raise ValueError("segmentation_source must be one of: none, ribbon, dseg, ribbon_or_dseg")


def _discover_t1w_images(fmriprep_root: Path) -> list[Path]:
    return sorted(
        path for path in fmriprep_root.glob("**/anat/sub-*_desc-preproc_T1w.nii.gz") if "_space-" not in path.name
    )


def _bold_mask_for_boldref(boldref: Path) -> Path:
    return boldref.with_name(boldref.name.replace("_boldref.nii.gz", "_desc-brain_mask.nii.gz"))


def _transform_for_boldref(boldref: Path, subject_id: str, task: str, run_label: str) -> Path:
    run_part = f"_{run_label}" if run_label else ""
    return boldref.with_name(
        f"{subject_id}_task-{task}{run_part}_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
    )


def build_fmriprep_boldref_t1w_manifest(
    config: FmriprepRegistrationManifestConfig,
) -> dict[str, object]:
    fmriprep_root = Path(config.fmriprep_root)
    out_csv = Path(config.out_csv)
    one_row_csv = Path(config.one_row_csv) if config.one_row_csv is not None else out_csv.with_name(out_csv.stem + "_one.csv")
    summary_json = Path(config.summary_json) if config.summary_json is not None else out_csv.with_suffix(".summary.json")

    if not fmriprep_root.exists():
        raise FileNotFoundError(f"fMRIPrep root not found: {fmriprep_root}")

    subject_filter = {_normalize_subject(x) for x in config.subjects}
    task_filter = set(config.tasks)
    run_filter = {_normalize_run_label(x) for x in config.run_labels}

    rows: list[dict[str, str]] = []
    skip_counts: dict[str, int] = {
        "subject_filter": 0,
        "missing_reference_mask": 0,
        "missing_func_dir": 0,
        "missing_coreg_mask": 0,
        "missing_or_rejected_boldref": 0,
    }
    segmentation_counts: dict[str, int] = {"ribbon": 0, "dseg": 0, "none": 0}
    t1w_paths = _discover_t1w_images(fmriprep_root)

    for t1w_path in t1w_paths:
        subject_id = _entity(t1w_path.name, "sub")
        subject_id = f"sub-{subject_id}" if subject_id else t1w_path.name.split("_", 1)[0]
        if subject_filter and subject_id not in subject_filter:
            skip_counts["subject_filter"] += 1
            continue

        anat_dir = t1w_path.parent
        subject_dir = anat_dir.parent
        func_dir = subject_dir / "func"
        ref_mask = anat_dir / f"{subject_id}_desc-brain_mask.nii.gz"
        if not ref_mask.exists():
            skip_counts["missing_reference_mask"] += 1
            continue
        if not func_dir.exists():
            skip_counts["missing_func_dir"] += 1
            continue

        seg_path = _pick_segmentation(anat_dir, subject_id, config.segmentation_source)
        if seg_path is None:
            segmentation_counts["none"] += 1
        elif "_desc-ribbon_mask" in seg_path.name:
            segmentation_counts["ribbon"] += 1
        elif "_dseg" in seg_path.name:
            segmentation_counts["dseg"] += 1

        boldrefs = sorted(func_dir.glob(f"{subject_id}_task-*_boldref.nii.gz"))
        if config.require_space_t1w:
            boldrefs = [p for p in boldrefs if "_space-T1w_" in p.name]
        if not boldrefs:
            skip_counts["missing_or_rejected_boldref"] += 1
            continue

        for boldref in boldrefs:
            task = _entity(boldref.name, "task")
            run_label = _normalize_run_label(_entity(boldref.name, "run"))
            if task_filter and task not in task_filter:
                continue
            if run_filter and run_label not in run_filter:
                continue
            coreg_mask = _bold_mask_for_boldref(boldref)
            if not coreg_mask.exists():
                skip_counts["missing_coreg_mask"] += 1
                continue
            transform = _transform_for_boldref(boldref, subject_id, task, run_label)
            rows.append(
                {
                    "subject_id": subject_id,
                    "task": task,
                    "run_label": run_label,
                    "reference_img_path": str(t1w_path),
                    "coreg_img_path": str(boldref),
                    "reference_mask_path": str(ref_mask),
                    "coreg_mask_path": str(coreg_mask),
                    "affine_transform_path": str(transform) if transform.exists() else "",
                    "segmentation_mask_path": str(seg_path) if seg_path is not None else "",
                    "source_label": str(config.source_label),
                }
            )

    rows.sort(key=lambda r: (r["subject_id"], r["task"], _run_sort_value(r["run_label"]), r["coreg_img_path"]))
    if not rows:
        raise ValueError(f"No manifest rows discovered under {fmriprep_root}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    one_row_csv.parent.mkdir(parents=True, exist_ok=True)
    with one_row_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(rows[0])

    summary = {
        "timestamp_utc": _now_iso(),
        "config": _json_ready(asdict(config)),
        "fmriprep_root": str(fmriprep_root),
        "out_csv": str(out_csv),
        "one_row_csv": str(one_row_csv),
        "summary_json": str(summary_json),
        "n_t1w_images": len(t1w_paths),
        "n_manifest_rows": len(rows),
        "n_subjects": len({r["subject_id"] for r in rows}),
        "skip_counts": skip_counts,
        "segmentation_counts_by_subject": segmentation_counts,
        "first_row": rows[0],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a registration QC manifest from fMRIPrep BOLDref/T1w outputs.")
    parser.add_argument("--fmriprep-root", required=True, type=Path, help="fMRIPrep derivatives root.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Full manifest CSV to write.")
    parser.add_argument("--one-row-csv", type=Path, default=None, help="Optional one-row smoke-test manifest path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--subjects", default="", help="Comma-separated subject IDs, e.g. sub-001,sub-002.")
    parser.add_argument("--tasks", default="", help="Comma-separated task labels without task-.")
    parser.add_argument("--run-labels", default="", help="Comma-separated run labels, e.g. run-1,run-2.")
    parser.add_argument(
        "--segmentation-source",
        choices=["none", "ribbon", "dseg", "ribbon_or_dseg"],
        default="ribbon_or_dseg",
        help="Contour source for APNG overlays. Prefer ribbon_or_dseg for fMRIPrep.",
    )
    parser.add_argument(
        "--allow-non-space-t1w",
        action="store_true",
        help="Include BOLDrefs not explicitly labeled space-T1w.",
    )
    parser.add_argument("--source-label", default="fmriprep_T1w", help="source_label value written to manifest.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_fmriprep_boldref_t1w_manifest(
        FmriprepRegistrationManifestConfig(
            fmriprep_root=args.fmriprep_root,
            out_csv=args.out_csv,
            one_row_csv=args.one_row_csv,
            summary_json=args.summary_json,
            subjects=_parse_filter(args.subjects),
            tasks=_parse_filter(args.tasks),
            run_labels=_parse_filter(args.run_labels),
            segmentation_source=args.segmentation_source,
            require_space_t1w=not bool(args.allow_non_space_t1w),
            source_label=str(args.source_label),
        )
    )
    print(f"Wrote manifest: {summary['out_csv']}")
    print(f"Wrote one-row manifest: {summary['one_row_csv']}")
    print(f"Wrote summary: {summary['summary_json']}")
    print(f"Rows: {summary['n_manifest_rows']} across {summary['n_subjects']} subjects")


__all__ = [
    "FmriprepRegistrationManifestConfig",
    "MANIFEST_COLUMNS",
    "build_fmriprep_boldref_t1w_manifest",
    "main",
]


if __name__ == "__main__":
    main()
