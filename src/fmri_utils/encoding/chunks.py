from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

from .config import EncodingConfig
from .data import SubjectData, load_run_manifest, load_subject_mask
from .model import EncodingResult
from .outputs import METRIC_NAMES, result_metric_arrays, write_metric_map


CHUNK_SCHEMA = "encoding_chunk_v1"
CHUNK_MANIFEST_COLUMNS = (
    "task_id",
    "subject_id",
    "chunk_id",
    "voxel_start",
    "voxel_stop",
    "n_voxels",
    "total_voxels",
    "run_manifest_path",
    "chunk_relpath",
    "mask_strategy",
    "analysis_mask_path",
)


def _relative_path(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(str(path.resolve()), str(parent.resolve()))).as_posix()


def _resolve_relative(value: str, source: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _subject_ids(run_manifest: Path) -> List[str]:
    with Path(run_manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    subjects = sorted({str(row.get("subject_id", "")).strip() for row in rows})
    return [subject for subject in subjects if subject]


def make_chunk_manifest(
    run_manifest: Path,
    output_path: Path,
    *,
    chunk_size: int,
    mask_strategy: str = "run_mask_union",
    analysis_mask_path: Optional[str] = None,
    subjects: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """Create one scheduler row per contiguous range of masked voxels."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    run_manifest = Path(run_manifest).resolve()
    output_path = Path(output_path).resolve()
    selected_subjects = list(subjects) if subjects else _subject_ids(run_manifest)
    if not selected_subjects:
        raise ValueError("No subjects were found in the run manifest")

    analysis_mask = None
    analysis_mask_reference = ""
    if analysis_mask_path:
        analysis_mask_file = Path(analysis_mask_path).expanduser().resolve()
        analysis_mask = str(analysis_mask_file)
        analysis_mask_reference = _relative_path(
            analysis_mask_file, output_path.parent
        )
    config = EncodingConfig(
        mask_strategy=mask_strategy,
        analysis_mask_path=analysis_mask,
    )
    config.validate()

    rows: List[Dict[str, str]] = []
    task_id = 1
    manifest_reference = _relative_path(run_manifest, output_path.parent)
    for subject_id in selected_subjects:
        specs = load_run_manifest(run_manifest, subject_id=subject_id)
        mask, _ = load_subject_mask(specs, config)
        total_voxels = int(np.count_nonzero(mask))
        for chunk_id, start in enumerate(range(0, total_voxels, chunk_size)):
            stop = min(start + chunk_size, total_voxels)
            relpath = (
                Path(subject_id)
                / "chunks"
                / "chunk-{:05d}_voxels-{:06d}-{:06d}.npz".format(
                    chunk_id, start, stop
                )
            )
            rows.append(
                {
                    "task_id": str(task_id),
                    "subject_id": subject_id,
                    "chunk_id": str(chunk_id),
                    "voxel_start": str(start),
                    "voxel_stop": str(stop),
                    "n_voxels": str(stop - start),
                    "total_voxels": str(total_voxels),
                    "run_manifest_path": manifest_reference,
                    "chunk_relpath": relpath.as_posix(),
                    "mask_strategy": mask_strategy,
                    "analysis_mask_path": analysis_mask_reference,
                }
            )
            task_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CHUNK_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_chunk_manifest(
    path: Path,
    *,
    task_id: Optional[int] = None,
    subject_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Read and optionally select rows from a chunk manifest."""

    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(CHUNK_MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Chunk manifest is missing columns: {}".format(sorted(missing)))
        rows = list(reader)
    if task_id is not None:
        rows = [row for row in rows if int(row["task_id"]) == int(task_id)]
    if subject_id is not None:
        rows = [row for row in rows if row["subject_id"] == subject_id]
    if not rows:
        selector = "task_id={}".format(task_id) if task_id is not None else subject_id or "*"
        raise ValueError("No chunk-manifest rows matched {}".format(selector))
    if task_id is not None and len(rows) != 1:
        raise ValueError("task_id={} is not unique in the chunk manifest".format(task_id))
    return rows


def resolve_run_manifest(chunk_manifest: Path, row: Dict[str, str]) -> Path:
    return _resolve_relative(row["run_manifest_path"], Path(chunk_manifest).resolve())


def chunk_output_path(output_root: Path, row: Dict[str, str]) -> Path:
    return Path(output_root).resolve() / Path(row["chunk_relpath"])


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _chunk_signature(result: EncodingResult, data: SubjectData) -> str:
    full_indices = np.flatnonzero(np.asarray(data.mask).ravel(order="C"))
    input_files = {}
    for run in data.input_provenance or []:
        for key in ("bold_path", "mask_path", "features_path", "confounds_path"):
            value = run.get(key)
            if not value:
                continue
            path = Path(str(value))
            stat = path.stat()
            input_files[str(path)] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    analysis_mask = result.metadata.get("analysis_mask_path")
    if analysis_mask:
        path = Path(str(analysis_mask))
        stat = path.stat()
        input_files[str(path)] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    payload = {
        "metadata": result.metadata,
        "input_provenance": data.input_provenance or [],
        "input_files": input_files,
        "reference_shape": list(data.reference_image.shape[:3]),
        "reference_affine": np.asarray(data.reference_image.affine).round(8).tolist(),
        "total_voxels": int(data.total_voxels or full_indices.size),
        "mask_indices_sha256": hashlib.sha256(full_indices.tobytes()).hexdigest(),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def save_encoding_chunk(
    result: EncodingResult,
    data: SubjectData,
    output_path: Path,
    *,
    save_traces: bool = False,
) -> Path:
    """Write a compact, stitchable voxel chunk."""

    start = int(data.voxel_start)
    stop = int(data.voxel_stop if data.voxel_stop is not None else data.voxel_indices.size)
    total = int(data.total_voxels or data.voxel_indices.size)
    if stop - start != data.voxel_indices.size:
        raise ValueError("Chunk range does not match the loaded voxel count")

    metadata = {
        "schema_version": CHUNK_SCHEMA,
        "subject_id": result.subject_id,
        "voxel_start": start,
        "voxel_stop": stop,
        "total_voxels": total,
        "signature": _chunk_signature(result, data),
        "run_ids": list(result.run_ids),
        "fold_ids": list(result.fold_ids),
        "encoding_metadata": result.metadata,
        "input_provenance": data.input_provenance or [],
        "traces_included": bool(save_traces),
    }
    payload: Dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "voxel_indices": np.asarray(data.voxel_indices, dtype=np.int64),
        "reference_shape": np.asarray(data.reference_image.shape[:3], dtype=np.int32),
        "reference_affine": np.asarray(data.reference_image.affine, dtype=np.float64),
    }
    for name, values in result_metric_arrays(result).items():
        metric = np.asarray(values, dtype=np.float32)
        if metric.ndim not in {1, 2} or metric.shape[-1] != data.voxel_indices.size:
            raise ValueError("Metric {} has an invalid chunk shape {}".format(name, metric.shape))
        payload["metric__{}".format(name)] = metric
    if save_traces:
        for index, (observed, predicted) in enumerate(
            zip(result.observed_by_fold, result.predicted_by_fold)
        ):
            payload["observed_{:02d}".format(index)] = np.asarray(observed, dtype=np.float32)
            payload["predicted_{:02d}".format(index)] = np.asarray(predicted, dtype=np.float32)
            payload["test_run_indices_{:02d}".format(index)] = np.asarray(
                result.test_run_indices_by_fold[index], dtype=np.int16
            )
            payload["source_rows_{:02d}".format(index)] = np.asarray(
                result.test_source_rows_by_fold[index], dtype=np.int32
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _load_chunk(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(arrays.pop("metadata_json").item()))
    if metadata.get("schema_version") != CHUNK_SCHEMA:
        raise ValueError("Unsupported chunk schema in {}".format(path))
    arrays["metadata"] = metadata
    arrays["path"] = path
    return arrays


def _validate_and_order_chunks(
    rows: Sequence[Dict[str, str]], output_root: Path
) -> List[Dict[str, object]]:
    ordered_rows = sorted(rows, key=lambda row: int(row["voxel_start"]))
    expected_start = 0
    loaded = []
    signature = None
    total_voxels = int(ordered_rows[0]["total_voxels"])
    for row in ordered_rows:
        start = int(row["voxel_start"])
        stop = int(row["voxel_stop"])
        if start != expected_start:
            raise ValueError(
                "Voxel coverage has a gap or overlap at {} (expected {})".format(
                    start, expected_start
                )
            )
        if int(row["total_voxels"]) != total_voxels:
            raise ValueError("Chunk manifest has inconsistent total_voxels")
        path = chunk_output_path(output_root, row)
        if not path.exists():
            raise FileNotFoundError("Missing encoding chunk: {}".format(path))
        chunk = _load_chunk(path)
        metadata = chunk["metadata"]
        if metadata["subject_id"] != row["subject_id"]:
            raise ValueError("Chunk subject does not match its manifest row: {}".format(path))
        if int(metadata["voxel_start"]) != start or int(metadata["voxel_stop"]) != stop:
            raise ValueError("Chunk voxel range does not match its manifest row: {}".format(path))
        if int(metadata["total_voxels"]) != total_voxels:
            raise ValueError("Chunk total_voxels does not match its manifest row: {}".format(path))
        if np.asarray(chunk["voxel_indices"]).size != stop - start:
            raise ValueError("Chunk voxel_indices length is incorrect: {}".format(path))
        if int(row["n_voxels"]) != stop - start:
            raise ValueError("Chunk-manifest n_voxels does not match its range")
        if signature is None:
            signature = metadata["signature"]
        elif metadata["signature"] != signature:
            raise ValueError("Chunks were produced with different inputs or settings")
        loaded.append(chunk)
        expected_start = stop
    if expected_start != total_voxels:
        raise ValueError(
            "Voxel coverage stops at {} but expected {}".format(expected_start, total_voxels)
        )
    return loaded


def stitch_encoding_chunks(
    chunk_manifest: Path,
    output_root: Path,
    *,
    subject_id: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Validate and stitch complete voxel coverage into final NIfTI maps."""

    rows = load_chunk_manifest(chunk_manifest)
    subjects = sorted({row["subject_id"] for row in rows})
    if subject_id:
        if subject_id not in subjects:
            raise ValueError("Subject {} is absent from the chunk manifest".format(subject_id))
        subjects = [subject_id]

    all_outputs: Dict[str, Dict[str, str]] = {}
    for subject in subjects:
        subject_rows = [row for row in rows if row["subject_id"] == subject]
        chunks = _validate_and_order_chunks(subject_rows, Path(output_root))
        first = chunks[0]
        first_meta = first["metadata"]
        metric_names = sorted(
            key.split("metric__", 1)[1]
            for key in first
            if isinstance(key, str) and key.startswith("metric__")
        )
        if set(metric_names) != set(METRIC_NAMES):
            raise ValueError("Chunk is missing or contains unknown encoding metrics")
        expected_keys = {"metric__{}".format(name) for name in metric_names}
        for chunk in chunks[1:]:
            keys = {
                key for key in chunk if isinstance(key, str) and key.startswith("metric__")
            }
            if keys != expected_keys:
                raise ValueError("Chunks contain different metric arrays")
            for key in expected_keys:
                if np.asarray(chunk[key]).shape[:-1] != np.asarray(first[key]).shape[:-1]:
                    raise ValueError("Chunk metric axes differ for {}".format(key))
            if not np.array_equal(chunk["reference_shape"], first["reference_shape"]):
                raise ValueError("Chunk reference shapes differ")
            if not np.allclose(chunk["reference_affine"], first["reference_affine"]):
                raise ValueError("Chunk reference affines differ")

        voxel_indices = np.concatenate(
            [np.asarray(chunk["voxel_indices"], dtype=np.int64) for chunk in chunks]
        )
        if np.unique(voxel_indices).size != voxel_indices.size:
            raise ValueError("Stitched voxel indices contain duplicates")
        if voxel_indices.size > 1 and not np.all(np.diff(voxel_indices) > 0):
            raise ValueError("Stitched voxel indices are not in mask order")
        stitched_dir = Path(output_root).resolve() / subject / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, str] = {}
        for metric in metric_names:
            values = np.concatenate(
                [np.asarray(chunk["metric__{}".format(metric)]) for chunk in chunks],
                axis=-1,
            )
            path = stitched_dir / "{}__{}.nii.gz".format(subject, metric)
            write_metric_map(
                values,
                voxel_indices,
                tuple(int(value) for value in first["reference_shape"]),
                np.asarray(first["reference_affine"], dtype=np.float64),
                path,
            )
            outputs[metric] = str(path)

        trace_flags = [bool(chunk["metadata"]["traces_included"]) for chunk in chunks]
        if len(set(trace_flags)) != 1:
            raise ValueError("Some chunks include traces and others do not")
        traces_included = trace_flags[0]
        if traces_included:
            fold_count = len(first_meta["fold_ids"])
            trace_payload: Dict[str, np.ndarray] = {
                "run_ids": np.asarray(first_meta["run_ids"], dtype="U"),
                "fold_ids": np.asarray(first_meta["fold_ids"], dtype="U"),
            }
            for index in range(fold_count):
                for stem in ("test_run_indices", "source_rows"):
                    key = "{}_{:02d}".format(stem, index)
                    reference = np.asarray(first[key])
                    if not all(np.array_equal(np.asarray(chunk[key]), reference) for chunk in chunks):
                        raise ValueError("Chunk trace row provenance differs")
                    trace_payload[key] = reference
                for stem in ("observed", "predicted"):
                    key = "{}_{:02d}".format(stem, index)
                    trace_payload[key] = np.concatenate(
                        [np.asarray(chunk[key]) for chunk in chunks], axis=1
                    )
            trace_path = stitched_dir / "{}__outer_fold_traces.npz".format(subject)
            np.savez_compressed(trace_path, **trace_payload)
            outputs["outer_fold_traces"] = str(trace_path)

        metadata = dict(first_meta["encoding_metadata"])
        metadata.update(
            {
                "schema_version": "encoding_stitched_v1",
                "subject_id": subject,
                "total_voxels": int(first_meta["total_voxels"]),
                "n_chunks": len(chunks),
                "chunk_signature": first_meta["signature"],
                "chunk_manifest": str(Path(chunk_manifest).resolve()),
                "chunk_files": [str(chunk["path"]) for chunk in chunks],
                "reference_shape": [int(value) for value in first["reference_shape"]],
                "reference_affine": np.asarray(first["reference_affine"]).tolist(),
                "input_provenance": first_meta["input_provenance"],
                "outer_fold_axis": first_meta["fold_ids"],
                "traces_included": traces_included,
                "outputs": outputs,
            }
        )
        metadata_path = stitched_dir / "{}__encoding.json".format(subject)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        outputs["metadata"] = str(metadata_path)
        all_outputs[subject] = outputs
    return all_outputs
