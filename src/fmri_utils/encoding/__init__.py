"""Reusable voxelwise fMRI encoding analysis."""

from .config import EncodingConfig, fmriprep_nuisance_columns
from .data import (
    EncodingRun,
    RunSpec,
    SubjectData,
    discover_fmriprep_runs,
    load_run_manifest,
    load_subject_mask,
    load_subject_data,
    write_run_manifest,
)
from .chunks import (
    CHUNK_SCHEMA,
    chunk_output_path,
    load_chunk_manifest,
    make_chunk_manifest,
    save_encoding_chunk,
    stitch_encoding_chunks,
)
from .model import EncodingResult, fit_encoding
from .outputs import (
    result_metric_arrays,
    save_encoding_result,
    write_metric_map,
)
from .splits import CVPlan, InnerFold, OuterFold, build_cv_plan, validate_cv_plan

__all__ = [
    "EncodingConfig",
    "EncodingResult",
    "EncodingRun",
    "RunSpec",
    "SubjectData",
    "CVPlan",
    "InnerFold",
    "OuterFold",
    "CHUNK_SCHEMA",
    "build_cv_plan",
    "validate_cv_plan",
    "discover_fmriprep_runs",
    "fit_encoding",
    "fmriprep_nuisance_columns",
    "chunk_output_path",
    "load_chunk_manifest",
    "load_run_manifest",
    "load_subject_mask",
    "load_subject_data",
    "make_chunk_manifest",
    "result_metric_arrays",
    "save_encoding_chunk",
    "save_encoding_result",
    "stitch_encoding_chunks",
    "write_metric_map",
    "write_run_manifest",
]
