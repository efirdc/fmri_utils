import sys

if sys.version_info < (3, 9):
    raise RuntimeError(
        "fmri-utils requires Python >= 3.9. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
    )

from .second_level_analysis import (
    SecondLevelOutputs,
    second_level_one_sample_ttest,
)
from .montage import (
    MontageConfig,
    OutlineRegion,
    OutlineSpec,
    SubjectMap,
    build_auto_outline_regions,
    render_axial_subject_montage_series,
)
from .registration_qc import (
    RegistrationQcConfig,
    RegistrationQcRow,
    run_registration_qc,
)
from .transformations import warp_to_mni_with_fmriprep_transform
from .encoding import (
    EncodingConfig,
    EncodingResult,
    CVPlan,
    InnerFold,
    OuterFold,
    build_cv_plan,
    validate_cv_plan,
    discover_fmriprep_runs,
    fit_encoding,
    fmriprep_nuisance_columns,
    load_chunk_manifest,
    load_run_manifest,
    load_subject_data,
    make_chunk_manifest,
    save_encoding_result,
    stitch_encoding_chunks,
)

__all__ = [
    "SecondLevelOutputs",
    "second_level_one_sample_ttest",
    "MontageConfig",
    "OutlineRegion",
    "OutlineSpec",
    "SubjectMap",
    "build_auto_outline_regions",
    "render_axial_subject_montage_series",
    "RegistrationQcConfig",
    "RegistrationQcRow",
    "run_registration_qc",
    "warp_to_mni_with_fmriprep_transform",
    "EncodingConfig",
    "EncodingResult",
    "CVPlan",
    "InnerFold",
    "OuterFold",
    "build_cv_plan",
    "validate_cv_plan",
    "discover_fmriprep_runs",
    "fit_encoding",
    "fmriprep_nuisance_columns",
    "load_chunk_manifest",
    "load_run_manifest",
    "load_subject_data",
    "make_chunk_manifest",
    "save_encoding_result",
    "stitch_encoding_chunks",
]
