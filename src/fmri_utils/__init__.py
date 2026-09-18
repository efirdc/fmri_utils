"""Reusable fMRI utilities.

Top-level names are imported lazily (PEP 562), so a subpackage only pulls in
the dependencies it actually needs: ``fmri_utils.story_ratings`` works without
nilearn or nibabel installed, while ``from fmri_utils import fit_encoding``
behaves exactly as before.
"""

import importlib
import sys
from typing import Any, Dict

if sys.version_info < (3, 9):
    raise RuntimeError(
        "fmri-utils requires Python >= 3.9. "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
    )

_EXPORTS: Dict[str, str] = {
    "SecondLevelOutputs": ".second_level_analysis",
    "second_level_one_sample_ttest": ".second_level_analysis",
    "FirstLevelConfig": ".first_level",
    "FirstLevelContrastOutputs": ".first_level",
    "FirstLevelOutputs": ".first_level",
    "combine_fixed_effects": ".first_level",
    "fit_first_level_run": ".first_level",
    "GroupContrastOutputs": ".group_level",
    "GroupLevelOutputs": ".group_level",
    "fit_second_level_glm": ".group_level",
    "MontageConfig": ".montage",
    "OutlineRegion": ".montage",
    "OutlineSpec": ".montage",
    "SubjectMap": ".montage",
    "build_auto_outline_regions": ".montage",
    "render_axial_subject_montage_series": ".montage",
    "RegistrationQcConfig": ".registration_qc",
    "RegistrationQcRow": ".registration_qc",
    "run_registration_qc": ".registration_qc",
    "warp_to_mni_with_fmriprep_transform": ".transformations",
    "TemporalCVPlan": ".temporal_decoding",
    "TemporalDecoderConfig": ".temporal_decoding",
    "TemporalDecodingResult": ".temporal_decoding",
    "TemporalInnerFold": ".temporal_decoding",
    "TemporalOuterFold": ".temporal_decoding",
    "TemporalRun": ".temporal_decoding",
    "build_loso_plan": ".temporal_decoding",
    "build_pooled_4x3_plan": ".temporal_decoding",
    "fit_temporal_decoder": ".temporal_decoding",
    "validate_temporal_plan": ".temporal_decoding",
    "EncodingConfig": ".encoding",
    "EncodingResult": ".encoding",
    "CVPlan": ".encoding",
    "InnerFold": ".encoding",
    "OuterFold": ".encoding",
    "build_cv_plan": ".encoding",
    "validate_cv_plan": ".encoding",
    "discover_fmriprep_runs": ".encoding",
    "fit_encoding": ".encoding",
    "fmriprep_nuisance_columns": ".encoding",
    "load_chunk_manifest": ".encoding",
    "load_run_manifest": ".encoding",
    "load_subject_data": ".encoding",
    "make_chunk_manifest": ".encoding",
    "save_encoding_result": ".encoding",
    "stitch_encoding_chunks": ".encoding",
}

__all__ = [*sorted(_EXPORTS), "story_ratings"]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        if name in {"story_ratings", "encoding"}:
            return importlib.import_module(f".{name}", __name__)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)


def __dir__() -> list:
    return sorted(__all__)
