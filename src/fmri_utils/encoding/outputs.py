from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np

from .data import SubjectData
from .model import EncodingResult


def _write_map(
    values: np.ndarray,
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: Path,
) -> None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        volume = np.full(mask.shape, np.nan, dtype=np.float32)
        volume[mask] = array
    elif array.ndim == 2:
        volume = np.full(mask.shape + (array.shape[0],), np.nan, dtype=np.float32)
        for index in range(array.shape[0]):
            volume[..., index][mask] = array[index]
    else:
        raise ValueError("Map values must be voxel or fold-by-voxel arrays")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(volume, reference.affine, header), str(path))


def save_encoding_result(
    result: EncodingResult,
    data: SubjectData,
    output_dir: Path,
    *,
    prefix: str = "",
) -> Dict[str, str]:
    """Write maps, fold traces, and provenance for one subject."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = prefix or result.subject_id
    maps = {
        "mean_correlation": result.mean_correlation,
        "mean_r2": result.mean_r2,
        "outer_fold_correlation": result.outer_fold_correlation,
        "outer_fold_r2": result.outer_fold_r2,
        "outer_selected_alpha": result.outer_selected_alpha,
        "outer_selected_pca_components": result.outer_selected_pca_components,
        "mean_selected_alpha": result.mean_selected_alpha,
        "mode_selected_pca_components": result.mode_selected_pca_components,
        "null_mean_fisher_z": result.null_mean_fisher_z,
        "null_std_fisher_z": result.null_std_fisher_z,
        "null_standardized_effect_fisher_z": result.null_standardized_effect_fisher_z,
        "empirical_p": result.empirical_p,
        "signed_z": result.signed_z,
    }
    outputs = {}
    for metric, values in maps.items():
        path = output_dir / "{}__{}.nii.gz".format(name_prefix, metric)
        _write_map(values, data.mask, data.reference_image, path)
        outputs[metric] = str(path)

    trace_path = output_dir / "{}__outer_fold_traces.npz".format(name_prefix)
    trace_payload = {
        "run_ids": np.asarray(result.run_ids, dtype="U"),
        "fold_ids": np.asarray(result.fold_ids, dtype="U"),
    }
    for index, (observed, predicted) in enumerate(
        zip(result.observed_by_fold, result.predicted_by_fold)
    ):
        trace_payload["observed_{:02d}".format(index)] = observed
        trace_payload["predicted_{:02d}".format(index)] = predicted
        trace_payload["test_run_indices_{:02d}".format(index)] = (
            result.test_run_indices_by_fold[index]
        )
        trace_payload["source_rows_{:02d}".format(index)] = (
            result.test_source_rows_by_fold[index]
        )
    np.savez_compressed(trace_path, **trace_payload)
    outputs["outer_fold_traces"] = str(trace_path)

    metadata = dict(result.metadata)
    metadata.update(
        {
            "subject_id": result.subject_id,
            "mask_voxels": int(np.count_nonzero(data.mask)),
            "reference_shape": list(data.reference_image.shape[:3]),
            "reference_affine": np.asarray(data.reference_image.affine).tolist(),
            "input_provenance": data.input_provenance or [],
            "outputs": outputs,
        }
    )
    metadata_path = output_dir / "{}__encoding.json".format(name_prefix)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs["metadata"] = str(metadata_path)
    return outputs
