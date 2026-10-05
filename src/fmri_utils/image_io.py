from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def save_float32_image(
    image: nib.spatialimages.SpatialImage,
    path: str | Path,
) -> Path:
    """Save a statistical image as unscaled float32, independent of mask-header dtype."""

    path = Path(path)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    output = nib.Nifti1Image(
        np.asarray(image.dataobj, dtype=np.float32),
        image.affine,
        header,
        extra=getattr(image, "extra", None),
    )
    output.to_filename(path)
    return path
