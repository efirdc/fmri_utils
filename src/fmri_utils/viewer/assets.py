"""Turning volumes into the files a browser can read quickly.

Two rules run through all of this. Anatomy is 8-bit, because nobody reads a
number off an underlay and it keeps a 1 mm brain near 3 MB instead of 12.
Statistical maps stay float32, because the reader thresholds them and a
quantised map would move the threshold under their hands.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats


def benjamini_hochberg(p_values: np.ndarray, q: float) -> float:
    """The largest p that survives BH at this q, or zero if none does."""
    ordered = np.sort(np.asarray(p_values, dtype=float))
    if not ordered.size:
        return 0.0
    ranks = np.arange(1, ordered.size + 1)
    passed = ordered <= (ranks / ordered.size) * q
    return float(ordered[passed].max()) if passed.any() else 0.0


def statistic_curves(values: np.ndarray, degrees_of_freedom: int) -> dict:
    """Curves the viewer interpolates to turn any p or q into a threshold.

    Storing curves rather than a few fixed levels lets the page accept whatever
    p or q the reader types. The p curve is analytic. The q curve is not: a BH
    threshold depends on this map's own distribution of p, so it is evaluated
    here on a grid and interpolated in the browser.
    """
    finite = values[np.isfinite(values) & (values != 0)]
    p_grid = np.geomspace(1e-12, 0.5, 96)
    curves = {
        "degrees_of_freedom": int(degrees_of_freedom),
        "p": [float(x) for x in p_grid],
        "p_t": [float(stats.t.isf(x / 2.0, degrees_of_freedom)) for x in p_grid],
    }
    if finite.size:
        p_values = 2.0 * stats.t.sf(np.abs(finite), degrees_of_freedom)
        q_grid = np.geomspace(1e-6, 0.5, 64)
        q_t = []
        for q in q_grid:
            cutoff = benjamini_hochberg(p_values, float(q))
            q_t.append(
                float(stats.t.isf(cutoff / 2.0, degrees_of_freedom)) if cutoff > 0 else None
            )
        curves["q"] = [float(x) for x in q_grid]
        curves["q_t"] = q_t
    return curves


def correlation_curves(values: np.ndarray, samples: int) -> dict:
    """The same curves for a correlation map, via its t equivalent."""
    degrees_of_freedom = max(int(samples) - 2, 1)
    finite = values[np.isfinite(values) & (values != 0)]
    t_values = finite * np.sqrt(degrees_of_freedom / np.maximum(1 - finite ** 2, 1e-9))
    curves = statistic_curves(t_values, degrees_of_freedom)

    def to_r(t_value):
        if t_value is None:
            return None
        return float(t_value / np.sqrt(degrees_of_freedom + t_value ** 2))

    curves["p_t"] = [to_r(x) for x in curves["p_t"]]
    if "q_t" in curves:
        curves["q_t"] = [to_r(x) for x in curves["q_t"]]
    return curves


def crop_to_content(image: nib.Nifti1Image, margin_mm: float = 8.0) -> nib.Nifti1Image:
    """Trim an anatomical to its brain plus a margin, keeping world space intact.

    Subject anatomicals are conformed volumes whose brains sit in quite
    different parts of the box, so a montage of them frames every brain
    differently. Cropping to content makes the panels comparable; the affine is
    shifted by the crop so overlays still land in the right place.
    """
    values = np.asarray(image.get_fdata(dtype=np.float32))
    mask = values > np.percentile(values[values > 0], 2) if (values > 0).any() else values > 0
    if not mask.any():
        return image
    zooms = image.header.get_zooms()[:3]
    bounds = []
    for axis in range(3):
        present = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        pad = int(round(margin_mm / max(zooms[axis], 0.1)))
        bounds.append((max(int(present.min()) - pad, 0),
                       min(int(present.max()) + 1 + pad, values.shape[axis])))
    cropped = values[bounds[0][0]:bounds[0][1],
                     bounds[1][0]:bounds[1][1],
                     bounds[2][0]:bounds[2][1]]
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(
        image.affine, [bounds[0][0], bounds[1][0], bounds[2][0]]
    )
    return nib.Nifti1Image(cropped, affine)


def write_underlay(source: Path, path: Path, crop: bool = False) -> int:
    """Anatomical underlay as uint8, at full resolution."""
    image = nib.load(str(source))
    if crop:
        image = crop_to_content(image)
    values = np.asarray(image.get_fdata(dtype=np.float32))
    inside = values[values > 0]
    high = float(np.percentile(inside, 99.5)) if inside.size else 1.0
    scaled = np.clip(np.round(values / high * 255.0), 0, 255).astype(np.uint8)
    out = nib.Nifti1Image(scaled, image.affine)
    out.header.set_data_dtype(np.uint8)
    out.header["scl_slope"] = high / 255.0
    out.header["scl_inter"] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path.stat().st_size


def write_map(source: Path, path: Path) -> tuple[int, np.ndarray]:
    """Copy a statistical map through as float32, and hand back its values."""
    image = nib.load(str(source))
    values = np.asarray(image.get_fdata(dtype=np.float32))
    clean = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    out = nib.Nifti1Image(clean, image.affine)
    out.header.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path.stat().st_size, clean


def write_labels(source: Path, path: Path) -> int:
    """An atlas of integer labels, as compactly as its values allow."""
    image = nib.load(str(source))
    values = np.asarray(image.get_fdata(dtype=np.float32))
    top = float(values.max()) if values.size else 0.0
    dtype = np.uint8 if top < 255 else np.uint16
    out = nib.Nifti1Image(values.astype(dtype), image.affine)
    out.header.set_data_dtype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path.stat().st_size


def coordinate_bins(path: Path, stride: int = 4) -> dict:
    """A thinned copy of a coordinate map, as raw float32 the page can fetch.

    The viewer reads these to carry a crosshair between brains. Full resolution
    is more than the job needs -- the field is smooth -- so every fourth voxel
    is kept and the browser interpolates.
    """
    image = nib.load(str(path))
    values = np.asarray(image.get_fdata(dtype=np.float32))
    if values.ndim != 4 or values.shape[3] != 3:
        raise ValueError(f"{path} is not a three-component coordinate map")
    thinned = values[::stride, ::stride, ::stride, :]
    affine = image.affine.copy()
    affine[:3, :3] = affine[:3, :3] * stride
    return {
        "dims": [int(n) for n in thinned.shape[:3]],
        "affine": [float(x) for x in affine.reshape(-1)],
        "values": np.ascontiguousarray(thinned.transpose(3, 0, 1, 2), dtype=np.float32),
    }


def voxel_size(path: Path) -> list[float]:
    """The map's voxel size in millimetres, for the page to size its underlay."""
    zooms = nib.load(str(path)).header.get_zooms()[:3]
    return [round(float(z), 4) for z in zooms]


def percentile_range(values: np.ndarray, percentile: float) -> list[float]:
    """A sensible starting window: the map's own spread, not its extremes."""
    finite = values[np.isfinite(values) & (values != 0)]
    if not finite.size:
        return [0.0, 1.0]
    high = float(np.percentile(np.abs(finite), percentile))
    return [0.0, high if high > 0 else float(np.abs(finite).max() or 1.0)]
