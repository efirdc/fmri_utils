from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from nibabel.processing import resample_from_to

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


SpatialImage = nib.spatialimages.SpatialImage


@dataclass(frozen=True)
class RegistrationQcRow:
    subject_id: str
    task: str
    run_label: str
    reference_img_path: str
    coreg_img_path: str
    reference_mask_path: str
    coreg_mask_path: str
    affine_transform_path: str | None = None
    segmentation_mask_path: str | None = None
    source_label: str = ""


@dataclass(frozen=True)
class RegistrationQcConfig:
    affine_tol: float = 1e-5
    require_matching_affine: bool = True
    auto_resample_to_reference: bool = True
    apng_enabled: bool = True
    apng_duration_ms: int = 1200
    n_slices: int = 7
    crop_margin_px: int = 8
    apng_tile_scale: int = 1
    font_size: int = 18
    title_font_scale: float = 1.15
    overlay_text_font_scale: float = 0.95
    contour_rgba: tuple[int, int, int, int] = (255, 0, 0, 220)
    contour_linewidth: float = 0.65
    write_overview_heatmap: bool = True
    overview_heatmap_name: str = "overview_heatmap.png"
    apng_subdir: str = "qualitative/apng"
    metric_name_map: dict[str, str] | None = None


REQUIRED_COLUMNS = (
    "subject_id",
    "task",
    "run_label",
    "reference_img_path",
    "coreg_img_path",
    "reference_mask_path",
    "coreg_mask_path",
)
OPTIONAL_COLUMNS = (
    "affine_transform_path",
    "segmentation_mask_path",
    "source_label",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_column(name: str) -> str:
    return str(name).strip().lstrip("\ufeff")


def _normalize_optional_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _load_manifest_rows(manifest: str | Path | Sequence[RegistrationQcRow | Mapping[str, Any]]) -> list[RegistrationQcRow]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest has no header: {manifest_path}")
            original_cols = [str(c) for c in reader.fieldnames]
            normalized = [_normalize_column(c) for c in original_cols]
            missing = [c for c in REQUIRED_COLUMNS if c not in normalized]
            if missing:
                raise ValueError(f"Manifest missing required columns: {missing}")
            remap = {orig: norm for orig, norm in zip(original_cols, normalized)}
            rows: list[RegistrationQcRow] = []
            for idx, src in enumerate(reader, start=2):
                row = {remap[k]: v for k, v in src.items() if k in remap}
                rows.append(
                    RegistrationQcRow(
                        subject_id=str(row["subject_id"]).strip(),
                        task=str(row["task"]).strip(),
                        run_label=str(row["run_label"]).strip(),
                        reference_img_path=str(row["reference_img_path"]).strip(),
                        coreg_img_path=str(row["coreg_img_path"]).strip(),
                        reference_mask_path=str(row["reference_mask_path"]).strip(),
                        coreg_mask_path=str(row["coreg_mask_path"]).strip(),
                        affine_transform_path=_normalize_optional_path(row.get("affine_transform_path")),
                        segmentation_mask_path=_normalize_optional_path(row.get("segmentation_mask_path")),
                        source_label=str(row.get("source_label", "")).strip(),
                    )
                )
                if not rows[-1].subject_id or not rows[-1].task:
                    raise ValueError(f"Manifest row {idx} missing subject_id/task.")
            return rows

    rows_out: list[RegistrationQcRow] = []
    for item in manifest:
        if isinstance(item, RegistrationQcRow):
            rows_out.append(item)
            continue
        if isinstance(item, Mapping):
            rows_out.append(
                RegistrationQcRow(
                    subject_id=str(item["subject_id"]).strip(),
                    task=str(item["task"]).strip(),
                    run_label=str(item["run_label"]).strip(),
                    reference_img_path=str(item["reference_img_path"]).strip(),
                    coreg_img_path=str(item["coreg_img_path"]).strip(),
                    reference_mask_path=str(item["reference_mask_path"]).strip(),
                    coreg_mask_path=str(item["coreg_mask_path"]).strip(),
                    affine_transform_path=_normalize_optional_path(item.get("affine_transform_path")),
                    segmentation_mask_path=_normalize_optional_path(item.get("segmentation_mask_path")),
                    source_label=str(item.get("source_label", "")).strip(),
                )
            )
            continue
        raise TypeError(f"Unsupported manifest row type: {type(item).__name__}")
    if not rows_out:
        raise ValueError("Manifest is empty.")
    return rows_out


def _to_path(value: str | None, row: RegistrationQcRow, field_name: str) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing file for subject={row.subject_id} task={row.task} run={row.run_label}: {field_name}={p}"
        )
    return p


def _load_img(path: Path) -> SpatialImage:
    img = nib.as_closest_canonical(nib.load(str(path)))
    if img.ndim > 3:
        img = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine)
    return img


def _align_to_reference_grid(
    moving_img: SpatialImage,
    reference_img: SpatialImage,
    *,
    interpolation_order: int,
    field_name: str,
    config: RegistrationQcConfig,
    row: RegistrationQcRow,
) -> tuple[SpatialImage, bool]:
    same_shape = moving_img.shape == reference_img.shape
    same_affine = np.allclose(
        np.asarray(moving_img.affine, dtype=np.float64),
        np.asarray(reference_img.affine, dtype=np.float64),
        atol=float(config.affine_tol),
        rtol=0.0,
    )
    if same_shape and (same_affine or not config.require_matching_affine):
        return moving_img, False

    if bool(config.auto_resample_to_reference):
        aligned = resample_from_to(moving_img, reference_img, order=int(interpolation_order))
        return nib.as_closest_canonical(aligned), True

    if not same_shape:
        raise ValueError(
            f"Shape mismatch for {field_name} at subject={row.subject_id} task={row.task} run={row.run_label}: "
            f"{reference_img.shape} vs {moving_img.shape}."
        )
    if config.require_matching_affine and not same_affine:
        raise ValueError(
            f"Affine mismatch for {field_name} at subject={row.subject_id} task={row.task} run={row.run_label}."
        )
    return moving_img, False


def _load_binary_mask(path: Path, row: RegistrationQcRow, field_name: str) -> tuple[SpatialImage, np.ndarray]:
    img = _load_img(path)
    data = np.asarray(img.dataobj)
    mask = np.isfinite(data) & (data > 0)
    if not np.any(mask):
        raise ValueError(
            f"Empty mask for {field_name} at subject={row.subject_id} task={row.task} run={row.run_label}: {path}"
        )
    return img, mask


def _load_nonempty_image(path: Path, row: RegistrationQcRow, field_name: str) -> SpatialImage:
    img = _load_img(path)
    data = np.asarray(img.dataobj)
    if not np.any(np.isfinite(data) & (data > 0)):
        raise ValueError(
            f"Empty image for {field_name} at subject={row.subject_id} task={row.task} run={row.run_label}: {path}"
        )
    return img


def _center_of_mass_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    com_vox = coords.mean(axis=0)
    return np.asarray(nib.affines.apply_affine(affine, com_vox), dtype=np.float64)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return float("nan")
    return inter / union


def _parse_itk_affine_txt(transform_path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = transform_path.read_text(encoding="utf-8", errors="ignore")
    line = next((ln for ln in text.splitlines() if ln.startswith("Parameters:")), "")
    if not line:
        raise ValueError(f"No Parameters line in affine transform: {transform_path}")
    values = [float(x) for x in line.replace("Parameters:", "").strip().split()]
    if len(values) < 12:
        raise ValueError(f"Unexpected affine parameter length={len(values)} in {transform_path}")
    matrix = np.array(values[:9], dtype=np.float64).reshape(3, 3)
    translation = np.array(values[9:12], dtype=np.float64)
    return matrix, translation


def _compute_affine_metrics(transform_path: Path | None) -> dict[str, float]:
    out = {
        "transform_translation_norm_mm": float("nan"),
        "transform_rotation_deg": float("nan"),
    }
    if transform_path is None:
        return out
    matrix, translation = _parse_itk_affine_txt(transform_path)
    out["transform_translation_norm_mm"] = float(np.linalg.norm(translation))
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    trace = float(np.trace(rot))
    trace = min(3.0, max(-1.0, trace))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = min(1.0, max(-1.0, cos_theta))
    out["transform_rotation_deg"] = float(np.degrees(math.acos(cos_theta)))
    return out


def _robust_normalize(vol: np.ndarray) -> np.ndarray:
    finite = np.isfinite(vol)
    if not np.any(finite):
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[finite]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if np.isclose(lo, hi):
            hi = lo + 1.0
    out = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32, copy=False)


def _extract_slice_for_display(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 2:  # axial: x-y plane
        raw = vol[:, :, idx]
    elif axis == 1:  # coronal: x-z plane
        raw = vol[:, idx, :]
    elif axis == 0:  # sagittal: y-z plane
        raw = vol[idx, :, :]
    else:
        raise ValueError(f"Invalid axis={axis}")
    # Match legacy coreg QC display convention across planes.
    return np.flipud(raw.T)


def _display_plane_content_mask(mask3d: np.ndarray, axis: int) -> np.ndarray:
    if axis == 2:
        raw = np.any(mask3d, axis=2)
    elif axis == 1:
        raw = np.any(mask3d, axis=1)
    elif axis == 0:
        raw = np.any(mask3d, axis=0)
    else:
        raise ValueError(f"Invalid axis={axis}")
    return np.flipud(raw.T)


def _compute_2d_bbox(mask2d: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    ys, xs = np.where(mask2d)
    if ys.size == 0 or xs.size == 0:
        return slice(0, mask2d.shape[0]), slice(0, mask2d.shape[1])
    y0 = max(0, int(np.min(ys)) - margin)
    y1 = min(mask2d.shape[0], int(np.max(ys)) + margin + 1)
    x0 = max(0, int(np.min(xs)) - margin)
    x1 = min(mask2d.shape[1], int(np.max(xs)) + margin + 1)
    return slice(y0, y1), slice(x0, x1)


def _crop_2d(arr: np.ndarray, bbox2d: tuple[slice, slice]) -> np.ndarray:
    return np.asarray(arr[bbox2d[0], bbox2d[1]])


def _pad_to_size(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_top = max(0, (target_h - h) // 2)
    pad_bottom = max(0, target_h - h - pad_top)
    pad_left = max(0, (target_w - w) // 2)
    pad_right = max(0, target_w - w - pad_left)
    if arr.ndim == 2:
        return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    return np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def _display_plane_spacings(affine: np.ndarray, axis: int) -> tuple[float, float]:
    voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
    if axis == 2:  # displayed rows=y, cols=x
        return float(voxel_sizes[1]), float(voxel_sizes[0])
    if axis == 1:  # displayed rows=z, cols=x
        return float(voxel_sizes[2]), float(voxel_sizes[0])
    if axis == 0:  # displayed rows=z, cols=y
        return float(voxel_sizes[2]), float(voxel_sizes[1])
    raise ValueError(f"Invalid axis={axis}")


def _resize_to_physical_aspect(
    arr: np.ndarray,
    *,
    row_spacing: float,
    col_spacing: float,
    base_spacing: float,
    is_mask: bool = False,
) -> np.ndarray:
    if arr.size == 0:
        return arr
    h, w = arr.shape[:2]
    out_h = max(1, int(round(h * float(row_spacing) / float(base_spacing))))
    out_w = max(1, int(round(w * float(col_spacing) / float(base_spacing))))
    if out_h == h and out_w == w:
        return arr
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if arr.dtype == bool:
        img = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
        out = np.asarray(img.resize((out_w, out_h), resample=mode)) > 0
        return out
    img = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    out = np.asarray(img.resize((out_w, out_h), resample=mode), dtype=np.float32)
    return out.astype(arr.dtype, copy=False) if np.issubdtype(arr.dtype, np.floating) else out


def _choose_slice_indices(mask3d: np.ndarray, axis: int, n_slices: int, lower_pct: float, upper_pct: float) -> np.ndarray:
    shape_n = mask3d.shape[axis]
    coords = np.argwhere(mask3d)
    if coords.size == 0:
        vals = np.linspace(0, max(0, shape_n - 1), n_slices)
    else:
        axis_vals = coords[:, axis].astype(np.float64)
        lo = float(np.percentile(axis_vals, lower_pct))
        hi = float(np.percentile(axis_vals, upper_pct))
        if hi <= lo:
            lo = float(np.min(axis_vals))
            hi = float(np.max(axis_vals))
        vals = np.linspace(lo, hi, n_slices)
    idx = np.clip(np.round(vals).astype(int), 0, max(0, shape_n - 1))
    return idx


def _axis_world_mm_for_index(affine: np.ndarray, shape: tuple[int, int, int], axis: int, idx: int) -> float:
    vox = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    vox[axis] = float(idx)
    world = nib.affines.apply_affine(affine, vox)
    return float(world[axis])


def _load_font(size: int) -> ImageFont.ImageFont:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for qualitative APNG outputs.")
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_labeled_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    text_fill: tuple[int, int, int, int] = (255, 255, 255, 255),
    bg_fill: tuple[int, int, int, int] = (0, 0, 0, 170),
    pad: int = 2,
) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - pad, top - pad, right + pad, bottom + pad), fill=bg_fill)
    draw.text((x, y), text, font=font, fill=text_fill)


def _render_contour_overlay_rgba(
    mask2d: np.ndarray,
    *,
    linewidth: float,
    color_rgba: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = mask2d.shape
    if h <= 1 or w <= 1 or not np.any(mask2d):
        return np.zeros((h, w, 4), dtype=np.uint8)
    rgb = tuple(float(c) / 255.0 for c in color_rgba[:3])
    alpha = float(color_rgba[3]) / 255.0
    dpi = 120
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor=(0, 0, 0, 0))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=(0, 0, 0, 0))
    ax.set_axis_off()
    ax.contour(
        mask2d.astype(np.float32),
        levels=[0.5],
        colors=[rgb],
        linewidths=float(linewidth),
        alpha=alpha,
        antialiased=True,
        corner_mask=False,
        origin="upper",
    )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    rgba = np.flipud(rgba)
    if rgba.shape[0] != h or rgba.shape[1] != w:
        rgba_img = Image.fromarray(rgba, mode="RGBA")
        rgba = np.asarray(rgba_img.resize((w, h), resample=Image.Resampling.BILINEAR), dtype=np.uint8)
    return rgba


def _segmentation_slice_to_contour_mask(seg2d: np.ndarray) -> np.ndarray:
    """Match the legacy DATT convention for segmentation contours.

    Binary masks are treated as an explicit ribbon/contour mask. Label maps are
    interpreted as fMRIPrep tissue dseg outputs, where label 3 is the WM class
    used by the old QC renderer. Using all nonzero labels turns dseg into a
    single whole-brain hull, which is not useful for registration QC.
    """
    arr = np.asarray(seg2d)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool)
    vals = np.unique(arr[finite])
    nonzero = vals[np.abs(vals) > 1e-6]
    if nonzero.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    if nonzero.size <= 1 and np.allclose(nonzero, 1.0):
        return finite & (arr > 0)
    if np.any(np.isclose(vals, 3.0)):
        return finite & np.isclose(arr, 3.0)
    return finite & (arr > 0)


def _render_mosaic_image(
    volume: np.ndarray,
    affine: np.ndarray,
    content_mask: np.ndarray,
    segmentation_mask: np.ndarray | None,
    title: str,
    config: RegistrationQcConfig,
) -> Image.Image:
    vol = _robust_normalize(volume)
    n_slices = int(config.n_slices)
    crop_margin_px = int(config.crop_margin_px)
    font_size = int(config.font_size)
    row_defs = [("axial", 2, "z"), ("coronal", 1, "y"), ("sagittal", 0, "x")]
    slice_pct_ranges = {"axial": (18.0, 96.0), "coronal": (8.0, 94.0), "sagittal": (6.0, 94.0)}

    plane_bboxes: dict[str, tuple[slice, slice]] = {}
    plane_sizes: list[tuple[int, int]] = []
    plane_spacings: dict[str, tuple[float, float]] = {}
    voxel_sizes = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
    finite_voxel_sizes = voxel_sizes[np.isfinite(voxel_sizes) & (voxel_sizes > 0)]
    base_spacing = float(np.min(finite_voxel_sizes)) if finite_voxel_sizes.size else 1.0
    for row_name, axis, _ in row_defs:
        plane_mask = _display_plane_content_mask(content_mask.astype(bool), axis)
        bbox2d = _compute_2d_bbox(plane_mask, margin=crop_margin_px)
        plane_bboxes[row_name] = bbox2d
        row_spacing, col_spacing = _display_plane_spacings(affine, axis)
        plane_spacings[row_name] = (row_spacing, col_spacing)
        plane_h = int(round((bbox2d[0].stop - bbox2d[0].start) * row_spacing / base_spacing))
        plane_w = int(round((bbox2d[1].stop - bbox2d[1].start) * col_spacing / base_spacing))
        plane_sizes.append((max(1, plane_h), max(1, plane_w)))

    target_h = max(h for h, _ in plane_sizes)
    target_w = max(w for _, w in plane_sizes)
    tile_scale = max(1, int(config.apng_tile_scale))
    tile_h = int(target_h * tile_scale)
    tile_w = int(target_w * tile_scale)
    gap = 2
    border = 3
    width = border * 2 + n_slices * tile_w + (n_slices - 1) * gap

    title_pt = max(14, int(font_size * float(config.title_font_scale)))
    tmp_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    while title_pt > 12:
        trial_font = _load_font(title_pt)
        trial_bbox = tmp_draw.textbbox((0, 0), title, font=trial_font)
        if int(trial_bbox[2] - trial_bbox[0]) <= (width - 14):
            break
        title_pt -= 1
    title_font = _load_font(title_pt)
    overlay_font = _load_font(max(11, int(font_size * float(config.overlay_text_font_scale))))
    title_bbox = tmp_draw.textbbox((0, 0), title, font=title_font)
    title_text_h = int(title_bbox[3] - title_bbox[1])
    title_h = max(int(font_size * 1.2), title_text_h + 8)

    height = border * 2 + title_h + len(row_defs) * tile_h + (len(row_defs) - 1) * gap
    canvas = Image.new("RGBA", (width, height), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((border + 2, 2), title, fill=(255, 255, 255, 255), font=title_font)

    y0 = border + title_h
    for r, (row_name, axis, axis_char) in enumerate(row_defs):
        lo_pct, hi_pct = slice_pct_ranges[row_name]
        indices = _choose_slice_indices(content_mask, axis, n_slices, lo_pct, hi_pct)
        bbox2d = plane_bboxes[row_name]
        row_spacing, col_spacing = plane_spacings[row_name]
        for c, idx in enumerate(indices):
            sl = _extract_slice_for_display(vol, axis, int(idx))
            sl = _crop_2d(sl, bbox2d)
            sl = _resize_to_physical_aspect(
                sl,
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                base_spacing=base_spacing,
                is_mask=False,
            )
            sl = _pad_to_size(sl, target_h, target_w)
            gray = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
            tile = Image.fromarray(gray, mode="L").convert("RGBA")

            if segmentation_mask is not None and np.any(segmentation_mask):
                seg2d_raw = _extract_slice_for_display(segmentation_mask.astype(np.float32), axis, int(idx))
                seg2d = _segmentation_slice_to_contour_mask(seg2d_raw)
                seg2d = _crop_2d(seg2d, bbox2d)
                seg2d = _resize_to_physical_aspect(
                    seg2d,
                    row_spacing=row_spacing,
                    col_spacing=col_spacing,
                    base_spacing=base_spacing,
                    is_mask=True,
                )
                seg2d = _pad_to_size(seg2d, target_h, target_w)
                if np.any(seg2d):
                    overlay = _render_contour_overlay_rgba(
                        seg2d,
                        linewidth=float(config.contour_linewidth),
                        color_rgba=tuple(int(x) for x in config.contour_rgba),
                    )
                    tile = Image.alpha_composite(tile, Image.fromarray(overlay, mode="RGBA"))

            if tile_scale > 1:
                tile = tile.resize((tile_w, tile_h), resample=Image.Resampling.BILINEAR)

            x = border + c * (tile_w + gap)
            y = y0 + r * (tile_h + gap)
            canvas.paste(tile, (x, y), tile)

            mm = _axis_world_mm_for_index(affine, vol.shape[:3], axis, int(idx))
            _draw_labeled_text(draw, (x + 10, y + tile_h - 26), f"{axis_char}={int(round(mm))}", font=overlay_font)
            if axis in (1, 2):
                _draw_labeled_text(draw, (x + 10, y + 12), "L", font=overlay_font)
                _draw_labeled_text(draw, (x + tile_w - 22, y + 12), "R", font=overlay_font)

    return canvas


def _build_apng(frame_a: Image.Image, frame_b: Image.Image, out_path: Path, duration_ms: int) -> bool:
    _ensure_dir(out_path.parent)
    try:
        frame_a.save(
            out_path,
            save_all=True,
            append_images=[frame_b],
            duration=[int(duration_ms), int(duration_ms)],
            loop=0,
            optimize=False,
        )
        return True
    except Exception:
        return False


def _run_axis_label(row: Mapping[str, Any]) -> str:
    task = str(row.get("task", "")).strip()
    run_label = str(row.get("run_label", "")).strip()
    return f"{task}:{run_label}" if run_label else task


def _run_sort_key(label: str) -> tuple[str, int, str]:
    task, run = (label.split(":", 1) + [""])[:2] if ":" in label else (label, "")
    m = re.search(r"(\d+)", run)
    run_num = int(m.group(1)) if m else 0
    return task, run_num, run


def _overview_metric_specs(config: RegistrationQcConfig, has_transform_metrics: bool) -> list[tuple[str, str, str, str]]:
    custom = config.metric_name_map or {}
    specs: list[tuple[str, str, str, str]] = [
        ("jaccard", custom.get("jaccard", "Jaccard"), "viridis", "{:.3f}"),
        (
            "pct_reference_covered_by_coreg_mask",
            custom.get("pct_reference_covered_by_coreg_mask", "Reference Voxels Covered by Coreg (%)"),
            "viridis",
            "{:.1f}",
        ),
        (
            "pct_coreg_mask_within_reference",
            custom.get("pct_coreg_mask_within_reference", "Coreg Voxels Inside Reference (%)"),
            "viridis",
            "{:.1f}",
        ),
        ("com_distance_mm", custom.get("com_distance_mm", "COM Distance (mm)"), "magma", "{:.2f}"),
    ]
    if has_transform_metrics:
        specs.extend(
            [
                (
                    "transform_translation_norm_mm",
                    custom.get("transform_translation_norm_mm", "Transform Translation Norm (mm)"),
                    "magma",
                    "{:.2f}",
                ),
                (
                    "transform_rotation_deg",
                    custom.get("transform_rotation_deg", "Transform Rotation (deg)"),
                    "magma",
                    "{:.2f}",
                ),
            ]
        )
    return specs


def _write_overview_heatmap(run_rows: list[dict[str, Any]], out_path: Path, config: RegistrationQcConfig) -> None:
    if not run_rows:
        return
    subjects = sorted({str(r["subject_id"]) for r in run_rows})
    run_labels = sorted({str(_run_axis_label(r)) for r in run_rows}, key=_run_sort_key)
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    run_to_idx = {r: i for i, r in enumerate(run_labels)}
    has_transform = any(np.isfinite(float(r.get("transform_translation_norm_mm", float("nan")))) for r in run_rows)
    specs = _overview_metric_specs(config, has_transform)

    n_metrics = len(specs)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(math.ceil(n_metrics / ncols))
    subplot_w = max(7.5, len(run_labels) * 0.34 + 3.4)
    subplot_h = max(5.2, len(subjects) * 0.26 + 2.4)
    fig_w = subplot_w * ncols
    fig_h = subplot_h * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), dpi=180, constrained_layout=True)
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).reshape(nrows, ncols)

    for idx, (metric_key, title, cmap, fmt) in enumerate(specs):
        ax = axes[idx // ncols, idx % ncols]
        arr = np.full((len(subjects), len(run_labels)), np.nan, dtype=np.float32)
        for row in run_rows:
            s_idx = subj_to_idx[str(row["subject_id"])]
            r_idx = run_to_idx[_run_axis_label(row)]
            value = row.get(metric_key, float("nan"))
            try:
                arr[s_idx, r_idx] = float(value)
            except Exception:
                arr[s_idx, r_idx] = np.nan

        finite = np.isfinite(arr)
        if np.any(finite):
            vmin = float(np.nanpercentile(arr, 5))
            vmax = float(np.nanpercentile(arr, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_yticks(np.arange(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=9)
        ax.set_xticks(np.arange(len(run_labels)))
        ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=9)
        ax.tick_params(length=0)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                value = arr[y, x]
                if not np.isfinite(value):
                    text = "NA"
                    color = "black"
                else:
                    text = fmt.format(float(value))
                    frac = 0.5 if np.isclose(vmax, vmin) else (float(value) - vmin) / (vmax - vmin)
                    color = "black" if frac > 0.55 else "white"
                ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.01)
        cbar.ax.tick_params(labelsize=9)

    # Hide any unused subplot slots.
    for idx in range(n_metrics, nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")

    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _summarize_numeric(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _subject_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in run_rows:
        rows_by_subject.setdefault(str(row["subject_id"]), []).append(row)

    subject_rows: list[dict[str, Any]] = []
    metric_keys = [
        "jaccard",
        "pct_reference_covered_by_coreg_mask",
        "pct_coreg_mask_within_reference",
        "com_distance_mm",
        "transform_translation_norm_mm",
        "transform_rotation_deg",
    ]
    for subject, items in sorted(rows_by_subject.items()):
        out: dict[str, Any] = {"subject_id": subject, "n_runs": len(items)}
        for key in metric_keys:
            vals = np.asarray([_safe_float(x.get(key, float("nan"))) for x in items], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[f"{key}_mean"] = float(np.mean(vals))
                out[f"{key}_min"] = float(np.min(vals))
                out[f"{key}_std"] = float(np.std(vals))
            else:
                out[f"{key}_mean"] = float("nan")
                out[f"{key}_min"] = float("nan")
                out[f"{key}_std"] = float("nan")
        subject_rows.append(out)
    return subject_rows


def run_registration_qc(
    manifest: str | Path | Sequence[RegistrationQcRow | Mapping[str, Any]],
    out_dir: str | Path,
    config: RegistrationQcConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RegistrationQcConfig()
    rows = _load_manifest_rows(manifest)
    out_root = Path(out_dir)
    _ensure_dir(out_root)
    apng_root = out_root / cfg.apng_subdir
    if cfg.apng_enabled:
        _ensure_dir(apng_root)

    run_metrics: list[dict[str, Any]] = []
    apng_created = 0
    auto_resample_applied_runs = 0

    for row in rows:
        ref_img_path = _to_path(row.reference_img_path, row, "reference_img_path")
        coreg_img_path = _to_path(row.coreg_img_path, row, "coreg_img_path")
        ref_mask_path = _to_path(row.reference_mask_path, row, "reference_mask_path")
        coreg_mask_path = _to_path(row.coreg_mask_path, row, "coreg_mask_path")
        xfm_path = _to_path(row.affine_transform_path, row, "affine_transform_path") if row.affine_transform_path else None
        seg_path = _to_path(row.segmentation_mask_path, row, "segmentation_mask_path") if row.segmentation_mask_path else None

        assert ref_img_path is not None and coreg_img_path is not None
        assert ref_mask_path is not None and coreg_mask_path is not None

        ref_img = _load_img(ref_img_path)
        coreg_img = None
        coreg_img_resampled = False
        if cfg.apng_enabled:
            coreg_img_raw = _load_img(coreg_img_path)
            coreg_img, coreg_img_resampled = _align_to_reference_grid(
                coreg_img_raw,
                ref_img,
                interpolation_order=1,
                field_name="coreg_img_path",
                config=cfg,
                row=row,
            )

        ref_mask_img_raw, _ref_mask_raw = _load_binary_mask(ref_mask_path, row, "reference_mask_path")
        ref_mask_img, ref_mask_resampled = _align_to_reference_grid(
            ref_mask_img_raw,
            ref_img,
            interpolation_order=0,
            field_name="reference_mask_path",
            config=cfg,
            row=row,
        )
        ref_mask = np.asarray(ref_mask_img.dataobj) > 0
        if not np.any(ref_mask):
            raise ValueError(
                f"Reference mask became empty after alignment for subject={row.subject_id} task={row.task} run={row.run_label}."
            )

        coreg_mask_img_raw, _coreg_mask_raw = _load_binary_mask(coreg_mask_path, row, "coreg_mask_path")
        coreg_mask_img, coreg_mask_resampled = _align_to_reference_grid(
            coreg_mask_img_raw,
            ref_img,
            interpolation_order=0,
            field_name="coreg_mask_path",
            config=cfg,
            row=row,
        )
        coreg_mask = np.asarray(coreg_mask_img.dataobj) > 0
        if not np.any(coreg_mask):
            raise ValueError(
                f"Coreg mask became empty after alignment for subject={row.subject_id} task={row.task} run={row.run_label}."
            )

        seg_mask = None
        seg_resampled = False
        if seg_path is not None:
            seg_img_raw = _load_nonempty_image(seg_path, row, "segmentation_mask_path")
            seg_img, seg_resampled = _align_to_reference_grid(
                seg_img_raw,
                ref_img,
                interpolation_order=0,
                field_name="segmentation_mask_path",
                config=cfg,
                row=row,
            )
            seg_mask = np.asarray(seg_img.dataobj)
            if not np.any(np.isfinite(seg_mask) & (seg_mask > 0)):
                seg_mask = None

        any_resampled = bool(coreg_img_resampled or ref_mask_resampled or coreg_mask_resampled or seg_resampled)
        if any_resampled:
            auto_resample_applied_runs += 1

        inter = int(np.logical_and(ref_mask, coreg_mask).sum())
        union = int(np.logical_or(ref_mask, coreg_mask).sum())
        ref_vox = int(ref_mask.sum())
        coreg_vox = int(coreg_mask.sum())
        com_ref = _center_of_mass_world(ref_mask, np.asarray(ref_img.affine, dtype=np.float64))
        com_coreg = _center_of_mass_world(coreg_mask, np.asarray(ref_img.affine, dtype=np.float64))
        com_dist = float(np.linalg.norm(com_ref - com_coreg)) if (com_ref is not None and com_coreg is not None) else float("nan")

        metric_row: dict[str, Any] = {
            "subject_id": row.subject_id,
            "task": row.task,
            "run_label": row.run_label,
            "source_label": row.source_label or "",
            "reference_img_path": str(ref_img_path),
            "coreg_img_path": str(coreg_img_path),
            "reference_mask_path": str(ref_mask_path),
            "coreg_mask_path": str(coreg_mask_path),
            "affine_transform_path": str(xfm_path) if xfm_path is not None else "",
            "segmentation_mask_path": str(seg_path) if seg_path is not None else "",
            "auto_resampled_to_reference": int(any_resampled),
            "coreg_img_resampled": int(coreg_img_resampled),
            "reference_mask_resampled": int(ref_mask_resampled),
            "coreg_mask_resampled": int(coreg_mask_resampled),
            "segmentation_mask_resampled": int(seg_resampled),
            "n_reference_mask_voxels": ref_vox,
            "n_coreg_mask_voxels": coreg_vox,
            "n_intersection_voxels": inter,
            "n_union_voxels": union,
            "jaccard": _jaccard(ref_mask, coreg_mask),
            "pct_reference_covered_by_coreg_mask": (inter / ref_vox * 100.0) if ref_vox > 0 else float("nan"),
            "pct_coreg_mask_within_reference": (inter / coreg_vox * 100.0) if coreg_vox > 0 else float("nan"),
            "com_distance_mm": com_dist,
        }
        metric_row.update(_compute_affine_metrics(xfm_path))
        run_metrics.append(metric_row)

        if cfg.apng_enabled:
            if not PIL_AVAILABLE:
                raise RuntimeError("Pillow is required for APNG output but is not installed.")
            if coreg_img is None:
                raise RuntimeError("Internal error: APNG requested but coreg image was not loaded.")
            ref_vol = np.asarray(ref_img.dataobj, dtype=np.float32)
            coreg_vol = np.asarray(coreg_img.dataobj, dtype=np.float32)
            content_mask = np.logical_or(ref_mask, coreg_mask)
            title = f"{row.subject_id} | task-{row.task}" + (f" | {row.run_label}" if row.run_label else "") + " | coreg vs reference"
            frame_coreg = _render_mosaic_image(coreg_vol, np.asarray(ref_img.affine), content_mask, seg_mask, title, cfg)
            frame_ref = _render_mosaic_image(ref_vol, np.asarray(ref_img.affine), content_mask, seg_mask, title, cfg)
            apng_name = f"{row.subject_id}__task-{row.task}" + (f"__{row.run_label}" if row.run_label else "") + ".png"
            if _build_apng(frame_coreg, frame_ref, apng_root / apng_name, int(cfg.apng_duration_ms)):
                apng_created += 1

    run_fields = [
        "subject_id",
        "task",
        "run_label",
        "source_label",
        "reference_img_path",
        "coreg_img_path",
        "reference_mask_path",
        "coreg_mask_path",
        "affine_transform_path",
        "segmentation_mask_path",
        "auto_resampled_to_reference",
        "coreg_img_resampled",
        "reference_mask_resampled",
        "coreg_mask_resampled",
        "segmentation_mask_resampled",
        "n_reference_mask_voxels",
        "n_coreg_mask_voxels",
        "n_intersection_voxels",
        "n_union_voxels",
        "jaccard",
        "pct_reference_covered_by_coreg_mask",
        "pct_coreg_mask_within_reference",
        "com_distance_mm",
        "transform_translation_norm_mm",
        "transform_rotation_deg",
    ]
    run_metrics_path = out_root / "run_metrics.csv"
    _write_csv(run_metrics_path, run_metrics, run_fields)

    subject_rows = _subject_summary(run_metrics)
    subject_fields = [
        "subject_id",
        "n_runs",
        "jaccard_mean",
        "jaccard_min",
        "jaccard_std",
        "pct_reference_covered_by_coreg_mask_mean",
        "pct_reference_covered_by_coreg_mask_min",
        "pct_reference_covered_by_coreg_mask_std",
        "pct_coreg_mask_within_reference_mean",
        "pct_coreg_mask_within_reference_min",
        "pct_coreg_mask_within_reference_std",
        "com_distance_mm_mean",
        "com_distance_mm_min",
        "com_distance_mm_std",
        "transform_translation_norm_mm_mean",
        "transform_translation_norm_mm_min",
        "transform_translation_norm_mm_std",
        "transform_rotation_deg_mean",
        "transform_rotation_deg_min",
        "transform_rotation_deg_std",
    ]
    subject_metrics_path = out_root / "subject_metrics.csv"
    _write_csv(subject_metrics_path, subject_rows, subject_fields)

    overview_heatmap_path = out_root / cfg.overview_heatmap_name
    if cfg.write_overview_heatmap:
        _write_overview_heatmap(run_metrics, overview_heatmap_path, cfg)

    summary = {
        "timestamp_utc": _now_iso(),
        "n_runs": len(run_metrics),
        "n_subjects": len({str(r["subject_id"]) for r in run_metrics}),
        "manifest_rows": len(rows),
        "auto_resample_applied_runs": int(auto_resample_applied_runs),
        "apng_enabled": bool(cfg.apng_enabled),
        "apng_created": int(apng_created),
        "outputs": {
            "run_metrics_csv": str(run_metrics_path),
            "subject_metrics_csv": str(subject_metrics_path),
            "qc_summary_json": str(out_root / "qc_summary.json"),
            "overview_heatmap_png": str(overview_heatmap_path) if cfg.write_overview_heatmap else "",
            "apng_dir": str(apng_root) if cfg.apng_enabled else "",
        },
        "metric_summary": {
            "jaccard": _summarize_numeric([_safe_float(x.get("jaccard", float("nan"))) for x in run_metrics]),
            "pct_reference_covered_by_coreg_mask": _summarize_numeric(
                [_safe_float(x.get("pct_reference_covered_by_coreg_mask", float("nan"))) for x in run_metrics]
            ),
            "pct_coreg_mask_within_reference": _summarize_numeric(
                [_safe_float(x.get("pct_coreg_mask_within_reference", float("nan"))) for x in run_metrics]
            ),
            "com_distance_mm": _summarize_numeric([_safe_float(x.get("com_distance_mm", float("nan"))) for x in run_metrics]),
            "transform_translation_norm_mm": _summarize_numeric(
                [_safe_float(x.get("transform_translation_norm_mm", float("nan"))) for x in run_metrics]
            ),
            "transform_rotation_deg": _summarize_numeric(
                [_safe_float(x.get("transform_rotation_deg", float("nan"))) for x in run_metrics]
            ),
        },
        "worst_runs_by_jaccard": [
            {
                "subject_id": x["subject_id"],
                "task": x["task"],
                "run_label": x["run_label"],
                "jaccard": _safe_float(x.get("jaccard", float("nan"))),
            }
            for x in sorted(
                run_metrics,
                key=lambda r: _safe_float(r.get("jaccard", float("nan")))
                if np.isfinite(_safe_float(r.get("jaccard", float("nan"))))
                else 999.0,
            )[:20]
        ],
        "config": asdict(cfg),
    }
    summary_path = out_root / "qc_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest-driven registration QC for aligned reference/coreg image pairs.")
    parser.add_argument("--manifest-csv", required=True, type=Path, help="Path to manifest CSV.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--skip-apng", action="store_true", help="Disable APNG output.")
    parser.add_argument("--apng-duration-ms", type=int, default=1200, help="Frame duration per APNG frame in milliseconds.")
    parser.add_argument("--slices", type=int, default=7, help="Slices per plane in mosaic.")
    parser.add_argument("--crop-margin-px", type=int, default=8, help="Per-plane crop margin in displayed pixels.")
    parser.add_argument(
        "--apng-tile-scale",
        type=int,
        default=1,
        help="Integer scale factor applied after rendering each APNG tile. Prefer 1 for native/reference-grid quality.",
    )
    parser.add_argument("--font-size", type=int, default=18, help="Base font size for APNG overlays.")
    parser.add_argument("--no-overview-heatmap", action="store_true", help="Disable overview run-by-subject heatmap.")
    parser.add_argument(
        "--overview-heatmap-name",
        default="overview_heatmap.png",
        help="Filename for overview heatmap (written under out-dir).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = RegistrationQcConfig(
        apng_enabled=not bool(args.skip_apng),
        apng_duration_ms=int(args.apng_duration_ms),
        n_slices=int(args.slices),
        crop_margin_px=int(args.crop_margin_px),
        apng_tile_scale=int(args.apng_tile_scale),
        font_size=int(args.font_size),
        write_overview_heatmap=not bool(args.no_overview_heatmap),
        overview_heatmap_name=str(args.overview_heatmap_name),
    )
    summary = run_registration_qc(args.manifest_csv, args.out_dir, config)
    print(f"Wrote run metrics: {summary['outputs']['run_metrics_csv']}")
    print(f"Wrote subject metrics: {summary['outputs']['subject_metrics_csv']}")
    print(f"Wrote summary: {summary['outputs']['qc_summary_json']}")
    if config.write_overview_heatmap:
        print(f"Wrote overview heatmap: {summary['outputs']['overview_heatmap_png']}")
    if config.apng_enabled:
        print(f"APNG output dir: {summary['outputs']['apng_dir']}")


__all__ = [
    "RegistrationQcRow",
    "RegistrationQcConfig",
    "run_registration_qc",
    "main",
]


if __name__ == "__main__":
    main()
