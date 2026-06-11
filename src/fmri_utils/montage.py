from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm
from scipy.stats import norm


SpatialImage = nib.spatialimages.SpatialImage


@dataclass(frozen=True)
class SubjectMap:
    subject_id: str
    effect_img: SpatialImage
    include_mask_img: SpatialImage | None = None


@dataclass(frozen=True)
class OutlineRegion:
    region_id: int
    label: str
    color: str


@dataclass(frozen=True)
class OutlineSpec:
    label_img: SpatialImage
    regions: Sequence[OutlineRegion]
    alpha: float = 1.0
    linewidth: float = 0.7


@dataclass(frozen=True)
class MontageConfig:
    mode: Literal["solid", "thresholded"] = "solid"
    tail: Literal["two_sided", "positive", "negative"] = "two_sided"
    p_thresholds: tuple[float, ...] = (0.05, 0.01, 0.005, 0.001)
    slice_indices: tuple[int, ...] | None = None
    slice_labels: tuple[int, ...] | None = None

    grid_rows: int = 4
    grid_cols: int = 6
    dpi: int = 240
    panel_scale: float = 4.0
    interpad_px: float = 1.0
    crop_pad_voxels: int = 2

    cbar_width_px: float = 20.0
    cbar_pad_px: float = 18.0
    title_height_px: float = 94.0
    legend_height_px: float | None = None
    legend_cols: int = 2
    margin_left_px: int = 2
    margin_right_px: int = 300
    margin_top_px: int = 10
    margin_bottom_px: int = 4

    overlay_alpha: float = 1.0
    # When True in solid mode, alpha increases with |effect| so values near 0
    # stay transparent and the anatomical underlay remains visible.
    solid_alpha_by_magnitude: bool = False
    solid_alpha_gamma: float = 1.0
    darken_outside_include_mask: bool = True
    outside_include_darken_factor: float = 0.58

    filename_prefix: str = "map"
    title_prefix: str = ""
    cbar_label: str = "value"
    show_legend: bool = True
    solid_title_suffix: str = "solid overlay"

    subject_fontsize: float = 11.0
    title_fontsize: float = 14.0
    legend_fontsize: float = 7.6
    cbar_tick_fontsize: float = 9.8
    cbar_ticks: tuple[float, ...] | None = None

    solid_cmap: str | Colormap = "viridis"
    solid_norm: Normalize | None = None
    solid_center_zero: bool = False
    threshold_cmap: str | Colormap = "bwr"

    reserve_missing_subject: str = ""
    reserve_panel_index: int = -1


def _validate_spatial_image(name: str, img: SpatialImage) -> None:
    if not isinstance(img, SpatialImage):
        raise TypeError(f"{name} must be a nibabel spatial image, got {type(img).__name__}.")


def _validate_alignment(reference_img: SpatialImage, name: str, other_img: SpatialImage, affine_tol: float) -> None:
    if reference_img.shape != other_img.shape:
        raise ValueError(
            f"{name} shape mismatch: expected {reference_img.shape}, got {other_img.shape}. "
            "Inputs must already be aligned."
        )
    if not np.allclose(reference_img.affine, other_img.affine, atol=affine_tol, rtol=0.0):
        raise ValueError(
            f"{name} affine mismatch relative to reference (tolerance={affine_tol}). "
            "Inputs must already be aligned."
        )


def _parse_thresholds(p_thresholds: Sequence[float]) -> tuple[list[float], list[float]]:
    vals = sorted(set(float(p) for p in p_thresholds), reverse=True)
    if not vals:
        raise ValueError("p_thresholds must contain at least one value.")
    for p in vals:
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"Invalid p-threshold {p}; expected 0 < p < 1.")
    return vals, [float(norm.isf(p)) for p in vals]


def _threshold_overlay(effect: np.ndarray, z_thr: float, tail: str) -> np.ndarray:
    if tail == "positive":
        return np.where(effect > z_thr, effect, np.nan)
    if tail == "negative":
        return np.where(effect < -z_thr, effect, np.nan)
    return np.where(np.abs(effect) > z_thr, effect, np.nan)


def _slug_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _derive_slices(config: MontageConfig, n_slices: int) -> list[int]:
    if config.slice_indices is None:
        return list(range(n_slices))
    out = [int(z) for z in config.slice_indices]
    if not out:
        raise ValueError("slice_indices was provided but empty.")
    for z in out:
        if z < 0 or z >= n_slices:
            raise ValueError(f"slice index {z} out of range [0, {n_slices - 1}].")
    if config.slice_labels is not None and len(config.slice_labels) != len(out):
        raise ValueError("slice_labels must have the same length as slice_indices.")
    return out


def _compute_crop_bounds(reference_data: np.ndarray, crop_pad_voxels: int) -> tuple[int, int, int, int]:
    brain_mask = np.isfinite(reference_data) & (reference_data > 0)
    if not np.any(brain_mask):
        brain_mask = np.isfinite(reference_data)
    x_idx = np.where(brain_mask.any(axis=(1, 2)))[0]
    y_idx = np.where(brain_mask.any(axis=(0, 2)))[0]
    if x_idx.size == 0 or y_idx.size == 0:
        raise RuntimeError("Failed to derive crop bounds from reference image.")
    x0 = max(0, int(x_idx[0]) - int(crop_pad_voxels))
    x1 = min(reference_data.shape[0] - 1, int(x_idx[-1]) + int(crop_pad_voxels))
    y0 = max(0, int(y_idx[0]) - int(crop_pad_voxels))
    y1 = min(reference_data.shape[1] - 1, int(y_idx[-1]) + int(crop_pad_voxels))
    return x0, x1, y0, y1


def _build_distinct_colors(n: int) -> list[str]:
    palettes = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors: list[str] = []
    for cmap in palettes:
        for i in np.linspace(0.0, 1.0, 20, endpoint=False):
            rgba = cmap(i)
            colors.append(
                "#{:02x}{:02x}{:02x}".format(
                    int(round(rgba[0] * 255)),
                    int(round(rgba[1] * 255)),
                    int(round(rgba[2] * 255)),
                )
            )
    if n > len(colors):
        extra = n - len(colors)
        for i in np.linspace(0.0, 1.0, extra, endpoint=False):
            rgba = plt.cm.hsv(i)
            colors.append(
                "#{:02x}{:02x}{:02x}".format(
                    int(round(rgba[0] * 255)),
                    int(round(rgba[1] * 255)),
                    int(round(rgba[2] * 255)),
                )
            )
    return colors[:n]


def _draw_legend_grid(
    ax,
    regions: Sequence[OutlineRegion],
    ncols: int,
    fontsize: float,
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    n = len(regions)
    if n == 0:
        return
    ncols = max(1, min(int(ncols), n))
    nrows = int(math.ceil(n / ncols))
    x_pad = 0.01
    y_pad = 0.03
    cell_w = (1.0 - 2.0 * x_pad) / ncols
    cell_h = (1.0 - 2.0 * y_pad) / nrows
    for k, region in enumerate(regions):
        r = k // ncols
        c = k % ncols
        x_left = x_pad + c * cell_w
        y_mid = 1.0 - y_pad - (r + 0.5) * cell_h
        x0 = x_left + 0.02 * cell_w
        x1 = x_left + 0.13 * cell_w
        xt = x_left + 0.16 * cell_w
        ax.plot([x0, x1], [y_mid, y_mid], color=region.color, linewidth=1.9, solid_capstyle="round", clip_on=False)
        ax.text(xt, y_mid, region.label, fontsize=fontsize, color="black", ha="left", va="center", clip_on=False)


def _compute_background_with_darken(
    background_slice: np.ndarray,
    include_slice: np.ndarray,
    darken_outside_include_mask: bool,
    darken_factor: float,
) -> np.ndarray:
    out = background_slice.copy()
    if not darken_outside_include_mask:
        return out
    brain = np.isfinite(background_slice) & (background_slice > 0)
    outside = brain & (~include_slice.astype(bool))
    out[outside] = out[outside] * float(np.clip(darken_factor, 0.0, 1.0))
    return out


def _resolve_solid_norm(config: MontageConfig, overlays: Sequence[np.ndarray]) -> Normalize:
    if config.solid_norm is not None:
        return config.solid_norm
    finite_vals: list[np.ndarray] = []
    for arr in overlays:
        vals = arr[np.isfinite(arr)]
        if vals.size:
            finite_vals.append(vals.astype(np.float32, copy=False))
    if not finite_vals:
        return Normalize(vmin=0.0, vmax=1.0)
    pooled = np.concatenate(finite_vals, axis=0)
    if config.solid_center_zero:
        vmax = float(np.nanmax(np.abs(pooled)))
        vmax = max(vmax, 1e-4)
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmin = float(np.nanpercentile(pooled, 2.0))
    vmax = float(np.nanpercentile(pooled, 98.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(pooled))
        vmax = float(np.nanmax(pooled))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def render_axial_subject_montage_series(
    subject_maps: Sequence[SubjectMap],
    reference_img: SpatialImage,
    out_dir: Path | str,
    config: MontageConfig,
    outline_spec: OutlineSpec | None = None,
    *,
    affine_tol: float = 1e-5,
) -> list[Path]:
    _validate_spatial_image("reference_img", reference_img)
    if not subject_maps:
        raise ValueError("subject_maps must be non-empty.")

    if config.mode not in ("solid", "thresholded"):
        raise ValueError(f"Unsupported mode: {config.mode}")
    if config.tail not in ("two_sided", "positive", "negative"):
        raise ValueError(f"Unsupported tail: {config.tail}")

    reference_data = np.asarray(reference_img.get_fdata(dtype=np.float32))
    n_slices = reference_data.shape[2]
    slices = _derive_slices(config, n_slices)

    if outline_spec is not None:
        _validate_spatial_image("outline_spec.label_img", outline_spec.label_img)
        _validate_alignment(reference_img, "outline_spec.label_img", outline_spec.label_img, affine_tol)
        outline_data = np.asarray(outline_spec.label_img.get_fdata(), dtype=np.int32)
        region_lookup = {int(r.region_id): r for r in outline_spec.regions}
    else:
        outline_data = None
        region_lookup = {}

    subjects: list[str] = []
    effect_stack: list[np.ndarray] = []
    include_stack: list[np.ndarray] = []
    for idx, subject_map in enumerate(subject_maps):
        _validate_spatial_image(f"subject_maps[{idx}].effect_img", subject_map.effect_img)
        _validate_alignment(reference_img, f"subject_maps[{idx}].effect_img", subject_map.effect_img, affine_tol)
        effect_data = np.asarray(subject_map.effect_img.get_fdata(dtype=np.float32))
        finite = np.isfinite(effect_data)
        effect_stack.append(np.where(finite, effect_data, np.nan).astype(np.float32, copy=False))

        if subject_map.include_mask_img is not None:
            _validate_spatial_image(f"subject_maps[{idx}].include_mask_img", subject_map.include_mask_img)
            _validate_alignment(reference_img, f"subject_maps[{idx}].include_mask_img", subject_map.include_mask_img, affine_tol)
            include_data = np.asarray(subject_map.include_mask_img.get_fdata(), dtype=np.float32) > 0
        else:
            include_data = finite
        include_stack.append(include_data.astype(bool, copy=False))
        subjects.append(str(subject_map.subject_id))

    n_panels = int(config.grid_rows) * int(config.grid_cols)
    if len(subjects) > n_panels:
        raise ValueError(f"Need at least {len(subjects)} panels, but grid provides {n_panels}.")

    reserved_blank_index = -1
    reserve_subject = config.reserve_missing_subject.strip()
    reserve_index = int(config.reserve_panel_index)
    if reserve_subject and reserve_index >= 0 and reserve_subject not in subjects and reserve_index <= len(subjects):
        subjects.insert(reserve_index, reserve_subject)
        effect_stack.insert(reserve_index, np.full(reference_data.shape, np.nan, dtype=np.float32))
        include_stack.insert(reserve_index, np.zeros(reference_data.shape, dtype=bool))
        reserved_blank_index = reserve_index

    x0, x1, y0, y1 = _compute_crop_bounds(reference_data, config.crop_pad_voxels)
    bg_finite = reference_data[np.isfinite(reference_data)]
    bg_vmin = float(np.percentile(bg_finite, 2.0))
    bg_vmax = float(np.percentile(bg_finite, 98.0))

    x_span = x1 - x0 + 1
    y_span = y1 - y0 + 1
    panel_w = int(round(x_span * float(config.panel_scale)))
    panel_h = int(round(y_span * float(config.panel_scale)))
    interpad_px = float(config.interpad_px)
    grid_w = int(round(int(config.grid_cols) * panel_w + (int(config.grid_cols) - 1) * interpad_px))
    grid_h = int(round(int(config.grid_rows) * panel_h + (int(config.grid_rows) - 1) * interpad_px))

    legend_cols = max(1, int(config.legend_cols))
    legend_rows = int(math.ceil(max(1, len(region_lookup)) / legend_cols))
    auto_legend_height = int(math.ceil(legend_rows * (float(config.legend_fontsize) * 5.8) + 30))
    legend_height_px = int(round(config.legend_height_px)) if config.legend_height_px is not None else auto_legend_height
    title_height_px = max(float(config.title_height_px), 40.0)

    fig_w_px = int(
        round(
            int(config.margin_left_px)
            + grid_w
            + float(config.cbar_pad_px)
            + float(config.cbar_width_px)
            + int(config.margin_right_px)
        )
    )
    fig_h_px = int(round(int(config.margin_bottom_px) + legend_height_px + grid_h + title_height_px + int(config.margin_top_px)))
    fig_w = fig_w_px / float(config.dpi)
    fig_h = fig_h_px / float(config.dpi)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    render_jobs: list[dict[str, object]] = []
    if config.mode == "thresholded":
        p_vals, z_vals = _parse_thresholds(config.p_thresholds)
        for p_thr, z_thr in zip(p_vals, z_vals):
            overlays = [_threshold_overlay(arr, z_thr, config.tail) for arr in effect_stack]
            finite_abs = [np.abs(o[np.isfinite(o)]) for o in overlays if np.isfinite(o).any()]
            if finite_abs:
                pooled = np.concatenate(finite_abs, axis=0)
                vmax = max(float(np.percentile(pooled, 99.0)), z_thr + 0.5)
            else:
                vmax = z_thr + 0.5
            if len(p_vals) == 1:
                job_dir = out_root
            else:
                job_dir = out_root / f"p_{_slug_number(p_thr)}_z_{_slug_number(round(z_thr, 3))}"
            render_jobs.append(
                {
                    "job_dir": job_dir,
                    "overlays": overlays,
                    "cmap": config.threshold_cmap,
                    "norm": TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                    "ticks": None,
                    "label": config.cbar_label,
                    "title_suffix": f"p<{p_thr:g} (|effect|>{z_thr:.3f})",
                }
            )
    else:
        overlays = [np.where(np.isfinite(arr), arr, np.nan).astype(np.float32, copy=False) for arr in effect_stack]
        render_jobs.append(
            {
                "job_dir": out_root,
                "overlays": overlays,
                "cmap": config.solid_cmap,
                "norm": _resolve_solid_norm(config, overlays),
                "ticks": config.cbar_ticks,
                "label": config.cbar_label,
                "title_suffix": config.solid_title_suffix,
            }
        )

    created: list[Path] = []
    ordered_regions = list(outline_spec.regions) if outline_spec is not None else []
    for job in render_jobs:
        job_dir = Path(job["job_dir"])
        job_dir.mkdir(parents=True, exist_ok=True)
        overlays = job["overlays"]
        cmap = job["cmap"]
        norm_obj = job["norm"]
        ticks = job["ticks"]
        title_suffix = str(job["title_suffix"])

        for slice_pos, z_idx in enumerate(slices):
            z_label = int(config.slice_labels[slice_pos]) if config.slice_labels is not None else int(z_idx)
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(config.dpi), facecolor="white")
            for i in range(n_panels):
                row = i // int(config.grid_cols)
                col = i % int(config.grid_cols)
                x_px = int(config.margin_left_px) + col * (panel_w + interpad_px)
                y_px = int(config.margin_bottom_px) + legend_height_px + (int(config.grid_rows) - 1 - row) * (panel_h + interpad_px)
                ax = fig.add_axes([x_px / fig_w_px, y_px / fig_h_px, panel_w / fig_w_px, panel_h / fig_h_px])

                if i >= len(subjects) or i == reserved_blank_index:
                    ax.axis("off")
                    continue

                bg = reference_data[x0 : x1 + 1, y0 : y1 + 1, z_idx].T
                include_slice = include_stack[i][x0 : x1 + 1, y0 : y1 + 1, z_idx].T
                bg_plot = _compute_background_with_darken(
                    bg,
                    include_slice,
                    config.darken_outside_include_mask,
                    config.outside_include_darken_factor,
                )
                ov = overlays[i][x0 : x1 + 1, y0 : y1 + 1, z_idx].T

                ax.imshow(bg_plot, cmap="gray", vmin=bg_vmin, vmax=bg_vmax, origin="lower", interpolation="none")
                ov_masked = np.ma.masked_invalid(ov)
                if config.mode == "solid" and bool(config.solid_alpha_by_magnitude):
                    vmax_abs = max(abs(float(norm_obj.vmin)), abs(float(norm_obj.vmax)))
                    if not np.isfinite(vmax_abs) or vmax_abs <= 0:
                        vmax_abs = 1.0
                    alpha_map = np.abs(ov.astype(np.float32, copy=False)) / float(vmax_abs)
                    alpha_map = np.clip(alpha_map, 0.0, 1.0)
                    gamma = max(float(config.solid_alpha_gamma), 1e-6)
                    alpha_map = np.power(alpha_map, gamma, dtype=np.float32)
                    alpha_map[~np.isfinite(ov)] = 0.0
                    alpha_use = alpha_map * float(np.clip(config.overlay_alpha, 0.0, 1.0))
                    ax.imshow(
                        ov_masked,
                        cmap=cmap,
                        norm=norm_obj,
                        alpha=alpha_use,
                        origin="lower",
                        interpolation="none",
                    )
                else:
                    ax.imshow(
                        ov_masked,
                        cmap=cmap,
                        norm=norm_obj,
                        alpha=float(config.overlay_alpha),
                        origin="lower",
                        interpolation="none",
                    )

                if outline_data is not None and ordered_regions:
                    label_slice = outline_data[x0 : x1 + 1, y0 : y1 + 1, z_idx].T
                    for region in ordered_regions:
                        roi_mask = (label_slice == int(region.region_id)).astype(np.float32)
                        if not np.any(roi_mask):
                            continue
                        ax.contour(
                            roi_mask,
                            levels=[0.5],
                            colors=[region.color],
                            linewidths=float(outline_spec.linewidth) if outline_spec is not None else 0.7,
                            alpha=float(outline_spec.alpha) if outline_spec is not None else 1.0,
                            antialiased=False,
                            corner_mask=False,
                        )

                ax.text(
                    0.01,
                    0.99,
                    subjects[i],
                    transform=ax.transAxes,
                    fontsize=float(config.subject_fontsize),
                    color="white",
                    ha="left",
                    va="top",
                    path_effects=[pe.withStroke(linewidth=1.15, foreground="black")],
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)

            cbar_x = (int(config.margin_left_px) + grid_w + float(config.cbar_pad_px)) / fig_w_px
            cbar_y = (int(config.margin_bottom_px) + legend_height_px) / fig_h_px
            cbar_w = float(config.cbar_width_px) / fig_w_px
            cbar_h = grid_h / fig_h_px
            cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
            sm = ScalarMappable(cmap=cmap, norm=norm_obj)
            sm.set_array(np.array([0.0, 1.0], dtype=np.float32))
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=float(config.cbar_tick_fontsize), colors="black", pad=4.5)
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.outline.set_edgecolor("black")
            cbar.ax.set_ylabel(str(job["label"]), fontsize=max(7.0, float(config.cbar_tick_fontsize) - 1.0), color="black", labelpad=10.0)
            if ticks is not None:
                cbar.set_ticks(np.asarray(ticks, dtype=np.float32))

            ax_title = fig.add_axes(
                [
                    int(config.margin_left_px) / fig_w_px,
                    (int(config.margin_bottom_px) + legend_height_px + grid_h) / fig_h_px,
                    grid_w / fig_w_px,
                    title_height_px / fig_h_px,
                ]
            )
            ax_title.axis("off")
            if config.title_prefix.strip():
                title_text = f"{config.title_prefix.strip()}\naxial z={z_label} | {title_suffix}"
            else:
                title_text = f"axial z={z_label} | {title_suffix}"
            ax_title.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=float(config.title_fontsize), color="black")

            if config.show_legend and ordered_regions:
                ax_legend = fig.add_axes(
                    [
                        int(config.margin_left_px) / fig_w_px,
                        int(config.margin_bottom_px) / fig_h_px,
                        grid_w / fig_w_px,
                        legend_height_px / fig_h_px,
                    ]
                )
                ax_legend.axis("off")
                _draw_legend_grid(ax_legend, ordered_regions, legend_cols, float(config.legend_fontsize))

            out_png = job_dir / f"{config.filename_prefix}_axial_z_{z_label:03d}.png"
            fig.savefig(out_png, dpi=int(config.dpi), facecolor="white")
            plt.close(fig)
            created.append(out_png)

    return created


def build_auto_outline_regions(label_img: SpatialImage) -> list[OutlineRegion]:
    _validate_spatial_image("label_img", label_img)
    data = np.asarray(label_img.get_fdata(), dtype=np.int32)
    ids = sorted(int(v) for v in np.unique(data) if int(v) > 0)
    colors = _build_distinct_colors(len(ids))
    return [OutlineRegion(region_id=i, label=f"ROI_{i}", color=colors[k]) for k, i in enumerate(ids)]


__all__ = [
    "SubjectMap",
    "OutlineRegion",
    "OutlineSpec",
    "MontageConfig",
    "render_axial_subject_montage_series",
    "build_auto_outline_regions",
]
