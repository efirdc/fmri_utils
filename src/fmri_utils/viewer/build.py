"""Building a viewer: spec in, a directory you can serve out.

The output is static. There is no server, no database and no session state, so
a viewer costs nothing to keep up and a link to it shows everyone the same
thing. Copy the directory anywhere that serves files over HTTP.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from . import assets
from .spec import Atlas, CoordinateMaps, Endpoint, RegionRow, Report, ViewerSpec, _matrix

PAGE = Path(__file__).parent / "resources" / "index.html"


def build_viewer(spec: ViewerSpec, output_root: Path, quiet: bool = False) -> Path:
    """Write the page, the manifest and every file the manifest points at.

    Returns the directory. Serve it, or copy it to wherever it is hosted.
    """
    spec.check()
    underlays.clear()
    output_root = Path(output_root)
    data = output_root / "data"
    data.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    manifest: dict = {
        "about": {
            "title": spec.about.title,
            "kicker": spec.about.kicker,
            "lede": spec.about.lede,
            "footnote": spec.about.footnote,
        },
        "features": {
            "surface": spec.features.surface,
            "regions": spec.features.regions,
            "montage": spec.features.montage,
            "subjectSpace": spec.features.subject_space,
            "fineUnderlay": spec.features.fine_underlay,
            "smoothShading": spec.features.smooth_shading,
        },
        "reports": [],
    }

    say("underlay")
    manifest["template"] = _relative(
        assets.write_underlay(spec.template, data / "template.nii.gz") and data / "template.nii.gz",
        output_root,
    )
    manifest["template_label"] = Path(spec.template).name
    if spec.template_coarse:
        assets.write_underlay(spec.template_coarse, data / "template_coarse.nii.gz")
        manifest["template_montage"] = _relative(data / "template_coarse.nii.gz", output_root)

    for report in spec.reports:
        say(f"report {report.id}")
        manifest["reports"].append(_build_report(report, spec, data, output_root, say))

    subject_space = _build_subject_space(spec, data, output_root, say)
    if subject_space:
        manifest["subject_space"] = subject_space

    if spec.atlases:
        say("atlases")
        manifest["atlas"] = _build_atlases(spec.atlases, data, output_root)

    if spec.region_info:
        manifest["region_info"] = dict(spec.region_info)

    if spec.cohorts:
        manifest["cohorts"] = [{"id": c.id, "label": c.label, "blurb": c.blurb} for c in spec.cohorts]
    if spec.excluded_subjects:
        manifest["excluded_subjects"] = dict(spec.excluded_subjects)
    if spec.template_space:
        manifest["template_space"] = spec.template_space

    if spec.surfaces:
        catalogue = Path(spec.surfaces) / "surfaces.json"
        if catalogue.exists():
            say("surfaces")
            manifest["surfaces"] = json.loads(catalogue.read_text(encoding="utf-8"))
            _copy_tree(Path(spec.surfaces) / "data" / "surfaces", data / "surfaces")
        else:
            say(f"no surfaces.json under {spec.surfaces}; skipping surfaces")

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copyfile(PAGE, output_root / "index.html")

    # A rebuild writes over what it needs and leaves the rest. That is fine
    # until the shape of the manifest changes, at which point the output holds
    # files nothing points at any more -- and a deploy is usually a copy, so
    # they travel. Naming them is enough; deleting other people's files out of
    # a directory they chose is not this function's business.
    for path in sorted(_orphans(manifest, output_root)):
        say(f"no longer referenced: {path}")

    say(f"viewer written to {output_root}")
    return output_root


def _orphans(manifest: dict, output_root: Path) -> list[str]:
    used = {manifest.get("template"), manifest.get("template_montage")}
    for report in manifest["reports"]:
        for endpoint in report["endpoints"]:
            for entry in endpoint["maps"]:
                used.add(entry["path"])
                used.add(entry.get("underlay"))
                used.update(image["path"] for image in entry.get("images", []))
                significance = entry.get("significance") or {}
                used.update(significance.get(mode) for mode in ("p", "q", "fwe"))
    space = manifest.get("subject_space") or {}
    for group in ("templates", "templates_2mm"):
        used.update((space.get(group) or {}).values())
    for record in (space.get("coords") or {}).values():
        used.update(side["path"] for side in record.values() if isinstance(side, dict))
    for entry in space.get("maps") or []:
        used.add(entry["path"])
    atlas = manifest.get("atlas") or {}
    for record in atlas.get("atlases") or []:
        used.add(record["mni"])
    for record in (atlas.get("subject") or {}).values():
        used.update(record.values())
    surfaces = manifest.get("surfaces") or {}
    for record in surfaces.values():
        for hemisphere in record.get("hemispheres", {}).values():
            used.add(hemisphere.get("mesh"))
            used.add(hemisphere.get("curvature"))
            used.add(hemisphere.get("sulc"))
            for geometry in hemisphere.get("geometries", {}).values():
                used.add(geometry.get("vertices"))
                used.add(geometry.get("face_mask"))
        for region in record.get("rois", {}).values():
            used.add(region.get("path"))
        for parcellation in record.get("parcellations", {}).values():
            for hemisphere in parcellation.get("hemispheres", {}).values():
                used.update(hemisphere.get(key) for key in
                            ("labels", "lines_index", "lines_weight", "lines_offsets"))
    used.discard(None)
    data = output_root / "data"
    present = {str(p.relative_to(output_root)).replace("\\", "/")
               for p in data.rglob("*") if p.is_file()}
    return sorted(present - used)


def _relative(path: Path, root: Path) -> str:
    return str(Path(path).relative_to(root)).replace("\\", "/")


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if item.is_dir():
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)


def _build_report(report: Report, spec: ViewerSpec, data: Path, root: Path, say) -> dict:
    endpoints = []
    for endpoint in report.endpoints:
        endpoints.append(_build_endpoint(report, endpoint, spec, data, root, say))
    return {
        "id": report.id,
        "label": report.label,
        "blurb": report.blurb,
        "endpoints": endpoints,
    }


def _file_stem(*parts: str) -> str:
    """A file name from labels, which may hold characters Windows refuses."""
    text = "_".join(part for part in parts if part)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def _curves(endpoint: Endpoint, entry, values: np.ndarray) -> dict | None:
    """p and q curves for a map that is a t (or r) statistic, else None."""
    dof = entry.degrees_of_freedom or endpoint.degrees_of_freedom.get(entry.subject)
    variant = next((v for v in endpoint.variants if v.id == entry.variant), None)
    if not dof or (variant is not None and not variant.analytic):
        return None
    if endpoint.statistic == "t":
        return assets.statistic_curves(values, int(dof))
    if endpoint.statistic == "r":
        return assets.correlation_curves(values, int(dof))
    return None


def _variant_control(control) -> dict:
    record = {"id": control.id, "label": control.label}
    if control.tip:
        record["tip"] = control.tip
    if control.is_toggle:
        record.update(kind="toggle", on=control.on, off=control.off)
    else:
        record["options"] = []
        for option in control.options:
            item = {"id": option.id, "label": option.label}
            if option.tip:
                item["tip"] = option.tip
            if option.unavailable:
                item["unavailable"] = option.unavailable
            record["options"].append(item)
    return record


# Underlays are shared: one T1 sits under eight runs' boldrefs, so each source
# is written once however many maps name it.
underlays: dict[str, str] = {}


def _underlay(source: Path, subject: str, data: Path, root: Path, written: dict) -> str:
    key = str(Path(source).resolve())
    if key not in written:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        target = data / "underlays" / subject / (_file_stem(Path(source).name.split(".")[0], digest) + ".nii.gz")
        assets.write_underlay(source, target, crop=True)
        written[key] = _relative(target, root)
    return written[key]


def _stack_image(image, subject: str, data: Path, root: Path) -> dict:
    """One image of a registration stack, written once however many maps use it."""
    path = _underlay(image.path, subject, data, root, underlays)
    values = np.asarray(nib.load(str(root / path)).get_fdata(dtype=np.float32))
    inside = values[values > 0]
    record = {"id": image.id, "label": image.label, "path": path, "frame": image.frame,
              "range": [0.0, float(np.percentile(inside, 99.5)) if inside.size else 1.0]}
    if image.affine is not None:
        record["affine"] = _matrix(image.affine)
    return record


def _content_corners(path: Path) -> np.ndarray | None:
    """World corners of an image's non-zero content, or None if it has none."""
    image = nib.load(str(path))
    values = np.asarray(image.dataobj)
    if values.ndim > 3:
        values = values[..., 0]
    inside = np.argwhere(values > 0)
    if not inside.size:
        return None
    low, high = inside.min(axis=0), inside.max(axis=0)
    corners = np.array([[i, j, k] for i in (low[0], high[0]) for j in (low[1], high[1])
                        for k in (low[2], high[2])], float)
    return corners @ image.affine[:3, :3].T + image.affine[:3, 3]


def _canvas(entry, spec: ViewerSpec, data: Path, root: Path) -> str:
    """A blank underlay spanning a stack's brain in both spaces.

    The page draws a stack itself, resampled into wherever the space slider
    is, so the underlay only gives the view its extent: the content of the
    subject's and the template's images (any "subject" or "template" frame
    image, and the subject's anatomy), with a margin for the warp between
    them. A boldref's field of view reaches well past the brain and is left
    out, so the view opens on the brain rather than on a box around the head.
    """
    sources = [image.path for image in entry.images if image.frame in ("subject", "template")]
    anatomy = spec.subject_space.templates.get(entry.subject) if spec.subject_space else None
    if anatomy:
        sources.append(anatomy)
    boxes = [box for box in (_content_corners(path) for path in sources) if box is not None]
    if not boxes:
        boxes = [_content_corners(spec.template)]
    points = np.concatenate(boxes)
    low, high = points.min(axis=0) - 8.0, points.max(axis=0) + 8.0
    key = f"canvas|{entry.subject}|{np.round(low).tolist()}|{np.round(high).tolist()}"
    if key not in underlays:
        shape = np.ceil((high - low) / 2.0).astype(int) + 1
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = low
        target = data / "underlays" / entry.subject / (_file_stem("canvas", hashlib.sha1(key.encode()).hexdigest()[:8]) + ".nii.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        blank = nib.Nifti1Image(np.zeros(tuple(shape), dtype=np.uint8), affine)
        blank.header.set_data_dtype(np.uint8)
        nib.save(blank, str(target))
        underlays[key] = _relative(target, root)
    return underlays[key]


def _build_endpoint(
    report: Report, endpoint: Endpoint, spec: ViewerSpec, data: Path, root: Path, say
) -> dict:
    features: list[dict] = []
    subjects: list[str] = []
    maps: list[dict] = []
    extremes: dict[str, list[float]] = {}
    own = endpoint.subject_display_ranges
    anatomy = endpoint.display == "anatomy"
    for entry in endpoint.maps:
        name = _file_stem(entry.feature or "map", entry.variant, entry.subject, entry.cohort) + ".nii.gz"
        target = data / report.id / endpoint.id / name
        if anatomy:
            # An image, not a statistic: stored as the underlays are (once,
            # however many maps show the same one -- a T1 under eight runs), and
            # windowed from black to its bright end rather than around zero.
            target = root / _underlay(entry.path, entry.subject, data, root, underlays)
            values = np.asarray(nib.load(str(target)).get_fdata(dtype=np.float32))
            inside = values[values > 0]
            window = [0.0, float(np.percentile(inside, 99.5)) if inside.size else 1.0]
        else:
            _, values = assets.write_map(entry.path, target)
            window = assets.percentile_range(values, spec.percentile)
        record = {
            "subject": entry.subject,
            "feature": entry.feature,
            "path": _relative(target, root),
            # The page reads this to decide how fine an underlay is worth
            # compositing onto: a 3 mm map on a 1 mm grid costs eight times the
            # blend texture and shows nothing more.
            "zooms": assets.voxel_size(entry.path),
            "range": window,
        }
        if entry.underlay:
            record["underlay"] = _underlay(entry.underlay, entry.subject, data, root, underlays)
        if entry.surface_frame:
            record["surface_frame"] = entry.surface_frame
        if entry.surface_affine is not None:
            record["surface_affine"] = _matrix(entry.surface_affine)
        if entry.warp:
            record["warp"] = True
        if entry.images:
            record["images"] = [_stack_image(image, entry.subject, data, root) for image in entry.images]
            if not entry.underlay:
                record["underlay"] = _canvas(entry, spec, data, root)
        if entry.variant:
            record["variant"] = entry.variant
        if entry.cohort:
            record["cohort"] = entry.cohort
        if entry.region_rows:
            record["region_rows"] = [_region_row(row) for row in entry.region_rows]
        significance = {}
        for mode, companion in (("p", entry.significance_p), ("q", entry.significance_q),
                                ("fwe", entry.significance_fwe)):
            if companion:
                sig_target = target.with_name(target.name.replace(".nii.gz", f"_neglog10{mode}.nii.gz"))
                assets.write_map(companion, sig_target)
                significance[mode] = _relative(sig_target, root)
        if significance:
            if entry.significance_label:
                significance["label"] = entry.significance_label
            if entry.significance_defaults:
                significance["defaults"] = {k: float(v) for k, v in entry.significance_defaults.items()}
            record["significance"] = significance
        curves = None if anatomy else _curves(endpoint, entry, values)
        if curves:
            record["thresholds"] = curves
        maps.append(record)
        # A subject with its own window does not set everyone else's.
        if entry.subject not in own:
            extremes.setdefault(entry.variant, []).append(record["range"][1])
        if entry.subject not in subjects:
            subjects.append(entry.subject)
        if entry.feature and entry.feature not in [f["id"] for f in features]:
            features.append({"id": entry.feature, "label": entry.feature})
        say(f"  {report.id}/{endpoint.id} {entry.subject} {entry.feature} {entry.variant}".rstrip())
    everyone = [value for values in extremes.values() for value in values]
    # A subject on its own scale (the group beside its participants) leads the
    # list, so it is what an endpoint opens on.
    subjects.sort(key=lambda subject: subject not in own)
    out = {
        "id": endpoint.id,
        "label": endpoint.label,
        "blurb": endpoint.blurb,
        "warp": endpoint.warp,
        "statistic": endpoint.statistic,
        "features": features or [{"id": "", "label": "—"}],
        "subjects": subjects,
        "range": list(endpoint.display_range) if endpoint.display_range
                 else [0.0, float(np.median(everyone)) if everyone else 1.0],
        "maps": maps,
    }
    if endpoint.display:
        out["display"] = endpoint.display
    if endpoint.outlines:
        out["outlines"] = endpoint.outlines
    if endpoint.template_space:
        out["template_space"] = endpoint.template_space
    if own:
        out["subject_ranges"] = {subject: [float(v) for v in window] for subject, window in own.items()}
    if endpoint.variants:
        out["variants"] = []
        for variant in endpoint.variants:
            record = {"id": variant.id, "label": variant.label, "blurb": variant.blurb}
            if variant.display_range:
                record["range"] = [float(v) for v in variant.display_range]
            if variant.statistic:
                record["statistic"] = variant.statistic
            if variant.values:
                record["values"] = dict(variant.values)
            if variant.subject_display_ranges:
                record["subject_ranges"] = {s: [float(v) for v in w]
                                            for s, w in variant.subject_display_ranges.items()}
            if variant.region_stats is not None:
                record["region_stats"] = _region_stats(variant.region_stats)
            if variant.threshold_default:
                record["threshold_default"] = {"mode": variant.threshold_default[0],
                                               "value": float(variant.threshold_default[1])}
            out["variants"].append(record)
        toggle = endpoint.variant_toggle
        if endpoint.variant_controls:
            out["variant_controls"] = [_variant_control(c) for c in endpoint.variant_controls]
        elif toggle is not None:
            out["variant_control"] = {"kind": "toggle", "label": toggle.label,
                                      "tip": toggle.tip, "on": toggle.on, "off": toggle.off}
        elif endpoint.variant_label:
            out["variant_control"] = {"label": endpoint.variant_label}
    if endpoint.region_stats is not None:
        out["region_stats"] = _region_stats(endpoint.region_stats)
    return out


def _region_stats(stats) -> dict:
    record = {
        "atlas": stats.atlas,
        "volume_atlas": list(stats.volume_atlas),
        "statistic": stats.statistic,
        "effect_label": stats.effect_label,
    }
    if stats.rows:
        record["rows"] = [_region_row(row) for row in stats.rows]
    return record


def _region_row(row: RegionRow) -> dict:
    """A region's result as the page reads it; unset fields are left out."""
    out: dict = {"name": row.name, "delta": float(row.delta), "significant": bool(row.significant)}
    for key in ("value", "t", "p", "q"):
        value = getattr(row, key)
        if value is not None:
            out[key] = float(value)
    for key in ("n_positive", "n"):
        value = getattr(row, key)
        if value is not None:
            out[key] = int(value)
    if row.atlas:
        out["atlas"] = row.atlas
    return out


def _build_subject_space(spec: ViewerSpec, data: Path, root: Path, say) -> dict | None:
    space = spec.subject_space
    if not space or not space.templates:
        return None
    say("subject anatomy")
    out: dict = {"templates": {}, "templates_2mm": {}, "coords": {}, "maps": []}
    for subject, path in space.templates.items():
        target = data / "subject_space" / subject / "template.nii.gz"
        assets.write_underlay(path, target, crop=True)
        out["templates"][subject] = _relative(target, root)
    for subject, path in space.templates_coarse.items():
        target = data / "subject_space" / subject / "template_coarse.nii.gz"
        assets.write_underlay(path, target, crop=True)
        out["templates_2mm"][subject] = _relative(target, root)
    for subject, maps in space.coordinates.items():
        record = {}
        for key, source in (("anat_to_mni", maps.to_template), ("mni_to_anat", maps.from_template)):
            bins = assets.coordinate_bins(source, stride=maps.stride)
            target = data / "subject_space" / subject / f"{key}.bin"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bins["values"].tobytes(order="C"))
            record[key] = {
                "path": _relative(target, root),
                "dims": bins["dims"],
                "affine": bins["affine"],
            }
        linear = _linear(maps)
        if linear is not None:
            record["linear_from_template"] = [v for row in linear for v in row]
            rigid = _rigid(linear, maps)
            record["rigid_from_template"] = [v for row in rigid for v in row]
        out["coords"][subject] = record
    for report in spec.reports:
        for endpoint in report.endpoints:
            for entry in endpoint.maps:
                if not entry.subject_space_path:
                    continue
                name = _file_stem(report.id, endpoint.id, entry.feature or "map",
                                  entry.variant, entry.cohort) + ".nii.gz"
                target = data / "subject_space" / entry.subject / name
                _, values = assets.write_map(entry.subject_space_path, target)
                record = {
                    "report": report.id,
                    "endpoint": endpoint.id,
                    "feature": entry.feature,
                    "subject": entry.subject,
                    "path": _relative(target, root),
                    "zooms": assets.voxel_size(entry.subject_space_path),
                }
                if entry.variant:
                    record["variant"] = entry.variant
                if entry.cohort:
                    record["cohort"] = entry.cohort
                # Its own curves, not the template map's: an FDR threshold is
                # a property of the distribution of p in the map being
                # corrected, and this map has a different voxel set.
                curves = _curves(endpoint, entry, values)
                if curves:
                    record["thresholds"] = curves
                out["maps"].append(record)
    return out


def _linear(maps: CoordinateMaps) -> list[list[float]] | None:
    """The registration's linear part, template mm -> subject mm, as a 4 x 4.

    Given, it is used as is. Otherwise it is the least-squares affine of the
    ``from_template`` field over the voxels it covers: the best linear account
    of the whole warp, so what is left for the slider to show is the
    nonlinear part and nothing else. None when the field covers too little to
    fit, and then the page offers no warp slider for the subject.
    """
    if maps.linear is not None:
        matrix = _matrix(maps.linear)
        if not matrix:
            raise ValueError("CoordinateMaps.linear must be 4 x 4")
        return matrix
    image = nib.load(str(maps.from_template))
    full = np.asarray(image.get_fdata(dtype=np.float32))
    if full.ndim != 4 or full.shape[3] != 3:
        return None
    step = 2 if full[..., 0].size > 1_000_000 else 1
    field = full[::step, ::step, ::step, :]
    ijk = np.stack(np.meshgrid(*[np.arange(0, n * step, step) for n in field.shape[:3]],
                               indexing="ij"), -1)
    target = field.reshape(-1, 3)
    covered = np.any(target != 0, axis=1)
    if covered.sum() < 12:
        return None
    points = ijk.reshape(-1, 3)[covered] @ image.affine[:3, :3].T + image.affine[:3, 3]
    design = np.c_[points, np.ones(len(points))]
    solution, *_ = np.linalg.lstsq(design, target[covered].astype(np.float64), rcond=None)
    matrix = np.eye(4)
    matrix[:3, :] = solution.T
    return matrix.tolist()


def _rigid(linear: list[list[float]], maps: CoordinateMaps) -> list[list[float]]:
    """The rigid part of the registration's linear part, template mm -> subject mm.

    The linear part's 3 x 3 is split by polar decomposition into a rotation
    and a stretch, and the rotation is anchored so that it and the whole
    linear part agree at the centre of the brain (the centroid of the voxels
    the field covers, else the template origin). A registration view that
    starts here shows the subject at their own size and shape, already
    turned and moved into the template's position, so its slider moves only
    what is not rigid: the scaling and shear, then the nonlinear warp.
    """
    matrix = np.asarray(linear, dtype=np.float64)
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    centre = np.zeros(3)
    try:
        image = nib.load(str(maps.from_template))
        field = np.asarray(image.get_fdata(dtype=np.float32))
        step = 4 if field[..., 0].size > 1_000_000 else 1
        thinned = field[::step, ::step, ::step, :]
        covered = np.argwhere(np.any(thinned != 0, axis=3)) * step
        if len(covered):
            centre = covered.mean(axis=0) @ image.affine[:3, :3].T + image.affine[:3, 3]
    except Exception:
        pass
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = matrix[:3, :3] @ centre + matrix[:3, 3] - rotation @ centre
    return out.tolist()


def _build_atlases(atlases: list[Atlas], data: Path, root: Path) -> dict:
    out: dict = {"atlases": [], "subject": {}}
    for atlas in atlases:
        target = data / "atlas" / f"{atlas.id}.nii.gz"
        assets.write_labels(atlas.template_path, target)
        out["atlases"].append({
            "id": atlas.id,
            "label": atlas.label,
            "mni": _relative(target, root),
            "regions": [{"value": int(value), "name": name} for value, name in atlas.labels],
        })
        for subject, path in atlas.subject_paths.items():
            subject_target = data / "atlas" / subject / f"{atlas.id}.nii.gz"
            assets.write_labels(path, subject_target)
            out["subject"].setdefault(subject, {})[atlas.id] = _relative(subject_target, root)
    return out
