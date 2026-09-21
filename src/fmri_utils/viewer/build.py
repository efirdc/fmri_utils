"""Building a viewer: spec in, a directory you can serve out.

The output is static. There is no server, no database and no session state, so
a viewer costs nothing to keep up and a link to it shows everyone the same
thing. Copy the directory anywhere that serves files over HTTP.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import assets
from .spec import Atlas, Endpoint, Report, ViewerSpec

PAGE = Path(__file__).parent / "resources" / "index.html"


def build_viewer(spec: ViewerSpec, output_root: Path, quiet: bool = False) -> Path:
    """Write the page, the manifest and every file the manifest points at.

    Returns the directory. Serve it, or copy it to wherever it is hosted.
    """
    spec.check()
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
    space = manifest.get("subject_space") or {}
    for group in ("templates", "templates_2mm"):
        used.update((space.get(group) or {}).values())
    for record in (space.get("coords") or {}).values():
        used.update(side["path"] for side in record.values())
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
            for geometry in hemisphere.get("geometries", {}).values():
                used.add(geometry.get("vertices"))
                used.add(geometry.get("face_mask"))
        for region in record.get("rois", {}).values():
            used.add(region.get("path"))
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


def _build_endpoint(
    report: Report, endpoint: Endpoint, spec: ViewerSpec, data: Path, root: Path, say
) -> dict:
    features: list[dict] = []
    subjects: list[str] = []
    maps: list[dict] = []
    extremes: list[float] = []
    for entry in endpoint.maps:
        name = f"{entry.feature or 'map'}_{entry.subject}.nii.gz"
        target = data / report.id / endpoint.id / name
        _, values = assets.write_map(entry.path, target)
        record = {
            "subject": entry.subject,
            "feature": entry.feature,
            "path": _relative(target, root),
            # The page reads this to decide how fine an underlay is worth
            # compositing onto: a 3 mm map on a 1 mm grid costs eight times the
            # blend texture and shows nothing more.
            "zooms": assets.voxel_size(entry.path),
            "range": assets.percentile_range(values, spec.percentile),
        }
        dof = endpoint.degrees_of_freedom.get(entry.subject)
        if endpoint.statistic == "t" and dof:
            record["thresholds"] = assets.statistic_curves(values, int(dof))
        elif endpoint.statistic == "r" and dof:
            record["thresholds"] = assets.correlation_curves(values, int(dof))
        maps.append(record)
        extremes.append(record["range"][1])
        if entry.subject not in subjects:
            subjects.append(entry.subject)
        if entry.feature and entry.feature not in [f["id"] for f in features]:
            features.append({"id": entry.feature, "label": entry.feature})
        say(f"  {report.id}/{endpoint.id} {entry.subject} {entry.feature}".rstrip())
    return {
        "id": endpoint.id,
        "label": endpoint.label,
        "blurb": endpoint.blurb,
        "warp": endpoint.warp,
        "statistic": endpoint.statistic,
        "features": features or [{"id": "", "label": "—"}],
        "subjects": subjects,
        "range": list(endpoint.display_range) if endpoint.display_range
                 else [0.0, float(np.median(extremes)) if extremes else 1.0],
        "maps": maps,
    }


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
            bins = assets.coordinate_bins(source)
            target = data / "subject_space" / subject / f"{key}.bin"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bins["values"].tobytes(order="C"))
            record[key] = {
                "path": _relative(target, root),
                "dims": bins["dims"],
                "affine": bins["affine"],
            }
        out["coords"][subject] = record
    for report in spec.reports:
        for endpoint in report.endpoints:
            for entry in endpoint.maps:
                if not entry.subject_space_path:
                    continue
                name = f"{endpoint.id}_{entry.feature or 'map'}_{entry.subject}.nii.gz"
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
                # Its own curves, not the template map's: an FDR threshold is
                # a property of the distribution of p in the map being
                # corrected, and this map has a different voxel set.
                dof = endpoint.degrees_of_freedom.get(entry.subject)
                if endpoint.statistic == "t" and dof:
                    record["thresholds"] = assets.statistic_curves(values, int(dof))
                elif endpoint.statistic == "r" and dof:
                    record["thresholds"] = assets.correlation_curves(values, int(dof))
                out["maps"].append(record)
    return out


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
