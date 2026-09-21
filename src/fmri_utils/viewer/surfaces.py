"""Cortical surfaces for the viewer, from a pycortex database.

Three steps, and each is re-runnable on its own: read the geometry out of a
pycortex database, write the pieces the browser fetches, and hand the catalogue
to the builder. fsaverage comes from nilearn instead, for group maps that
belong to no single brain.

The one subtlety is the flat map. It is not the same mesh as the folded one --
flattening cuts the medial wall and drops about five per cent of the faces --
so its face list travels as a per-face keep mask, and the viewer swaps face
lists when the unfold slider reaches it.
"""

from __future__ import annotations

import gzip
import json
import shutil
import struct
from pathlib import Path

import numpy as np

FSAVERAGE = "fsaverage"
FSAVERAGE_KEYS = {"wm": "white", "pia": "pial", "inflated": "infl", "flat": "flat"}
SIDES = {"lh": "left", "rh": "right"}

# Fischl's MNI305-to-MNI152 affine. fsaverage vertices are in MNI305 and the
# group maps are in MNI152, and the two differ by enough (up to ~4 mm at the
# edges) to smear a surface projection if it is skipped.
MNI305_TO_MNI152 = [
    [0.9975, -0.0073, 0.0176, -0.0429],
    [0.0146, 1.0009, -0.0024, 1.5496],
    [-0.0130, -0.0093, 0.9971, 1.1840],
    [0.0, 0.0, 0.0, 1.0],
]

_fsaverage_files = {}


def fsaverage_files():
    """nilearn's fsaverage, downloaded once and cached by nilearn itself."""
    if not _fsaverage_files:
        from nilearn import datasets

        _fsaverage_files.update(datasets.fetch_surf_fsaverage("fsaverage"))
    return _fsaverage_files


def load_pycortex(db_root: Path):
    """pycortex if it imports, otherwise None and we read the database directly."""
    try:
        from .warp_correlation_maps_to_mni import configure_pycortex
    except Exception:
        return None
    try:
        return configure_pycortex(db_root)
    except Exception as error:
        print(f"pycortex unavailable ({error}); reading the database as GIfTI", flush=True)
        return None


def read_surface(cortex, db_root: Path, subject: str, geometry: str, hemisphere: str):
    """Vertices and faces for one geometry, from pycortex or straight off disk.

    A pycortex database stores every surface as a GIfTI under the subject's
    surfaces directory, so the fallback reads exactly what pycortex would have
    handed back and keeps this runnable on a machine without pycortex.
    """
    if subject == FSAVERAGE:
        import nibabel as nb

        key = f"{FSAVERAGE_KEYS[geometry]}_{SIDES[hemisphere]}"
        arrays = nb.load(str(fsaverage_files()[key])).darrays
        points, polys = arrays[0].data, arrays[1].data
    elif cortex is not None:
        index = HEMISPHERES.index(hemisphere)
        points, polys = cortex.db.get_surf(subject, geometry, merge=False)[index]
    else:
        # A database holds each surface either as a GIfTI or as pycortex's own
        # npz of pts and polys, and which one depends on how the subject was
        # imported, so both are read.
        stem = db_root / subject / "surfaces" / f"{geometry}_{hemisphere}"
        gifti, npz = stem.with_suffix(".gii"), stem.with_suffix(".npz")
        if gifti.exists():
            import nibabel as nb

            arrays = nb.load(str(gifti)).darrays
            points, polys = arrays[0].data, arrays[1].data
        elif npz.exists():
            stored = np.load(npz)
            points, polys = stored["pts"], stored["polys"]
        else:
            raise FileNotFoundError(gifti)
    return np.asarray(points, dtype=np.float32), np.asarray(polys, dtype=np.uint32)


def face_keys(faces: np.ndarray) -> np.ndarray:
    """One integer per face, invariant to winding, so two lists can be compared."""
    ordered = np.sort(faces, axis=1).astype(np.int64)
    return ordered[:, 0] + (ordered[:, 1] << 21) + (ordered[:, 2] << 42)


def face_keep_mask(base: np.ndarray, subset: np.ndarray):
    """Which of base's faces survive in subset, or None if subset is not one."""
    keep = np.isin(face_keys(base), face_keys(subset))
    if int(keep.sum()) != subset.shape[0]:
        return None
    return keep.astype(np.uint8)


def read_curvature(cortex, freesurfer_dir, subject: str, hemisphere: str, n: int):
    """Per-vertex curvature, positive in sulci, or None if it cannot be found."""
    if subject == FSAVERAGE:
        import nibabel as nb

        values = nb.load(str(fsaverage_files()[f"curv_{SIDES[hemisphere]}"])).darrays[0].data
        return np.asarray(values, dtype=np.float32)
    if freesurfer_dir is not None:
        path = Path(freesurfer_dir) / subject / "surf" / f"{hemisphere}.curv"
        if path.exists():
            from nibabel.freesurfer.io import read_morph_data

            values = np.asarray(read_morph_data(str(path)), dtype=np.float32)
            if values.size == n:
                return values
            print(f"  {hemisphere} curvature has {values.size} values, not {n}", flush=True)
    if cortex is not None:
        try:
            surfinfo = cortex.db.get_surfinfo(subject, "curvature")
            values = surfinfo.left if hemisphere == "lh" else surfinfo.right
            values = np.asarray(values, dtype=np.float32)
            if values.size == n:
                return values
        except Exception as error:
            print(f"  {hemisphere} curvature unavailable ({error})", flush=True)
    return None




MAGIC = 23117
ATTR_FACE = 1
ATTR_VERT = 2


def write_mz3(vertices: np.ndarray, faces: np.ndarray, path: Path) -> int:
    header = struct.pack(
        "<HHIII", MAGIC, ATTR_FACE | ATTR_VERT, faces.shape[0], vertices.shape[0], 0
    )
    payload = header + faces.astype("<u4").tobytes() + vertices.astype("<f4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, 6))
    return path.stat().st_size


def copy_binary(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target.stat().st_size




GEOMETRIES = ("wm", "pia", "inflated", "flat")
HEMISPHERES = ("lh", "rh")


def export_surfaces(
    db_root: Path,
    output_root: Path,
    subjects,
    freesurfer_dir: Path | None = None,
    fsaverage: bool = False,
) -> Path:
    """Read geometry, curvature and region vertex sets out of a pycortex database.

    Works with or without pycortex installed: a database stores each surface
    either as a GIfTI or as pycortex's own npz, and both are read directly.
    Only the hand-drawn region overlay needs pycortex itself, and a previous
    run's regions are carried forward when it is absent, so re-exporting
    geometry never silently drops them.

    Pass ``fsaverage=True`` to export nilearn's fsaverage instead, which is
    what group maps are projected onto.
    """
    db_root = Path(db_root)
    output_root = Path(output_root)
    cortex = load_pycortex(db_root)
    subjects = [FSAVERAGE] if fsaverage else list(subjects)
    for subject in subjects:
        print(f"[{subject}]", flush=True)
        out = output_root / subject
        out.mkdir(parents=True, exist_ok=True)
        record: dict = {"subject": subject, "hemispheres": {}}
        if subject == FSAVERAGE:
            record["volume_transform"] = MNI305_TO_MNI152
            record["volume_space"] = "MNI152"
        for hemisphere in HEMISPHERES:
            entry: dict = {"geometries": {}}
            base_faces = None
            for geometry in GEOMETRIES:
                try:
                    points, polys = read_surface(
                        cortex, db_root, subject, geometry, hemisphere
                    )
                except Exception as error:  # a subject may lack a geometry
                    print(f"  {hemisphere} {geometry}: unavailable ({error})", flush=True)
                    continue
                geometry_record: dict = {}
                if base_faces is None:
                    base_faces = polys
                    (out / f"{hemisphere}_faces.bin").write_bytes(polys.tobytes(order="C"))
                    entry["faces"] = f"{hemisphere}_faces.bin"
                    entry["n_faces"] = int(polys.shape[0])
                    entry["n_vertices"] = int(points.shape[0])
                elif polys.shape != base_faces.shape or not np.array_equal(polys, base_faces):
                    mask = face_keep_mask(base_faces, polys)
                    if mask is None:
                        name = f"{hemisphere}_{geometry}_faces.bin"
                        (out / name).write_bytes(polys.tobytes(order="C"))
                        geometry_record["faces"] = name
                    else:
                        name = f"{hemisphere}_{geometry}_face_mask.bin"
                        (out / name).write_bytes(mask.tobytes(order="C"))
                        geometry_record["face_mask"] = name
                    geometry_record["n_faces"] = int(polys.shape[0])
                    print(
                        f"  {hemisphere} {geometry}: {polys.shape[0]} of "
                        f"{base_faces.shape[0]} faces kept",
                        flush=True,
                    )
                name = f"{hemisphere}_{geometry}.bin"
                (out / name).write_bytes(points.tobytes(order="C"))
                geometry_record.update(
                    {
                        "path": name,
                        "anatomical": geometry != "flat",
                        "bounds": [
                            [float(v) for v in points.min(axis=0)],
                            [float(v) for v in points.max(axis=0)],
                        ],
                    }
                )
                entry["geometries"][geometry] = geometry_record
                print(f"  {hemisphere} {geometry}: {points.shape[0]} vertices", flush=True)
            if entry["geometries"]:
                curvature = read_curvature(
                    cortex, freesurfer_dir, subject, hemisphere, entry["n_vertices"]
                )
                if curvature is not None:
                    name = f"{hemisphere}_curv.bin"
                    (out / name).write_bytes(curvature.astype(np.float32).tobytes(order="C"))
                    entry["curvature"] = name
                    print(
                        f"  {hemisphere} curvature: {float((curvature > 0).mean()):.0%} sulcal",
                        flush=True,
                    )
                record["hemispheres"][hemisphere] = entry

        # ROI vertex sets, so the surface view can outline the same regions the
        # flatmap figures use.
        try:
            if subject == FSAVERAGE:
                raise RuntimeError("fsaverage carries no pycortex ROI overlay")
            if cortex is None:
                raise RuntimeError("pycortex is needed to read the ROI overlay")
            rois = cortex.utils.get_roi_verts(subject)
            roi_dir = out / "rois"
            roi_dir.mkdir(exist_ok=True)
            catalogue = {}
            for name, vertices in rois.items():
                vertices = np.asarray(vertices, dtype=np.uint32).ravel()
                if vertices.size == 0:
                    continue
                safe = "".join(c for c in name if c.isalnum() or c in "-_")
                (roi_dir / f"{safe}.bin").write_bytes(vertices.tobytes(order="C"))
                catalogue[name] = {"path": f"rois/{safe}.bin", "n_vertices": int(vertices.size)}
            record["rois"] = catalogue
            print(f"  {len(catalogue)} ROI vertex sets", flush=True)
        except Exception as error:
            # Without pycortex the ROI overlay cannot be re-read, but a previous
            # run's vertex sets are still on disk; carry their catalogue forward
            # rather than dropping regions from a geometry-only re-export.
            previous = out / "surfaces.json"
            carried = {}
            if previous.exists():
                stored = json.loads(previous.read_text(encoding="utf-8")).get("rois", {})
                carried = {
                    name: roi for name, roi in stored.items() if (out / roi["path"]).exists()
                }
            if carried:
                record["rois"] = carried
            print(f"  ROIs unavailable: {error}; kept {len(carried)} from the last run", flush=True)

        (out / "surfaces.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


    return output_root


def package_surfaces(
    input_root: Path,
    output_root: Path,
    subjects=(),
    base_geometry: str = "wm",
) -> Path:
    """Pack an export into what the browser fetches, and write the catalogue.

    Each hemisphere ships one mz3 -- the base geometry, which carries the face
    list -- plus a raw vertex block per geometry. The viewer morphs by swapping
    vertices on a mesh it already has, which is both far faster than reloading
    and the only way a continuous unfold can work.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    subjects = list(subjects) or sorted(
        path.name for path in input_root.iterdir() if path.is_dir()
    )
    catalogue = {}
    for subject in subjects:
        source = input_root / subject
        record = json.loads((source / "surfaces.json").read_text(encoding="utf-8"))
        entry = {"hemispheres": {}, "rois": {}}
        if record.get("volume_transform"):
            # fsaverage sits in MNI305; the viewer needs the affine to sample a
            # map that sits in MNI152.
            entry["volume_transform"] = record["volume_transform"]
            entry["volume_space"] = record.get("volume_space")
        target_dir = Path("data/surfaces") / subject
        for hemisphere, info in record["hemispheres"].items():
            faces = np.frombuffer((source / info["faces"]).read_bytes(), dtype="<u4")
            faces = faces.reshape(-1, 3)
            base = info["geometries"][base_geometry]
            base_vertices = np.frombuffer(
                (source / base["path"]).read_bytes(), dtype="<f4"
            ).reshape(-1, 3)
            mesh_path = target_dir / f"{hemisphere}_{base_geometry}.mz3"
            mesh_kb = round(write_mz3(base_vertices, faces, output_root / mesh_path) / 1024)

            geometries = {}
            for name, geometry in info["geometries"].items():
                relative = target_dir / f"{hemisphere}_{name}_vertices.bin"
                size = copy_binary(source / geometry["path"], output_root / relative)
                geometries[name] = {
                    "vertices": str(relative).replace("\\", "/"),
                    "anatomical": geometry["anatomical"],
                    "kb": round(size / 1024),
                }
                # Only the flat geometry cuts faces today, but the mask is
                # written from whatever the export found rather than assumed.
                mask_name = geometry.get("face_mask")
                if mask_name:
                    mask_relative = target_dir / f"{hemisphere}_{name}_face_mask.bin"
                    copy_binary(source / mask_name, output_root / mask_relative)
                    geometries[name]["face_mask"] = str(mask_relative).replace("\\", "/")
                    geometries[name]["n_faces"] = geometry["n_faces"]

            hemisphere_entry = {
                "n_vertices": info["n_vertices"],
                "n_faces": info["n_faces"],
                "mesh": str(mesh_path).replace("\\", "/"),
                "mesh_kb": mesh_kb,
                "base_geometry": base_geometry,
                "geometries": geometries,
            }
            if info.get("curvature"):
                relative = target_dir / f"{hemisphere}_curv.bin"
                copy_binary(source / info["curvature"], output_root / relative)
                hemisphere_entry["curvature"] = str(relative).replace("\\", "/")
            entry["hemispheres"][hemisphere] = hemisphere_entry

        for name, roi in record.get("rois", {}).items():
            relative = target_dir / "rois" / Path(roi["path"]).name
            copy_binary(source / roi["path"], output_root / relative)
            entry["rois"][name] = {
                "path": str(relative).replace("\\", "/"),
                "n_vertices": roi["n_vertices"],
            }
        catalogue[subject] = entry
        print(
            f"{subject}: {len(entry['hemispheres'])} hemispheres, {len(entry['rois'])} rois",
            flush=True,
        )

    (output_root / "surfaces.json").write_text(
        json.dumps(catalogue, indent=1), encoding="utf-8"
    )


    return output_root
