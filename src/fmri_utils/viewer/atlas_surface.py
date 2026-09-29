"""Any volume atlas as a surface parcellation, for fsnative or fsaverage.

A viewer parcellation (``parcellation.write_parcellation``) needs one label per
vertex. This module gets those labels from a label volume -- any integer atlas
-- and adds the parcellation to a surface export, so ``package_surfaces``
carries it into the viewer next to the geometry:

    from fmri_utils.viewer.atlas_surface import add_atlas_parcellation, fsl_atlas_labels

    from fmri_utils.viewer.region_info import harvard_oxford_cortical

    regions = fsl_atlas_labels("HarvardOxford-Cortical.xml", harvard_oxford_cortical())
    add_atlas_parcellation("surfaces_export/sub-01", "ho", "Harvard-Oxford",
                           atlas="sub-01/HarvardOxford-cort_space-T1.nii.gz", regions=regions)
    add_atlas_parcellation("surfaces_export/fsaverage", "ho", "Harvard-Oxford",
                           atlas="HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", regions=regions)

How a vertex is labelled (``freesurfer.sample_volume_labels``): the atlas is
sampled, nearest voxel, at several depths between the white and pial surfaces,
and the commonest label wins; a vertex whose depths all land on unlabelled
voxels takes the nearest labelled vertex's label within ``fill_mm``. Then
``write_parcellation`` makes the labelling complete, smooth and one piece per
region, and traces the shared boundary network on the export's own faces.

Spaces. Surface coordinates are the export's world millimetres:

* **fsnative**: the subject's own anatomy, the same millimetres the viewer
  samples subject-space maps in. Give the atlas already warped into that
  subject's T1 (any grid; only its affine matters).
* **fsaverage**: MNI305. The export's ``volume_transform`` (MNI305 to MNI152)
  is applied automatically, so an MNI152 atlas can be used as it is.

``transform`` overrides either: a 4x4 from surface millimetres to the atlas's
world millimetres.

Tissue classes that some atlases carry alongside their regions (the
Harvard-Oxford subcortical atlas labels cerebral cortex and white matter)
would swallow the cortex; leave them out of ``regions`` or name them in
``exclude``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .freesurfer import sample_volume_labels
from .parcellation import write_parcellation

HEMISPHERES = ("lh", "rh")


# ---- region lists -------------------------------------------------------------

def fsl_atlas_labels(xml_path, abbreviations: Mapping[str, str] | None = None) -> list[dict]:
    """Regions of an FSL atlas XML (``<label index="0">Frontal Pole</label>``).

    FSL numbers labels from 0 in the XML and from 1 in the maxprob volumes, so
    each region's value is its index plus one.
    """
    root = ElementTree.parse(str(xml_path)).getroot()
    out = []
    for node in root.iter("label"):
        name = (node.text or "").strip()
        value = int(node.get("index")) + 1
        out.append(_region(value, name, abbreviations))
    return out


def table_labels(path, abbreviations: Mapping[str, str] | None = None) -> list[dict]:
    """Regions from a text table: ``value<TAB>name[<TAB>abbrev]`` per line.

    Lines starting with ``#`` and a header whose first field is not a number
    are skipped. Space-separated FreeSurfer colour tables (``value name r g b
    a``) also work: without tabs, the second field is the name and the rest is
    ignored.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if value == 0 or len(fields) < 2:
            continue
        region = _region(value, fields[1].strip(), abbreviations)
        if "\t" in line and len(fields) > 2 and fields[2].strip():
            region["abbrev"] = fields[2].strip()
        out.append(region)
    return out


def _region(value: int, name: str, abbreviations: Mapping | None) -> dict:
    # ``abbreviations`` is {name: abbrev}, or a region_info dict ({name:
    # {"abbrev": ...}}), so ``region_info.harvard_oxford_cortical()`` works
    # as it is. A "Left "/"Right " prefix is tried without it too.
    table = abbreviations or {}
    bare = name.replace("Left ", "", 1).replace("Right ", "", 1)
    entry = table.get(name, table.get(bare))
    if isinstance(entry, Mapping):
        entry = entry.get("abbrev")
    return {"value": int(value), "name": name, "abbrev": entry or name}


# ---- reading an export --------------------------------------------------------

def read_export_hemisphere(subject_dir, record: dict, hemisphere: str) -> dict:
    """White and pial vertices, full faces and the flat map's kept faces."""
    subject_dir = Path(subject_dir)
    info = record["hemispheres"][hemisphere]
    geometries = info["geometries"]
    for needed in ("wm", "pia"):
        if needed not in geometries:
            raise ValueError(f"{subject_dir.name} {hemisphere}: the export has no {needed} surface")

    def vertices(name: str) -> np.ndarray:
        path = subject_dir / geometries[name]["path"]
        return np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 3).astype(np.float64)

    faces = np.frombuffer((subject_dir / info["faces"]).read_bytes(), dtype="<u4").reshape(-1, 3)
    flat = geometries.get("flat", {})
    if flat.get("face_mask"):
        mask = np.frombuffer((subject_dir / flat["face_mask"]).read_bytes(), dtype=np.uint8).astype(bool)
        flat_faces = faces[mask]
    elif flat.get("faces"):
        flat_faces = np.frombuffer((subject_dir / flat["faces"]).read_bytes(), dtype="<u4").reshape(-1, 3)
    else:
        flat_faces = faces
    return {"white": vertices("wm"), "pial": vertices("pia"),
            "faces": faces.astype(np.int64), "flat_faces": flat_faces.astype(np.int64)}


# ---- the parcellation ---------------------------------------------------------

def atlas_vertex_labels(subject_dir, atlas, transform: np.ndarray | None = None,
                        depths: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                        fill_mm: float = 3.0, keep_values: Iterable[int] | None = None) -> dict:
    """Raw per-vertex labels for each hemisphere of an export, before cleaning.

    ``keep_values`` limits the labels to those values (others become 0), which
    is how tissue classes are dropped before they are sampled.
    """
    import nibabel as nib

    subject_dir = Path(subject_dir)
    record = json.loads((subject_dir / "surfaces.json").read_text(encoding="utf-8"))
    image = nib.load(str(atlas)) if isinstance(atlas, (str, Path)) else atlas
    if keep_values is not None:
        values = np.asarray(image.dataobj).astype(np.int32)
        values[~np.isin(values, list(keep_values))] = 0
        image = nib.Nifti1Image(values, image.affine)
    if transform is None and record.get("volume_transform") is not None:
        transform = np.asarray(record["volume_transform"], dtype=np.float64)
    out = {}
    for hemisphere in HEMISPHERES:
        if hemisphere not in record.get("hemispheres", {}):
            continue
        mesh = read_export_hemisphere(subject_dir, record, hemisphere)
        labels = sample_volume_labels(image, mesh["white"], mesh["pial"], depths=depths,
                                      transform=transform, fill_mm=fill_mm)
        out[hemisphere] = dict(mesh, labels=labels)
    return out


def add_atlas_parcellation(subject_dir, atlas_id: str, label: str, atlas, regions: list[dict],
                           transform: np.ndarray | None = None,
                           depths: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                           fill_mm: float = 3.0, smooth_rounds: int = 20,
                           exclude: Iterable[str] = (), drop_absent: bool = True,
                           sparse: bool = False) -> dict:
    """Sample ``atlas`` onto one subject's export and add it as a parcellation.

    ``subject_dir`` is one subject's folder of an ``export_surfaces`` output
    (it holds ``surfaces.json``). ``regions`` is ``[{"value", "name",
    "abbrev"}]`` -- ``fsl_atlas_labels`` and ``table_labels`` make one. Regions
    named in ``exclude`` are dropped before sampling, and with ``drop_absent``
    so is any region no vertex ended up in, so the viewer's list only offers
    regions that are on this surface. ``sparse`` marks an atlas that covers
    only part of the cortex (``write_parcellation``); give it a small
    ``fill_mm`` too, or the gap filling grows every region by that much.
    Returns the parcellation record, which is also written into
    ``surfaces.json`` under ``parcellations[atlas_id]``.
    """
    subject_dir = Path(subject_dir)
    excluded = set(exclude)
    regions = [dict(r) for r in regions if r["name"] not in excluded]
    meshes = atlas_vertex_labels(subject_dir, atlas, transform=transform, depths=depths,
                                 fill_mm=fill_mm, keep_values=[r["value"] for r in regions])
    if not meshes:
        raise ValueError(f"{subject_dir}: the export has no hemispheres")
    if drop_absent:
        present = set()
        for mesh in meshes.values():
            present.update(int(v) for v in np.unique(mesh["labels"]) if v)
        regions = [r for r in regions if r["value"] in present]
    parcellation = write_parcellation(subject_dir, atlas_id, label, regions,
                                      {h: {"labels": m["labels"], "faces": m["faces"],
                                           "flat_faces": m["flat_faces"]}
                                       for h, m in meshes.items()},
                                      smooth_rounds=smooth_rounds, sparse=sparse)
    record_path = subject_dir / "surfaces.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.setdefault("parcellations", {})[atlas_id] = parcellation
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return parcellation


def add_atlas_parcellations(export_root, subjects: Sequence[str], atlas_id: str, label: str,
                            atlas_for, regions: list[dict], **options) -> dict:
    """``add_atlas_parcellation`` for several subjects of one export root.

    ``atlas_for`` is a path with ``{subject}`` in it (each subject's own atlas),
    a mapping from subject to path, or one path for all (an MNI atlas on
    fsaverage). Returns ``{subject: record}``.
    """
    out = {}
    for subject in subjects:
        if isinstance(atlas_for, Mapping):
            atlas = atlas_for[subject]
        else:
            atlas = str(atlas_for).format(subject=subject)
        out[subject] = add_atlas_parcellation(Path(export_root) / subject, atlas_id, label,
                                              atlas, regions, **options)
        lines = {h: info["n_lines"] for h, info in out[subject]["hemispheres"].items()}
        print(f"{subject}: {label} written, {len(out[subject]['regions'])} regions, "
              f"boundary lines {lines}", flush=True)
    return out
