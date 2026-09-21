"""A viewer built from nothing, to check the tooling and to copy from.

``fmri-viewer demo --output-root somewhere`` writes a working viewer from
synthetic volumes: no dataset, no cluster, no surfaces. Serve the directory and
you get the real page with two subjects, two endpoints and an atlas, which is
enough to see what a project of your own would look like and to lift the
spec below as a starting point.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .build import build_viewer
from .spec import About, Atlas, Endpoint, MapEntry, Report, ViewerSpec


def _blob(shape, centre, width, amplitude):
    grid = np.indices(shape).astype(np.float32)
    distance = sum((grid[a] - centre[a]) ** 2 for a in range(3))
    return amplitude * np.exp(-distance / (2 * width ** 2))


def _write(values, affine, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values.astype(np.float32), affine), str(path))
    return path


def build_demo(output_root: Path) -> Path:
    """Write synthetic inputs next to the viewer and build it."""
    output_root = Path(output_root)
    scratch = output_root / "_demo_inputs"
    shape = (64, 76, 64)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-64, -76, -64]

    anatomy = _blob(shape, (32, 38, 32), 18, 900) + _blob(shape, (32, 38, 44), 10, 300)
    template = _write(anatomy, affine, scratch / "template.nii.gz")

    reports = []
    for endpoint_id, label, seed in (("t_main", "main effect", 0), ("t_control", "control", 7)):
        rng = np.random.default_rng(seed)
        maps = []
        for index, subject in enumerate(("sub-01", "sub-02")):
            values = (
                _blob(shape, (22 + index * 4, 46, 34), 6, 6.0)
                - _blob(shape, (44, 30, 30), 5, 4.0)
                + rng.normal(0, 0.7, shape)
            )
            values[anatomy < 120] = 0
            maps.append(MapEntry(
                subject=subject,
                path=_write(values, affine, scratch / f"{endpoint_id}_{subject}.nii.gz"),
            ))
        reports.append(Endpoint(
            id=endpoint_id, label=label, statistic="t",
            degrees_of_freedom={"sub-01": 120, "sub-02": 120},
            blurb="Synthetic data. Nothing here means anything.",
            maps=maps,
        ))

    labels = np.zeros(shape, dtype=np.float32)
    labels[_blob(shape, (22, 46, 34), 6, 1.0) > 0.3] = 1
    labels[_blob(shape, (44, 30, 30), 5, 1.0) > 0.3] = 2
    atlas = Atlas(
        id="demo", label="demo atlas",
        labels=[(1, "Left blob"), (2, "Right blob")],
        template_path=_write(labels, affine, scratch / "atlas.nii.gz"),
    )

    spec = ViewerSpec(
        about=About(
            title="Demo Browser",
            kicker="fmri-utils · viewer",
            lede="Synthetic maps on a synthetic brain. Drag to move the crosshair.",
            footnote="Built by <code>fmri-viewer demo</code>.",
        ),
        template=template,
        reports=[Report(id="demo", label="Demo report", endpoints=reports)],
        atlases=[atlas],
    )
    return build_viewer(spec, output_root)
