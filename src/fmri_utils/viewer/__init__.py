"""Static brain-map viewers: describe what you have, get a page you can host.

A viewer is one HTML file, one manifest and the files it points at. Nothing
runs on the server, so it can live on any static host -- a lab web directory, a
GitHub page, a shared drive -- and a link shows everyone the same thing.

The shortest useful build is a template and one map per subject::

    from pathlib import Path
    from fmri_utils.viewer import (
        About, Endpoint, MapEntry, Report, ViewerSpec, build_viewer,
    )

    spec = ViewerSpec(
        about=About(title="Motion localiser", kicker="study 42"),
        template=Path("MNI152_T1_1mm_brain.nii.gz"),
        reports=[Report(id="loc", label="Localiser", endpoints=[
            Endpoint(
                id="t_motion", label="motion > static", statistic="t",
                degrees_of_freedom={"sub-01": 240, "sub-02": 240},
                maps=[
                    MapEntry(subject="sub-01", path=Path("sub-01_t.nii.gz")),
                    MapEntry(subject="sub-02", path=Path("sub-02_t.nii.gz")),
                ],
            ),
        ])],
    )
    build_viewer(spec, Path("viewer"))

Everything else is optional and switches itself on when you supply it: subject
anatomy and coordinate maps give the subject-space views and a linked montage
crosshair, an atlas gives the region panel, exported surfaces give the surface
mode. ``Features`` turns any of them off again.

See ``docs/viewer.md`` for the manifest contract, which is what the page
actually reads -- a project with its own build pipeline can write that file
itself and skip this module entirely.
"""

from __future__ import annotations

from .build import PAGE, build_viewer
from .spec import (
    About,
    Atlas,
    CoordinateMaps,
    Endpoint,
    Features,
    MapEntry,
    Report,
    SubjectSpace,
    ViewerSpec,
)
from .surfaces import export_surfaces, package_surfaces

__all__ = [
    "About",
    "Atlas",
    "CoordinateMaps",
    "Endpoint",
    "Features",
    "MapEntry",
    "PAGE",
    "Report",
    "SubjectSpace",
    "ViewerSpec",
    "build_viewer",
    "export_surfaces",
    "package_surfaces",
]
