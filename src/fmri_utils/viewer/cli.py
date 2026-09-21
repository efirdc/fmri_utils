"""Command line for the parts of a viewer that are the same in every project.

Building the viewer itself is a few lines of Python, because only the project
knows what its reports and endpoints are. Exporting surfaces is not: it is the
same three steps every time, so they are here.

    fmri-viewer export-surfaces --pycortex-db DB --output-root OUT --subjects sub-01,sub-02
    fmri-viewer export-surfaces --pycortex-db DB --output-root OUT --fsaverage
    fmri-viewer package-surfaces --input-root OUT --output-root VIEWER
    fmri-viewer demo --output-root VIEWER
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fmri-viewer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-surfaces", help="read geometry out of a pycortex database")
    export.add_argument("--pycortex-db", type=Path, required=True,
                        help="a pycortex database directory")
    export.add_argument("--output-root", type=Path, required=True,
                        help="where the raw export goes")
    export.add_argument("--subjects", default="",
                        help="comma separated; default is every subject in the database")
    export.add_argument("--freesurfer-dir", type=Path, default=None,
                        help="SUBJECTS_DIR, for curvature FreeSurfer computed")
    export.add_argument("--fsaverage", action="store_true",
                        help="export nilearn fsaverage instead, with its MNI transform")

    package = sub.add_parser("package-surfaces", help="pack an export for the browser")
    package.add_argument("--input-root", type=Path, required=True,
                         help="an export-surfaces output root")
    package.add_argument("--output-root", type=Path, required=True,
                         help="gets data/surfaces/... and surfaces.json")
    package.add_argument("--subjects", default="",
                         help="comma separated; default is everything exported")
    package.add_argument("--base-geometry", default="wm",
                         help="the geometry whose mesh carries the face list")

    demo = sub.add_parser("demo", help="build a small viewer from synthetic data")
    demo.add_argument("--output-root", type=Path, required=True,
                      help="where the demo viewer is written")

    args = parser.parse_args(argv)
    names = [s.strip() for s in getattr(args, "subjects", "").split(",") if s.strip()]

    if args.command == "export-surfaces":
        from .surfaces import export_surfaces

        export_surfaces(
            args.pycortex_db, args.output_root, names,
            freesurfer_dir=args.freesurfer_dir, fsaverage=args.fsaverage,
        )
    elif args.command == "package-surfaces":
        from .surfaces import package_surfaces

        package_surfaces(args.input_root, args.output_root, names, args.base_geometry)
    elif args.command == "demo":
        from .demo import build_demo

        build_demo(args.output_root)


if __name__ == "__main__":
    main()
