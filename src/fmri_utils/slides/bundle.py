"""Copy a deck and everything it embeds into one folder that runs offline.

The bundle needs nothing but Python 3 (standard library) on the machine that shows it::

    <bundle>/<name>/        the deck (with its vendored KaTeX and NiiVue)
    <bundle>/<mount>/       each mounted folder; a viewer build (manifest.json + surfaces.json)
                            contributes only the files its manifest and surface catalogue
                            reference, not stale files from older builds
    <bundle>/serve.py       the local server (a copy of fmri_utils.slides.server)
    <bundle>/start.bat      double-click: starts the server and opens the deck (Windows)
    <bundle>/README.txt

Files already in the bundle with the same size are not copied again, so rebuilding after an edit
is quick. Files that no longer exist in the deck are not removed from the bundle.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .server import load_config

VIEWER_INDEXES = ("manifest.json", "surfaces.json")


def referenced(node, root: Path, found: set[str]) -> None:
    """Every string in a JSON tree that names a file under ``root``."""
    if isinstance(node, dict):
        for value in node.values():
            referenced(value, root, found)
    elif isinstance(node, list):
        for value in node:
            referenced(value, root, found)
    elif isinstance(node, str) and "/" in node and len(node) < 300 and not node.startswith("http"):
        if (root / node).is_file():
            found.add(node)


def copy(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
    return source.stat().st_size


def mount_files(folder: Path) -> list[str]:
    """The files of a mounted folder worth shipping."""
    if all((folder / name).is_file() for name in VIEWER_INDEXES):
        files = {"index.html", *VIEWER_INDEXES}
        for name in VIEWER_INDEXES:
            referenced(json.loads((folder / name).read_text(encoding="utf-8")), folder, files)
        return sorted(f for f in files if (folder / f).is_file())
    return sorted(str(p.relative_to(folder).as_posix()) for p in folder.rglob("*") if p.is_file())


def build_bundle(deck: Path, output: Path) -> int:
    """Writes the bundle; returns its size in bytes."""
    deck, output = Path(deck).resolve(), Path(output)
    config = load_config(deck)
    name = config["name"]
    total = 0
    for path in deck.rglob("*"):
        if path.is_file() and not path.name.endswith(".bak") and path.name != "deck.json":
            total += copy(path, output / name / path.relative_to(deck))
    for mount, folder in config["mounts"].items():
        files = mount_files(folder)
        for relative in files:
            total += copy(folder / relative, output / mount / relative)
        print(f"{mount}: {len(files)} files", flush=True)
    # The bundle's deck.json mounts the copies beside it.
    bundled = json.loads((deck / "deck.json").read_text(encoding="utf-8")) if (deck / "deck.json").is_file() else {}
    bundled["name"] = name
    bundled["mounts"] = {mount: f"../{mount}" for mount in config["mounts"]}
    (output / name / "deck.json").write_text(json.dumps(bundled, indent=1), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("server.py"), output / "serve.py")
    (output / "start.bat").write_text("@echo off\r\ncd /d \"%~dp0\"\r\npython serve.py\r\npause\r\n", encoding="ascii")
    (output / "README.txt").write_text(
        f"{config.get('title', name)}, offline.\n\n"
        "Double-click start.bat (or run: python serve.py). The deck opens at\n"
        f"http://127.0.0.1:8740/{name}/ . Arrow keys / Space / a clicker step through, F toggles\n"
        "full screen, E edits (Ctrl+S saves into the bundle's index.html). Needs only Python 3.\n",
        encoding="utf-8")
    print(f"bundle: {total / 1e9:.2f} GB in {output}", flush=True)
    return total
