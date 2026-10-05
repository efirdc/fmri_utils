"""Serve a slide deck locally, with the pages it embeds, and save the editor's changes.

A deck is a folder with ``index.html`` (the slides between ``<!--SLIDES-->`` and
``<!--/SLIDES-->``), ``deck.json`` and the deck's assets (``deck.js``,
``editor.js``, ``deck.css``, ``vendor/``). ``deck.json`` names the deck's URL
folder and the sibling folders its embeds refer to as ``../<mount>/``::

    {"name": "talk-2026-10-06",
     "public_base": "https://example.org/~me/",
     "mounts": {"viewer": "../../results/viewer",
                "reader/audio": "../../results/reader_audio",
                "reader": "../../results/reader_site"}}

The server mounts them side by side, as they sit when published::

    /talk-2026-10-06/   the deck
    /viewer/            the mounted folder
    ...

so embedded pages are same-origin and the deck can drive them. It also makes a
local copy work offline: the viewer's NiiVue (from a CDN) is served from the
deck's ``vendor/``, Google Fonts are dropped, and range requests are answered
(audio seeks). Two POST endpoints serve the editor: ``/<name>/__save`` writes
the slides back into ``index.html`` between the markers (keeping a ``.bak``),
and ``/<name>/__upload`` saves a picture under ``figures/uploads/``.

The module only needs the standard library, so an offline bundle ships a copy
of it as ``serve.py``::

    python -m fmri_utils.slides.server DECK_DIR [--port 8740] [--no-browser]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

NIIVUE_CDN = re.compile(r"https://cdn\.jsdelivr\.net/npm/@niivue/niivue@[^\"']+/dist/niivue\.umd\.js")
FONTS = re.compile(r"<link[^>]+fonts\.(googleapis|gstatic)\.com[^>]*>")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("application/javascript", ".js")


def load_config(deck: Path) -> dict:
    """``deck.json``, with its mounts resolved against the deck folder."""
    path = deck / "deck.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config.setdefault("name", deck.name)
    config["mounts"] = {name.strip("/"): (deck / folder).resolve()
                        for name, folder in config.get("mounts", {}).items()}
    return config


def routes(deck: Path, config: dict) -> list[tuple[str, Path]]:
    """URL prefix -> folder, longest prefix first (so ``a/audio/`` wins over ``a/``)."""
    table = [(f"/{config['name']}/", deck.resolve())]
    table += [(f"/{name}/", folder) for name, folder in config["mounts"].items()]
    return sorted(table, key=lambda item: -len(item[0]))


def make_handler(deck: Path, config: dict):
    table = routes(deck, config)
    name = config["name"]
    vendor_niivue = f"/{name}/vendor/niivue.umd.js"

    def resolve(path: str) -> Path | None:
        for prefix, root in table:
            if path.startswith(prefix):
                relative = path[len(prefix):] or "index.html"
                if relative.endswith("/"):
                    relative += "index.html"
                target = (root / relative).resolve()
                if root in target.parents or target == root:
                    return target
        return None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def send_body(self, body: bytes, kind: str, status=HTTPStatus.OK, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", ""):
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/{name}/")
                self.end_headers()
                return
            target = resolve(path)
            if target is None or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if target.suffix == ".html":
                text = target.read_text(encoding="utf-8")
                text = NIIVUE_CDN.sub(vendor_niivue, text)
                text = FONTS.sub("", text)
                self.send_body(text.encode("utf-8"), "text/html; charset=utf-8", extra={"Cache-Control": "no-store"})
                return
            self.send_file(target)

        def send_file(self, target: Path):
            kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            size = target.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if match:
                start = int(match.group(1)) if match.group(1) else 0
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_POST(self):
            path = urlparse(self.path).path
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if path == f"/{name}/__save":
                save_slides(deck, data.decode("utf-8"))
                self.send_body(b"saved", "text/plain")
            elif path == f"/{name}/__upload":
                saved = save_upload(deck, unquote(self.headers.get("X-Filename", "image.png")), data)
                self.send_body(json.dumps({"path": saved}).encode("utf-8"), "application/json")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

    return Handler


def save_slides(deck: Path, slides: str) -> None:
    """Writes the slides into ``index.html`` between the markers, keeping a ``.bak``."""
    page = deck / "index.html"
    text = page.read_text(encoding="utf-8")
    start, stop = text.index("<!--SLIDES-->") + len("<!--SLIDES-->"), text.index("<!--/SLIDES-->")
    shutil.copy(page, page.with_suffix(".html.bak"))
    page.write_text(text[:start] + "\n" + slides.strip() + "\n" + text[stop:], encoding="utf-8")
    print(f"saved slides ({len(slides)} characters) to {page}", flush=True)


def save_upload(deck: Path, filename: str, data: bytes) -> str:
    """Saves an inserted picture as ``figures/uploads/<name>`` (a new name if taken); returns its path."""
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"
    suffix = Path(name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png"
    folder = deck / "figures" / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target, n = folder / f"{stem}{suffix}", 1
    while target.exists():
        n += 1
        target = folder / f"{stem}-{n}{suffix}"
    target.write_bytes(data)
    print(f"saved image {target}", flush=True)
    return f"figures/uploads/{target.name}"


def serve(deck: Path, port: int = 8740, open_browser: bool = True) -> None:
    deck = Path(deck).resolve()
    config = load_config(deck)
    missing = [str(folder) for folder in config["mounts"].values() if not folder.is_dir()]
    if missing:
        print("warning: mounted folders not found: " + ", ".join(missing), flush=True)
    url = f"http://127.0.0.1:{port}/{config['name']}/"
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(deck, config))
    print(f"serving {deck} at {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def default_deck() -> Path:
    """Next to this file (an offline bundle), the one folder holding a deck.json; else the current folder."""
    here = Path(__file__).resolve().parent
    decks = [p.parent for p in here.glob("*/deck.json")]
    return decks[0] if len(decks) == 1 else Path.cwd()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("deck", nargs="?", type=Path, help="the deck folder (default: the bundle's deck)")
    parser.add_argument("--port", type=int, default=8740)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(args.deck or default_deck(), args.port, not args.no_browser)


if __name__ == "__main__":
    main()
