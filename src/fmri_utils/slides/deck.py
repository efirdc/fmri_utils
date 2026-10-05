"""Create a deck folder, and refresh a deck's copy of the slide assets."""

from __future__ import annotations

import html
import json
import shutil
from importlib import resources
from pathlib import Path

ASSETS = ("deck.js", "editor.js", "deck.css")
# A deck's own charts (not an asset: "assets" never overwrites it).
CHARTS_STUB = """// This deck's interactive charts: deck.js draws window.DeckCharts[name](element) into each
// <div class="chart" data-chart="name">.
window.DeckCharts = Object.assign(window.DeckCharts || {}, {
});
"""


def resource_root() -> Path:
    return Path(str(resources.files("fmri_utils.slides") / "resources"))


def copy_assets(deck: Path) -> list[str]:
    """Copies deck.js, editor.js, deck.css and vendor/ (KaTeX, NiiVue) into the deck, overwriting."""
    root, deck = resource_root(), Path(deck)
    deck.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        shutil.copy2(root / name, deck / name)
    shutil.copytree(root / "vendor", deck / "vendor", dirs_exist_ok=True)
    return [*ASSETS, "vendor/"]


def new_deck(deck: Path, name: str | None = None, title: str = "Untitled talk", byline: str = "",
             public_base: str = "", mounts: dict[str, str] | None = None) -> Path:
    """A new deck: index.html with one title slide, deck.json and the assets."""
    deck = Path(deck)
    if (deck / "index.html").exists():
        raise FileExistsError(f"{deck / 'index.html'} exists; refresh its assets with `fmri-slides assets` instead")
    name = name or deck.name
    copy_assets(deck)
    page = (resource_root() / "template.html").read_text(encoding="utf-8")
    page = (page.replace("{{title}}", html.escape(title))
                .replace("{{byline}}", html.escape(byline))
                .replace("{{public_base}}", html.escape(public_base, quote=True)))
    (deck / "index.html").write_text(page, encoding="utf-8")
    (deck / "deck.json").write_text(json.dumps(
        {"name": name, "title": title, "public_base": public_base, "mounts": mounts or {}}, indent=1), encoding="utf-8")
    for folder in ("data", "figures"):
        (deck / folder).mkdir(exist_ok=True)
    (deck / "charts.js").write_text(CHARTS_STUB, encoding="utf-8")
    return deck
