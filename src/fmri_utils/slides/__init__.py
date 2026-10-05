"""Web slide decks that embed fmri_utils viewers, with an in-browser editor.

See ``docs/slides.md``. The pieces:

- ``resources/``: ``deck.js`` (navigation, steps and animations, embeds, brain views and cluster
  lists driving a same-origin viewer, KaTeX, charts), ``editor.js`` (slide sorter and editor),
  ``deck.css``, ``template.html`` and ``vendor/`` (KaTeX, NiiVue for offline use).
- ``server``: a standard-library server for a deck and the folders it embeds; the editor saves
  through it.
- ``bundle``: an offline copy of a deck and everything it embeds.
- ``deck``: create a deck folder; refresh a deck's assets.
"""

from .bundle import build_bundle
from .deck import copy_assets, new_deck
from .server import load_config, serve

__all__ = ["build_bundle", "copy_assets", "load_config", "new_deck", "serve"]
