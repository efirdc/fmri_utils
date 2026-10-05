"""fmri-slides: make, serve, edit and bundle web slide decks that embed fmri_utils viewers.

    fmri-slides new DECK --title "My talk" [--name talk-2026-10-06] [--public-base URL]
                         [--mount viewer=../results/viewer ...]
    fmri-slides serve DECK [--port 8740] [--no-browser]      # local copy; E edits, Ctrl+S saves
    fmri-slides assets DECK                                  # refresh deck.js / editor.js / css / vendor
    fmri-slides bundle DECK --output DIR                     # offline copy (needs only Python 3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import bundle, deck, server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="fmri-slides", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("new", help="create a deck folder")
    p.add_argument("deck", type=Path)
    p.add_argument("--name", help="the deck's URL folder (default: the folder's name)")
    p.add_argument("--title", default="Untitled talk")
    p.add_argument("--byline", default="")
    p.add_argument("--public-base", default="", help="where the deck and its mounts are published")
    p.add_argument("--mount", action="append", default=[], metavar="NAME=FOLDER",
                   help="a sibling folder the slides embed as ../NAME/ (relative to the deck)")
    p = sub.add_parser("serve", help="serve a deck locally, with its mounts and the editor")
    p.add_argument("deck", type=Path)
    p.add_argument("--port", type=int, default=8740)
    p.add_argument("--no-browser", action="store_true")
    p = sub.add_parser("assets", help="refresh a deck's copy of the slide assets")
    p.add_argument("deck", type=Path)
    p = sub.add_parser("bundle", help="copy a deck and what it embeds into an offline folder")
    p.add_argument("deck", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "new":
        mounts = dict(item.split("=", 1) for item in args.mount)
        made = deck.new_deck(args.deck, args.name, args.title, args.byline, args.public_base, mounts)
        print(f"created {made}; serve it with: fmri-slides serve {made}")
    elif args.command == "serve":
        server.serve(args.deck, args.port, not args.no_browser)
    elif args.command == "assets":
        print("copied " + ", ".join(deck.copy_assets(args.deck)) + f" into {args.deck}")
    elif args.command == "bundle":
        bundle.build_bundle(args.deck, args.output)


if __name__ == "__main__":
    main()
