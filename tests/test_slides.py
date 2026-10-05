"""fmri_utils.slides: scaffolding, the server's routing and endpoints, and the offline bundle."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from fmri_utils.slides import bundle, deck, server


class Slides(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.viewer = root / "results" / "viewer"
        (self.viewer / "data").mkdir(parents=True)
        (self.viewer / "index.html").write_text("<html>viewer <script src=\"https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/niivue.umd.js\"></script></html>")
        (self.viewer / "data" / "used.bin").write_bytes(b"x" * 100)
        (self.viewer / "data" / "stale.bin").write_bytes(b"y")
        (self.viewer / "manifest.json").write_text(json.dumps({"maps": [{"path": "data/used.bin"}]}))
        (self.viewer / "surfaces.json").write_text("{}")
        self.deck = deck.new_deck(root / "talks" / "my-talk", title="My talk", byline="Me",
                                  mounts={"viewer": "../../results/viewer"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_deck(self):
        for name in ("index.html", "deck.json", "deck.js", "editor.js", "deck.css", "vendor/katex/katex.min.js",
                     "vendor/niivue.umd.js"):
            self.assertTrue((self.deck / name).exists(), name)
        page = (self.deck / "index.html").read_text(encoding="utf-8")
        self.assertIn("<!--SLIDES-->", page)
        self.assertIn("My talk", page)
        with self.assertRaises(FileExistsError):
            deck.new_deck(self.deck)

    def test_config_and_routes(self):
        config = server.load_config(self.deck)
        self.assertEqual(config["name"], "my-talk")
        self.assertEqual(config["mounts"]["viewer"], self.viewer.resolve())
        prefixes = [p for p, _ in server.routes(self.deck, {**config, "mounts": {"a": self.viewer, "a/audio": self.viewer}})]
        self.assertLess(prefixes.index("/a/audio/"), prefixes.index("/a/"))

    def test_save_and_upload(self):
        server.save_slides(self.deck, '<section class="slide"><h2>New</h2></section>')
        page = (self.deck / "index.html").read_text(encoding="utf-8")
        self.assertIn("<h2>New</h2>", page)
        self.assertEqual(page.count("<!--SLIDES-->"), 1)
        self.assertTrue((self.deck / "index.html.bak").exists())
        first = server.save_upload(self.deck, "My Figure!.PNG", b"png")
        second = server.save_upload(self.deck, "My Figure!.PNG", b"png")
        self.assertEqual(first, "figures/uploads/My_Figure.png")
        self.assertEqual(second, "figures/uploads/My_Figure-2.png")

    def test_serving(self):
        config = server.load_config(self.deck)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(self.deck, config))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            viewer_page = urllib.request.urlopen(f"{base}/viewer/").read().decode()
            self.assertIn("/my-talk/vendor/niivue.umd.js", viewer_page)   # CDN rewritten to the deck's copy
            request = urllib.request.Request(f"{base}/viewer/data/used.bin", headers={"Range": "bytes=10-19"})
            response = urllib.request.urlopen(request)
            self.assertEqual(response.status, 206)
            self.assertEqual(len(response.read()), 10)
            self.assertIn("deck.js", urllib.request.urlopen(f"{base}/my-talk/").read().decode())
        finally:
            httpd.shutdown()

    def test_bundle(self):
        out = Path(self.tmp.name) / "bundle"
        bundle.build_bundle(self.deck, out)
        self.assertTrue((out / "my-talk" / "index.html").exists())
        self.assertTrue((out / "viewer" / "data" / "used.bin").exists())
        self.assertFalse((out / "viewer" / "data" / "stale.bin").exists())
        self.assertEqual(json.loads((out / "my-talk" / "deck.json").read_text())["mounts"], {"viewer": "../viewer"})
        self.assertTrue((out / "serve.py").exists())


if __name__ == "__main__":
    unittest.main()
