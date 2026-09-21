from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_utils.viewer import (
    About,
    Atlas,
    CoordinateMaps,
    Endpoint,
    Features,
    MapEntry,
    Report,
    SubjectSpace,
    ViewerSpec,
    build_viewer,
)
from fmri_utils.viewer import assets, build
from fmri_utils.viewer.demo import build_demo


def _write(values: np.ndarray, path: Path, affine: np.ndarray | None = None) -> Path:
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine), str(path))
    return path


def _spec(tmp: Path, **overrides) -> ViewerSpec:
    rng = np.random.default_rng(0)
    template = _write(rng.random((8, 8, 8)) * 100, tmp / "template.nii.gz")
    maps = [
        MapEntry(subject="sub-01", path=_write(rng.normal(0, 1, (8, 8, 8)), tmp / "a.nii.gz")),
        MapEntry(subject="sub-02", path=_write(rng.normal(0, 1, (8, 8, 8)), tmp / "b.nii.gz")),
    ]
    fields = dict(
        about=About(title="Test Browser", kicker="tests"),
        template=template,
        reports=[Report(id="r", label="Report", endpoints=[
            Endpoint(
                id="e", label="Endpoint", statistic="t",
                degrees_of_freedom={"sub-01": 40, "sub-02": 40}, maps=maps,
            ),
        ])],
    )
    fields.update(overrides)
    return ViewerSpec(**fields)


class SpecTests(unittest.TestCase):
    def test_subjects_are_listed_in_order_of_appearance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_spec(Path(tmp)).subjects(), ["sub-01", "sub-02"])

    def test_check_names_the_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[
                Endpoint(id="e", label="E", maps=[
                    MapEntry(subject="sub-01", path=root / "nowhere.nii.gz"),
                ]),
            ])])
            with self.assertRaises(FileNotFoundError) as caught:
                spec.check()
            self.assertIn("nowhere.nii.gz", str(caught.exception))

    def test_check_rejects_an_endpoint_with_no_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(Path(tmp), reports=[Report(id="r", label="R", endpoints=[
                Endpoint(id="e", label="E", maps=[]),
            ])])
            with self.assertRaises(ValueError):
                spec.check()


class AssetTests(unittest.TestCase):
    def test_underlay_is_written_as_uint8_with_the_scale_in_the_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write(np.linspace(0, 400, 512).reshape((8, 8, 8)), root / "t1.nii.gz")
            assets.write_underlay(source, root / "out.nii.gz")
            image = nib.load(str(root / "out.nii.gz"))
            self.assertEqual(image.get_data_dtype(), np.uint8)
            # Stored at full 8-bit range, and still reading back in the
            # original units once the header's scale is applied.
            self.assertEqual(int(image.dataobj.get_unscaled().max()), 255)
            self.assertAlmostEqual(float(np.asarray(image.dataobj).max()), 400.0, delta=6.0)

    def test_maps_keep_float_precision_and_lose_their_nans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.full((4, 4, 4), 1.2345678, dtype=np.float32)
            values[0, 0, 0] = np.nan
            source = _write(values, root / "map.nii.gz")
            _, written = assets.write_map(source, root / "out.nii.gz")
            self.assertEqual(nib.load(str(root / "out.nii.gz")).get_data_dtype(), np.float32)
            self.assertEqual(written[0, 0, 0], 0.0)
            self.assertAlmostEqual(float(written[1, 1, 1]), 1.2345678, places=6)

    def test_threshold_curves_rise_as_p_falls(self) -> None:
        rng = np.random.default_rng(1)
        curves = assets.statistic_curves(rng.normal(0, 1, 4000).astype(np.float32), 40)
        self.assertEqual(curves["degrees_of_freedom"], 40)
        self.assertTrue(all(a > b for a, b in zip(curves["p_t"], curves["p_t"][1:])))
        self.assertIn("q", curves)

    def test_benjamini_hochberg_is_zero_when_nothing_survives(self) -> None:
        self.assertEqual(assets.benjamini_hochberg(np.full(100, 0.9), 0.05), 0.0)
        self.assertGreater(assets.benjamini_hochberg(np.full(100, 1e-8), 0.05), 0.0)

    def test_coordinate_bins_thin_the_grid_and_scale_the_affine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write(np.zeros((16, 16, 16, 3)), root / "coords.nii.gz")
            bins = assets.coordinate_bins(source, stride=4)
            self.assertEqual(bins["dims"], [4, 4, 4])
            self.assertEqual(bins["affine"][0], 8.0)
            self.assertEqual(bins["values"].shape, (3, 4, 4, 4))


class BuildTests(unittest.TestCase):
    def test_a_minimal_build_writes_a_page_a_manifest_and_the_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_viewer(_spec(root), root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

            self.assertTrue((out / "index.html").exists())
            self.assertEqual(manifest["about"]["title"], "Test Browser")
            endpoint = manifest["reports"][0]["endpoints"][0]
            self.assertEqual(endpoint["subjects"], ["sub-01", "sub-02"])
            self.assertEqual(len(endpoint["maps"]), 2)
            for entry in endpoint["maps"]:
                self.assertTrue((out / entry["path"]).exists())
                self.assertIn("thresholds", entry)
            self.assertNotIn("subject_space", manifest)
            self.assertNotIn("atlas", manifest)
            self.assertNotIn("surfaces", manifest)

    def test_an_endpoint_with_no_statistic_gets_no_threshold_curves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _spec(root)
            plain = Endpoint(
                id="plain", label="Plain", maps=spec.reports[0].endpoints[0].maps
            )
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[plain])])
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            for entry in manifest["reports"][0]["endpoints"][0]["maps"]:
                self.assertNotIn("thresholds", entry)

    def test_optional_parts_appear_only_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anatomy = _write(np.random.default_rng(2).random((8, 8, 8)) * 100,
                             root / "anat.nii.gz")
            coords = _write(np.zeros((8, 8, 8, 3)), root / "coords.nii.gz")
            labels = np.zeros((8, 8, 8), dtype=np.float32)
            labels[2:5, 2:5, 2:5] = 1
            atlas_path = _write(labels, root / "atlas.nii.gz")

            base = _spec(root)
            entry = base.reports[0].endpoints[0].maps[0]
            endpoint = Endpoint(
                id="e", label="E", statistic="t", degrees_of_freedom={"sub-01": 40},
                maps=[MapEntry(subject="sub-01", path=entry.path,
                               subject_space_path=entry.path)],
            )
            spec = _spec(
                root,
                reports=[Report(id="r", label="R", endpoints=[endpoint])],
                subject_space=SubjectSpace(
                    templates={"sub-01": anatomy},
                    templates_coarse={"sub-01": anatomy},
                    coordinates={"sub-01": CoordinateMaps(coords, coords)},
                ),
                atlases=[Atlas(id="demo", label="demo atlas",
                               labels=[(1, "Blob")], template_path=atlas_path)],
                features=Features(montage=False),
            )
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

            self.assertFalse(manifest["features"]["montage"])
            space = manifest["subject_space"]
            self.assertTrue((out / space["templates"]["sub-01"]).exists())
            self.assertTrue((out / space["coords"]["sub-01"]["anat_to_mni"]["path"]).exists())
            self.assertEqual(space["maps"][0]["endpoint"], "e")
            atlas = manifest["atlas"]["atlases"][0]
            self.assertEqual(atlas["regions"], [{"value": 1, "name": "Blob"}])
            self.assertTrue((out / atlas["mni"]).exists())


    def test_a_rebuild_names_the_files_nothing_points_at_any_more(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_viewer(_spec(root), root / "viewer", quiet=True)
            stale = out / "data" / "r" / "e" / "map_sub-99.nii.gz"
            stale.write_bytes(b"left over from an earlier shape")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("data/r/e/map_sub-99.nii.gz", build._orphans(manifest, out))
            # and the files the manifest does point at are not named
            live = manifest["reports"][0]["endpoints"][0]["maps"][0]["path"]
            self.assertNotIn(live, build._orphans(manifest, out))


class DemoTests(unittest.TestCase):
    def test_the_demo_builds_something_servable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = build_demo(Path(tmp) / "viewer")
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            page = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn("manifest.json", page)
            self.assertEqual(manifest["about"]["title"], "Demo Browser")
            self.assertEqual(len(manifest["reports"][0]["endpoints"]), 2)
            self.assertTrue((out / manifest["template"]).exists())


if __name__ == "__main__":
    unittest.main()
