"""Variant toggles, region-level endpoints, and atlas parcellations on surfaces."""

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
    Endpoint,
    MapEntry,
    RegionRow,
    RegionStats,
    Report,
    Variant,
    VariantToggle,
    ViewerSpec,
    add_atlas_parcellation,
    build_viewer,
    fsl_atlas_labels,
    table_labels,
)


def _write(values: np.ndarray, path: Path, affine: np.ndarray | None = None) -> Path:
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(values), affine), str(path))
    return path


def _base(tmp: Path) -> dict:
    rng = np.random.default_rng(1)
    return {
        "template": _write((rng.random((8, 8, 8)) * 100).astype(np.float32), tmp / "t.nii.gz"),
        "net": _write(rng.normal(0, 1, (8, 8, 8)).astype(np.float32), tmp / "net.nii.gz"),
        "raw": _write(rng.normal(0, 2, (8, 8, 8)).astype(np.float32), tmp / "raw.nii.gz"),
    }


class VariantToggleTests(unittest.TestCase):
    def test_a_toggle_and_a_variant_statistic_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = _base(tmp)
            endpoint = Endpoint(
                id="delta", label="Δr", maps=[
                    MapEntry(subject="s1", path=files["net"], variant="net"),
                    MapEntry(subject="s1", path=files["raw"], variant="raw"),
                ],
                variants=[Variant(id="net", label="net", display_range=(0.1, 1.0), statistic="q"),
                          Variant(id="raw", label="raw", blurb="Raw.")],
                variant_toggle=VariantToggle(label="net of control", on="net", off="raw",
                                             tip="Subtract the control."),
            )
            spec = ViewerSpec(template=files["template"], about=About(),
                              reports=[Report(id="r", label="R", endpoints=[endpoint])])
            manifest = json.loads((build_viewer(spec, tmp / "out", quiet=True)
                                   / "manifest.json").read_text(encoding="utf-8"))
            out = manifest["reports"][0]["endpoints"][0]
            self.assertEqual(out["variant_control"], {"kind": "toggle", "label": "net of control",
                                                      "tip": "Subtract the control.",
                                                      "on": "net", "off": "raw"})
            self.assertEqual(out["variants"][0]["statistic"], "q")
            self.assertEqual(out["variants"][0]["range"], [0.1, 1.0])
            self.assertNotIn("statistic", out["variants"][1])
            self.assertEqual(sorted(m["variant"] for m in out["maps"]), ["net", "raw"])

    def test_a_toggle_must_name_the_endpoints_two_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _base(Path(tmp))
            endpoint = Endpoint(
                id="e", label="E", maps=[MapEntry(subject="s1", path=files["net"], variant="a")],
                variants=[Variant(id="a", label="a"), Variant(id="b", label="b")],
                variant_toggle=VariantToggle(label="x", on="a", off="c"),
            )
            spec = ViewerSpec(template=files["template"],
                              reports=[Report(id="r", label="R", endpoints=[endpoint])])
            with self.assertRaises(ValueError):
                spec.check()


class RegionStatsTests(unittest.TestCase):
    def _atlas(self, tmp: Path) -> Atlas:
        labels = np.zeros((8, 8, 8), dtype=np.int16)
        labels[:4], labels[4:] = 1, 2
        return Atlas(id="cort", label="cortical", labels=[(1, "Left Area"), (2, "Right Area")],
                     template_path=_write(labels, tmp / "atlas.nii.gz"))

    def test_region_stats_and_per_map_rows_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = _base(tmp)
            rows = [RegionRow(name="Left Area", delta=0.004, value=2.5, t=5.1, p=1e-4, q=3e-3,
                              significant=True, n_positive=8, n=8, atlas="cortical")]
            endpoint = Endpoint(
                id="region", label="Region test",
                maps=[MapEntry(subject="group", path=files["net"], variant="q", region_rows=rows),
                      MapEntry(subject="group", path=files["raw"], variant="p")],
                variants=[Variant(id="q", label="FDR q", statistic="q"),
                          Variant(id="p", label="uncorrected p", statistic="p")],
                region_stats=RegionStats(atlas="ho", volume_atlas=["cort"], effect_label="Δr",
                                         rows=[RegionRow(name="Right Area", delta=-0.001)]),
            )
            spec = ViewerSpec(template=files["template"], atlases=[self._atlas(tmp)],
                              reports=[Report(id="r", label="R", endpoints=[endpoint])])
            manifest = json.loads((build_viewer(spec, tmp / "out", quiet=True)
                                   / "manifest.json").read_text(encoding="utf-8"))
            out = manifest["reports"][0]["endpoints"][0]
            self.assertEqual(out["region_stats"]["atlas"], "ho")
            self.assertEqual(out["region_stats"]["volume_atlas"], ["cort"])
            self.assertEqual(out["region_stats"]["effect_label"], "Δr")
            self.assertEqual(out["region_stats"]["rows"],
                             [{"name": "Right Area", "delta": -0.001, "significant": False}])
            row = out["maps"][0]["region_rows"][0]
            self.assertEqual((row["name"], row["q"], row["n_positive"], row["significant"]),
                             ("Left Area", 3e-3, 8, True))
            self.assertNotIn("region_rows", out["maps"][1])

    def test_region_stats_must_name_atlases_in_the_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = _base(Path(tmp))
            endpoint = Endpoint(id="e", label="E", maps=[MapEntry(subject="g", path=files["net"])],
                                region_stats=RegionStats(atlas="ho", volume_atlas=["missing"]))
            spec = ViewerSpec(template=files["template"],
                              reports=[Report(id="r", label="R", endpoints=[endpoint])])
            with self.assertRaises(ValueError):
                spec.check()


def _grid_export(root: Path, subject: str, offset=(0.0, 0.0, 0.0), transform=None) -> Path:
    """One hemisphere: a 10 x 10 vertex sheet, pial 2 mm above white."""
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    n = 10
    xs, ys = np.meshgrid(np.arange(n) * 2.0, np.arange(n) * 2.0, indexing="ij")
    white = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)]) + np.asarray(offset)
    pial = white + [0.0, 0.0, 2.0]
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            faces += [(a, b, c), (a, c, d)]
    faces = np.asarray(faces, dtype="<u4")
    (folder / "lh_faces.bin").write_bytes(faces.tobytes())
    (folder / "lh_wm.bin").write_bytes(white.astype("<f4").tobytes())
    (folder / "lh_pia.bin").write_bytes(pial.astype("<f4").tobytes())
    record = {"subject": subject, "hemispheres": {"lh": {
        "faces": "lh_faces.bin", "n_faces": int(len(faces)), "n_vertices": n * n,
        "geometries": {"wm": {"path": "lh_wm.bin"}, "pia": {"path": "lh_pia.bin"}}}}}
    if transform is not None:
        record["volume_transform"] = transform
    (folder / "surfaces.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def _split_atlas(path: Path) -> Path:
    """Label 1 for x < 9 mm, 2 above; label 3 (a tissue class) nowhere on the sheet."""
    values = np.zeros((12, 12, 4), dtype=np.int16)
    values[:5], values[5:] = 1, 2
    values[:, :, 3] = 3
    return _write(values, path)


class AtlasSurfaceTests(unittest.TestCase):
    regions = [{"value": 1, "name": "West", "abbrev": "W"}, {"value": 2, "name": "East", "abbrev": "E"},
               {"value": 3, "name": "White Matter", "abbrev": "WM"}]

    def test_an_atlas_becomes_a_parcellation_in_the_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = _grid_export(tmp, "sub-01")
            record = add_atlas_parcellation(folder, "split", "Split", _split_atlas(tmp / "a.nii.gz"),
                                            self.regions, exclude=["White Matter"], smooth_rounds=0)
            self.assertEqual([r["name"] for r in record["regions"]], ["West", "East"])
            labels = np.frombuffer((folder / record["hemispheres"]["lh"]["labels"]).read_bytes(),
                                   dtype="<i2").reshape(10, 10)
            self.assertTrue((labels[:4] == 1).all() and (labels[5:] == 2).all())
            self.assertGreater(record["hemispheres"]["lh"]["n_lines"], 0)
            saved = json.loads((folder / "surfaces.json").read_text(encoding="utf-8"))
            self.assertIn("split", saved["parcellations"])

    def test_a_sparse_atlas_leaves_uncovered_cortex_unlabelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = _grid_export(tmp, "sub-01")
            values = np.zeros((12, 12, 4), dtype=np.int16)
            values[1:4, 1:4] = 1  # one small parcel in a corner of the sheet
            atlas = _write(values, tmp / "parcel.nii.gz")
            region = [{"value": 1, "name": "Parcel", "abbrev": "P"}]
            record = add_atlas_parcellation(folder, "p", "Parcel", atlas, region,
                                            fill_mm=0.0, smooth_rounds=2, sparse=True)
            labels = np.frombuffer((folder / record["hemispheres"]["lh"]["labels"]).read_bytes(),
                                   dtype="<i2")
            self.assertGreater((labels == 1).sum(), 0)
            self.assertGreater((labels == 0).sum(), 50)
            self.assertGreater(record["hemispheres"]["lh"]["n_lines"], 0)
            self.assertEqual(list(record["hemispheres"]["lh"]["anchors"]), ["1"])
            dense = add_atlas_parcellation(folder, "d", "Dense", atlas, region,
                                           fill_mm=0.0, smooth_rounds=2)
            filled = np.frombuffer((folder / dense["hemispheres"]["lh"]["labels"]).read_bytes(),
                                   dtype="<i2")
            self.assertTrue((filled == 1).all())

    def test_the_exports_volume_transform_is_applied(self) -> None:
        # Surface millimetres 100 mm to the left of the atlas, as fsaverage's
        # MNI305 is (slightly) off MNI152; the export's transform brings them back.
        shift = [[1, 0, 0, 100.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folder = _grid_export(tmp, "fsaverage", offset=(-100.0, 0.0, 0.0), transform=shift)
            record = add_atlas_parcellation(folder, "split", "Split", _split_atlas(tmp / "a.nii.gz"),
                                            self.regions[:2], smooth_rounds=0)
            self.assertEqual(len(record["regions"]), 2)

    def test_region_lists_from_fsl_xml_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "atlas.xml").write_text(
                '<atlas><data><label index="0" x="1" y="1" z="1">Frontal Pole</label>'
                '<label index="1">Left Thalamus</label></data></atlas>', encoding="utf-8")
            regions = fsl_atlas_labels(tmp / "atlas.xml",
                                       {"Frontal Pole": {"abbrev": "FP"}, "Thalamus": "Th"})
            self.assertEqual(regions, [{"value": 1, "name": "Frontal Pole", "abbrev": "FP"},
                                       {"value": 2, "name": "Left Thalamus", "abbrev": "Th"}])
            (tmp / "lut.txt").write_text("# comment\n0 Unknown 0 0 0 0\n17 Left-Hippocampus 220 216 20 0\n",
                                         encoding="utf-8")
            (tmp / "table.tsv").write_text("value\tname\tabbrev\n5\tArea Five\tA5\n", encoding="utf-8")
            self.assertEqual(table_labels(tmp / "lut.txt"),
                             [{"value": 17, "name": "Left-Hippocampus", "abbrev": "Left-Hippocampus"}])
            self.assertEqual(table_labels(tmp / "table.tsv"),
                             [{"value": 5, "name": "Area Five", "abbrev": "A5"}])


if __name__ == "__main__":
    unittest.main()
