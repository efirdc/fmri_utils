from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_utils.viewer import (
    About,
    AnatomyImage,
    Atlas,
    Cohort,
    CoordinateMaps,
    Endpoint,
    Features,
    MapEntry,
    Report,
    SubjectSpace,
    Variant,
    VariantControl,
    VariantOption,
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

    def test_variant_controls_and_significance_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            effect = _write(rng.normal(0, 1, (8, 8, 8)), root / "effect.nii.gz")
            neglog = _write(rng.random((8, 8, 8)) * 2, root / "neglog10p.nii.gz")
            controls = [
                VariantControl(id="delta", label="Δr", on="1", off="0"),
                VariantControl(id="test", label="Group test", options=[
                    VariantOption("none", "none"),
                    VariantOption("tfce", "TFCE", unavailable="mean of features only"),
                ]),
            ]
            variants = [
                Variant("r", "r", values={"delta": "0", "test": "none"}),
                Variant("d", "Δr", values={"delta": "1", "test": "none"}),
                Variant("dt", "Δr, TFCE", values={"delta": "1", "test": "tfce"},
                        threshold_default=("p", 0.05),
                        subject_display_ranges={"group": (0.0, 0.01)}),
            ]
            maps = [
                MapEntry(subject="group", path=effect, variant="r"),
                MapEntry(subject="group", path=effect, variant="d"),
                MapEntry(subject="group", path=effect, variant="dt", significance_p=neglog,
                         significance_label="TFCE FWE"),
            ]
            endpoint = Endpoint(id="e", label="E", maps=maps, variants=variants,
                                variant_controls=controls)
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[endpoint])])
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            built = manifest["reports"][0]["endpoints"][0]
            self.assertEqual([c["id"] for c in built["variant_controls"]], ["delta", "test"])
            self.assertEqual(built["variant_controls"][0]["kind"], "toggle")
            self.assertEqual(built["variant_controls"][1]["options"][1]["unavailable"],
                             "mean of features only")
            self.assertNotIn("variant_control", built)
            tfce = built["variants"][2]
            self.assertEqual(tfce["values"], {"delta": "1", "test": "tfce"})
            self.assertEqual(tfce["threshold_default"], {"mode": "p", "value": 0.05})
            self.assertEqual(tfce["subject_ranges"], {"group": [0.0, 0.01]})
            with_sig = [m for m in built["maps"] if m.get("significance")]
            self.assertEqual(len(with_sig), 1)
            self.assertEqual(with_sig[0]["significance"]["label"], "TFCE FWE")
            self.assertTrue((out / with_sig[0]["significance"]["p"]).exists())
            self.assertNotIn("q", with_sig[0]["significance"])

    def test_cohorts_reach_the_manifest_with_their_own_files_and_curves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(1)
            new = _write(rng.normal(0, 1, (8, 8, 8)), root / "group_new.nii.gz")
            old = _write(rng.normal(0, 1, (8, 8, 8)), root / "group_old.nii.gz")
            subject = _write(rng.normal(0, 1, (8, 8, 8)), root / "sub.nii.gz")
            spec = _spec(root, cohorts=[Cohort("n2", "2 participants"), Cohort("all", "All 3")],
                         excluded_subjects={"sub-03": "example reason"},
                         reports=[Report(id="r", label="R", endpoints=[Endpoint(
                             id="e", label="E", statistic="t", degrees_of_freedom={"group": 1},
                             maps=[MapEntry(subject="group", path=new),
                                   MapEntry(subject="group", path=old, cohort="all", degrees_of_freedom=2),
                                   MapEntry(subject="sub-01", path=subject)])])])
            spec.check()
            out = root / "viewer"
            build_viewer(spec, out, quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([c["id"] for c in manifest["cohorts"]], ["n2", "all"])
            self.assertEqual(manifest["excluded_subjects"], {"sub-03": "example reason"})
            maps = manifest["reports"][0]["endpoints"][0]["maps"]
            groups = [m for m in maps if m["subject"] == "group"]
            self.assertEqual(sorted(m.get("cohort", "") for m in groups), ["", "all"])
            self.assertEqual(len({m["path"] for m in groups}), 2)
            # Each group map's p curve uses its own degrees of freedom.
            by_cohort = {m.get("cohort", ""): m for m in groups}
            self.assertNotEqual(by_cohort[""]["thresholds"], by_cohort["all"]["thresholds"])

    def test_fwe_companion_and_defaults_reach_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(3)
            t = _write(rng.normal(0, 1, (8, 8, 8)), root / "t.nii.gz")
            p = _write(rng.random((8, 8, 8)) * 3, root / "p.nii.gz")
            q = _write(rng.random((8, 8, 8)), root / "q.nii.gz")
            fwe = _write(rng.random((8, 8, 8)), root / "fwe.nii.gz")
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[Endpoint(
                id="e", label="E", statistic="t", degrees_of_freedom={"group": 9},
                maps=[MapEntry(subject="group", path=t, significance_p=p, significance_q=q,
                               significance_fwe=fwe, significance_label="permutation",
                               significance_defaults={"p": 0.005})])])])
            spec.check()
            out = root / "viewer"
            build_viewer(spec, out, quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            sig = manifest["reports"][0]["endpoints"][0]["maps"][0]["significance"]
            self.assertEqual(set(sig), {"p", "q", "fwe", "label", "defaults"})
            self.assertEqual(sig["defaults"], {"p": 0.005})
            self.assertTrue((out / sig["fwe"]).exists())
            bad = _spec(root, reports=[Report(id="r", label="R", endpoints=[Endpoint(
                id="e", label="E", maps=[MapEntry(subject="group", path=t, significance_p=p,
                                                  significance_defaults={"fdr": 0.05})])])])
            with self.assertRaises(ValueError):
                bad.check()

    def test_template_space_reaches_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = _spec(root)
            endpoint = base.reports[0].endpoints[0]
            other = Endpoint(id="e2", label="E2", maps=list(endpoint.maps), template_space="MNI152NLin2009cAsym")
            spec = _spec(root, template_space="MNI152NLin6Asym",
                         reports=[Report(id="r", label="R", endpoints=[endpoint, other])])
            out = root / "viewer"
            build_viewer(spec, out, quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["template_space"], "MNI152NLin6Asym")
            endpoints = manifest["reports"][0]["endpoints"]
            self.assertNotIn("template_space", endpoints[0])
            self.assertEqual(endpoints[1]["template_space"], "MNI152NLin2009cAsym")
            plain = root / "plain"
            build_viewer(_spec(root), plain, quiet=True)
            self.assertNotIn("template_space", json.loads((plain / "manifest.json").read_text(encoding="utf-8")))

    def test_a_shared_map_and_a_cohort_map_may_share_a_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write(np.ones((8, 8, 8)), root / "m.nii.gz")
            spec = _spec(root, cohorts=[Cohort("a", "A"), Cohort("b", "B")], reports=[
                Report(id="r", label="R", endpoints=[Endpoint(id="e", label="E", maps=[
                    MapEntry(subject="s", path=path),                # shared
                    MapEntry(subject="s", path=path, cohort="b"),    # b's own
                    MapEntry(subject="t", path=path, cohort="a"),    # only in a
                ])])])
            spec.check()
            twice = _spec(root, cohorts=[Cohort("a", "A")], reports=[Report(id="r", label="R", endpoints=[
                Endpoint(id="e", label="E", maps=[MapEntry(subject="s", path=path, cohort="a"),
                                                  MapEntry(subject="s", path=path, cohort="a")])])])
            with self.assertRaises(ValueError):
                twice.check()

    def test_check_rejects_an_unknown_cohort_and_a_repeated_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write(np.ones((8, 8, 8)), root / "m.nii.gz")
            unknown = _spec(root, cohorts=[Cohort("a", "A")], reports=[Report(id="r", label="R", endpoints=[
                Endpoint(id="e", label="E", maps=[MapEntry(subject="s", path=path, cohort="b")])])])
            with self.assertRaises(ValueError):
                unknown.check()
            repeated = _spec(root, reports=[Report(id="r", label="R", endpoints=[
                Endpoint(id="e", label="E", maps=[MapEntry(subject="s", path=path),
                                                  MapEntry(subject="s", path=path)])])])
            with self.assertRaises(ValueError):
                repeated.check()

    def test_check_rejects_a_variant_value_no_control_offers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = _spec(root)
            maps = [MapEntry(subject="sub-01", path=spec.reports[0].endpoints[0].maps[0].path,
                             variant="a")]
            endpoint = Endpoint(
                id="e", label="E", maps=maps,
                variants=[Variant("a", "a", values={"test": "bogus"})],
                variant_controls=[VariantControl(id="test", label="T",
                                                 options=[VariantOption("none", "none")])],
            )
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[endpoint])])
            with self.assertRaises(ValueError):
                spec.check()

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


    def test_registration_maps_carry_their_underlay_frame_and_warp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(4)
            anatomy = _write(rng.random((12, 12, 12)) * 100 + 1, root / "anat.nii.gz")
            boldref = _write(rng.random((6, 6, 6)) * 50 + 1, root / "boldref.nii.gz",
                             np.diag([2.0, 2.0, 2.0, 1.0]))
            # A field that is exactly an affine: template mm -> 2 x + 3.
            ijk = np.stack(np.meshgrid(*[np.arange(12.0)] * 3, indexing="ij"), -1)
            field = _write(2.0 * ijk + 3.0, root / "field.nii.gz")
            shift = [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            endpoint = Endpoint(
                id="reg", label="Reg", display="anatomy", outlines="both",
                maps=[MapEntry(subject="sub-01", path=anatomy, underlay=boldref,
                               surface_affine=shift, variant="bold"),
                      MapEntry(subject="sub-01", path=anatomy, underlay=boldref,
                               surface_frame="subject", variant="t1w"),
                      MapEntry(subject="sub-01", path=anatomy, warp=True)],
                variants=[Variant("bold", "BOLD"), Variant("t1w", "T1w")],
            )
            spec = _spec(
                root, reports=[Report(id="r", label="R", endpoints=[endpoint])],
                subject_space=SubjectSpace(
                    templates={"sub-01": anatomy},
                    coordinates={"sub-01": CoordinateMaps(field, field, stride=1)},
                ),
            )
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            record = manifest["reports"][0]["endpoints"][0]
            self.assertEqual(record["display"], "anatomy")
            self.assertEqual(record["outlines"], "both")
            bold, t1w, warp = record["maps"]
            # Anatomy is stored as the underlays are, and never gets p curves.
            self.assertEqual(nib.load(str(out / bold["path"])).get_data_dtype(), np.uint8)
            self.assertNotIn("thresholds", bold)
            self.assertEqual(bold["range"][0], 0.0)
            # One underlay file however many maps share it.
            self.assertEqual(bold["underlay"], t1w["underlay"])
            self.assertTrue((out / bold["underlay"]).exists())
            self.assertEqual(bold["surface_affine"], [[float(v) for v in row] for row in shift])
            self.assertEqual(t1w["surface_frame"], "subject")
            self.assertTrue(warp["warp"])
            linear = np.array(manifest["subject_space"]["coords"]["sub-01"]["linear_from_template"]).reshape(4, 4)
            # Fitted to the field when not given. The field is 2 ijk + 3 on a
            # 2 mm grid (mm = 2 ijk), so in millimetres it is mm + 3.
            np.testing.assert_allclose(linear[:3, :3], np.eye(3), atol=1e-6)
            np.testing.assert_allclose(linear[:3, 3], [3.0, 3.0, 3.0], atol=1e-5)

    def test_a_given_linear_part_is_used_as_is_and_bad_fields_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anatomy = _write(np.ones((6, 6, 6)), root / "anat.nii.gz")
            field = _write(np.ones((6, 6, 6, 3)), root / "field.nii.gz")
            given = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
            spec = _spec(root, subject_space=SubjectSpace(
                templates={"sub-01": anatomy},
                coordinates={"sub-01": CoordinateMaps(field, field, stride=1, linear=given)}))
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["subject_space"]["coords"]["sub-01"]["linear_from_template"],
                             [float(v) for row in given for v in row])
            # A pure translation is its own rigid part.
            np.testing.assert_allclose(manifest["subject_space"]["coords"]["sub-01"]["rigid_from_template"],
                                       [float(v) for row in given for v in row], atol=1e-9)
            # A scaled, rotated linear part: the rigid part is a rotation that
            # agrees with the linear part at the brain's centre.
            angle = np.radians(20)
            rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            scaled = np.eye(4)
            scaled[:3, :3] = rot @ np.diag([0.9, 1.1, 1.0])
            scaled[:3, 3] = [4, -2, 7]
            rigid = np.asarray(build._rigid(scaled.tolist(), CoordinateMaps(field, field, stride=1)))
            np.testing.assert_allclose(rigid[:3, :3] @ rigid[:3, :3].T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rigid[:3, :3]), 1.0)
            entry = spec.reports[0].endpoints[0].maps[0]
            for bad in (dict(display="fancy"), dict(outlines="grey")):
                with self.assertRaises(ValueError):
                    _spec(root, reports=[Report(id="r", label="R", endpoints=[
                        Endpoint(id="e", label="E", maps=[entry], **bad)])]).check()
            with self.assertRaises(ValueError):
                _spec(root, reports=[Report(id="r", label="R", endpoints=[
                    Endpoint(id="e", label="E", maps=[MapEntry(subject="sub-01", path=entry.path,
                                                               warp=True)])])]).check()

    def test_an_image_stack_reaches_the_manifest_with_a_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(6)
            anatomy = _write(rng.random((12, 12, 12)) * 100 + 1, root / "anat.nii.gz")
            boldref = _write(rng.random((6, 6, 6)) * 50 + 1, root / "boldref.nii.gz")
            ijk = np.stack(np.meshgrid(*[np.arange(12.0)] * 3, indexing="ij"), -1)
            field = _write(2.0 * ijk + 3.0, root / "field.nii.gz")
            shift = [[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            base = _spec(root)
            images = (AnatomyImage("bold", "boldref", boldref, frame="affine", affine=shift),
                      AnatomyImage("t1", "T1", anatomy),
                      AnatomyImage("mni", "MNI", base.template, frame="template"))
            endpoint = Endpoint(id="reg", label="Reg", display="anatomy", maps=[
                MapEntry(subject="sub-01", path=anatomy, feature=f"run {run}", images=images)
                for run in (1, 2)])
            spec = _spec(root, reports=[Report(id="r", label="R", endpoints=[endpoint])],
                         subject_space=SubjectSpace(
                             templates={"sub-01": anatomy},
                             coordinates={"sub-01": CoordinateMaps(field, field, stride=1)}))
            out = build_viewer(spec, root / "viewer", quiet=True)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            first, second = manifest["reports"][0]["endpoints"][0]["maps"]
            self.assertEqual([i["frame"] for i in first["images"]], ["affine", "subject", "template"])
            self.assertEqual(first["images"][0]["affine"], [[float(v) for v in row] for row in shift])
            self.assertTrue(all(i["range"][1] > 0 for i in first["images"]))
            # Written once: both runs share the images, the map file and the canvas.
            self.assertEqual(first["images"], second["images"])
            self.assertEqual(first["path"], second["path"])
            self.assertEqual(first["underlay"], second["underlay"])
            canvas = nib.load(str(out / first["underlay"]))
            self.assertEqual(int(np.asarray(canvas.dataobj).max()), 0)
            # A stack needs an anatomy endpoint and the subject's coordinates.
            with self.assertRaises(ValueError):
                _spec(root, reports=[Report(id="r", label="R", endpoints=[Endpoint(
                    id="e", label="E", maps=[MapEntry(subject="sub-01", path=anatomy, images=images)])])],
                    subject_space=spec.subject_space).check()
            with self.assertRaises(ValueError):
                _spec(root, reports=[Report(id="r", label="R", endpoints=[Endpoint(
                    id="e", label="E", display="anatomy",
                    maps=[MapEntry(subject="sub-01", path=anatomy, images=images)])])]).check()

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
