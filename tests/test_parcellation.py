from __future__ import annotations

import unittest

import numpy as np

from fmri_utils.viewer.parcellation import (
    boundary_network,
    clean_parcellation,
    region_anchors,
)


def grid(n: int = 30):
    """An n x n vertex grid triangulated into 2(n-1)^2 faces."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    index = np.arange(n * n).reshape(n, n)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = index[i, j], index[i + 1, j], index[i, j + 1], index[i + 1, j + 1]
            faces += [(a, b, c), (b, d, c)]
    return points, np.asarray(faces)


def three_regions(points: np.ndarray) -> np.ndarray:
    """Left half 1; right half split into 2 (top) and 3 (bottom)."""
    x, y = points[:, 0], points[:, 1]
    labels = np.where(x < 15, 1, np.where(y < 15, 2, 3))
    return labels.astype(np.int32)


class CleanTests(unittest.TestCase):
    def test_unlabelled_cortex_is_filled(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        rng = np.random.default_rng(0)
        holes = rng.random(labels.size) < 0.3
        labels[holes] = 0
        cleaned = clean_parcellation(labels, faces)
        self.assertTrue((cleaned > 0).all())

    def test_medial_wall_stays_unlabelled(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        keep = points[:, 0] > 2
        cleaned = clean_parcellation(labels, faces, keep=keep)
        self.assertTrue((cleaned[~keep] == 0).all())
        self.assertTrue((cleaned[keep] > 0).all())

    def test_an_island_is_absorbed(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        island = (np.abs(points[:, 0] - 6) <= 1) & (np.abs(points[:, 1] - 6) <= 1)
        labels[island] = 3  # a stray piece of region 3 inside region 1
        cleaned = clean_parcellation(labels, faces, smooth_rounds=0)
        self.assertTrue((cleaned[island] == 1).all())

    def test_each_region_is_one_piece(self) -> None:
        from scipy.sparse.csgraph import connected_components
        from fmri_utils.viewer.parcellation import adjacency

        points, faces = grid()
        labels = three_regions(points)
        rng = np.random.default_rng(1)
        labels[rng.random(labels.size) < 0.05] = 2  # speckle
        cleaned = clean_parcellation(labels, faces)
        graph = adjacency(faces, labels.size)
        for value in np.unique(cleaned):
            members = np.flatnonzero(cleaned == value)
            count, _ = connected_components(graph[members][:, members], directed=False)
            self.assertEqual(count, 1, f"region {value} is in {count} pieces")


class NetworkTests(unittest.TestCase):
    def positions(self, points, indices, weights):
        return (points[indices] * weights[..., None]).sum(axis=1)

    def test_lines_meet_at_the_three_way_junction(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        indices, weights, offsets = boundary_network(labels, faces, relax=2)
        xy = self.positions(points, indices, weights)
        ends = [xy[offsets[i]] for i in range(len(offsets) - 1)] + \
               [xy[offsets[i + 1] - 1] for i in range(len(offsets) - 1)]
        ends = np.asarray(ends)
        # Three lines end at the junction near (15, 15), at the same point.
        near = ends[np.linalg.norm(ends - [14.5, 14.5], axis=1) < 1.5]
        self.assertGreaterEqual(len(near), 3)
        self.assertLess(np.ptp(near, axis=0).max(), 1e-6)

    def test_boundary_lies_between_the_regions(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        indices, weights, offsets = boundary_network(labels, faces, relax=0)
        xy = self.positions(points, indices, weights)
        # Every point of the 1|2 and 1|3 boundary is within a vertex of x = 14.5.
        vertical = xy[np.abs(xy[:, 1] - 14.5) > 1.5]
        vertical = vertical[vertical[:, 0] < 15.5]
        self.assertTrue(np.all(np.abs(vertical[:, 0] - 14.5) <= 0.5 + 1e-9))

    def test_weights_sum_to_one(self) -> None:
        points, faces = grid()
        _indices, weights, _offsets = boundary_network(three_regions(points), faces)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)

    def test_anchor_is_inside_and_away_from_the_boundary(self) -> None:
        points, faces = grid()
        labels = three_regions(points)
        anchors = region_anchors(labels, faces)
        self.assertEqual(set(anchors), {1, 2, 3})
        for value, vertex in anchors.items():
            self.assertEqual(labels[vertex], value)
        x, y = points[anchors[1]]
        self.assertTrue(4 <= x <= 11 and 5 <= y <= 24)



class WriteAndPackageTests(unittest.TestCase):
    def test_parcellation_round_trips_through_packaging(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from fmri_utils.viewer.parcellation import write_parcellation
        from fmri_utils.viewer.surfaces import package_surfaces

        points, faces = grid(12)
        labels = three_regions(points * 2.5)
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "export" / "sub-01"
            export.mkdir(parents=True)
            vertices = np.column_stack([points, np.zeros(len(points))]).astype("<f4")
            (export / "lh_faces.bin").write_bytes(faces.astype("<u4").tobytes())
            (export / "lh_wm.bin").write_bytes(vertices.tobytes())
            record = {"subject": "sub-01", "hemispheres": {"lh": {
                "faces": "lh_faces.bin", "n_faces": len(faces), "n_vertices": len(points),
                "geometries": {"wm": {"path": "lh_wm.bin", "anatomical": True}}}}}
            regions = [{"value": v, "name": f"R{v}", "abbrev": f"R{v}"} for v in (1, 2, 3)]
            record["parcellations"] = {"demo": write_parcellation(
                export, "demo", "Demo atlas", regions,
                {"lh": {"labels": labels, "faces": faces}})}
            (export / "surfaces.json").write_text(json.dumps(record))
            packed = Path(tmp) / "packed"
            package_surfaces(export.parent, packed)
            catalogue = json.loads((packed / "surfaces.json").read_text())
            parcellation = catalogue["sub-01"]["parcellations"]["demo"]
            self.assertEqual(parcellation["label"], "Demo atlas")
            hemisphere = parcellation["hemispheres"]["lh"]
            stored = np.frombuffer((packed / hemisphere["labels"]).read_bytes(), dtype="<i2")
            self.assertEqual(stored.size, len(points))
            self.assertEqual(set(np.unique(stored)), {1, 2, 3})
            offsets = np.frombuffer((packed / hemisphere["lines_offsets"]).read_bytes(), dtype="<u4")
            self.assertEqual(len(offsets) - 1, hemisphere["n_lines"])
            self.assertEqual(set(hemisphere["anchors"]), {"1", "2", "3"})


class RegionInfoTests(unittest.TestCase):
    def test_every_region_has_a_description_and_a_link(self) -> None:
        from fmri_utils.viewer.region_info import harvard_oxford_cortical

        info = harvard_oxford_cortical()
        self.assertEqual(len(info), 48)
        for name, entry in info.items():
            self.assertTrue(entry["abbrev"] and entry["description"], name)
            self.assertTrue(entry["related"], name)
            for work in entry["related"]:
                self.assertTrue(work["url"].startswith("https://scholar.google.com/"))


if __name__ == "__main__":
    unittest.main()
