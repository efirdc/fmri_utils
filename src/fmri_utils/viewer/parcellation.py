"""Surface parcellations: one clean label per vertex, and one boundary network.

A volume atlas sampled onto a cortical surface arrives ragged: vertices whose
voxel fell below the atlas threshold carry no label, voxel-sized speckles of
one region sit inside another, and boundaries follow the voxel grid. Drawing
each region's outline separately from that makes two slightly different lines
wherever regions meet, and gaps at every corner where three meet.

This module fixes the parcellation first and draws once:

``clean_parcellation``
    fills unlabelled cortex from its labelled neighbours, smooths boundaries
    by diffusing each region's indicator over the mesh, and folds away islands, so every
    cortical vertex has a label and each region is one piece per hemisphere
    (apart from genuinely large second pieces).
``boundary_network``
    the lines between regions, traced through the mesh's mixed triangles
    (marching triangles), so each stretch of boundary is one polyline shared by
    the two regions on either side and the lines meet exactly at junctions.
    Points are stored as vertex indices plus weights, so any geometry --
    folded, inflated or flat -- places them.
``region_anchors``
    for each region, the vertex deepest inside its largest piece, where a label
    sits without touching the boundary.

Everything works on one hemisphere's mesh at a time: ``labels`` has one entry
per vertex (0 = unlabelled / medial wall), ``faces`` is the triangle list.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def adjacency(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    """Symmetric vertex adjacency (no self loops) from a triangle list."""
    faces = np.asarray(faces, dtype=np.int64)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    matrix = sparse.coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                               shape=(n_vertices, n_vertices)).tocsr()
    matrix.data[:] = 1.0
    return matrix


def _one_hot(labels: np.ndarray, values: np.ndarray) -> sparse.csr_matrix:
    index = np.searchsorted(values, labels)
    valid = labels > 0
    rows = np.flatnonzero(valid)
    return sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, index[valid])),
                             shape=(labels.size, values.size))


def _majority(labels: np.ndarray, graph: sparse.csr_matrix, values: np.ndarray,
              self_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Most common label among each vertex's neighbours (plus itself, weighted)."""
    votes = graph @ _one_hot(labels, values)
    if self_weight:
        votes = votes + self_weight * _one_hot(labels, values)
    votes = votes.toarray()
    best = votes.argmax(axis=1)
    return values[best], votes.max(axis=1)


def clean_parcellation(labels: np.ndarray, faces: np.ndarray, keep: np.ndarray | None = None,
                       smooth_rounds: int = 8, keep_fraction: float = 0.3,
                       max_rounds: int = 500) -> np.ndarray:
    """A complete, smooth, island-free labelling of the vertices in ``keep``.

    ``keep`` marks cortex (for a pycortex subject: the vertices the flat map
    uses; the rest is medial wall and stays 0). Unlabelled kept vertices take
    the majority label of their labelled neighbours, repeatedly, until none are
    left. Boundaries are then smoothed by diffusing each region's indicator
    over the mesh for ``smooth_rounds`` rounds of neighbour averaging and giving
    each vertex the region whose indicator is largest: the boundary becomes a
    level set of a smoothed field rather than the voxel staircase the atlas
    arrived with, and speckle goes with it. Finally, any connected piece of a
    region smaller than ``keep_fraction`` of that region's largest piece is
    relabelled from its
    surroundings, which is what removes an unlabelled-looking hole or a stray
    fragment of one region inside another.
    """
    labels = np.asarray(labels).astype(np.int32).copy()
    n = labels.size
    keep = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    labels[~keep] = 0
    graph = adjacency(faces, n)
    # Only cortex votes and is voted on: the medial wall must not grow regions.
    cortex = sparse.diags(keep.astype(np.float32))
    graph = (cortex @ graph @ cortex).tocsr()
    values = np.unique(labels[labels > 0])
    if values.size == 0:
        return labels

    for _ in range(max_rounds):
        missing = keep & (labels == 0)
        if not missing.any():
            break
        proposal, support = _majority(labels, graph, values, 0.0)
        fill = missing & (support > 0)
        if not fill.any():
            break
        labels[fill] = proposal[fill]

    if smooth_rounds:
        with_self = (graph + sparse.diags(keep.astype(np.float32))).tocsr()
        degree = np.asarray(with_self.sum(axis=1)).ravel()
        step = (sparse.diags(1.0 / np.maximum(degree, 1.0)) @ with_self).tocsr()
        field = _one_hot(labels, values).toarray()
        for _ in range(smooth_rounds):
            field = step @ field
        smoothed = values[field.argmax(axis=1)]
        labels[keep & (field.max(axis=1) > 0)] = smoothed[keep & (field.max(axis=1) > 0)]

    for _ in range(8):
        changed = False
        for value in values:
            members = np.flatnonzero(labels == value)
            if members.size == 0:
                continue
            sub = graph[members][:, members]
            count, component = connected_components(sub, directed=False)
            if count <= 1:
                continue
            sizes = np.bincount(component)
            small = sizes < keep_fraction * sizes.max()
            if not small.any():
                continue
            stray = members[small[component]]
            labels[stray] = 0
            changed = True
        if not changed:
            break
        for _ in range(max_rounds):
            missing = keep & (labels == 0)
            if not missing.any():
                break
            proposal, support = _majority(labels, graph, values, 0.0)
            fill = missing & (support > 0)
            if not fill.any():
                break
            labels[fill] = proposal[fill]
    return labels


def boundary_network(labels: np.ndarray, faces: np.ndarray, relax: int = 0):
    """Lines between regions, traced through mixed triangles and chained.

    A triangle whose three vertices carry two labels contributes one segment,
    between the midpoints of its two mixed edges; one with three labels
    contributes three, from each edge midpoint to its centroid, which is where
    three regions meet. Faces touching an unlabelled vertex (medial wall) are
    skipped, so the flat map's cut edge is not drawn as a boundary.

    Segments sharing an endpoint are chained into polylines that stop at
    junctions (three or more segments) and dead ends. ``relax_polylines``
    smooths them in whatever 2D or 3D coordinates they are drawn in, with the
    ends pinned so neighbouring lines still meet; ``relax`` > 0 instead relaxes
    here, in vertex-weight space, which serves every geometry at once but is
    slow for more than a few passes.

    Returns ``(indices, weights, offsets)``: point ``k`` sits at
    ``sum(weights[k, j] * vertex[indices[k, j]])``; polyline ``i`` is points
    ``offsets[i]:offsets[i + 1]``.
    """
    labels = np.asarray(labels)
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    la, lb, lc = labels[a], labels[b], labels[c]
    usable = (la > 0) & (lb > 0) & (lc > 0)
    mixed = usable & ~((la == lb) & (lb == lc))

    # A point is either an edge midpoint (key: sorted vertex pair) or a face
    # centroid (key: face index); both map to vertex indices and weights.
    point_id: dict = {}
    point_verts: list = []
    point_weights: list = []

    def edge_point(u: int, v: int) -> int:
        key = ("e", min(u, v), max(u, v))
        if key not in point_id:
            point_id[key] = len(point_verts)
            point_verts.append((u, v, v))
            point_weights.append((0.5, 0.5, 0.0))
        return point_id[key]

    def face_point(f: int) -> int:
        key = ("f", f)
        if key not in point_id:
            point_id[key] = len(point_verts)
            point_verts.append(tuple(faces[f]))
            point_weights.append((1 / 3, 1 / 3, 1 / 3))
        return point_id[key]

    segments = []
    for f in np.flatnonzero(mixed):
        u, v, w = int(a[f]), int(b[f]), int(c[f])
        lu, lv, lw = labels[u], labels[v], labels[w]
        edges = [(u, v, lu != lv), (v, w, lv != lw), (w, u, lw != lu)]
        crossing = [edge_point(x, y) for x, y, differs in edges if differs]
        if len(crossing) == 2:
            segments.append((crossing[0], crossing[1]))
        else:  # three labels meet in this face
            centre = face_point(int(f))
            segments.extend((point, centre) for point in crossing)

    neighbours = defaultdict(list)
    for p, q in segments:
        neighbours[p].append(q)
        neighbours[q].append(p)
    used = set()
    polylines = []

    def walk(start: int, first: int) -> list:
        line = [start, first]
        used.add((min(start, first), max(start, first)))
        previous, current = start, first
        while len(neighbours[current]) == 2:
            nxt = neighbours[current][0] if neighbours[current][0] != previous else neighbours[current][1]
            key = (min(current, nxt), max(current, nxt))
            if key in used:
                break
            used.add(key)
            line.append(nxt)
            previous, current = current, nxt
        return line

    # Lines that end at junctions or dead ends first, then closed loops.
    for point, near in neighbours.items():
        if len(near) != 2:
            for other in near:
                if (min(point, other), max(point, other)) not in used:
                    polylines.append(walk(point, other))
    for point, near in neighbours.items():
        for other in near:
            if (min(point, other), max(point, other)) not in used:
                polylines.append(walk(point, other))

    verts = np.asarray(point_verts, dtype=np.int64).reshape(-1, 3)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1, 3)
    out_indices, out_weights, offsets = [], [], [0]
    for line in polylines:
        # Relaxation in weight space: a point's position is a weighted sum of
        # vertices, so averaging neighbouring points' weight vectors (expanded
        # over the union of their vertices) averages their positions on every
        # geometry at once. Pinned ends keep junctions exact.
        points = [dict(zip(verts[p], weights[p])) for p in line]
        closed = len(line) > 2 and line[0] == line[-1]
        for _ in range(relax):
            relaxed = [dict(points[0])]
            for i in range(1, len(points) - 1):
                blend = defaultdict(float)
                for vertex, weight in points[i].items():
                    blend[vertex] += 0.5 * weight
                for side in (points[i - 1], points[i + 1]):
                    for vertex, weight in side.items():
                        blend[vertex] += 0.25 * weight
                relaxed.append(dict(blend))
            relaxed.append(dict(points[-1]))
            points = relaxed
        for point in points:
            # Keep the three heaviest vertices; renormalise.
            top = sorted(point.items(), key=lambda item: -item[1])[:3]
            while len(top) < 3:
                top.append((top[0][0], 0.0))
            total = sum(weight for _, weight in top) or 1.0
            out_indices.append([vertex for vertex, _ in top])
            out_weights.append([weight / total for _, weight in top])
        offsets.append(len(out_indices))
        del closed
    return (np.asarray(out_indices, dtype=np.uint32).reshape(-1, 3),
            np.asarray(out_weights, dtype=np.float32).reshape(-1, 3),
            np.asarray(offsets, dtype=np.uint32))


def region_anchors(labels: np.ndarray, faces: np.ndarray) -> dict[int, int]:
    """For each region, the vertex deepest inside its largest connected piece.

    Depth is graph distance to the nearest vertex of another label (or of the
    medial wall), found by breadth-first search from the boundary inwards, so
    the anchor sits in the middle of the region however concave it is.
    """
    labels = np.asarray(labels)
    n = labels.size
    graph = adjacency(faces, n)
    indptr, indices = graph.indptr, graph.indices
    # Boundary vertices: any neighbour with a different label.
    neighbour_labels = labels[indices]
    owner = np.repeat(np.arange(n), np.diff(indptr))
    differs = np.zeros(n, dtype=bool)
    np.logical_or.at(differs, owner, neighbour_labels != labels[owner])
    # A mesh border (an edge with one face) is a boundary too: a cut surface
    # has one, and a label should not sit against it.
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    differs[unique[counts == 1].ravel()] = True
    depth = np.full(n, -1, dtype=np.int32)
    frontier = differs & (labels > 0)
    depth[frontier] = 0
    level = 0
    while frontier.any():
        level += 1
        reached = (graph @ frontier.astype(np.float32)) > 0
        frontier = reached & (depth < 0) & (labels > 0)
        depth[frontier] = level
    anchors = {}
    for value in np.unique(labels[labels > 0]):
        members = np.flatnonzero(labels == value)
        sub = graph[members][:, members]
        count, component = connected_components(sub, directed=False)
        largest = members[component == np.bincount(component).argmax()]
        anchors[int(value)] = int(largest[np.argmax(depth[largest])])
    return anchors


def relax_polylines(points: np.ndarray, offsets: np.ndarray, passes: int = 20,
                    weight: float = 0.5) -> np.ndarray:
    """Laplacian smoothing of each polyline with its two ends pinned.

    ``points`` are positions (any dimension) laid out as ``boundary_network``
    returns them. Ends are junctions or dead ends and stay exactly where they
    are, so lines that met before relaxing still meet after.
    """
    points = np.array(points, dtype=np.float64, copy=True)
    for i in range(len(offsets) - 1):
        start, stop = int(offsets[i]), int(offsets[i + 1])
        if stop - start < 3:
            continue
        line = points[start:stop]
        for _ in range(passes):
            line[1:-1] = (1 - weight) * line[1:-1] + weight * 0.5 * (line[:-2] + line[2:])
        points[start:stop] = line
    return points


def write_parcellation(out_dir, atlas_id: str, label: str, regions: list[dict],
                       hemispheres: dict, smooth_rounds: int = 20, sparse: bool = False) -> dict:
    """Clean, trace and write one parcellation for the viewer; return its record.

    ``hemispheres`` maps ``"lh"``/``"rh"`` to ``{"labels": raw per-vertex labels,
    "faces": the full mesh's faces, "flat_faces": the faces the flat map keeps}``
    -- the same face lists the surface export wrote, so the lines are traced on
    exactly the mesh the viewer draws. ``regions`` is ``[{"value", "name",
    "abbrev"}]``. Files go under ``out_dir/parcellations/<atlas_id>/``:

    ``<hemi>_labels.bin``          int16 per vertex, 0 = medial wall
    ``<hemi>_lines_index.bin``     uint32 [points][3] vertex indices
    ``<hemi>_lines_weight.bin``    float32 [points][3] weights (sum to one)
    ``<hemi>_lines_offsets.bin``   uint32 [lines + 1] polyline starts

    The record lists each hemisphere's files and one label anchor per region.
    Paths in it are relative to ``out_dir``.

    ``sparse`` is for an atlas that covers only part of the cortex (a handful
    of functional parcels): cortex no region reaches is not filled from its
    neighbours but held as one background region while the labels are
    cleaned and traced, so each region's edge against uncovered cortex is
    smoothed and drawn like any other boundary. The background is written
    back as 0 and gets no anchor.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    target = out_dir / "parcellations" / atlas_id
    target.mkdir(parents=True, exist_ok=True)
    record = {"id": atlas_id, "label": label, "regions": regions, "hemispheres": {}}
    for hemisphere, info in hemispheres.items():
        faces = np.asarray(info["faces"])
        flat_faces = np.asarray(info.get("flat_faces", faces))
        keep = np.zeros(np.asarray(info["labels"]).size, dtype=bool)
        keep[np.unique(flat_faces)] = True
        raw = np.asarray(info["labels"]).astype(np.int32)
        background = None
        if sparse:
            background = int(max([r["value"] for r in regions] + [int(raw.max())])) + 1
            raw = raw.copy()
            raw[keep & (raw == 0)] = background
        labels = clean_parcellation(raw, faces, keep=keep, smooth_rounds=smooth_rounds)
        indices, weights, offsets = boundary_network(labels, flat_faces)
        anchors = region_anchors(labels, flat_faces)
        if background is not None:
            anchors.pop(background, None)
            labels[labels == background] = 0
        files = {}
        for name, array in (("labels", labels.astype("<i2")),
                            ("lines_index", indices.astype("<u4")),
                            ("lines_weight", weights.astype("<f4")),
                            ("lines_offsets", offsets.astype("<u4"))):
            path = target / f"{hemisphere}_{name}.bin"
            path.write_bytes(np.ascontiguousarray(array).tobytes())
            files[name] = str(path.relative_to(out_dir)).replace("\\", "/")
        record["hemispheres"][hemisphere] = {
            **files,
            "n_vertices": int(labels.size),
            "n_lines": int(len(offsets) - 1),
            "anchors": {str(value): vertex for value, vertex in anchors.items()},
        }
    return record
