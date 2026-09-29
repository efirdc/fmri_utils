"""FreeSurfer subjects on the fsaverage mesh, for the viewer's fsnative mode.

A FreeSurfer reconstruction has no flat map, and a high-resolution one has far
more vertices than a browser wants (half a million per hemisphere from a 0.5 mm
T1). Decimating the folded white surface is the obvious fix and the wrong one:
on cortex that folded, a decimator lays triangles across sulci and flips them,
and the white and pial surfaces render as shattered glass.

This module resamples instead. Every subject already carries its registration
to fsaverage, ``?h.sphere.reg``. Each fsaverage vertex is located on the
subject's registered sphere, and the subject's own white, pial and inflated
positions are read off at that point by barycentric interpolation within the
subject triangle that contains it. The result is the subject's anatomy on the
fsaverage triangulation -- the same idea as HCP's fs_LR meshes -- which buys
three things at once:

* well-shaped triangles at about 1 mm spacing, whatever the recon resolution;
* one vertex correspondence across subjects;
* a flat map: fsaverage's flat patch is defined on this triangulation, so its
  cut and its flat coordinates apply to every subject unchanged. The cut on a
  subject is exactly where registration puts fsaverage's cut.

Positions are written in scanner (T1w) millimetres, the space subject-level
maps are resampled into, so the page samples a T1-space map at each vertex
directly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .surfaces import FSAVERAGE, SIDES, fsaverage_files

GEOMETRY_FILES = {"wm": "white", "pia": "pial", "inflated": "inflated"}
# FreeSurfer 7 writes ?h.pial as a link to ?h.pial.T1; a copied subjects
# directory often carries the link broken, and the file it named beside it.
FALLBACKS = {"pial": ("pial.T1",)}


def read_freesurfer_geometry(surf_dir: Path, hemisphere: str, name: str):
    """``?h.<name>`` from a FreeSurfer surf directory, or its known fallback."""
    import nibabel as nib

    tried = []
    for candidate in (name, *FALLBACKS.get(name, ())):
        path = Path(surf_dir) / f"{hemisphere}.{candidate}"
        try:
            return nib.freesurfer.read_geometry(str(path))
        except OSError as error:
            tried.append(f"{path.name} ({error.__class__.__name__})")
    raise FileNotFoundError(f"no readable {hemisphere}.{name} in {surf_dir}: " + ", ".join(tried))


def _unit(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def sphere_correspondence(source_sphere: np.ndarray, source_faces: np.ndarray,
                          target_points: np.ndarray, candidates: int = 24,
                          chunk: int = 16384) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate each target point on a source sphere mesh.

    Returns ``(face, weights, worst)``: the source face containing each target
    point, barycentric weights over that face's three vertices (non-negative,
    summing to one), and the most negative raw weight seen, which is how far
    outside its best triangle the worst point fell (0 for a clean match).

    Both spheres are treated as unit spheres about the origin, which is what
    FreeSurfer's ``sphere`` and ``sphere.reg`` are up to their radius. A target
    point is the ray from the origin through it; the triangle it pierces is
    found among the ``candidates`` triangles with the nearest centroids.
    """
    from scipy.spatial import cKDTree

    source = _unit(source_sphere)
    faces = np.asarray(source_faces, dtype=np.int64)
    targets = _unit(target_points)
    centroids = _unit(source[faces].mean(axis=1))
    tree = cKDTree(centroids)
    face_out = np.empty(len(targets), dtype=np.int64)
    weight_out = np.empty((len(targets), 3), dtype=np.float64)
    worst = 0.0
    for start in range(0, len(targets), chunk):
        direction = targets[start:start + chunk]
        _, near = tree.query(direction, k=candidates)
        a = source[faces[near, 0]]
        b = source[faces[near, 1]]
        c = source[faces[near, 2]]
        d = direction[:, None, :]
        # Moller-Trumbore from the origin along d.
        e1, e2 = b - a, c - a
        p = np.cross(d, e2)
        det = np.einsum("tki,tki->tk", e1, p)
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        to_origin = -a
        u = np.einsum("tki,tki->tk", to_origin, p) / det
        q = np.cross(to_origin, e1)
        v = np.einsum("tki,tki->tk", d, q) / det
        s = np.einsum("tki,tki->tk", e2, q) / det
        w0 = 1.0 - u - v
        score = np.minimum(np.minimum(w0, u), v)
        score = np.where(s > 0, score, -np.inf)
        best = np.argmax(score, axis=1)
        rows = np.arange(len(direction))
        chosen = np.stack([w0[rows, best], u[rows, best], v[rows, best]], axis=1)
        worst = min(worst, float(score[rows, best].min()))
        chosen = np.clip(chosen, 0.0, None)
        chosen /= chosen.sum(axis=1, keepdims=True)
        face_out[start:start + chunk] = near[rows, best]
        weight_out[start:start + chunk] = chosen
    return face_out, weight_out, worst


def interpolate(values: np.ndarray, faces: np.ndarray, face: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """Per-vertex source values (N, or N x 3) read off at the matched points."""
    values = np.asarray(values)
    corners = values[np.asarray(faces)[face]]  # T x 3 (x 3)
    if values.ndim == 1:
        return (corners * weights).sum(axis=1)
    return (corners * weights[..., None]).sum(axis=1)


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                            points[faces[:, 2]] - points[faces[:, 0]])
    out = np.zeros_like(points)
    for corner in range(3):
        np.add.at(out, faces[:, corner], face_normals)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def neighbour_average(faces: np.ndarray, n_vertices: int):
    """Sparse matrix averaging each vertex's neighbours, and the graph Laplacian."""
    from scipy import sparse

    faces = np.asarray(faces, dtype=np.int64)
    rows = np.concatenate([faces[:, i] for i in (0, 1, 2, 1, 2, 0)])
    cols = np.concatenate([faces[:, i] for i in (1, 2, 0, 0, 1, 2)])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices))
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(1.0 / degree) @ adjacency, sparse.diags(degree) - adjacency


def taubin(points: np.ndarray, average, iterations: int, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin's lambda|mu smoothing: removes the finest wrinkles without shrinking."""
    points = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        points = points + lam * (average @ points - points)
        points = points + mu * (average @ points - points)
    return points


def bending(points: np.ndarray, faces: np.ndarray, average) -> float:
    """How folded a surface is: the mean normal part of the Laplacian, per edge length.

    Only the normal part counts. The tangential part measures how unevenly the
    vertices are spread, which registration makes uneven on every subject and
    says nothing about folding.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normal_part = np.abs(np.einsum("ij,ij->i", average @ points - points, vertex_normals(points, faces)))
    edge = np.linalg.norm(points[faces[:, 0]] - points[faces[:, 1]], axis=1).mean()
    return float(normal_part.mean() / edge)


def _area(points: np.ndarray, faces: np.ndarray) -> float:
    corners = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                               corners[:, 2] - corners[:, 0]), axis=1).sum())


def inflate_to(points: np.ndarray, faces: np.ndarray, target: float, step: float = 10.0,
               max_steps: int = 30) -> tuple[np.ndarray, int]:
    """Smooth an inflated surface until it bends no more than ``target``.

    FreeSurfer's mris_inflate runs a fixed number of neighbourhood smoothing
    steps, so on a mesh several times denser than usual -- a recon from a
    0.5 mm T1 -- each step covers a fraction of the usual cortex and the result
    stays visibly folded. This continues it with implicit steps, (I + t L) p' = p,
    which remove long wavelengths in one solve where explicit steps would need
    hundreds, and rescales to the starting area after each so the surface does
    not shrink. Returns the surface and the number of steps taken.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    points = np.asarray(points, dtype=np.float64)
    average, laplacian = neighbour_average(faces, len(points))
    solve = splu((sparse.identity(len(points)) + step * laplacian).tocsc())
    area = _area(points, faces)
    steps = 0
    while steps < max_steps and bending(points, faces, average) > target:
        points = np.column_stack([solve.solve(points[:, axis]) for axis in range(3)])
        centre = points.mean(axis=0)
        points = centre + (points - centre) * np.sqrt(area / _area(points, faces))
        steps += 1
    return points, steps


def sign_changes(values: np.ndarray, faces: np.ndarray) -> float:
    """Fraction of mesh edges across which a field changes sign."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return float(np.mean(np.sign(values[edges[:, 0]]) != np.sign(values[edges[:, 1]])))


def smooth_curvature(values: np.ndarray, faces: np.ndarray, target: float, step: float = 2.0,
                     max_steps: int = 20) -> tuple[np.ndarray, int]:
    """Diffuse curvature until its sign changes no more often than ``target``.

    The viewer shades by the sign of curvature, sulci dark and gyri light. A
    sub-millimetre recon's curvature changes sign at a fine scale, and on the
    fsaverage mesh -- whose triangles registration stretches along sulcal
    walls -- that renders as speckle and saw-toothed stripes. Implicit
    diffusion steps smooth it to the scale of fsaverage's own shading.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    values = np.asarray(values, dtype=np.float64)
    _, laplacian = neighbour_average(faces, len(values))
    solve = splu((sparse.identity(len(values)) + step * laplacian).tocsc())
    steps = 0
    while steps < max_steps and sign_changes(values, faces) > target:
        values = solve.solve(values)
        steps += 1
    return values, steps


def tkr_to_scanner(freesurfer_subject_dir: Path) -> np.ndarray:
    """FreeSurfer surface (tkRAS) millimetres to scanner millimetres."""
    import nibabel as nib

    orig = nib.load(str(Path(freesurfer_subject_dir) / "mri" / "orig.mgz"))
    return orig.affine @ np.linalg.inv(orig.header.get_vox2ras_tkr())


def _apply(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ affine[:3, :3].T + affine[:3, 3]


def export_freesurfer_subject(
    freesurfer_subject_dir: Path,
    output_root: Path,
    subject: str,
    fsaverage_export: Path,
    geometry_overrides: dict | None = None,
    antialias_iterations: int = 5,
    post_smooth_iterations: int = 2,
    inflate: bool = True,
    curvature_scale: float = 0.0,
    quiet: bool = False,
) -> dict:
    """Export one FreeSurfer subject on the fsaverage mesh, in scanner millimetres.

    ``fsaverage_export`` is the directory ``export_surfaces(fsaverage=True)``
    wrote; its flat geometry and flat face mask are copied for this subject,
    since they are defined on the same triangulation. ``geometry_overrides``
    maps ``(hemisphere, geometry)`` to ``(points, faces)`` already in scanner
    millimetres on the subject's native mesh -- for a surface whose FreeSurfer
    file is missing but which another pipeline wrote (fMRIPrep's GIfTIs).

    ``antialias_iterations`` Taubin-smooths white and pial on the native mesh
    before they are sampled. A recon from a sub-millimetre T1 carries detail
    finer than the ~0.9 mm fsaverage mesh can hold, and point-sampling it folds
    that detail into jagged, wrinkled shading; smoothing first removes it the
    way a low-pass filter precedes any downsampling. It moves vertices about a
    tenth of a millimetre. ``post_smooth_iterations`` of Taubin on the fsaverage
    mesh then soften the facets where registration stretches triangles along
    sulcal walls. ``inflate`` continues FreeSurfer's inflation until the surface
    is as smooth as fsaverage's inflated one (see ``inflate_to``).

    The shading curvature is FreeSurfer's own ``?h.curv``, resampled and left
    raw by default. ``curvature_scale`` > 0 diffuses it until its sign changes
    at most that many times as often as fsaverage's (``smooth_curvature``), but
    that is not recommended: diffusion on this mesh runs in the sphere's metric,
    so where registration stretches triangles along a sulcal wall a few steps
    carry curvature millimetres across the crown. Measured on a 0.5 mm recon,
    the raw curvature's sign agrees with the displayed white surface's folding
    at about 80% of vertices (fsaverage's own: 82%), and every smoothing tried
    -- on this mesh, on the native mesh, or of curvature taken from the
    displayed surface -- lowered that. Its fine speckle is real curvature.

    Writes ``output_root/subject`` in the layout ``package_surfaces`` reads and
    returns the record, including per-hemisphere QC numbers.
    """
    import nibabel as nib

    fs_dir = Path(freesurfer_subject_dir)
    out = Path(output_root) / subject
    out.mkdir(parents=True, exist_ok=True)
    fsaverage_export = Path(fsaverage_export)
    fsaverage_record = json.loads((fsaverage_export / "surfaces.json").read_text(encoding="utf-8"))
    overrides = geometry_overrides or {}
    to_scanner = tkr_to_scanner(fs_dir)
    record: dict = {"subject": subject, "mesh": "fsaverage", "hemispheres": {}, "qc": {}}

    def say(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    for hemisphere in ("lh", "rh"):
        side = SIDES[hemisphere]
        target_sphere = nib.load(str(fsaverage_files()[f"sphere_{side}"])).darrays
        target_points = np.asarray(target_sphere[0].data)
        target_faces = np.asarray(target_sphere[1].data, dtype=np.uint32)
        source_sphere, source_faces = nib.freesurfer.read_geometry(
            str(fs_dir / "surf" / f"{hemisphere}.sphere.reg"))
        face, weights, worst = sphere_correspondence(source_sphere, source_faces, target_points)

        entry: dict = {"n_vertices": int(len(target_points)), "n_faces": int(len(target_faces)),
                       "geometries": {}}
        (out / f"{hemisphere}_faces.bin").write_bytes(target_faces.tobytes(order="C"))
        entry["faces"] = f"{hemisphere}_faces.bin"
        positions = {}
        inverted = {}
        moved = {}
        native_average = (neighbour_average(source_faces, len(source_sphere))[0]
                          if antialias_iterations else None)
        for geometry, fs_name in GEOMETRY_FILES.items():
            if (hemisphere, geometry) in overrides:
                points, native_faces = overrides[(hemisphere, geometry)]
                if not np.array_equal(np.asarray(native_faces), np.asarray(source_faces)):
                    raise ValueError(f"{subject} {hemisphere} {geometry}: override topology differs")
                scanner = np.asarray(points, dtype=np.float64)
            else:
                points, native_faces = read_freesurfer_geometry(fs_dir / "surf", hemisphere, fs_name)
                if len(points) != len(source_sphere):
                    raise ValueError(f"{subject} {hemisphere} {fs_name}: vertex count differs from sphere.reg")
                scanner = _apply(to_scanner, points)
            smoothed = scanner
            if native_average is not None and geometry in ("wm", "pia"):
                smoothed = taubin(scanner, native_average, antialias_iterations)
                moved[geometry] = float(np.median(np.linalg.norm(smoothed - scanner, axis=1)))
            resampled = interpolate(smoothed, source_faces, face, weights)
            if post_smooth_iterations and geometry in ("wm", "pia"):
                resampled = taubin(resampled, neighbour_average(target_faces, len(target_points))[0],
                                   post_smooth_iterations)
            # A resampled triangle is inverted where its normal disagrees with
            # the native surface's normal at the same place; on white and pial
            # these are what would read as broken shading.
            native_normals = vertex_normals(scanner, source_faces)
            reference = interpolate(native_normals, source_faces, face, weights)[target_faces].mean(axis=1)
            corners = resampled[target_faces]
            normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            inverted[geometry] = float(np.mean(np.einsum("ij,ij->i", normals, reference) < 0))
            if geometry == "inflated" and inflate:
                target_average = neighbour_average(target_faces, len(target_points))[0]
                reference_inflated = np.asarray(
                    nib.load(str(fsaverage_files()[f"infl_{side}"])).darrays[0].data, dtype=np.float64)
                target = bending(reference_inflated, target_faces, target_average)
                before = bending(resampled, target_faces, target_average)
                resampled, steps = inflate_to(resampled, target_faces, target)
                moved["inflation"] = {"bending_before": before, "bending_after":
                                      bending(resampled, target_faces, target_average),
                                      "fsaverage_bending": target, "steps": steps}
            if geometry == "inflated":
                # fsaverage's inflated surfaces are centred on the origin, and
                # the page separates the hemispheres by their bounds; match it.
                resampled = resampled - (resampled.min(axis=0) + resampled.max(axis=0)) / 2
            positions[geometry] = resampled.astype(np.float32)
        for geometry, points in positions.items():
            name = f"{hemisphere}_{geometry}.bin"
            (out / name).write_bytes(points.tobytes(order="C"))
            entry["geometries"][geometry] = {
                "path": name, "anatomical": True,
                "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            }

        # The flat map is fsaverage's own: same vertices, same kept faces.
        flat = fsaverage_record["hemispheres"][hemisphere]["geometries"].get("flat")
        if flat:
            shutil.copyfile(fsaverage_export / flat["path"], out / f"{hemisphere}_flat.bin")
            flat_entry = {"path": f"{hemisphere}_flat.bin", "anatomical": False,
                          "bounds": flat["bounds"], "n_faces": flat["n_faces"]}
            if flat.get("face_mask"):
                shutil.copyfile(fsaverage_export / flat["face_mask"], out / f"{hemisphere}_flat_face_mask.bin")
                flat_entry["face_mask"] = f"{hemisphere}_flat_face_mask.bin"
            entry["geometries"]["flat"] = flat_entry

        curvature_path = fs_dir / "surf" / f"{hemisphere}.curv"
        if curvature_path.exists():
            curvature = nib.freesurfer.read_morph_data(str(curvature_path))
            resampled = interpolate(np.asarray(curvature, dtype=np.float64), source_faces, face, weights)
            if curvature_scale:
                reference = np.asarray(nib.load(str(fsaverage_files()[f"curv_{side}"])).darrays[0].data,
                                       dtype=np.float64)
                target = curvature_scale * sign_changes(reference, target_faces)
                before = sign_changes(resampled, target_faces)
                resampled, steps = smooth_curvature(resampled, target_faces, target)
                moved["curvature"] = {"sign_changes_before": before,
                                      "sign_changes_after": sign_changes(resampled, target_faces),
                                      "target": target, "steps": steps}
            (out / f"{hemisphere}_curv.bin").write_bytes(resampled.astype(np.float32).tobytes())
            entry["curvature"] = f"{hemisphere}_curv.bin"
        # Sulcal depth as well: the page can shade by either. A sub-millimetre
        # recon's curvature changes sign at a scale finer than the fsaverage
        # mesh shows, so it reads as speckle there; depth follows the sulci.
        sulc_path = fs_dir / "surf" / f"{hemisphere}.sulc"
        if sulc_path.exists():
            depth = nib.freesurfer.read_morph_data(str(sulc_path))
            resampled = interpolate(np.asarray(depth, dtype=np.float64), source_faces, face, weights)
            (out / f"{hemisphere}_sulc.bin").write_bytes(resampled.astype(np.float32).tobytes())
            entry["sulc"] = f"{hemisphere}_sulc.bin"

        white = positions["wm"].astype(np.float64)
        tri = white[target_faces]
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        record["qc"][hemisphere] = {
            "native_vertices": int(len(source_sphere)),
            "worst_barycentric": worst,
            "median_edge_mm": float(np.median(np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1))),
            "degenerate_faces": int(np.sum(areas < 1e-4)),
            "inverted_faces": inverted,
            "antialias_iterations": antialias_iterations,
            "antialias_median_move_mm": {k: v for k, v in moved.items() if k in ("wm", "pia")},
            "post_smooth_iterations": post_smooth_iterations,
            "inflation": moved.get("inflation"),
            "curvature": moved.get("curvature"),
        }
        record["hemispheres"][hemisphere] = entry
        inflation = moved.get("inflation") or {}
        say(f"  {subject} {hemisphere}: {len(source_sphere)} -> {len(target_points)} vertices; "
            f"inverted faces white {inverted['wm']:.2%}, pial {inverted['pia']:.2%}; "
            f"inflated bending {inflation.get('bending_before', float('nan')):.4f} -> "
            f"{inflation.get('bending_after', float('nan')):.4f} in {inflation.get('steps', 0)} steps")

    (out / "surfaces.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def export_curvature(freesurfer_subject_dir: Path, subject_export_dir: Path) -> None:
    """Rewrite an export's shading fields from ``?h.curv`` and ``?h.sulc``, resampled raw.

    For refreshing an export without redoing its geometry.
    """
    import nibabel as nib

    fs_dir = Path(freesurfer_subject_dir)
    out = Path(subject_export_dir)
    record_path = out / "surfaces.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for hemisphere, entry in record["hemispheres"].items():
        target = np.asarray(nib.load(str(fsaverage_files()[f"sphere_{SIDES[hemisphere]}"])).darrays[0].data)
        sphere, faces = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{hemisphere}.sphere.reg"))
        face, weights, _ = sphere_correspondence(sphere, faces, target)
        curvature = nib.freesurfer.read_morph_data(str(fs_dir / "surf" / f"{hemisphere}.curv"))
        values = interpolate(np.asarray(curvature, dtype=np.float64), faces, face, weights)
        (out / f"{hemisphere}_curv.bin").write_bytes(values.astype(np.float32).tobytes())
        entry["curvature"] = f"{hemisphere}_curv.bin"
        sulc_path = fs_dir / "surf" / f"{hemisphere}.sulc"
        if sulc_path.exists():
            depth = nib.freesurfer.read_morph_data(str(sulc_path))
            values = interpolate(np.asarray(depth, dtype=np.float64), faces, face, weights)
            (out / f"{hemisphere}_sulc.bin").write_bytes(values.astype(np.float32).tobytes())
            entry["sulc"] = f"{hemisphere}_sulc.bin"
        record.setdefault("qc", {}).setdefault(hemisphere, {})["curvature"] = None
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def sample_volume_labels(atlas, white: np.ndarray, pial: np.ndarray,
                         depths=(0.1, 0.3, 0.5, 0.7, 0.9), transform: np.ndarray | None = None,
                         fill_mm: float = 3.0) -> np.ndarray:
    """One integer label per vertex, the commonest among several cortical depths.

    ``atlas`` is a NIfTI image of integer labels; ``white`` and ``pial`` are
    matching vertex arrays in millimetres. ``transform`` (4x4), if given,
    carries those millimetres into the atlas's world space first -- MNI305 to
    MNI152 for fsaverage. A vertex whose depths all miss a label takes the
    label of the nearest labelled vertex within ``fill_mm`` (a thr25 atlas
    leaves the grey/white boundary partly unlabelled). This is for drawing;
    region statistics should use the voxel labels.
    """
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree

    values = np.asarray(atlas.dataobj).astype(np.int32)
    inverse = np.linalg.inv(atlas.affine)
    if transform is not None:
        inverse = inverse @ np.asarray(transform, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    pial = np.asarray(pial, dtype=np.float64)
    samples = []
    for depth in depths:
        points = white + depth * (pial - white)
        voxels = points @ inverse[:3, :3].T + inverse[:3, 3]
        samples.append(map_coordinates(values, voxels.T, order=0, mode="constant", cval=0))
    stacked = np.stack(samples).astype(np.int64)
    labels = np.zeros(stacked.shape[1], dtype=np.int32)
    top = int(stacked.max()) + 1 if stacked.size else 1
    counts = np.zeros((stacked.shape[1], top), dtype=np.int32)
    for row in stacked:
        counts[np.arange(len(row)), row] += 1
    counts[:, 0] = 0
    labelled = counts.max(axis=1) > 0
    labels[labelled] = counts[labelled].argmax(axis=1)
    if fill_mm and labelled.any() and (~labelled).any():
        middle = white + 0.5 * (pial - white)
        distance, nearest = cKDTree(middle[labelled]).query(middle[~labelled])
        labels[~labelled] = np.where(distance <= fill_mm, labels[labelled][nearest], 0)
    return labels


def fsaverage_record_path(export_root: Path) -> Path:
    return Path(export_root) / FSAVERAGE / "surfaces.json"
