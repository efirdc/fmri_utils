"""FSL registrations, applied without FSL: functional grid <-> anatomy <-> MNI152.

For data whose functional grid is registered to an anatomy by an FSL (FLIRT-convention) affine and
whose anatomy is warped to MNI152 by FNIRT:

``warp_affine``        resample an image with a FLIRT source->reference matrix (NaN-aware)
``column_labels``      an anatomical-space atlas's label at each response column of a ``RunTable``
``ColumnMNIWarp``      response columns -> MNI152 2 mm as a sparse trilinear matrix, built from the
                       FNIRT coefficient field (needs ``fslpy``), saved once and applied to any map:
                       values trilinear, NaN treated as absent, a voxel kept when its nearest column
                       has data (what ``applywarp --premat`` gives with a separately warped support mask)

and, on a machine with FSL, the commands that estimate the transforms (``register_anatomical``,
``atlas_to_anatomical``). ``pycortex_func_to_anat`` exports a pycortex functional-to-anatomical
transform in FSL's convention.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

MNI_SHAPE = (91, 109, 91)
MNI_AFFINE = np.array([[-2.0, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])


def voxel_to_fsl(image) -> np.ndarray:
    """Voxel indices to FSL scaled-mm coordinates (FLIRT flips the first axis of neurological images)."""
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    scale = np.diag(np.append(np.abs(zooms), 1.0))
    if np.linalg.det(np.asarray(image.affine)[:3, :3]) > 0:
        flip = np.eye(4)
        flip[0, 0], flip[0, 3] = -1.0, float(image.shape[0]) - 1.0
        return scale @ flip
    return scale


def warp_affine(source, matrix: np.ndarray, reference, order: int = 1) -> np.ndarray:
    """Resample ``source`` onto ``reference`` with a FLIRT source->reference matrix."""
    from scipy.ndimage import map_coordinates
    pull = np.linalg.inv(voxel_to_fsl(source)) @ np.linalg.inv(np.asarray(matrix, dtype=np.float64)) @ voxel_to_fsl(reference)
    grid = np.indices(reference.shape[:3], dtype=np.float64).reshape(3, -1)
    coords = (pull @ np.vstack([grid, np.ones((1, grid.shape[1]))]))[:3]
    values = np.asarray(source.get_fdata(dtype=np.float32), dtype=np.float64)
    support = np.isfinite(values)
    out = map_coordinates(np.where(support, values, 0.0), coords, order=order, mode="constant", cval=0.0)
    keep = map_coordinates(support.astype(np.float64), coords, order=0, mode="constant", cval=0.0) >= 0.5
    out = out.reshape(reference.shape[:3]).astype(np.float32)
    out[~keep.reshape(reference.shape[:3])] = np.nan
    return out


def column_labels(table, subject: str, atlas_anat: Path, func_to_anat: Path) -> np.ndarray:
    """An anatomical-space atlas's label at each response column (nearest neighbour)."""
    import nibabel as nib
    mask = table.mask_image(subject)
    order = table.column_volume(subject)
    atlas = nib.load(atlas_anat)
    source = nib.Nifti1Image(np.asarray(atlas.dataobj).astype(np.float32), atlas.affine, atlas.header)
    labels = np.nan_to_num(warp_affine(source, np.linalg.inv(np.loadtxt(func_to_anat)), mask, order=0)).round().astype(int)
    out = np.zeros(table.n_columns(subject), dtype=int)
    inside = order > 0
    out[order[inside] - 1] = labels[inside]
    return out


def func_to_mni_coordinates(functional_image, anatomical: Path, warpcoef: Path, func_to_anat: Path, template: Path) -> np.ndarray:
    """For every template voxel, the (fractional) functional voxel it pulls from: 3 x n."""
    from fsl.data.image import Image
    from fsl.transform import fnirt
    ref, src = Image(str(template)), Image(str(anatomical))
    field = fnirt.readFnirt(str(warpcoef), src, ref)
    anat = field.transform(np.indices(ref.shape[:3], dtype=np.float64).reshape(3, -1).T, "voxel", "fsl")
    to_func = np.linalg.inv(voxel_to_fsl(functional_image)) @ np.linalg.inv(np.loadtxt(func_to_anat))
    return (np.hstack([anat, np.ones((anat.shape[0], 1))]) @ to_func.T)[:, :3].T


class ColumnMNIWarp:
    """Response columns -> template voxels, as a sparse trilinear matrix."""

    def __init__(self, template_index, columns, weights, nearest_column, n_columns: int, template_shape=MNI_SHAPE):
        from scipy import sparse
        self.template_index = np.asarray(template_index, dtype=np.int64)
        self.nearest = np.asarray(nearest_column, dtype=np.int64)
        self.columns, self.weights, self.n_columns = columns, weights, int(n_columns)
        self.template_shape = tuple(int(x) for x in template_shape)
        rows = np.repeat(np.arange(columns.shape[0]), columns.shape[1]).reshape(columns.shape)
        ok = columns >= 0
        self.matrix = sparse.csr_matrix((weights[ok].astype(np.float32), (rows[ok], columns[ok])),
                                        shape=(columns.shape[0], self.n_columns))

    @classmethod
    def build(cls, table, subject: str, anatomical: Path, warpcoef: Path, func_to_anat: Path, template: Path) -> "ColumnMNIWarp":
        import nibabel as nib
        func = table.mask_image(subject)
        func_shape = func.shape[:3]
        column = table.column_volume(subject).ravel() - 1                 # -1 off the mask
        coords = func_to_mni_coordinates(func, anatomical, warpcoef, func_to_anat, template)
        nearest = np.rint(coords).astype(np.int64)
        # Points outside [0, n - 1] have no data (as in scipy's constant mode, and applywarp), even when
        # they round onto the edge.
        inside = np.all((coords >= 0) & (coords <= np.array(func_shape)[:, None] - 1), axis=0)
        flat = np.full(nearest.shape[1], -1, dtype=np.int64)
        flat[inside] = np.ravel_multi_index(tuple(nearest[:, inside]), func_shape)
        nearest_column = np.where(flat >= 0, column[np.maximum(flat, 0)], -1)
        index = np.flatnonzero(nearest_column >= 0)
        c = coords[:, index]
        base = np.floor(c).astype(np.int64)
        frac = c - base
        columns, weights = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = base + np.array([dx, dy, dz])[:, None]
                    w = (frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) * (frac[2] if dz else 1 - frac[2])
                    ok = np.all((idx >= 0) & (idx < np.array(func_shape)[:, None]), axis=0)
                    corner = np.full(idx.shape[1], -1, dtype=np.int64)
                    corner[ok] = column[np.ravel_multi_index(tuple(idx[:, ok]), func_shape)]
                    columns.append(corner)
                    weights.append(np.where(ok & (corner >= 0), w, 0.0))
        return cls(index, np.stack(columns, axis=1), np.stack(weights, axis=1).astype(np.float32), nearest_column[index],
                   table.n_columns(subject), nib.load(template).shape[:3])

    def save(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, template_index=self.template_index.astype(np.int32), columns=self.columns.astype(np.int32),
                            weights=self.weights, nearest=self.nearest.astype(np.int32), n_columns=self.n_columns,
                            template_shape=np.array(self.template_shape))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "ColumnMNIWarp":
        data = np.load(path)
        return cls(data["template_index"], data["columns"].astype(np.int64), data["weights"], data["nearest"],
                   int(data["n_columns"]), tuple(data["template_shape"]))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Column values (n_columns,) or (n_columns, k) -> values on ``template_index``."""
        values = np.asarray(values, dtype=np.float32)
        support = np.isfinite(values)
        out = np.asarray(self.matrix @ np.where(support, values, 0.0), dtype=np.float32)
        near = support[self.nearest] if values.ndim == 1 else support[self.nearest].all(axis=-1)
        out[~near] = np.nan
        return out

    def volume(self, values: np.ndarray) -> np.ndarray:
        out = np.full(int(np.prod(self.template_shape)), np.nan, dtype=np.float32)
        out[self.template_index] = self.apply(values)
        return out.reshape(self.template_shape)


def save_template_map(volume: np.ndarray, path: Path, affine: np.ndarray = MNI_AFFINE) -> Path:
    import nibabel as nib
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), affine)
    image.header.set_data_dtype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    return Path(path)


# ---- estimating transforms (FSL; pycortex) --------------------------------------------------------
def _run(command: list[str]) -> None:
    print("  $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def register_anatomical(head: Path, brain: Path, output: Path, fsldir: Path, fnirt_config: str = "T1_2_MNI152_2mm") -> Path:
    """FLIRT 12-DOF then FNIRT (anatomy -> MNI152 2 mm), the anatomy at 2 mm, and the inverse warp:
    ``anat_to_MNI152_{affine.mat,warpcoef.nii.gz}``, ``T1_2mm.nii.gz``, ``MNI_to_anat_warpcoef.nii.gz``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    standard = Path(fsldir) / "data" / "standard"
    affine, warp = output / "anat_to_MNI152_affine.mat", output / "anat_to_MNI152_warpcoef.nii.gz"
    t1_2mm, inverse = output / "T1_2mm.nii.gz", output / "MNI_to_anat_warpcoef.nii.gz"
    if not affine.exists():
        _run(["flirt", "-in", str(brain), "-ref", str(standard / "MNI152_T1_2mm_brain.nii.gz"), "-omat", str(affine), "-dof", "12"])
    if not warp.exists():
        _run(["fnirt", f"--in={head}", f"--aff={affine}", f"--cout={warp}", f"--config={fnirt_config}"])
    if not t1_2mm.exists():
        _run(["flirt", "-in", str(head), "-ref", str(head), "-applyisoxfm", "2", "-out", str(t1_2mm), "-interp", "trilinear"])
    if not inverse.exists():
        _run(["invwarp", f"--warp={warp}", f"--out={inverse}", f"--ref={t1_2mm}"])
    (output / "registration.json").write_text(json.dumps({"head": str(head), "brain": str(brain), "fnirt_config": fnirt_config,
                                                          "method": "FLIRT 12-DOF then FNIRT; invwarp onto T1_2mm"}, indent=1))
    return output


def atlas_to_anatomical(atlas_mni: Path, registration: Path, output: Path) -> Path:
    """An MNI atlas onto the anatomy's 2 mm grid (``T1_2mm``) through the inverse warp, nearest neighbour."""
    registration = Path(registration)
    _run(["applywarp", f"--ref={registration / 'T1_2mm.nii.gz'}", f"--in={atlas_mni}",
          f"--warp={registration / 'MNI_to_anat_warpcoef.nii.gz'}", f"--out={output}", "--interp=nn"])
    return Path(output)


def pycortex_func_to_anat(pycortex_db: Path, subject: str, xfm: str, anatomical: Path, output: Path) -> Path:
    """A pycortex functional-to-anatomical transform, in FSL's convention (``func_to_anat.mat``)."""
    import cortex
    cortex.db.filestore = str(pycortex_db)
    matrix = np.asarray(cortex.db.get_xfm(subject, xfm).to_fsl(str(anatomical)), dtype=np.float64)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, matrix, fmt="%.10f")
    return Path(output)
