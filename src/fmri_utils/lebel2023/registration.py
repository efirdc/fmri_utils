"""Registration: the functional grid, the subject's anatomy, and MNI152.

Per subject, a registration folder holds (``<registration>/<subject>/``):

``func_to_anat.mat``               the pycortex functional-to-anatomical transform, in FSL's convention
``anat_to_MNI152_warpcoef.nii.gz`` FNIRT from the anatomical (pycortex ``raw.nii.gz``) to MNI152 2 mm
``MNI_to_anat_warpcoef.nii.gz``    its inverse (``invwarp``), for atlases into the anatomy

and ``<registration>/MNI152_T1_2mm.nii.gz`` the template. ``register_subject`` estimates them on a
machine with FSL and pycortex (FLIRT 12-DOF, then FNIRT with ``T1_2_MNI152_2mm``); everything else
here only needs fslpy (``pip install fslpy``) to read the FNIRT field.

``column_labels`` gives each response column the label of an atlas in the subject's anatomical
space (``atlas_to_anat`` puts Harvard-Oxford there). ``MNIWarp`` is a sparse matrix from response
columns to MNI 2 mm voxels: trilinear, NaN treated as absent, a voxel kept when its nearest column
has data, which is what ``applywarp --premat`` with a separately warped support mask produces.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .dataset import Dataset

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
    """Resample ``source`` onto ``reference`` with a FLIRT source->reference matrix (no FSL needed)."""
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


def column_labels(dataset: Dataset, subject: str, atlas_anat: Path, registration: Path) -> np.ndarray:
    """The label of an anatomical-space atlas at each response column (nearest neighbour)."""
    import nibabel as nib
    mask = nib.load(dataset.mask_path(subject))
    order = dataset.column_volume(subject)
    atlas = nib.load(atlas_anat)
    matrix = np.loadtxt(Path(registration) / subject / "func_to_anat.mat")
    source = nib.Nifti1Image(np.asarray(atlas.dataobj).astype(np.float32), atlas.affine, atlas.header)
    labels = np.nan_to_num(warp_affine(source, np.linalg.inv(matrix), mask, order=0)).round().astype(int)
    out = np.zeros(dataset.n_columns(subject), dtype=int)
    inside = order > 0
    out[order[inside] - 1] = labels[inside]
    return out


def func_to_mni_coordinates(dataset: Dataset, subject: str, registration: Path) -> np.ndarray:
    """For every MNI 2 mm voxel, the (fractional) functional voxel it pulls from: 3 x n."""
    import nibabel as nib
    from fsl.data.image import Image
    from fsl.transform import fnirt
    registration = Path(registration)
    ref = Image(str(registration / "MNI152_T1_2mm.nii.gz"))
    src = Image(str(dataset.anatomical(subject)))
    field = fnirt.readFnirt(str(registration / subject / "anat_to_MNI152_warpcoef.nii.gz"), src, ref)
    grid = np.indices(ref.shape[:3], dtype=np.float64).reshape(3, -1).T
    anat = field.transform(grid, "voxel", "fsl")
    premat = np.loadtxt(registration / subject / "func_to_anat.mat")
    to_func = np.linalg.inv(voxel_to_fsl(nib.load(dataset.mask_path(subject)))) @ np.linalg.inv(premat)
    return (np.hstack([anat, np.ones((anat.shape[0], 1))]) @ to_func.T)[:, :3].T


class MNIWarp:
    """Response columns -> MNI152 2 mm, as a sparse trilinear matrix (build once per subject, save, reuse)."""

    def __init__(self, mni_index, columns, weights, nearest_column, n_columns: int):
        from scipy import sparse
        self.mni_index = np.asarray(mni_index, dtype=np.int64)
        self.nearest = np.asarray(nearest_column, dtype=np.int64)
        rows = np.repeat(np.arange(columns.shape[0]), columns.shape[1]).reshape(columns.shape)
        ok = columns >= 0
        self.matrix = sparse.csr_matrix((weights[ok].astype(np.float32), (rows[ok], columns[ok])),
                                        shape=(columns.shape[0], n_columns))
        self.columns, self.weights, self.n_columns = columns, weights, n_columns

    @classmethod
    def build(cls, dataset: Dataset, subject: str, registration: Path) -> "MNIWarp":
        import nibabel as nib
        func_shape = nib.load(dataset.mask_path(subject)).shape[:3]
        column = dataset.column_volume(subject).ravel() - 1                 # -1 off the mask
        coords = func_to_mni_coordinates(dataset, subject, registration)
        nearest = np.rint(coords).astype(np.int64)
        inside = np.all((nearest >= 0) & (nearest < np.array(func_shape)[:, None]), axis=0)
        flat = np.full(nearest.shape[1], -1, dtype=np.int64)
        flat[inside] = np.ravel_multi_index(tuple(nearest[:, inside]), func_shape)
        nearest_column = np.where(flat >= 0, column[np.maximum(flat, 0)], -1)
        mni_index = np.flatnonzero(nearest_column >= 0)
        c = coords[:, mni_index]
        base = np.floor(c).astype(np.int64)
        frac = c - base
        columns, weights = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = base + np.array([dx, dy, dz])[:, None]
                    w = ((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) * (frac[2] if dz else 1 - frac[2]))
                    ok = np.all((idx >= 0) & (idx < np.array(func_shape)[:, None]), axis=0)
                    corner = np.full(idx.shape[1], -1, dtype=np.int64)
                    corner[ok] = column[np.ravel_multi_index(tuple(idx[:, ok]), func_shape)]
                    columns.append(corner)
                    weights.append(np.where(ok & (corner >= 0), w, 0.0))
        return cls(mni_index, np.stack(columns, axis=1), np.stack(weights, axis=1).astype(np.float32),
                   nearest_column[mni_index], dataset.n_columns(subject))

    def save(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mni_index=self.mni_index.astype(np.int32), columns=self.columns.astype(np.int32),
                            weights=self.weights, nearest=self.nearest.astype(np.int32), n_columns=self.n_columns)
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "MNIWarp":
        data = np.load(path)
        return cls(data["mni_index"], data["columns"].astype(np.int64), data["weights"], data["nearest"], int(data["n_columns"]))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Column values (n_columns,) or (n_columns, k) -> values on ``mni_index`` (NaN where the nearest column is NaN)."""
        values = np.asarray(values, dtype=np.float32)
        support = np.isfinite(values)
        out = np.asarray(self.matrix @ np.where(support, values, 0.0), dtype=np.float32)
        near = support[self.nearest] if values.ndim == 1 else support[self.nearest].all(axis=-1)
        out[~near] = np.nan
        return out

    def volume(self, values: np.ndarray) -> np.ndarray:
        """Column values as a full MNI 2 mm volume (NaN outside)."""
        out = np.full(int(np.prod(MNI_SHAPE)), np.nan, dtype=np.float32)
        out[self.mni_index] = self.apply(values)
        return out.reshape(MNI_SHAPE)


def save_mni(volume: np.ndarray, path: Path) -> Path:
    import nibabel as nib
    image = nib.Nifti1Image(np.asarray(volume, dtype=np.float32), MNI_AFFINE)
    image.header.set_data_dtype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    return Path(path)


# ---- estimating the registration (FSL and pycortex) -------------------------------------------------
def _run(command: list[str]) -> None:
    print("  $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def register_subject(dataset: Dataset, subject: str, output: Path, fsldir: Path,
                     fnirt_config: str = "T1_2_MNI152_2mm") -> Path:
    """FLIRT + FNIRT anatomical -> MNI152 2 mm, the inverse warp, and func_to_anat.mat from pycortex."""
    import cortex
    output = Path(output)
    folder = output / subject
    folder.mkdir(parents=True, exist_ok=True)
    head, brain = dataset.anatomical(subject, "raw.nii.gz"), dataset.anatomical(subject, "brainmask.nii.gz")
    standard = Path(fsldir) / "data" / "standard"
    cortex.database.default_filestore = str(dataset.derivatives / "pycortex-db")
    cortex.db.filestore = str(dataset.derivatives / "pycortex-db")
    transforms = sorted(p.name for p in (dataset.derivatives / "pycortex-db" / subject / "transforms").iterdir() if p.is_dir())
    matrix = np.asarray(cortex.db.get_xfm(subject, transforms[0]).to_fsl(str(head)), dtype=np.float64)
    np.savetxt(folder / "func_to_anat.mat", matrix, fmt="%.10f")
    affine, warp, inverse = folder / "anat_to_MNI152_affine.mat", folder / "anat_to_MNI152_warpcoef.nii.gz", folder / "MNI_to_anat_warpcoef.nii.gz"
    if not affine.exists():
        _run(["flirt", "-in", str(brain), "-ref", str(standard / "MNI152_T1_2mm_brain.nii.gz"), "-omat", str(affine), "-dof", "12"])
    if not warp.exists():
        _run(["fnirt", f"--in={head}", f"--aff={affine}", f"--cout={warp}", f"--config={fnirt_config}"])
    t1_2mm = folder / "T1_2mm.nii.gz"
    if not t1_2mm.exists():
        _run(["flirt", "-in", str(head), "-ref", str(head), "-applyisoxfm", "2", "-out", str(t1_2mm), "-interp", "trilinear"])
    if not inverse.exists():
        _run(["invwarp", f"--warp={warp}", f"--out={inverse}", f"--ref={t1_2mm}"])
    template = output / "MNI152_T1_2mm.nii.gz"
    if not template.exists():
        template.write_bytes((standard / "MNI152_T1_2mm.nii.gz").read_bytes())
    (folder / "registration.json").write_text(json.dumps({"subject": subject, "xfm": transforms[0], "fnirt_config": fnirt_config,
                                                          "method": "FLIRT 12-DOF then FNIRT; func_to_anat from pycortex"}, indent=1))
    return folder


def atlas_to_anat(atlas_mni: Path, subject: str, registration: Path, output: Path) -> Path:
    """An MNI atlas into the subject's anatomical space (``T1_2mm``), nearest neighbour, with FSL's applywarp."""
    folder = Path(registration) / subject
    _run(["applywarp", f"--ref={folder / 'T1_2mm.nii.gz'}", f"--in={atlas_mni}",
          f"--warp={folder / 'MNI_to_anat_warpcoef.nii.gz'}", f"--out={output}", "--interp=nn"])
    return Path(output)
