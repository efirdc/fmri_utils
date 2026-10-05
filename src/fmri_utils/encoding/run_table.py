"""Responses stored as one matrix per run, described by a run table.

Many naturalistic datasets release their preprocessed responses as matrices (time x voxels) per run,
in HDF5 or NumPy files, with a mask that says which voxel each column is. A run table is a CSV with
one row per (subject, run):

``subject``       subject id
``run``           run id (a story, a movie part); runs with the same id share a stimulus
``role``          ``train`` or ``test``
``response``      the run's file (.hf5/.h5/.hdf5, .npy, .npz)
``response_key``  HDF5/npz dataset of the responses (default ``data``)
``repeats_key``   for a test run: dataset of its single repeats (repeats x time x voxels), if released
``mask``          NIfTI whose nonzero voxels are the columns
``column_order``  ``C`` (numpy order of the mask, default), ``F``, or ``pycortex`` (pycortex's
                  masked-vector order: the mask transposed to z, y, x)

Paths may be relative to the table. Rows are kept in table order: the training runs' order sets
the inner cross-validation folds (run i in fold i mod k), so write them in a fixed order.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

COLUMN_ORDERS = ("C", "F", "pycortex")


class RunTable:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        with self.path.open(newline="", encoding="utf-8") as handle:
            self.rows = [dict(r) for r in csv.DictReader(handle)]
        missing = {"subject", "run", "role", "response", "mask"} - set(self.rows[0] if self.rows else {})
        if missing:
            raise ValueError(f"{self.path}: missing columns {sorted(missing)}")
        for row in self.rows:
            if row["role"] not in ("train", "test"):
                raise ValueError(f"{self.path}: role must be train or test, not {row['role']!r}")
            row["column_order"] = row.get("column_order") or "C"
            if row["column_order"] not in COLUMN_ORDERS:
                raise ValueError(f"{self.path}: column_order must be one of {COLUMN_ORDERS}")

    @classmethod
    def write(cls, rows: list[dict], path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["subject", "run", "role", "response", "response_key", "repeats_key", "mask", "column_order"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({f: row.get(f, "") for f in fields})
        return path

    # ---- runs ---------------------------------------------------------------------------------
    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.path.parent / path)

    def subjects(self) -> list[str]:
        return list(dict.fromkeys(r["subject"] for r in self.rows))

    def runs(self, subject: str, role: str = "train") -> list[str]:
        return [r["run"] for r in self.rows if r["subject"] == subject and r["role"] == role]

    def test_run(self, subject: str) -> str:
        tests = self.runs(subject, "test")
        if len(tests) != 1:
            raise ValueError(f"{subject}: expected one test run, found {tests}")
        return tests[0]

    def shared_runs(self, subjects=None, role: str = "train") -> list[str]:
        """The runs every one of ``subjects`` has, in the first subject's order."""
        subjects = subjects or self.subjects()
        shared = None
        for subject in subjects:
            have = self.runs(subject, role)
            shared = have if shared is None else [r for r in shared if r in have]
        return shared or []

    def row(self, subject: str, run: str) -> dict:
        for r in self.rows:
            if r["subject"] == subject and r["run"] == run:
                return r
        raise KeyError(f"no run {run} for {subject}")

    # ---- responses --------------------------------------------------------------------------------
    @staticmethod
    def _read(path: Path, key: str, index):
        suffix = path.suffix.lower()
        if suffix in (".hf5", ".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as handle:
                return np.asarray(handle[key][index], dtype=np.float32)
        if suffix == ".npz":
            return np.asarray(np.load(path)[key][index], dtype=np.float32)
        return np.asarray(np.load(path, mmap_mode="r")[index], dtype=np.float32)

    def response(self, subject: str, run: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Time x columns ``start:stop`` of a run (HDF5 reads only that slab)."""
        row = self.row(subject, run)
        data = self._read(self._resolve(row["response"]), row.get("response_key") or "data", (slice(None), slice(start, stop)))
        if data.ndim != 2 or np.isinf(data).any():
            raise ValueError(f"invalid response {subject} {run}: {data.shape}")
        return data

    def repeats(self, subject: str, run: str, start: int = 0, stop: int | None = None) -> np.ndarray | None:
        """A test run's single repeats (repeats x time x columns), or None if none were released."""
        row = self.row(subject, run)
        if not row.get("repeats_key"):
            return None
        data = self._read(self._resolve(row["response"]), row["repeats_key"], (slice(None), slice(None), slice(start, stop)))
        if data.ndim != 3 or np.isinf(data).any():
            raise ValueError(f"invalid repeats {subject} {run}: {data.shape}")
        return data

    def n_rows(self, subject: str, run: str) -> int:
        row = self.row(subject, run)
        path = self._resolve(row["response"])
        if path.suffix.lower() in (".hf5", ".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as handle:
                return int(handle[row.get("response_key") or "data"].shape[0])
        return int(self._read(path, row.get("response_key") or "data", (slice(None), slice(0, 1))).shape[0])

    # ---- columns and the volume -------------------------------------------------------------------
    @lru_cache(maxsize=16)
    def _mask(self, subject: str):
        import nibabel as nib
        rows = [r for r in self.rows if r["subject"] == subject]
        image = nib.load(self._resolve(rows[0]["mask"]))
        return image, np.asarray(image.dataobj) > 0, rows[0]["column_order"]

    def mask_image(self, subject: str):
        return self._mask(subject)[0]

    def n_columns(self, subject: str) -> int:
        return int(self._mask(subject)[1].sum())

    def unmask(self, subject: str, values: np.ndarray) -> np.ndarray:
        """A vector over columns as a volume (NaN off the mask)."""
        _, mask, order = self._mask(subject)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != int(mask.sum()):
            raise ValueError(f"{values.size} values for {int(mask.sum())} columns")
        if order == "pycortex":
            out = np.full(mask.T.shape, np.nan, dtype=np.float32)
            out[mask.T] = values
            return out.T
        out = np.full(mask.size, np.nan, dtype=np.float32)
        flat = mask.ravel(order=order)
        out[flat] = values
        return out.reshape(mask.shape, order=order)

    def column_volume(self, subject: str) -> np.ndarray:
        """Each voxel's column, 1-based (0 off the mask)."""
        order = self.unmask(subject, np.arange(1, self.n_columns(subject) + 1, dtype=np.float32))
        return np.rint(np.nan_to_num(order)).astype(np.int64)

    def save_map(self, subject: str, values: np.ndarray, path: Path | str) -> Path:
        import nibabel as nib
        image = self.mask_image(subject)
        header = image.header.copy()
        header.set_data_dtype(np.float32)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.unmask(subject, values), image.affine, header), path)
        return path


def common_finite(matrices) -> np.ndarray:
    """Columns finite in every matrix (2-D, or 3-D with columns last)."""
    valid = None
    for matrix in matrices:
        if matrix is None:
            continue
        ok = np.isfinite(matrix).reshape(-1, matrix.shape[-1]).all(axis=0)
        valid = ok if valid is None else valid & ok
    if valid is None or not valid.any():
        raise ValueError("no column is finite in every matrix")
    return valid
