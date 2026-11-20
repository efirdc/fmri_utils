from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_RIDGE_ALPHAS = (
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
)


@dataclass(frozen=True)
class EncodingConfig:
    """Scientific settings for nested voxelwise encoding.

    A value of ``None`` in ``pca_components`` means that PCA is skipped for
    that grid point. Hyperparameters are selected independently per voxel from
    inner-fold prediction correlations.

    ``unpenalized_features`` marks the first ``k`` columns of every run's
    feature matrix (at every lag) as a base fitted by ordinary least squares.
    Within each training fold the base is fitted first; PCA and ridge on the
    remaining columns then fit what it leaves, and a prediction is the sum of
    the two. Use it for a few strong regressors -- an event's on/off response,
    say -- that would otherwise be shrunk along with many weak ones.

    ``band_columns`` and ``band_scales`` make the ridge banded (two penalties): after the per-fold
    standardisation, the listed columns of the feature matrix as the model sees it (after lags) are
    multiplied by a scale ``s`` from ``band_scales``. Ridge with penalty ``alpha`` on the scaled design is
    ridge with penalty ``alpha`` on the other columns and ``alpha / s**2`` on the band, so the grid over
    ``(s, alpha)``, selected per voxel in the inner folds like alpha alone, is banded ridge. Include
    ``0.0`` to let a voxel drop the band. Needs ``pca_components=(None,)`` and no unpenalised base.

    ``gate_columns`` and ``gated_blocks`` make the band *gated* instead: block ``k`` (columns
    ``gated_blocks[k][0]`` to ``gated_blocks[k][1] - 1``) times gate ``k`` (column ``gate_columns[k]``), row by
    row. The gates are read from the unstandardised features, so a gate of 0 switches its block off, and are
    not model columns themselves; the blocks stay in the model as ordinary columns too. With lagged
    features, give one block and one gate per lag. The same ``band_scales`` grid then applies to the gated
    band. A plain and a gated band cannot be combined.
    """

    lags: Tuple[int, ...] = (2, 3, 4, 5)
    lag_mode: str = "concat"
    ridge_alphas: Tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    pca_components: Tuple[Optional[int], ...] = (None,)
    unpenalized_features: int = 0
    fisher_z_selection: bool = True
    nuisance_columns: Tuple[str, ...] = ()
    add_run_intercept: bool = True
    cv_scheme: str = "leave_one_run_out"
    outer_splits: int = 5
    inner_splits: int = 4
    cv_shuffle: bool = False
    cv_seed: int = 0
    embargo_rows: int = 0
    mask_strategy: str = "run_mask_union"
    analysis_mask_path: Optional[str] = None
    n_permutations: int = 0
    permutation_block_length: int = 24
    random_seed: int = 0
    band_columns: Tuple[int, ...] = ()
    band_scales: Tuple[float, ...] = (1.0,)
    gate_columns: Tuple[int, ...] = ()
    gated_blocks: Tuple[Tuple[int, int], ...] = ()

    def validate(self) -> None:
        if len(self.lags) == 0 or any(lag < 0 for lag in self.lags):
            raise ValueError("lags must contain one or more non-negative integers")
        if self.lag_mode not in {"concat", "mean"}:
            raise ValueError("lag_mode must be 'concat' or 'mean'")
        if len(self.ridge_alphas) == 0 or any(alpha < 0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must contain non-negative values")
        if len(self.pca_components) == 0:
            raise ValueError("pca_components must contain at least one grid value")
        if any(value is not None and value <= 0 for value in self.pca_components):
            raise ValueError("PCA component counts must be positive or None")
        if self.cv_scheme not in {
            "leave_one_run_out",
            "grouped_run_kfold",
            "blocked_kfold",
        }:
            raise ValueError(
                "cv_scheme must be leave_one_run_out, grouped_run_kfold, or "
                "blocked_kfold"
            )
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits and inner_splits must both be >= 2")
        if self.unpenalized_features < 0:
            raise ValueError("unpenalized_features must be >= 0")
        if len(self.band_scales) == 0 or any(scale < 0 for scale in self.band_scales):
            raise ValueError("band_scales must contain non-negative values")
        if self.band_columns:
            if tuple(self.pca_components) != (None,):
                raise ValueError("band_columns needs pca_components=(None,)")
            if self.unpenalized_features:
                raise ValueError("band_columns cannot be combined with unpenalized_features")
            if any(column < 0 for column in self.band_columns):
                raise ValueError("band_columns must be non-negative indices")
        if self.gate_columns or self.gated_blocks:
            if self.band_columns:
                raise ValueError("a plain band (band_columns) and a gated band cannot be combined")
            if len(self.gate_columns) != len(self.gated_blocks):
                raise ValueError("give one gate column per gated block")
            if tuple(self.pca_components) != (None,):
                raise ValueError("a gated band needs pca_components=(None,)")
            if self.unpenalized_features:
                raise ValueError("a gated band cannot be combined with unpenalized_features")
            for start, stop in self.gated_blocks:
                if not 0 <= start < stop:
                    raise ValueError("gated_blocks must be (start, stop) with 0 <= start < stop")
            gates = set(self.gate_columns)
            if any(start <= g < stop for g in gates for start, stop in self.gated_blocks):
                raise ValueError("a gate column cannot lie inside a gated block")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows must be >= 0")
        if self.mask_strategy not in {
            "run_mask_union",
            "run_mask_intersection",
            "analysis_mask_only",
        }:
            raise ValueError(
                "mask_strategy must be run_mask_union, run_mask_intersection, "
                "or analysis_mask_only"
            )
        if self.mask_strategy == "analysis_mask_only" and not self.analysis_mask_path:
            raise ValueError("analysis_mask_only requires analysis_mask_path")
        if self.n_permutations < 0:
            raise ValueError("n_permutations must be >= 0")
        if self.permutation_block_length <= 0:
            raise ValueError("permutation_block_length must be positive")


def fmriprep_nuisance_columns(
    *,
    n_acompcor: int = 4,
    include_cosines: bool = True,
    include_motion: bool = True,
    include_fd: bool = True,
) -> Tuple[str, ...]:
    """Return a conventional fMRIPrep nuisance-column specification.

    ``cosine*`` is a wildcard expanded against each confounds TSV. Motion
    rotations are kept in fMRIPrep's native radians.
    """

    if n_acompcor < 0:
        raise ValueError("n_acompcor must be >= 0")
    columns = []
    if include_cosines:
        columns.append("cosine*")
    if include_motion:
        columns.extend(
            ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        )
    if include_fd:
        columns.append("framewise_displacement")
    columns.extend("a_comp_cor_{:02d}".format(index) for index in range(n_acompcor))
    return tuple(columns)
