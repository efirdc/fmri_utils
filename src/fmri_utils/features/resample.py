"""Putting unit-level values on a regular clock.

Kept apart from extraction on purpose. A model run is expensive and a
resampling is not, so the expensive thing should happen once and the cheap
thing should be a choice you can revisit. Every kernel here takes unit times
and returns a weight matrix, so comparing two of them costs a matrix multiply
rather than another pass over a language model.

Two of these are not interchangeable, and the difference is easy to miss.
``lanczos_sum`` is the Huth lab's published resampler, and its weights sum to
the local event rate rather than to one: each output sample is a rate-weighted
*sum*, so it carries how fast the stimulus was arriving as well as what it was.
Everything else here normalises, giving a weighted average on the same scale as
the input.
"""

from __future__ import annotations

import numpy as np


def lanczos_kernel(delta: np.ndarray, window: int = 3) -> np.ndarray:
    """Three-lobe Lanczos, in units of the output sample spacing."""
    weights = np.zeros_like(delta, dtype=np.float64)
    nonzero = (np.abs(delta) <= window) & (delta != 0)
    weights[delta == 0] = 1.0
    values = delta[nonzero]
    weights[nonzero] = (
        window * np.sin(np.pi * values) * np.sin(np.pi * values / window)
        / (np.pi ** 2 * values ** 2)
    )
    return weights


def hann_kernel(delta: np.ndarray, half_width: float = 2.0) -> np.ndarray:
    """Raised cosine, strictly non-negative, compact support.

    Non-negativity is what makes a normalised average safe: the denominator can
    never vanish or change sign, which is exactly how a normalised Lanczos
    fails in sparse stretches.
    """
    return np.where(np.abs(delta) <= half_width,
                    0.5 * (1.0 + np.cos(np.pi * delta / half_width)), 0.0)


def gaussian_kernel(delta: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    return np.exp(-0.5 * (delta / sigma) ** 2)


def boxcar_kernel(delta: np.ndarray, half_width: float = 0.5) -> np.ndarray:
    return (np.abs(delta) <= half_width).astype(np.float64)


KERNELS = {"lanczos": lanczos_kernel, "hann": hann_kernel,
           "gaussian": gaussian_kernel, "boxcar": boxcar_kernel}


def weights(unit_times, sample_times, kernel: str = "hann", normalise: bool = True,
            **kernel_args) -> tuple[np.ndarray, np.ndarray]:
    """Weight matrix from unit times to sample times, plus its row support.

    ``delta`` is in units of the output spacing, which is how a Lanczos cutoff
    is conventionally defined, so the same kernel arguments mean the same thing
    at any TR.

    Returns the matrix and a boolean of which output samples any unit reached.
    A row with no support is left at zero rather than divided by nothing.
    """
    unit_times = np.asarray(unit_times, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    spacing = float(np.mean(np.diff(sample_times)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("sample_times must be increasing and regularly spaced")
    delta = (sample_times[:, None] - unit_times[None, :]) / spacing
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; have {sorted(KERNELS)}")
    matrix = KERNELS[kernel](delta, **kernel_args)
    support = np.abs(matrix).sum(axis=1) > 1e-9
    if normalise:
        total = matrix.sum(axis=1)
        safe = np.abs(total) > 1e-9
        matrix = np.divide(matrix, total[:, None], out=np.zeros_like(matrix),
                           where=safe[:, None])
        support &= safe
    return matrix, support


def resample(values, unit_times, sample_times, kernel: str = "hann",
             normalise: bool = True, **kernel_args) -> np.ndarray:
    """Unit-level values onto the sample clock. ``values`` is (units, dims)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    matrix, _ = weights(unit_times, sample_times, kernel=kernel,
                        normalise=normalise, **kernel_args)
    return (matrix @ values).astype(np.float32)


def lanczos_sum(values, unit_times, sample_times, window: int = 3) -> np.ndarray:
    """The published resampler, reproduced exactly: unnormalised Lanczos.

    Here for comparison and for reproducing prior work. Prefer a normalised
    kernel for anything new; see the module docstring for why.
    """
    return resample(values, unit_times, sample_times, kernel="lanczos",
                    normalise=False, window=window)


def event_rate(unit_times, sample_times, kernel: str = "hann",
               **kernel_args) -> np.ndarray:
    """Units per second on the sample clock, through the same kernel.

    Worth carrying alongside any normalised feature set: normalising removes
    stimulus density from the features, and this is where it goes if you want
    the model to have it as its own regressor rather than smuggled into every
    dimension.
    """
    unit_times = np.asarray(unit_times, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    spacing = float(np.mean(np.diff(sample_times)))
    matrix, _ = weights(unit_times, sample_times, kernel=kernel,
                        normalise=False, **kernel_args)
    integral = matrix.sum(axis=1)
    # The kernel's own time integral, so the result is per second rather than
    # per unit of kernel gain.
    width = float(np.abs(KERNELS[kernel](
        np.linspace(-8, 8, 4001), **kernel_args)).sum() * (16.0 / 4000.0) * spacing)
    return (integral / max(width, 1e-9)).astype(np.float32)
