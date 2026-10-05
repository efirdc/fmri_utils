"""The LeBel et al. (2023) encoding pipeline (OpenNeuro ds003020): stimulus and cross-participant
encoding models, and the ablation of a story rating from either. See ``docs/lebel2023.md``.

- ``dataset``            loaders, the response clock, TextGrid words, column <-> volume
- ``features``           English1000, BERT, GPT-2 XL and word rate on the response clock
- ``regressors``         a ``story_ratings`` run -> a rating regressor per story
- ``ablation``           the rating, orthogonalised on word rate, projected out of a feature space
- ``encoding``           the design (PCA 256 + rate column + FIR 1-4) and voxelwise ridge, in chunks
- ``cross_participant``  aCompCor cleaning, the other seven brains as predictors, the rating removed
- ``registration``       functional <-> anatomical <-> MNI152 (FSL to estimate, fslpy to apply)
- ``group``              sign-flip voxel t, region tests, null draws, in MNI
- ``nulls``              the optional variance-matched null
- ``cli``                ``fmri-lebel``
"""

from .dataset import FIR_DELAYS, RIDGE_ALPHAS, SUBJECTS, TEST_STORY, TR, Dataset, lanczos_weights, tr_times

__all__ = ["Dataset", "FIR_DELAYS", "RIDGE_ALPHAS", "SUBJECTS", "TEST_STORY", "TR", "lanczos_weights", "tr_times"]
