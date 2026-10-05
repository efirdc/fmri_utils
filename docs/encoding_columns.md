# Encoding On Column Data

Many naturalistic fMRI datasets release preprocessed responses as one matrix
per run (time × voxels, in HDF5 or NumPy files), with a mask that says which
voxel each column is. `fmri_utils.encoding` fits voxelwise encoding models on
that format directly, with a fixed held-out test run, in column chunks that
stitch exactly into the whole fit.

```bash
fmri-encoding fit-columns --run-table runs.csv --subject S01 --feature-root features \
  --feature bert --extra-root features/word_rate --output fits --chunk 0 --n-chunks 5
fmri-encoding stitch-columns --run-table runs.csv --subject S01 --folder fits/bert/S01
```

## The Run Table

A CSV with one row per subject and run (`encoding.run_table.RunTable`):

| column | meaning |
|---|---|
| `subject` | subject id |
| `run` | run id. Runs with the same id share a stimulus |
| `role` | `train` or `test` (one test run per subject) |
| `response` | the run's file: `.hf5`/`.h5`/`.hdf5`, `.npy` or `.npz` |
| `response_key` | HDF5/npz dataset of the responses (default `data`) |
| `repeats_key` | for the test run: dataset of its single repeats (repeats × time × voxels), if released. Gives the noise ceiling |
| `mask` | NIfTI whose nonzero voxels are the columns |
| `column_order` | `C` (numpy order of the mask, default), `F`, or `pycortex` (pycortex's masked-vector order: the mask transposed to z, y, x) |

Paths may be relative to the table. Rows keep their table order, and the
training runs' order sets the cross-validation folds (run i in fold i mod 5).
Write them in a fixed order.

`RunTable` reads:
- runs: `subjects`, `runs`, `test_run`, `shared_runs`;
- responses: `response` (a column slab: HDF5 reads only that part), `repeats`;
- the grid: `n_columns`, `unmask` and `column_volume` (column ↔ voxel), and
  `save_map` (a column vector as a NIfTI on the mask's grid).

## The Model

`encoding.design.prepare_design` builds the classic naturalistic-stimulus
design. Every step is fitted on the training runs only:

1. z-score the features, then PCA (256 components, randomized, seed 2023);
2. append extra columns, such as a word rate, after the PCA, z-scored. A single
   regressor would otherwise be one of hundreds of PCA inputs, smeared across
   components;
3. FIR delays per run (1-4 samples by default; zero-padded, so nothing crosses a
   run boundary);
4. z-score the delayed design.

`encoding.columns.fit_voxelwise` fits per-voxel ridge through `fit_encoding`
with `splits.fixed_test_plan`:
- alpha comes from 10 … 1e5 in half-decade steps (`WIDE_ALPHAS`), chosen per
  voxel by cross-validation over whole training runs (Fisher-z-averaged r);
- the model is refitted on every training run and scored by Pearson r on the test
  run.

`encoding.noise_ceiling.noise_ceiling` estimates the correlation ceiling from
the test repeats (Schoppe et al.'s signal/noise split), and `normalise` gives
r / max(ceiling, 0.3).

`design.design_from_blocks` takes predictors that are already on the response
clock (other subjects' responses, say) as they are: lag 0, z-scored with the
training statistics.

## Chunks

`fit_feature_space` fits one subject's column chunk on
`<feature_root>/<feature>/<run>.npy` (rows = response rows) and writes
`<output>/<model>/<subject>/chunks/chunk_<c>_of_<n>.{npz,json}`. The `npz`
holds:
- `voxel_index`, `correlation_raw`, `selected_alpha`;
- with repeats, `noise_ceiling` and `correlation_noise_ceiling_normalized`.

Every voxelwise quantity depends only on that voxel, so chunks are independent
cluster tasks. `read_chunks` returns them as column vectors and refuses
incomplete sets. `stitch` writes them as maps.

`encoding.columns.Chunk` loads a chunk's responses once and refits them on any
feature set. The [ablation](ablation.md) null uses it to fit hundreds of
feature spaces per task.
