# Chunked Encoding And Stitching

Use voxel chunking when a whole-subject fit is too slow or memory-intensive,
especially when nested CV is repeated for many permutations. Each array task
fits one contiguous range of masked voxels. A separate command verifies exact
coverage and writes the final NIfTI maps.

Chunking changes scheduling, not the statistical model. Every chunk must use
the same random seed and scientific arguments. This gives each chunk the same
permutation schedule; only its target voxels differ.

The current implementation chunks the voxel axis only. Each task runs all
requested permutations for its voxel range; permutation results are not split
across separate tasks.

## 1. Create The Run Manifest

Prepare the generic run manifest as described in
[Voxelwise Encoding](encoding.md). Audit feature/BOLD alignment before running
anything expensive.

## 2. Create A Chunk Manifest

```bash
fmri-encoding make-chunks \
  --manifest manifests/story_runs.csv \
  --output manifests/story_chunks.csv \
  --chunk-size 1000 \
  --mask-strategy run_mask_union
```

For an explicit analysis mask:

```bash
fmri-encoding make-chunks \
  --manifest manifests/story_runs.csv \
  --output manifests/story_chunks.csv \
  --chunk-size 1000 \
  --mask-strategy analysis_mask_only \
  --analysis-mask masks/analysis_mask.nii.gz
```

The command loads masks but not BOLD. It writes one row per subject and voxel
range and prints the required 1-based SLURM array range. `voxel_start` is
inclusive and `voxel_stop` is exclusive.

## 3. Benchmark A Chunk

Run one task on a compute node with the exact production settings:

```bash
srun fmri-encoding fit-chunk \
  --chunk-manifest manifests/story_chunks.csv \
  --task-id 1 \
  --output-root results/story_perm200 \
  --lags 2,3,4,5 \
  --pca-components none,16,32,64 \
  --ridge-alphas 0.01,0.1,1,10,100 \
  --cv-scheme leave_one_run_out \
  --nuisance-profile fmriprep \
  --n-acompcor 4 \
  --n-permutations 200 \
  --permutation-block-length 24 \
  --seed 1234
```

Choose chunk size from measured runtime. Very small chunks repeat BOLD loading,
feature loading, fold construction, PCA, and permutation setup too often. Very
large chunks reduce scheduler parallelism and may exceed memory or wall time.

## 4. Submit The Array

Set the array end to the task count printed by `make-chunks`. The complete
example is [`encoding_chunk_array.sbatch`](../examples/encoding_chunk_array.sbatch).

```bash
job_id=$(sbatch --parsable examples/encoding_chunk_array.sbatch)
```

The array task ID directly selects a chunk-manifest row. Do not change model,
CV, nuisance, permutation, or seed arguments between tasks.

Chunk files are compact NPZ files under:

```text
OUTPUT_ROOT/
  sub-001/
    chunks/
      chunk-00000_voxels-000000-001000.npz
      chunk-00001_voxels-001000-002000.npz
```

They contain only modeled voxel values, not mostly-empty whole-brain NIfTIs.

## 5. Stitch In A Separate Job

Submit stitching only after every array task succeeds:

```bash
sbatch --dependency=afterok:${job_id} \
  examples/encoding_stitch.sbatch
```

Or run the command directly on a compute node:

```bash
srun fmri-encoding stitch \
  --chunk-manifest manifests/story_chunks.csv \
  --output-root results/story_perm200
```

The stitcher fails rather than producing a partial result when it detects:

- a missing chunk;
- a gap or overlap in voxel ranges;
- changed mask geometry;
- different model/CV/permutation settings or input provenance;
- inconsistent metric or fold axes.

Final outputs are written to:

```text
OUTPUT_ROOT/sub-001/stitched/
```

See [Encoding Output Reference](encoding_outputs.md) for every map definition.

## Prediction Traces

Chunk tasks omit observed/predicted fold traces by default because they can be
large. Add `--save-traces` to every `fit-chunk` call if they are required. The
stitcher validates row provenance and concatenates the voxel columns into one
`outer_fold_traces.npz`.

## Reproducibility Rules

- Keep `--seed` identical across all chunks so permutation index `p` means the
  same block ordering at every voxel.
- Regenerate the chunk manifest after changing a mask.
- Use a new output root after changing scientific settings.
- Keep the run and chunk manifests with the final result.
- Do not mix chunk files from separate campaigns. The stitch signature is a
  guardrail, not a substitute for clean output directories.
