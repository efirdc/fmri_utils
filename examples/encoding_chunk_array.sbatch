#!/usr/bin/env bash
#SBATCH --job-name=encoding_chunks
#SBATCH --array=1-100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/encoding_%A_%a.out

set -euo pipefail

# Load site-specific modules and activate the environment here.
CHUNK_MANIFEST="manifests/story_chunks.csv"
OUTPUT_ROOT="results/story_perm200"

srun --cpu-bind=cores fmri-encoding fit-chunk \
  --chunk-manifest "${CHUNK_MANIFEST}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --lags 2,3,4,5 \
  --pca-components none,16,32,64 \
  --ridge-alphas 0.01,0.1,1,10,100 \
  --cv-scheme leave_one_run_out \
  --nuisance-profile fmriprep \
  --n-acompcor 4 \
  --n-permutations 200 \
  --permutation-block-length 24 \
  --seed 1234
