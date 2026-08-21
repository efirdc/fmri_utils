#!/usr/bin/env bash
#SBATCH --job-name=fmri-encoding
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Subject labels sub-001 through sub-020.
#SBATCH --array=1-20

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the generic run-manifest CSV}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the result directory}"

subject_id=$(printf "sub-%03d" "$SLURM_ARRAY_TASK_ID")

srun --cpu-bind=cores --hint=nomultithread fmri-encoding fit \
  --manifest "$MANIFEST" \
  --subject "$subject_id" \
  --output-dir "$OUTPUT_ROOT/$subject_id" \
  --lags 2,3,4,5 \
  --pca-components none,16,32,64 \
  --cv-scheme leave_one_run_out \
  --nuisance-profile fmriprep
