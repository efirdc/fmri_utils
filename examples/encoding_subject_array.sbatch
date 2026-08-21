#!/usr/bin/env bash
#SBATCH --job-name=fmri-encoding
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the generic run-manifest CSV}"
: "${SUBJECTS_FILE:?Set SUBJECTS_FILE to one subject ID per line}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the result directory}"

subject_id=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
if [[ -z "$subject_id" ]]; then
  echo "No subject for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

srun --cpu-bind=cores --hint=nomultithread fmri-encoding fit \
  --manifest "$MANIFEST" \
  --subject "$subject_id" \
  --output-dir "$OUTPUT_ROOT/$subject_id" \
  --lags 2,3,4,5 \
  --pca-components none,16,32,64 \
  --cv-scheme leave_one_run_out \
  --nuisance-profile fmriprep
