#!/usr/bin/env bash
#SBATCH --job-name=encoding_stitch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/encoding_stitch_%j.out

set -euo pipefail

# Load site-specific modules and activate the environment here.
srun --cpu-bind=cores fmri-encoding stitch \
  --chunk-manifest manifests/story_chunks.csv \
  --output-root results/story_perm200
