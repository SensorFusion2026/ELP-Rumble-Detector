#!/usr/bin/env bash

# Usage:
#   bash slurm_scripts/submit-train.sh <shared|debug>
# Example:
#   bash slurm_scripts/submit-train.sh shared

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <shared|debug>"
  exit 1
fi

PARTITION="$1"

if [[ "${PARTITION}" == "shared" ]]; then
  JOB_SCRIPT="slurm_scripts/run-train-gpu-shared.sh"
  JOB_NAME="train_cnn_shared"
elif [[ "${PARTITION}" == "debug" ]]; then
  JOB_SCRIPT="slurm_scripts/run-train-gpu-debug.sh"
  JOB_NAME="train_cnn_debug"
else
  echo "Partition must be 'shared' or 'debug'"
  exit 1
fi

sbatch --job-name="${JOB_NAME}" "${JOB_SCRIPT}"
