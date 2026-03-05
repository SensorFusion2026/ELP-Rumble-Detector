#!/usr/bin/env bash
#SBATCH --job-name=train_cnn_debug
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x.o%j.%N

set -euo pipefail

declare -xr SINGULARITY_MODULE="singularitypro/3.11"
SANDBOX_PATH="${SANDBOX_PATH:-../sandbox}"

mkdir -p slurm_logs

module purge
module load "${SINGULARITY_MODULE}"
module list

export NVIDIA_DISABLE_REQUIRE=true

# Debug-size runtime budget for quick path checks.
time -p singularity exec --bind /expanse,/scratch --nv "${SANDBOX_PATH}" \
  python -u -m elp_rumble.training.train_cnn
