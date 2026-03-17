#!/usr/bin/env bash
#SBATCH --job-name=elp_rumble_train
#SBATCH --account=cso100
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/expanse/lustre/projects/cso100/%u/ElephantListeningProject/slurm_logs/%x.o%j.%N

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/_run-train-gpu.sh" "$@"