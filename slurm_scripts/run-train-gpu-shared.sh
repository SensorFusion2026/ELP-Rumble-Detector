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
#SBATCH --output=/expanse/lustre/projects/cso100/%u/ElephantListeningProject/ELP-Rumble-Detector/slurm_logs/%x.o%j.%N

exec bash "/expanse/lustre/projects/cso100/$USER/ElephantListeningProject/ELP-Rumble-Detector/slurm_scripts/_run-train-gpu.sh" "$@"