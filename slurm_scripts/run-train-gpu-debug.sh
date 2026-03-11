#!/usr/bin/env bash
#SBATCH --job-name=elp_rumble_train_debug
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x.o%j.%N

set -euo pipefail

usage() {
  cat <<EOF >&2
Usage: $0 <cnn|rnn> <model1|model2|model3> [epochs]
  cnn|rnn         selects the trainer module (train_cnn / train_rnn)
  model1..model3  picks a dataset split definition
  epochs          optional whole-number epoch count override
EOF
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi

MODELTYPE="${1,,}"
MODEL="${2,,}"
EPOCHS="${3:-}"

if [[ ! "cnn rnn" =~ (^| )${MODELTYPE}( |$) ]]; then
  echo "MODELTYPE must be 'cnn' or 'rnn'" >&2
  usage
fi

if [[ ! "model1 model2 model3" =~ (^| )${MODEL}( |$) ]]; then
  echo "MODEL must be one of model1, model2, model3" >&2
  usage
fi

if [ -n "$EPOCHS" ] && ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  usage
fi

module purge
module load singularitypro
module list

export PROJECT_ROOT="/expanse/lustre/projects/cso100/$USER/ElephantListeningProject"
export SIF="/cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif"
export PYTHONUSERBASE="$PROJECT_ROOT/.pythonuserbase"
mkdir -p slurm_logs
export NVIDIA_DISABLE_REQUIRE=true

# Install the editable repo into the persistent user base.
singularity exec --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  python -m pip install --upgrade --user pip setuptools wheel && \
  python -m pip install --user -e '$PROJECT_ROOT/ELP-Rumble-Detector' \
"

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    sys.exit('No GPUs detected inside container')
print('Detected GPUs:', gpus)
PY

TRAIN_MODULE="elp_rumble.training.train_${MODELTYPE}"
EPOCH_EXPORT=""
if [ -n "$EPOCHS" ]; then
  EPOCH_EXPORT="export EPOCHS='$EPOCHS' && \
"
fi

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  export MODEL='$MODEL' && \
  $EPOCH_EXPORT\
  python -m $TRAIN_MODULE \
"
