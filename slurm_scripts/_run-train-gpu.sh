#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF >&2
Usage: $0 <cnn|rnn> <model1|model2|model3> [epochs]
  cnn|rnn         selects the trainer module (train_cnn / train_rnn)
  model1..model3  picks a dataset split definition
  epochs          optional positive integer epoch count override
EOF
  exit 1
}

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  usage
fi

MODELTYPE="${1,,}"
MODEL="${2,,}"
EPOCHS="${3:-}"

if [[ "$MODELTYPE" != "cnn" && "$MODELTYPE" != "rnn" ]]; then
  echo "MODELTYPE must be 'cnn' or 'rnn'" >&2
  usage
fi

if [[ "$MODEL" != "model1" && "$MODEL" != "model2" && "$MODEL" != "model3" ]]; then
  echo "MODEL must be one of model1, model2, model3" >&2
  usage
fi

if [[ -n "$EPOCHS" && ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  usage
fi

module purge
module load singularitypro/3.11
module list

export PROJECT_ROOT="/expanse/lustre/projects/cso100/$USER/ElephantListeningProject"
export SIF="$PROJECT_ROOT/tensorflow-2.15.0-gpu.sif"
export PYTHONUSERBASE="$PROJECT_ROOT/.pythonuserbase"
export NVIDIA_DISABLE_REQUIRE=true

SLURM_LOG_DIR="$PROJECT_ROOT/slurm_logs"
if [[ ! -d "$SLURM_LOG_DIR" ]]; then
  echo "ERROR: $SLURM_LOG_DIR does not exist." >&2
  echo "Create it before submitting the job:" >&2
  echo "  mkdir -p $SLURM_LOG_DIR" >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: Container not found: $SIF" >&2
  exit 1
fi

# Install the editable repo into the persistent user base.
singularity exec --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  python -m pip install --upgrade --user pip setuptools wheel && \
  python -m pip install --user -e '$PROJECT_ROOT/ELP-Rumble-Detector'
"

GPU_CHECK_SCRIPT="python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    sys.exit('No GPUs detected inside container')
print('Detected GPUs:', gpus)
PY"

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "$GPU_CHECK_SCRIPT"

TRAIN_MODULE="elp_rumble.training.train_${MODELTYPE}"
EPOCH_EXPORT=""
[ -n "$EPOCHS" ] && EPOCH_EXPORT="export EPOCHS='$EPOCHS' && \
"

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  export MODEL='$MODEL' && \
  ${EPOCH_EXPORT} \
  python -m $TRAIN_MODULE \
"