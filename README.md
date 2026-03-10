# ELP Rumble Detector

A CNN-based detector for African forest elephant rumble vocalizations, built for the [Elephant Listening Project](https://elephantlisteningproject.org/) at Cornell. Audio clips are converted to spectrograms and classified as rumble / non-rumble. Training runs locally or on the SDSC Expanse ACCESS GPU supercomputer.

> **Python baseline:** This repo targets **Python 3.11.9**, the version proven on Expanse for rumble model training. Dependencies are managed exclusively through `pyproject.toml` — there is no `requirements.txt`.

---

## Local Setup

### Prerequisites

- **macOS** (primary development platform; untested on Windows)
- [Homebrew](https://brew.sh)
- [pyenv](https://github.com/pyenv/pyenv) for managing Python versions
- Access to the Cornell ELP data

### Install Python 3.11.9

```bash
brew install pyenv
pyenv install 3.11.9
```

### Clone and configure

```bash
cd /path/to/your/ElephantListeningProject
git clone <repo https url>
cd ELP-Rumble-Detector
```

### Create the virtual environment and install

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure your local data path

```bash
cp .env.example .env
```

Edit `.env` and set `CORNELL_DATA_ROOT` to point to your local copy of the ELP Cornell Data folder. For example:

```
CORNELL_DATA_ROOT="/Users/username/ELP_Cornell_Data"
```

---

## Data Creation

> **Step 1 creates shared, version-controlled artifacts.**
> Do **NOT** rerun it unless the team agrees to change the dataset.
> In normal use, only run steps 2 and 3.

### Pipeline overview

```
create_data_plan.py  ──►  clips_plan.csv + splits/model{1,2,3}.csv  (committed)
cut_wav_clips.py     ──►  data/wav_clips/{pos,neg}/...               (local)
create_tfrecords.py  ──►  data/tfrecords/...                         (local)
```

### Steps

1. **Data plan** (committed — do not rerun casually)
   ```bash
   python -m elp_rumble.data_creation.create_data_plan
   ```

2. **Cut clips** (safe to run)
   ```bash
   python -m elp_rumble.data_creation.cut_wav_clips
   ```

3. **TFRecords** (safe to run)
   ```bash
   python -m elp_rumble.data_creation.create_tfrecords
   ```
  Output in one run for all three models (`model1`, `model2`, `model3`):
  `data/tfrecords/tfrecords_audio/{model}/` and `data/tfrecords/tfrecords_spectrogram/{model}/`

### Data plan policy (summary)

- **Sources:** Rumble PNNN and Dzanga folders only (`pnnn1`, `pnnn2`, `dzanga`).
- **Neg:pos ratio:** 3:1 in every split, enforced by per-split trimming.
- **Split assignment:** WAV-level grouping (80/10/10 train/val/test) prevents recording-condition leakage.
- **Model hierarchy:** `model1 ⊂ model2 ⊂ model3` — feasibility → scaled → full dataset.

---

## Model Training via SDSC Expanse

### Prerequisites

- Expanse project storage access (e.g. `/expanse/lustre/projects/cso100/`)
- Processed data tfrecords

### Initial setup on Expanse

Log in to Expanse via the [portal](https://portal.expanse.sdsc.edu/) or SSH, then clone the repo to project storage:

```bash
mkdir -p /expanse/lustre/projects/cso100/$USER/elp_container
cd /expanse/lustre/projects/cso100/$USER/elp_container
git clone <repo https url>
```

### Build the Singularity container (SLURM job — recommended)

Create `build_and_setup_container.slurm` in the `elp_container` directory:

```bash
#!/bin/bash
#SBATCH --job-name=elp_sandbox
#SBATCH --output=elp_sandbox.out
#SBATCH --error=elp_sandbox.err
#SBATCH --partition=gpu
#SBATCH --account=cso100
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=lustre

module load singularitypro

SCRATCH_DIR=/expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_$SLURM_JOB_ID
PROJECT_DIR=/expanse/lustre/projects/cso100/$USER/elp_container

mkdir -p $SCRATCH_DIR/tmp || { echo "Failed to create scratch tmp dir"; exit 1; }
export SINGULARITY_TMPDIR=$SCRATCH_DIR/tmp

rm -rf $SCRATCH_DIR/sandbox/ $SCRATCH_DIR/tmp/* $SCRATCH_DIR/build-temp-*/

singularity build --sandbox $SCRATCH_DIR/sandbox/ \
  /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif \
  || { echo "Singularity build failed"; exit 1; }

singularity exec --writable $SCRATCH_DIR/sandbox/ bash -c "\
  pip install --upgrade pip && \
  pip install -e /expanse/lustre/projects/cso100/$USER/elp_container/ELP-Rumble-Detector" \
  || { echo "pip install failed"; exit 1; }

rsync -av $SCRATCH_DIR/sandbox/ $PROJECT_DIR/sandbox/ \
  || { echo "rsync to project dir failed"; exit 1; }
```

Submit the job:

```bash
sbatch build_and_setup_container.slurm
```

### Verify the container environment

After the build job completes, confirm Python version, TensorFlow, and GPU visibility:

```bash
singularity exec --nv /expanse/lustre/projects/cso100/$USER/elp_container/sandbox/ \
python -c "
import sys, tensorflow as tf
print('Python', sys.version)
print('TensorFlow', tf.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
"
```

### Upload processed data tfrecords to Expanse

From your local machine:

```bash
rsync -avh --progress \
  "/path/to/local/ELP-Rumble-Detector/data/tfrecords" \
  your_username@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/your_username/elp_container/ELP-Rumble-Detector/data/tfrecords
```

### Running training jobs

SLURM scripts live in `slurm_scripts/`:

```bash
# Debug (quick path check, 20 min)
sbatch slurm_scripts/run-train-gpu-debug.sh

# Full training
sbatch slurm_scripts/run-train-gpu-shared.sh

# Convenience wrapper
bash slurm_scripts/submit-train.sh shared   # or: debug
```

### Monitor jobs

```bash
squeue -u $USER -l                          # queue status
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS   # job details
cat slurm_logs/<job_name>.o<job_id>.<node>  # output logs
scancel <job_id>                            # cancel
```

---

## Running Experiments (Local)

```bash
python -m elp_rumble.training.train_cnn
python -m elp_rumble.training.train_compare
```

With overrides:

```bash
python -m elp_rumble.training.train_cnn --epochs 10 --batch_size 32 --lr 1e-4
```

---

## Legacy

The `Legacy/` directory contains earlier training and utility scripts from the 2024–2025 CNN-vs-RNN research phase. These are **deprecated** and will be removed in a future cleanup. Use the `elp_rumble` package entrypoints above instead.

---

## Tools and Resources

- **[RavenPro / RavenLite](https://www.ravensoundsoftware.com/software/)** — view and annotate audio waveforms and spectrograms
- **[SDSC Expanse User Guide](https://www.sdsc.edu/systems/expanse/user_guide.html)**
- **[SDSC Basic Skills](https://github.com/sdsc-hpc-training-org/basic_skills)** — Linux, interactive computing, Jupyter on Expanse
- **[SDSC On-Demand Learning](https://www.sdsc.edu/education/on-demand-learning/index.html)** — webinars and educational archive

### Related research

- [2024–2025 ELP CNN vs RNN (SSIF 2025)](https://www.ecst.csuchico.edu/~sbsiewert/extra/research/elephant/SSIF-2025-ELP-Presentation.pdf)
- [Dr. Siewert's research group](https://sites.google.com/csuchico.edu/research/home)