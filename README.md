# SDSC Setup (Automated Version)

If you are using the SDSC Expanse/ACCESS system and need to build a Singularity container (e.g., with additional packages like Ray Tune), you can now automate the full process via a SLURM job to convert a prexisting container into a singularity sandbox and add the needed packages. Doing this as a SLURM job as outlined below avoids login node timeouts, handles disk quotas correctly, and ensures everything is saved in project storage. Once the container is built and configured, scripts can be launched using that container.

## Prerequisites

- You must be a member of a project with access to Expanse project storage (e.g., `/expanse/lustre/projects/cso100/`).
- Your Python dependencies should be listed in a `requirements.txt` file which should be available in the project repo.
- You must have access to the Cornell ELP data

## Step-by-Step Instructions

First, login and run the NSF ACCESS expanse shell either via the online user portal: https://portal.expanse.sdsc.edu/ or from your terminal's command line via ssh. See documentation for details: https://www.sdsc.edu/systems/expanse/user_guide.html

Next, make a directory for the container in project storage and clone the github repo (or your fork of the repo) to project storage.
```bash
mkdir /expanse/lustre/projects/cso100/$USER/elp_container
cd /expanse/lustre/projects/cso100/$USER/elp_container
git clone <repo web url clone link>
```

### 1. Create the SLURM Script - Scratch Drive Option (recommended method)

📝 **Scratch Drive Option:** If you prefer to use scratch space for faster builds and larger temporary capacity (up to 10 TB), the script now supports using the scratch drive `/expanse/scratch/$USER` for the build process. It will copy the final container to project storage after building.

Ensure you are in the directory `/expanse/lustre/projects/cso100/$USER/elp_container/`

Create a job script called `build_and_setup_container.slurm`:
```bash
nano build_and_setup_container.slurm
```

Paste in the following:
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

# Load Singularity module
module load singularitypro

# Set working paths
SCRATCH_DIR=/expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_$SLURM_JOB_ID
PROJECT_DIR=/expanse/lustre/projects/cso100/$USER/elp_container

echo "Hostname: $(hostname)"
echo "Running on: $(pwd)"
echo "Scratch build directory: $SCRATCH_DIR"
echo "Project directory: $PROJECT_DIR"

# Create scratch build directory
mkdir -p $SCRATCH_DIR/tmp || { echo "❌ Failed to create scratch tmp dir"; exit 1; }
export SINGULARITY_TMPDIR=$SCRATCH_DIR/tmp

# Cleanup old builds if needed
rm -rf $SCRATCH_DIR/sandbox/
rm -rf $SCRATCH_DIR/tmp/*
rm -rf $SCRATCH_DIR/build-temp-*/

# Build the container in scratch
singularity build --sandbox $SCRATCH_DIR/sandbox/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif || { echo "❌ Singularity build failed"; exit 1; }

# Copy requirements file - Update your repo name accordingly
cp /expanse/lustre/projects/cso100/$USER/ELP-CNNvsRNN-v2/requirements.txt $SCRATCH_DIR/requirements.txt || { echo "❌ Could not copy requirements.txt"; exit 1; }

# Install dependencies inside container
singularity exec --writable $SCRATCH_DIR/sandbox/ bash -c "\
  pip install --upgrade pip && \
  pip install -r /requirements.txt && \
  pip install -e /expanse/lustre/projects/cso100/$USER/ELP-CNNvsRNN-v2 && \
  rm /requirements.txt" || { echo "❌ pip install failed"; exit 1; }

# Copy completed sandbox to project storage
rsync -av $SCRATCH_DIR/sandbox/ $PROJECT_DIR/sandbox/ || { echo "❌ rsync to project dir failed"; exit 1; }
```

### 2. Submit the Job to Build and Setup the Container

```bash
sbatch build_and_setup_container.slurm
```

### 📊 Monitor Your Job

To check the queue status of your job:
```bash
squeue -u $USER
```

To see details about the most recent job submission:
```bash
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,AllocCPUs
```

To follow logs in real time:
```bash
tail -f elp_sandbox.out
```

Or to view errors:
```bash
tail -f elp_sandbox.err
```

To stop following:
```bash
Ctrl+C
```

If you prefer to just view a snapshot of these files, use
```bash
cat elp_sandbox.out
cat elp_sandbox.err
```

To cancel a running job:
```bash
scancel <job_id>
```

### 🔍 Live Monitoring with htop

To check how much space your scratch build is using:
```bash
du -sh /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_*
```
This will give you a summary of total disk space used by each job-specific build folder.

For a deeper look at what's inside:
```bash
du -h --max-depth=1 /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_<jobid>
```

You can also watch space usage live:
```bash
watch -n 2 'du -sh /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_*'
```

If your job is running and you want to monitor live system resource usage:

1. Get the node where your job is running:
```bash
squeue -u $USER
```

2. SSH into the node:
```bash
ssh <nodelist>
```
(e.g., `ssh exp-9-57`)

3. Launch htop:
```bash
htop
```
Use `F6` to sort, and `F10` to exit htop view.

To exit the node and return to your main SDSC session, type:
```bash
exit
```

### 3. Use the Container

Once built, you can use the container with:

```bash
singularity exec --writable /expanse/lustre/projects/cso100/$USER/elp_container/sandbox/ bash
```

Now you're inside the container and ready to run training or experiments.

---

## Legacy Manual Build (Not Recommended)

If you prefer to use the login node to build the container (not recommended):
```bash
module load singularitypro
singularity build --sandbox sandbox_container/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif
singularity exec --writable sandbox_container/ bash
pip install -r requirements.txt
exit
```

⚠️ **This method is not recommended** as it may get killed due to I/O, time, or memory limits.

---

### 🧼 Optional: Clean Up Old Scratch Builds With Script: `cleanup_scratch_builds.sh

To remove old container build folders in your scratch space (e.g., from previous failed or old SLURM jobs), you can run a script that automatically skips active jobs and logs deleted folders.

To do so, ensure you are in the directory `/expanse/lustre/projects/cso100/$USER/elp_container/`

Create a job script called `cleanup_scratch_builds.sh`:
```bash
nano cleanup_scratch_builds.sh
```

Paste in the following and save the script:
```bash
#!/bin/bash

SCRATCH_BASE=/expanse/lustre/scratch/$USER/temp_project
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$SCRIPT_DIR/scratch_cleanup_$(date +%Y%m%d_%H%M%S).log"

echo "🧹 Cleaning up scratch containers in $SCRATCH_BASE" | tee -a $LOGFILE
mkdir -p empty_temp_dir

for dir in "$SCRATCH_BASE"/elp_sandbox_tmp_*; do
  if [ -d "$dir" ]; then
    JOB_ID=$(basename "$dir" | awk -F_ '{print $NF}')
    if squeue -j "$JOB_ID" -u $USER | grep -q "$JOB_ID"; then
      echo "🔒 Skipping active job dir: $dir" | tee -a $LOGFILE
      continue
    fi
    echo "→ Processing $dir" | tee -a $LOGFILE
    rsync -a --delete --omit-dir-times empty_temp_dir/ "$dir/" 2>> $LOGFILE
    chmod -R u+w "$dir" 2>> $LOGFILE
    rm -rf "$dir"
    echo "✅ Deleted $dir" | tee -a $LOGFILE
  fi
done

rm -rf empty_temp_dir

# Keep only most recent 3 cleanup logs
cd "$SCRIPT_DIR"
ls -tp scratch_cleanup_*.log | grep -v '/$' | tail -n +4 | xargs -r rm --

echo "✅ Scratch cleanup complete at $(date)" | tee -a $LOGFILE
```

Make it executable:
```bash
chmod +x cleanup_scratch_builds.sh
```

Run it in the background:
```bash
nohup ./cleanup_scratch_builds.sh > cleanup_scratch.log 2>&1 &
```

Check on progress:
```bash
tail -f cleanup_scratch.log
```

Stop monitoring:
```bash
Ctrl+C
```

View the logs:
```bash
less cleanup_scratch_*.log
```

---

# General Setup (Local Machine)

This project requires Python 3.11. These instructions will guide you through setting up a local development environment on macOS.

**Install Prerequisites (Homebrew & pyenv)**

If you don't already have them, install Homebrew (the macOS package manager) and then use it to install `pyenv` for managing Python versions.
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv
brew install pyenv
```

**Install Python 3.11**
Use `pyenv` to install the specific Python version for this project. This may take a few minutes.
```bash
pyenv install 3.11.9
```

**Configure the Project Environment**
Navigate to the root folder which will hold your ELP-CNNvsRNN-v2 cloned repository and your venv
```bash
cd /path/to/your/ElephantListeningProject
```

Clone this repo from github
```bash
git clone <insert repo's https link>
```

Create a venv with python 3.11, activate it, cd into repo, and install dependencies.
```bash
~/.pyenv/versions/3.11.9/bin/python -m venv elp-venv
source elp-venv/bin/activate
cd ELP-CNNvsRNN-v2
pip install -r requirements.txt
pip install -e .
```

Create your personal .env configuration file from the example
```bash
cp .env.example .env
```

Edit the .env file with your local data path:
Open the .env file in a text editor and change the path to point
to where you stored the ELP Cornell Data folder. For example:
CORNELL_DATA_ROOT="/Users/username/ELP_Cornell_Data"
You can either use nano or the editor of your choosing.
```bash
nano .env
```

---

# Data Preprocessing (Suggested to do on your local machine)
The `elp_rumble.data_creation` package now follows the same staged structure as the gunshot repo:

1. `create_clips_plan.py`: builds one source-of-truth clip plan CSV.
2. `cut_wav_clips.py`: cuts both positive and negative 5s clips from raw WAVs.
3. `create_splits.py`: builds split CSV(s) from the clip plan.
4. `create_tfrecords.py`: converts split-defined clips directly into spectrogram TFRecords.

These scripts require access to the Cornell raw data paths configured in `.env`.

## Rumble Data Split Policy (Current)

This project uses a positive-driven split policy:

- Seen positives: PNNN-derived rumble clips are split into `train/validate` at `80/20`.
- Holdout positives: all Dzanga rumble clips are assigned to `test`.
- Seen negatives: sampled to match seen positive totals, then split to match `train/validate` positive counts.
- Holdout negatives: sampled to match Dzanga positive count and assigned to `test`.
- Result: `test` can be larger than `train` by design because Dzanga holdout is intentionally kept intact.

### Split Controls

- `TRAIN_FRAC`: fraction for seen positive training split (default `0.8`).
- `SPLIT_SEED`: random seed for reproducible split sampling (default `42`).

### Split Outputs

- `model1.csv`: feasibility set with exactly `60` positive + `60` negative clips.
  : Dzanga positives are only used in `test` for holdout semantics.
- `model3.csv`: full production split using all available clips under current policy.

### Pipeline Order

Run from repo root with the project virtual environment active:

```bash
python -m elp_rumble.data_creation.create_clips_plan
python -m elp_rumble.data_creation.cut_wav_clips
python -m elp_rumble.data_creation.create_splits
MODEL=model1 python -m elp_rumble.data_creation.create_tfrecords
MODEL=model3 python -m elp_rumble.data_creation.create_tfrecords
```

### Verification Checklist

After generation, verify counts/sizes:

```bash
find data/wav_clips -type f -name '*.wav' | wc -l
ls -lh src/elp_rumble/data_creation/splits/model3.csv
ls -lh src/elp_rumble/data_creation/splits/model1.csv
ls -lh data/tfrecords/tfrecords_spectrogram/*.tfrecord
ls -lh data/tfrecords/tfrecords_spectrogram/model1/*.tfrecord
```

Split artifacts are written to:

- `src/elp_rumble/data_creation/clips_plan.csv`
- `src/elp_rumble/data_creation/splits/model1.csv`
- `src/elp_rumble/data_creation/splits/model3.csv`
- `data/tfrecords/tfrecords_spectrogram/clip_splits.csv`
- `data/tfrecords/tfrecords_spectrogram/model1/clip_splits.csv`

---

## After Preprocessing Data Locally, Upload to Expanse Project Storage

To upload your local preprocessed tfrecords data to SDSC Expanse project storage, use the `rsync` command from your local terminal:
```bash
rsync -avh --progress \
"/path/to/local/project/ELP-CNNvsRNN-v2/data" \
your_username@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/your_username/elp_container/ELP-CNNvsRNN-v2/data

rsync -avh --progress \
"/path/to/local/project/repo/ELP-CNNvsRNN-v2/data/tfrecords_spectrogram" \
your_username@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/your_username/elp_container/ELP-CNNvsRNN-v2/data/
```

Replace `"/path/to/local/project/repo/"` with the full path to your local git project repo and replace `your_username` with your ACCESS Expanse username.

Note: This may take a while!

If interrupted, you can re-run the same command to resume.

You can check storage usage (snapshot) on Expanse with:
```bash
du -sh /expanse/lustre/projects/cso100/$USER/elp_container/ELP-CNNvsRNN-v2/data/
```

Or to monitor it continuously as it grows:
```bash
watch -n 5 'du -sh /expanse/lustre/projects/cso100/$USER/elp_container/ELP-CNNvsRNN-v2/data/'
```

---

# Running Experiments

### Local Terminal (current package entrypoints):

```bash
python3 -m elp_rumble.training.train_compare
python3 -m elp_rumble.training.train_cnn
```

Optional train command with overrides:
```bash
python3 -m elp_rumble.training.train_cnn --epochs 10 --batch_size 32 --lr 1e-4
```

### Legacy scripts (deprecated, kept under `Legacy/`):

```bash
python3 Legacy/cross_validation_experiment.py cnn  # or rnn
```

### SLURM Batch Job:

SLURM scripts now live in `slurm_scripts/`.

For training debugging, making sure paths are correct, etc.:
```bash
sbatch slurm_scripts/run-train-gpu-debug.sh
```

For full training:
```bash
sbatch slurm_scripts/run-train-gpu-shared.sh
```

Convenience submitters:
```bash
bash slurm_scripts/submit-train.sh shared
```

#### Monitor your job:

The job id, job name, status, node, and other info about the job can be found via:
```
squeue -u $USER -l
``` 

SSH into node to check GPU:
```bash
ssh <node>
nvtop
```

#### Check output logs:
```bash
ls -lh train.o*
```

Note: Replace train with whatever the job name is, which can be found in the script or with the squeue command above. For example, for a script, the job-name is assigned here:
```
#SBATCH --job-name=train-debug
```
Therefore, use:
```bash
ls -lh train-debug.o*
```

To see the logs:
```bash
cat <name of file>
```
Ex:
```bash
cat train.o41166992.exp-14-58
```

---

## View Results

```bash
python3 Legacy/view_cross_validation_results.py
vim Legacy/train.py  # edit best config
```

---

## Train Final Model

```bash
python3 -m elp_rumble.training.train_cnn
```

Or submit with:

```bash
sbatch slurm_scripts/run-train-gpu-shared.sh
```

# Other Tools/Resources

## RavenPro (or RavenLite - free) 
- Can be used to view and annotate audio waveforms and spectrograms

https://www.ravensoundsoftware.com/software/
https://www.ravensoundsoftware.com/knowledge-base/

## San Diego Supercomputer Center

#### SDSC User Guide
https://www.sdsc.edu/systems/expanse/user_guide.html#narrow-wysiwyg-7

#### SDSC Basic Skills
- Includes basic Linux skills, interactive computing, running Jupyter notebooks on Expanse, and info on using git/Github

https://github.com/sdsc-hpc-training-org/basic_skills

#### Intermediate Linux Workshop Slides
- Useful for navigating ACCESS Expanse server. Slides are from July 2025. Webinar video not yet available (as of July 2025) however they will soon be uploaded to SDSC On-Demand Learning archive. Previous Intermediate Linux webinars can be found there as well.

https://drive.google.com/file/d/1t8WwPcnAsieVc-3jisJiQyw6jFBv4Hmb/view?usp=sharing


#### SDSC On-Demand Learning
- Archive of webinars, educational videos, github repos and other educational resources related to the SDSC.

https://www.sdsc.edu/education/on-demand-learning/index.html


## Previous and current ELP Research, as well as related research with Dr. Siewert

#### 2024-2025 ELP CNN vs RNN research from which this repo builds upon:
https://www.ecst.csuchico.edu/~sbsiewert/extra/research/elephant/SSIF-2025-ELP-Presentation.pdf

#### Other research:
https://sites.google.com/csuchico.edu/research/home

 