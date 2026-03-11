# ELP Rumble Detector

A CNN-based detector for African forest elephant rumble vocalizations, built for the [Elephant Listening Project](https://elephantlisteningproject.org/) at Cornell. Audio clips are converted to spectrograms and classified as rumble / non-rumble. Training runs locally or on the SDSC Expanse ACCESS GPU supercomputer.

> **Python baseline:** This repo targets **Python 3.10**, the version proven on Expanse for rumble model training. Dependencies are managed exclusively through `pyproject.toml` — there is no `requirements.txt`.

---

## Local Setup

### Prerequisites

- **macOS** (primary development platform; untested on Windows)
- [Homebrew](https://brew.sh)
- [pyenv](https://github.com/pyenv/pyenv) for managing Python versions
- Access to the Cornell ELP data

### Install Python 3.10 (local dev)

```bash
brew install pyenv
pyenv install 3.10.13
```

### Clone and configure

```bash
cd /path/to/your/ElephantListeningProject
git clone <repo https url>
cd ELP-Rumble-Detector
```

### Create the virtual environment and install

```bash
~/.pyenv/versions/3.10.13/bin/python -m venv .venv
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
- Processed data tfrecords uploaded to the project tree

### Project layout on Expanse

Clone the repo under the shared project root so it co-resides with other Elephant Listening Project material:

```
/expanse/lustre/projects/cso100/<your_username>/ElephantListeningProject/
   └── ELP-Rumble-Detector/
```

### Upload processed data tfrecords

Use [Globus Connect Personal](https://docs.globus.org/globus-connect-personal/) with your SDSC Expanse credentials to transfer files between your local machine and the remote server. [Tutorial](https://docs.globus.org/guides/tutorials/manage-files/transfer-files/). Note: In the [Globus file manager tab](https://app.globus.org/file-manager), search for the collection `SDSC HPC - Expanse Lustre`, then either append path to direct it to your project storage or navigate to your project storage via the UI.

To ensure consistency between local and remote environments, use the same relative data folder structure on both systems.
```
ELP-Rumble-Detector/
├── data/
│   ├── tfrecords/
│   │   ├── tfrecords_audio/
│   │   │   ├── model1/
│   │   │   ├── model2/
│   │   │   └── model3/
│   │   └── tfrecords_spectrogram/
│   │       ├── model1/
│   │       ├── model2/
│   │       └── model3/
├── slurm_scripts/
├── src/
└── ...
```

### Remote training workflow (Singularity + GPU validation)

Both `run-train-gpu-shared.sh` and `run-train-gpu-debug.sh` live in `slurm_scripts/`. They perform the following actions automatically:

1. Load `singularitypro` and use `/cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif`
2. Install the repo into `/expanse/lustre/projects/cso100/$USER/ElephantListeningProject/.pythonuserbase` (editable install)
3. Run a TensorFlow GPU detection check inside the container and fail fast if no GPUs are visible
4. Invoke either `elp_rumble.training.train_cnn` or `train_rnn` with your requested split and optional epoch override

Invoke the scripts with `MODELTYPE` (`cnn` or `rnn`) and `MODEL` (`model1`, `model2`, or `model3`). An optional `EPOCHS` value overrides the default epoch count. Example invocations:

```bash
# Run the shared job with the CNN trainer on model3 (default epochs)
sbatch slurm_scripts/run-train-gpu-shared.sh cnn model3

# Run the debug job on model1 with 2 epochs for a quick sanity check
sbatch slurm_scripts/run-train-gpu-debug.sh cnn model1 2

# Run the RNN trainer on the shared partition
sbatch slurm_scripts/run-train-gpu-shared.sh rnn model2
```

Each script prints the usage string and exits if `MODELTYPE` or `MODEL` are missing/invalid. The GPU check occurs before training so you won’t waste time running if the container cannot see any devices.

### Monitor jobs

```bash
squeue -u $USER -l                          # queue status
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS   # job details
cat slurm_logs/<job_name>.o<job_id>.<node>  # output logs
scancel <job_id>                            # cancel
```

---

## Training (Local)

Select a model split via the `MODEL` environment variable (`model1`, `model2`, `model3`). Results are saved under `runs/{cnn,rnn}/<run_name>/` depending on the trainer.

### CNN

```bash
MODEL=model3 python -m elp_rumble.training.train_cnn
```

Override epoch count (useful for quick smoke-tests):

```bash
MODEL=model1 EPOCHS=2 python -m elp_rumble.training.train_cnn
```

### RNN

```bash
MODEL=model3 python -m elp_rumble.training.train_rnn
```

### Artifacts saved per run

Each completed run produces a directory `runs/{cnn,rnn}/{MODEL}_bs{BS}_lr{LR}_e{EPOCHS}_{TIMESTAMP}/` containing:

| File | Description |
|------|-------------|
| `params.json` | All hyperparameters, TFRecord paths, class weights |
| `history.csv` | Per-epoch loss, accuracy, precision, recall, AUC (train + val) |
| `best_model.keras` | Best checkpoint (monitored by val AUC) |
| `final_model.keras` | Last-epoch model |
| `test_metrics.json` | Test-set accuracy, precision, recall, AUC, confusion matrix |
| `test_predictions.csv` | Per-clip: `clip_wav_relpath`, `y_true`, `y_pred`, `y_score` |
| `logs/` | TensorBoard event files |

---

## Evaluation

Generate publication-quality figures from a completed run (no model reload needed):

```bash
python -m elp_rumble.evaluate_cnn --run_dir runs/cnn/<run_name>
```

Figures are saved to `results/figures/` as both PDF (for LaTeX) and PNG at 300 DPI:

- `training_curves.{pdf,png}` — loss + AUC vs. epoch
- `confusion_matrix.{pdf,png}` — counts and percentages
- `roc_curve.{pdf,png}` — ROC with AUC annotated
- `pr_curve.{pdf,png}` — precision-recall with AP annotated

A display-only notebook at `notebooks/cnn_results.ipynb` renders these figures alongside a metrics summary table.

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