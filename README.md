# Sparse-Gated Band Distillation for Single-Channel EEG Emotion Recognition

This repository contains the reference implementation for the paper **“Sparse-Gated Band Distillation for Single-Channel EEG Emotion Recognition in Neuroeducation.”** The code trains a multichannel EEG teacher, distills knowledge into a single-channel student through a deterministic sparse gate, and then performs deployment-consistent retraining on the finally selected channel. The implementation is designed around the **DEAP** dataset and supports both **valence** and **arousal** binary classification.

## 1. What this project does

The code implements a three-stage pipeline:

1. **Teacher pretraining**  
   A multichannel classifier is trained on full EEG windows.

2. **Sparse-gated distillation**  
   A global sparse gate compresses multichannel input into a single-channel surrogate input.  
   The student is trained with:
   - supervised classification loss,
   - temperature-scaled logit distillation,
   - optional band-consistency distillation over **theta / alpha / beta / gamma** bands,
   - optional one-hot regularization to progressively harden the gate.

3. **Deployment-consistent retraining**  
   After the gate converges, the selected channel is fixed and the student is retrained on the true single-channel signal from that electrode.

This matches the method described in the accompanying paper, including:
- windowed DEAP loading,
- separate models for **valence** and **arousal**,
- support for **holdout** and **LOOCV** evaluation,
- automatic metric logging, CSV/JSON export, gate history export, and figure generation.

## 2. Repository structure

The code is organized into five main modules plus the main entry script:

```text
.
├── main.py                         # Main training / evaluation entry point
├── datasets/
│   ├── __init__.py
│   └── deap_windows.py             # DEAP loader with sliding windows and grouping metadata
├── models/
│   ├── __init__.py
│   ├── classifier.py               # Wrapper around EEG backbone + classification head
│   ├── eegnet.py                   # Lightweight 1D EEG backbone
│   └── sparse_gate.py              # Sparsemax gate and deterministic channel compression
├── losses/
│   ├── __init__.py
│   ├── distillation.py             # CE + temperature-scaled KL distillation
│   ├── band_consistency.py         # Theta/alpha/beta/gamma consistency loss
│   └── regularizers.py             # One-hot encouraging regularizer for the gate
├── train/
│   ├── __init__.py
│   ├── train_teacher.py            # One-epoch teacher training
│   ├── train_distill.py            # One-epoch sparse-gated distillation
│   ├── retrain_single.py           # One-epoch retraining on fixed single channel
│   └── reporting.py                # Result storage, CSV/JSON export, gate logging, plotting hooks
└── utils/
    ├── __init__.py
    ├── deap_channels.py            # DEAP channel index → name mapping
    ├── io.py                       # Simple filesystem / CSV / JSON helpers
    ├── metrics.py                  # Binary metrics and confusion-matrix utilities
    └── plotting.py                 # Matplotlib plotting utilities
```

## 3. File-by-file summary

### `main.py`
The top-level runner. It:
- parses CLI arguments,
- builds the DEAP window dataset,
- creates holdout or leave-one-subject-out splits,
- trains the teacher,
- runs the requested baselines,
- aggregates metrics,
- writes summary files and figures.

Supported baselines:
- `proposed`: logits KD + band consistency + sparse gate + retraining
- `logits_only`: logits KD + sparse gate + retraining, without band-consistency loss
- `fixed`: fixed predefined single-channel baseline

### `datasets/deap_windows.py`
Implements `DEAPWindowDataset`.

Key behavior:
- accepts `.pt` or original DEAP `.dat`,
- loads all trials once and windows them lazily,
- uses a 4-second window and 2-second stride by default,
- binarizes **valence** and **arousal** with threshold `5.0`,
- keeps both `groups_trial` and `groups_subject` for leakage-safe splitting.

Each returned sample is:
- `x_win`: `(C, win_len)`
- `y_val`
- `y_aro`
- `subject`
- `trial`

### `models/eegnet.py`
A lightweight temporal CNN backbone with:
- Conv1d → BN → ReLU
- Conv1d → BN → ReLU
- Adaptive average pooling

Despite the filename, this is a compact EEG-style 1D backbone rather than a verbatim copy of any specific public EEGNet implementation.

### `models/classifier.py`
Wraps the backbone and adds the final linear classifier.

### `models/sparse_gate.py`
Implements:
- `sparsemax`
- `GlobalSparseGate`

The gate is global and sample-independent. It produces nonnegative weights summing to 1, and compresses multichannel input into a surrogate single-channel waveform by weighted summation across channels.

### `losses/distillation.py`
Implements:
- `kd_kl`: temperature-scaled KL divergence
- `supervised_ce`: supervised cross-entropy

### `losses/band_consistency.py`
Computes band-power features for:
- theta: 4–7 Hz
- alpha: 8–13 Hz
- beta: 14–30 Hz
- gamma: 31–45 Hz

It then aligns multichannel teacher band summaries with the student’s single-channel band features through MSE loss.

### `losses/regularizers.py`
Implements a one-hot encouraging regularizer:
- under simplex constraints, maximizing `||g||²` encourages one-hot selection,
- the implemented loss is `1 - ||g||²`.

### `train/train_teacher.py`
Runs one training epoch for the multichannel teacher.

### `train/train_distill.py`
Runs one distillation epoch for the student + sparse gate with the selected loss terms.

### `train/retrain_single.py`
Retrains a single-channel student using the actual selected electrode.

### `train/reporting.py`
Handles:
- summary row collection,
- CSV / JSON export,
- gate history export to `.npy` and `.csv`,
- chosen-channel summaries,
- downstream figure generation.

### `utils/*`
General helper code for:
- filesystem I/O,
- channel naming,
- metrics,
- plotting.

## 4. Dependencies

The code is written in **Python 3** and depends on:

- `python >= 3.9`
- `torch`
- `numpy`
- `matplotlib`

Standard-library modules used include:
- `argparse`
- `csv`
- `json`
- `os`
- `pickle`
- `random`
- `time`
- `collections`
- `dataclasses`
- `typing`

A minimal installation is:

```bash
pip install torch numpy matplotlib
```

If you want a pinned environment, you can start from:

```bash
pip install torch==2.2.0 numpy==1.26.4 matplotlib==3.8.4
```

Version pinning is optional; the code does not currently require a specific `requirements.txt` file.

## 5. Expected data format

Dataset download link: `https://www.kaggle.com/datasets/manh123df/deap-dataset`

The loader accepts either:

### Option A: preprocessed `.pt`
A `.pt` file containing something like:

```python
{
    "x": Tensor[trial, C, T],
    "labels": Tensor[trial, 4],   # or key "y"
    "fs": int
}
```

### Option B: original DEAP `.dat`
A file with keys:
- `data`
- `labels`

The code assumes:
- the first **32 channels** are EEG channels,
- valence is label index `0`,
- arousal is label index `1`,
- binary threshold is `5.0`.

By default, the script looks for data under:

```text
data/DEAP/processed_subsample
```

You can change this with `--data_dir`.

## 6. How to run

## 6.1 Quick sanity check

```bash
python main.py \
  --data_dir data/DEAP/processed_subsample \
  --out_dir runs_quick \
  --protocols holdout \
  --tasks valence \
  --baselines proposed \
  --preset quick
```

This is the fastest way to confirm the whole pipeline works.

## 6.2 Standard holdout run

```bash
python main.py \
  --data_dir data/DEAP/processed_subsample \
  --out_dir runs_holdout \
  --protocols holdout \
  --tasks valence,arousal \
  --baselines proposed,logits_only,fixed
```

## 6.3 LOOCV run

```bash
python main.py \
  --data_dir data/DEAP/processed_subsample \
  --out_dir runs_loocv \
  --protocols loocv \
  --tasks valence,arousal \
  --baselines proposed,logits_only,fixed
```

## 6.4 Faster LOOCV debugging with only a few folds

```bash
python main.py \
  --data_dir data/DEAP/processed_subsample \
  --out_dir runs_loocv_debug \
  --protocols loocv \
  --tasks arousal \
  --baselines proposed \
  --preset quick \
  --loocv_max_folds 3
```

## 6.5 Example with custom hyperparameters

```bash
python main.py \
  --data_dir data/DEAP/processed_subsample \
  --out_dir runs_custom \
  --protocols holdout \
  --tasks valence,arousal \
  --baselines proposed,logits_only,fixed \
  --teacher_epochs 20 \
  --distill_epochs 30 \
  --retrain_epochs 10 \
  --fixed_epochs 20 \
  --batch_size 64 \
  --test_batch_size 128 \
  --alpha_band 0.5 \
  --T 4.0 \
  --fixed_channel 0
```

## 7. Important CLI arguments

### Data and output
- `--data_dir`: path to DEAP `.pt` or `.dat` files
- `--out_dir`: output directory
- `--device`: `cuda`, `cpu`, or empty for auto-detect
- `--seed`: random seed

### Experiment settings
- `--protocols`: `holdout`, `loocv`, or both
- `--tasks`: `valence`, `arousal`, or both
- `--baselines`: `proposed`, `logits_only`, `fixed`

### Windowing
- `--win_sec`: window length in seconds
- `--step_sec`: stride in seconds
- `--dataset_zscore`: whether to z-score inside the dataset loader

### Training
- `--batch_size`
- `--test_batch_size`
- `--teacher_epochs`
- `--distill_epochs`
- `--retrain_epochs`
- `--fixed_epochs`

### Distillation
- `--T`
- `--alpha_kd`
- `--alpha_band`
- `--lambda_onehot_start`
- `--lambda_onehot_end`
- `--onehot_warmup_epochs`

### Speed / debugging
- `--preset quick`
- `--loocv_max_folds`

## 8. Output files

After a run, the code writes artifacts under `out_dir`, typically including:

```text
out_dir/
├── figures/
│   ├── channel_freq_*.png
│   ├── gate_heatmap_*.png
│   ├── gate_prob_*.png
│   ├── metric_*.png
│   ├── confusion_*.png
│   └── curve_*.png
└── results/
    ├── summary.csv
    ├── summary.json
    ├── chosen_channels.csv
    ├── curves/
    │   └── curve_*.csv
    └── gate/
        ├── gate_hist_*.npy
        ├── gate_hist_*.csv
        ├── gate_final_*.npy
        └── gate_final_*.csv
```

### `summary.csv`
Contains per-fold / per-split metrics such as:
- `acc`
- `bacc`
- `f1`
- `precision`
- `recall`
- `tnr`
- confusion counts
- selected channel
- trial/window counts

### Gate exports
- `gate_hist_*`: gate probabilities across epochs
- `gate_final_*`: final gate distribution per fold

These are useful for reproducing the plots and verifying hardening behavior.

## 9. Reproducibility notes

The code sets seeds for:
- Python `random`
- NumPy
- PyTorch CPU / CUDA

Still, exact reproducibility can vary across:
- GPU type,
- CUDA / cuDNN version,
- PyTorch version,
- dataloader behavior.

For the most stable comparisons:
- keep the same seed,
- keep the same DEAP preprocessing,
- keep the same split protocol,
- report holdout and LOOCV separately.

## 10. Code provenance and authorship statement

**All source code in this repository is original project code.**  
According to the project author’s statement, the implementation was written specifically for this work and **was not copied from external repositories**.

More specifically:

### Written for this project
The following parts are original project implementations for this repository:

- `main.py`
- `datasets/deap_windows.py`
- `models/classifier.py`
- `models/eegnet.py`
- `models/sparse_gate.py`
- `losses/distillation.py`
- `losses/band_consistency.py`
- `losses/regularizers.py`
- `train/train_teacher.py`
- `train/train_distill.py`
- `train/retrain_single.py`
- `train/reporting.py`
- `utils/deap_channels.py`
- `utils/io.py`
- `utils/metrics.py`
- `utils/plotting.py`

### Adapted from prior code
**None.**  
This repository does not contain files declared as adapted from prior internal codebases or earlier public implementations.

### Copied from external repositories
**None.**  
No repository files are declared as copied from GitHub, papers’ supplementary code, or other external codebases.

All source code in this repository is original.
No code was copied from or adapted from external repositories.

## 11. Suggested citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{yourpaper2026_sparsegated,
  title={Sparse-Gated Band Distillation for Single-Channel EEG Emotion Recognition in Neuroeducation},
  author={Author(s)},
  journal={Under review},
  year={2026}
}
```

## 12. Known assumptions and limitations

- The code is tailored to DEAP-style data organization.
- Labels are treated as **binary** with threshold `5.0`.
- Evaluation is based on **trial-level aggregation of window predictions**.
- The model is lightweight and intentionally simple; it is not a large-capacity architecture.
- `models/eegnet.py` is a compact temporal CNN, not a claim of exact reproduction of any official EEGNet reference implementation.
- The paper reports holdout results prominently; LOOCV is implemented in code and can be run from the CLI.

## 13. Practical checklist before running

1. Prepare DEAP `.pt` or `.dat` files.
2. Put them under a folder such as `data/DEAP/processed_subsample`.
3. Install PyTorch, NumPy, and Matplotlib.
4. Start with a `--preset quick` run.
5. Check:
   - `results/summary.csv`
   - `results/summary.json`
   - generated figures
6. Then launch the full holdout or LOOCV experiments.

---

For questions about the method, use the paper together with `main.py`, `train/train_distill.py`, `losses/band_consistency.py`, and `models/sparse_gate.py` as the primary entry points.
