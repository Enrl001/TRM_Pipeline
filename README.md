# Pain EEG pipeline — Part 1: data preprocessing (shared)

This is the shared data-preparation stage. Both models (the Modares-Haghighi
graph-metric + SVM baseline, and the GAT) will read its output, so they are
compared on identical data.

## Files

| File | What it does |
|---|---|
| `paineeg/config.py` | Every setting in one place (bands, filters, ICA, windows, MVAR order, threshold density) |
| `paineeg/preprocess.py` | Load → pick channels → montage → resample → notch → 0.5–45 Hz band-pass → bad channels → average reference → ICA + ICLabel → interpolate → 1 s windows, 50 % overlap, ±100 µV rejection |
| `paineeg/connectivity.py` | Per-window **PDC** (MVAR, BIC order — as in Modares-Haghighi), per-window **wPLI**, and log band power |
| `paineeg/data.py` | Loads the saved per-subject files for the models |
| `01_build_dataset.py` | Runs everything over a manifest of subjects and saves one `.npz` per subject |
| `make_synthetic.py` | Makes fake test EEG with a planted theta-coupling effect (for testing only) |

## Install

```bash
pip install -r requirements.txt
```

### Google Colab

`colab_pipeline.ipynb` contains all the code (one cell per module). Upload only
the `.gdf` files to `MyDrive/TRM/generic dataset`, open the notebook in Colab
and run it top to bottom (it writes `manifest.csv` there if missing). Features
and results are written to `MyDrive/TRM/features/` and `MyDrive/TRM/results/`. If you edit code in the notebook, copy the
change back here (or the other way round) so the two stay the same.

## Run

1. Make a `manifest.csv` next to your data:

```csv
subject,path,label
sub-01,sub-01.edf,0
sub-02,sub-02.edf,1
```

2. Build the dataset:

```bash
python 01_build_dataset.py --manifest data/mydata/manifest.csv --out features/mydata \
    --conn both --channels Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 \
    --line-freq 50
```

Useful flags: `--crop 300,600` (e.g. keep only the eyes-closed segment),
`--line-freq 60` (data recorded in the USA/Mexico), `--no-ica` (quick tests),
`--snr 20` (add white noise — the paper's robustness test; use a separate `--out`).

3. Test on synthetic data first:

```bash
python make_synthetic.py --n-per-group 6
python 01_build_dataset.py --manifest data/synthetic/manifest.csv --out features/synthetic --conn both
```

## Output (one `.npz` per subject)

| Key | Shape | Meaning |
|---|---|---|
| `conn_pdc` | (windows, bands, C, C) | PDC, `W[source, target]`, directed |
| `conn_wpli` | (windows, bands, C, C) | wPLI, symmetric |
| `bandpower` | (windows, bands, C) | log10 band power (GAT node features / band-power baseline) |
| `label`, `subject`, `ch_names`, `bands`, `sfreq`, `t_start`, `mvar_order` | | metadata |

Plus `preprocessing_log.csv` (bad channels, ICA components removed, windows kept)
and `config_used.json` (record these in your methods section).

## Where this differs from Modares-Haghighi et al. (2021) — say so in your methods

| Paper | Here | Why |
|---|---|---|
| SOBI + fractal dimension / Fp1-spectrum artifact removal | ICA (extended infomax) + ICLabel | MNE has no SOBI; ICLabel removes the same artifact classes automatically and reproducibly |
| Alpha band only | Theta + alpha (edit `bands` to add more) | Theta is central to dysmenorrhea (Lee et al., 2017) |
| MVAR fitted to alpha-filtered signal | MVAR fitted to the broadband window, PDC averaged over each band | Fitting an MVAR to a narrow-band signal is ill-conditioned |
| 100 surrogate realisations to keep significant PDC links | Not included; proportional thresholding is used in the model step | Too slow for per-window graphs; can be added later |
| Reference at FPz | Average reference | Standard for connectivity and required by ICLabel |
| — | Bad-channel detection + interpolation, resampling to 250 Hz | Robustness on clinical data |

## Synthetic test result

All 18 synthetic subjects processed (119 windows each; ICLabel removed the
planted eye blinks). Theta wPLI between Fz and C3 rose with the planted effect
(control 0.40 → mild 0.54 → severe 0.74), so the step recovers a known
connectivity difference. PDC on 1 s windows changed much less (0.214 → 0.226),
because estimating a 19-channel MVAR from 250 samples is noisy — expect wPLI to
be the more reliable measure, and report both.

---

# Part 2: Modares-Haghighi et al. (2021) baseline

Reproduction of *Quantification of pain severity using EEG-based functional
connectivity* (Biomed. Signal Process. Control 69:102840): graph metrics of the
per-window connectivity graphs → SFFS feature selection → RBF SVM. It reads the
`.npz` files from Part 1, so it sees the same data as the GAT will.

## Files

| File | What it does |
|---|---|
| `paineeg/graph_metrics.py` | Graph filtering (9 cumulative + 7 windowed versions) and the five bctpy metrics per window and band. Reusable by the GAT |
| `02_baseline_modares.py` | Feature selection, SVM / hierarchical SVM tree, subject-wise validation, permutation test, noise-robustness table |

## Run

```bash
pip install -r requirements.txt     # needs scikit-learn, mlxtend, bctpy, matplotlib

# binary (label column stored in the .npz files)
python 02_baseline_modares.py --features features/mydata --conn wpli
python 02_baseline_modares.py --features features/mydata --conn pdc

# multi-class: labels come from the manifest; tree peels off 0, then 2, then 1 is left
python 02_baseline_modares.py --features features/mydata --conn wpli \
    --manifest data/mydata/manifest.csv --label-col label_3class --tree-order 0,2,1 \
    --out results/baseline_wpli_3class
```

Results go to `--out` (default `results/baseline_<conn>/`).

| Flag | Meaning |
|---|---|
| `--density 0.9` | Cumulative threshold: keep the strongest 90 % of edges (the paper's best) |
| `--sweep` | Run all 16 graph versions, write `sweep.csv`, then do the full analysis on the best one. The best version is chosen on the same CV, so its score is optimistic: report it as exploratory |
| `--loso` | Leave-one-subject-out instead of stratified 5-fold GroupKFold |
| `--paper-split` | Hold out ~25 % of subjects as a test set (paper: 17 train / 6 test), tune with 5-fold CV on the rest |
| `--n-perm 100` | Permutation test: subject labels shuffled, whole pipeline rerun. p-values for pooled subject- and window-level balanced accuracy, and subject AUC (binary) |
| `--max-features 20` | Cap on the number of SFFS features |
| `--fine-grid` | C from 0.1 to 10 in 0.05 steps (paper) instead of `0.1, 0.3, 1, 3, 10` |
| `--prefilter 30` | Before SFFS, keep the 30 best features by ANOVA F, computed inside the training fold (0 = off) |
| `--sel-windows 20` | Windows per subject used by SFFS and C tuning (the final model is trained on all windows; 0 = all) |
| `--subject-rule vote` | Subject prediction = majority vote of its windows (`mean` = mean SVM score) |
| `--shuffle-labels` | Leakage check: shuffle subject labels once; results should be at chance |
| `--n-jobs -1` | CPU cores to use |

Graph metrics are cached in `<features>/graph_cache/`, so reruns, sweeps and
permutations do not recompute them. Delete that folder if you rebuild the features.

### What happens inside each outer fold (no leakage)

Only the training subjects are used for: dropping zero-variance features,
the ANOVA pre-screen, standardisation (inside every SVM pipeline), SFFS
(inner CV grouped by subject, scored by balanced accuracy) and tuning C.
The test subjects are only used to score the final model.

### Outputs

| File | Contents |
|---|---|
| `metrics.json` | Pooled window- and subject-level metrics, mean ± SD over folds, permutation p-values |
| `per_fold.csv` | Accuracy, balanced accuracy, sensitivity, specificity, AUC (binary) per fold, windows and subjects, chosen C |
| `subject_predictions.csv` | One row per subject: true label, prediction, vote fractions, mean SVM score, fold |
| `selected_features.csv` | Features chosen by SFFS (e.g. `theta_clustering_Fz`) with how many folds chose them, per tree node |
| `confusion_matrix.png` | Window- and subject-level confusion matrices, pooled over folds |
| `run_config.json` | Every setting used |

## Noise robustness (paper section 5)

Build noisy copies of the features with Part 1, then evaluate them all with the
same settings in one run. This writes `noise_robustness.csv` next to the clean
results:

```bash
for snr in 10 20 30 40; do
  python 01_build_dataset.py --manifest data/mydata/manifest.csv \
      --out features/mydata_snr$snr --conn both --snr $snr \
      --channels Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2
done
python 02_baseline_modares.py --features features/mydata --conn wpli --n-perm 0 \
    --noise-dirs features/mydata_snr10,features/mydata_snr20,features/mydata_snr30,features/mydata_snr40
```

Each noisy set goes through the whole pipeline (selection, training and testing
on noisy data).

## Where Part 2 differs from Modares-Haghighi et al. (2021) — say so in your methods

| Paper | Here | Why |
|---|---|---|
| Hierarchical "bio-inspired" tree of binary SVMs | Same tree for multi-class labels (`--tree-order`); a single RBF SVM for binary labels | With two classes the tree is one node, so it reduces to one SVM |
| Train/test split of 17/6 subjects | Default: stratified 5-fold GroupKFold over subjects; `--paper-split` reproduces a ~25 % held-out test set; `--loso` available | With few subjects one split is very noisy; every subject is tested once and windows of one person never sit in both training and test |
| Window-level accuracy | Window-level and subject-level (majority vote) results | Clinical decisions are per person; overlapping windows of one subject are not independent |
| — | Permutation test (labels shuffled per subject, whole pipeline rerun) | Tells you whether the accuracy is above chance for this sample size |
| SFFS over all features | Zero-variance features dropped, then the 30 best by ANOVA F kept, then SFFS (max 20 features), all inside each training fold; SFFS and C tuning use 20 evenly spaced windows per subject, the final SVM uses all windows | Makes SFFS (and 100 permutation reruns) finish in minutes; selection still never sees test subjects |
| C tuned from 0.1 to 10 in 0.05 steps | Coarse grid 0.1, 0.3, 1, 3, 10 by default; `--fine-grid` for the paper's grid | Speed; the fine grid rarely changes the result |
| — | SVM with `class_weight="balanced"`, SFFS and tuning scored by balanced accuracy | Groups are usually unequal in size; without it the SVM predicts the larger group for everyone |
| Alpha band only | Theta + alpha (set in Part 1) | Theta is central to dysmenorrhea (Lee et al., 2017) |
| SOBI artifact removal | ICA + ICLabel (Part 1) | MNE has no SOBI |
| MVAR on alpha-filtered signal; surrogate test for significant PDC links | MVAR on the broadband window; no surrogate test (Part 1) | See Part 1 |
| Weight normalisation not specified | Each filtered window graph is divided by its strongest edge, so weights are in [0, 1] (needed by bctpy) | Metrics then describe the graph's shape; overall connectivity level is not a feature |
| Local efficiency (BCT) | bctpy `efficiency_wei(local=True)`, the Wang et al. (2016) version; for directed PDC it symmetrises each node's neighbourhood | bctpy's recommended version; bctpy has no separate directed local efficiency |
| Noise added to the EEG, then classification repeated | Same: `01_build_dataset.py --snr` then `--noise-dirs`; each noisy set is selected, trained and tested on its own | Follows the paper; train-clean/test-noisy is not implemented |

Note on degree: bctpy degree is binary (number of kept edges). At density 0.9
most nodes keep almost all their edges, so degree varies little; it becomes
more informative at lower densities (see `--sweep`).

## Part 2 synthetic test result

18 synthetic subjects (6 control, 6 mild, 6 severe), default settings
(density 0.9, stratified 5-fold GroupKFold, 100 permutations). Chance is 0.50
for binary, 0.33 for 3-class.

| Run | Window bal. acc. | Subject bal. acc. | Subject AUC | Perm. p (subject bal. acc.) |
|---|---|---|---|---|
| wPLI, binary | 0.70 | 0.79 | 1.00 (p = 0.01) | 0.02 |
| PDC, binary | 0.69 | 0.83 | 0.89 (p = 0.03) | 0.02 |
| wPLI, 3-class (tree 0,2,1) | 0.52 | 0.67 | — | 0.01 |
| PDC, 3-class (tree 0,2,1) | 0.47 | 0.44 | — | 0.22 |

- wPLI selected theta clustering / local efficiency at Fz, F3, F4, C3, C4: the
  channels with the planted theta coupling.
- PDC leaned on alpha features (the generator also lowers posterior alpha with
  severity) and found the theta coupling less well, as expected from Part 1.
- Binary wPLI ranks every subject correctly (AUC 1.0), but the mild group sits
  just below the vote threshold, so subject accuracy is lower than AUC.
- `--sweep` (wPLI): top-10/20 % graphs did best (subject bal. acc. 0.875);
  graphs built only from weak edges (rank 21-70 %) were at chance.
- Leakage check: the 100 shuffled-label runs averaged 0.48 (subject) and 0.49
  (window) balanced accuracy, i.e. chance.
- Runtime: ~10-20 min per run with 100 permutations on a 10-core laptop
  (seconds with `--n-perm 0`). Use `caffeinate -i` on a Mac so it does not sleep.

---

# Part 3: Graph attention network (GAT)

Same `.npz` files, same graph filtering (`--density`), same subject-wise outer
folds and metrics as Part 2 (it imports them from `02_baseline_modares.py`), so
the two models are compared on identical splits.

| File | What it does |
|---|---|
| `paineeg/gat.py` | Filtered graphs per window, node features, batching to PyG, GATv2 model |
| `03_gat.py` | Subject-wise CV training with early stopping, metrics, permutation test, attention per edge |

- Nodes = channels; node features = log band power (z-scored on training subjects),
  optionally plus each node's connectivity profile (`--node-features bp+conn`).
- Edges = filtered connectivity with one weight per band as edge features; PDC stays directed.
- 2 × GATv2 (4 heads, 32 units) → mean + max pooling → MLP; class-weighted loss.
- Inside each outer fold 20 % of the training subjects are held out for early stopping.
- Multi-class: one softmax over all classes (no tree).

```bash
python 03_gat.py --features features/mydata --conn wpli --manifest data/mydata/manifest.csv
python 03_gat.py --features features/mydata --conn pdc --manifest data/mydata/manifest.csv --label-col label_3class
```

Outputs as in Part 2 (`metrics.json`, `per_fold.csv`, `subject_predictions.csv`,
`confusion_matrix.png`, `run_config.json`) plus `training_log.csv` (loss per epoch)
and `attention_edges.csv` (mean last-layer attention per edge and true class).
Permutation test with `--n-perm N` retrains the whole CV per shuffle (use a GPU).
