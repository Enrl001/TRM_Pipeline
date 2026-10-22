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
