"""Make a small synthetic EEG dataset to test the pipeline end-to-end.

This is NOT real data. It only checks that every step runs and that the models
can find a planted effect. Each subject gets 19-channel 10-20 EEG (1/f noise +
posterior alpha + eye blinks). In the "pain" groups, a theta (6 Hz) source at
Fz drives fronto-central channels with a short delay, so theta connectivity
(and directed PDC Fz -> others) increases with severity.

Labels written to the manifest:
  label_3class : 0 = control, 1 = mild, 2 = severe  (tests the SVM tree)
  label        : 0 = control, 1 = pain (mild or severe)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne

CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
      "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
DRIVEN = {"F3": 0.008, "F4": 0.012, "Cz": 0.016, "C3": 0.020, "C4": 0.020}


def pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def narrowband(n, sfreq, lo, hi, rng):
    x = rng.standard_normal(n)
    x = mne.filter.filter_data(x[None], sfreq, lo, hi, verbose="error")[0]
    return x / x.std()


def make_subject(severity, sfreq, dur, rng):
    n = int(sfreq * dur)
    data = np.stack([pink(n, rng) for _ in CH]) * 8.0          # ~8 uV background
    alpha = narrowband(n, sfreq, 9, 11, rng)
    for i, ch in enumerate(CH):
        if ch[0] in "PO":
            data[i] += (6.0 - 1.5 * severity) * alpha * rng.uniform(0.8, 1.2)
        data[i] += 2.0 * narrowband(n, sfreq, 4, 8, rng)        # independent theta
    coupling = [0.0, 0.5, 1.0][severity]
    driver = narrowband(n, sfreq, 5, 7, rng) * 5.0
    data[CH.index("Fz")] += driver
    for ch, lag in DRIVEN.items():
        shift = int(lag * sfreq)
        data[CH.index(ch)] += coupling * np.roll(driver, shift)
    # eye blinks on frontal channels (gives ICA something to remove)
    for t in rng.uniform(1, dur - 1, size=int(dur / 4)):
        idx = int(t * sfreq)
        blink = 60 * np.hanning(int(0.3 * sfreq))
        for ch, g in (("Fp1", 1.0), ("Fp2", 1.0), ("F7", 0.4), ("F8", 0.4), ("F3", 0.3), ("F4", 0.3)):
            seg = data[CH.index(ch), idx:idx + len(blink)]
            seg += g * blink[:len(seg)]
    return data * 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-per-group", type=int, default=8)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(CH, args.sfreq, "eeg")
    from paineeg.preprocess import make_montage
    info.set_montage(make_montage("standard_1020"))
    rows = []
    k = 0
    for sev in (0, 1, 2):
        for _ in range(args.n_per_group):
            k += 1
            sid = f"sub-{k:02d}"
            raw = mne.io.RawArray(make_subject(sev, args.sfreq, args.duration, rng), info, verbose="error")
            fname = f"{sid}_eeg.fif"
            raw.save(out / fname, overwrite=True, verbose="error")
            rows.append(dict(subject=sid, path=fname, label=int(sev > 0), label_3class=sev))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} subjects to {out}/ and manifest.csv")


if __name__ == "__main__":
    main()
