"""STEP 1 (shared): preprocess every recording and save connectivity + band power.

Both models read the output of this script, so they see identical data.

Usage
-----
python 01_build_dataset.py --manifest data/manifest.csv --out features/ \
       --conn both --label-col label

The manifest is a CSV with at least these columns:
  subject  - participant ID (used for subject-wise cross-validation)
  path     - path to the EEG file (.fif/.edf/.bdf/.gdf/.set/.vhdr ...)
  label    - integer class (e.g. 0 = control, 1 = pain), or any column named by --label-col

Optional flags:
  --snr 20      add white noise at this SNR (dB) after cleaning
                (repeats the paper's robustness test; save to a separate --out)
  --no-ica      skip ICA/ICLabel (faster; for quick tests only)
  --channels Fp1,Fp2,...   fix the node set (strongly recommended for real data)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paineeg.config import Config
from paineeg import preprocess as pp
from paineeg import connectivity as cx


def add_white_noise(raw, snr_db, rng):
    data = raw.get_data()
    p_sig = np.mean(data ** 2, axis=1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    raw._data = data + rng.standard_normal(data.shape) * np.sqrt(p_noise)
    return raw


def process_subject(row, cfg, conn, label_col, snr=None):
    raw = pp.load_raw(row["path"])
    raw, log = pp.preprocess(raw, cfg)
    if snr is not None:
        raw = add_white_noise(raw, snr, np.random.default_rng(cfg.random_seed))
    sfreq = raw.info["sfreq"]
    windows, t_start, starts = pp.make_windows(raw, cfg)
    win = windows.shape[-1]
    out = dict(subject=str(row["subject"]), label=int(row[label_col]),
               ch_names=np.array(raw.ch_names), bands=np.array(list(cfg.bands)),
               sfreq=sfreq, t_start=t_start)
    out["bandpower"] = cx.band_power_windows(windows, sfreq, cfg.bands) if len(windows) else \
        np.empty((0, len(cfg.bands), len(raw.ch_names)))
    if conn in ("pdc", "both"):
        order = cx.choose_subject_order(windows, cfg) if len(windows) else 0
        out["mvar_order"] = order
        out["conn_pdc"] = cx.pdc_windows(windows, sfreq, cfg.bands, order, cfg.mvar_ridge) \
            if len(windows) else np.empty((0, len(cfg.bands), len(raw.ch_names), len(raw.ch_names)))
    if conn in ("wpli", "both"):
        analytic = cx.band_analytic(raw, cfg.bands)
        out["conn_wpli"] = cx.wpli_windows(analytic, starts, win)
    log.update(n_windows=int(len(windows)), mvar_order=int(out.get("mvar_order", 0)))
    return out, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conn", choices=["pdc", "wpli", "both"], default="both")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--channels", default=None, help="comma-separated channel list")
    ap.add_argument("--line-freq", type=float, default=None)
    ap.add_argument("--crop", default=None, help="tmin,tmax in seconds")
    ap.add_argument("--no-ica", action="store_true")
    ap.add_argument("--snr", type=float, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.channels:
        cfg.channels = [c.strip() for c in args.channels.split(",")]
    if args.line_freq is not None:
        cfg.line_freq = args.line_freq
    if args.crop:
        cfg.crop = tuple(float(v) for v in args.crop.split(","))
    if args.no_ica:
        cfg.use_ica = False

    man = pd.read_csv(args.manifest)
    base = Path(args.manifest).parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    for _, row in man.iterrows():
        row = row.copy()
        p = Path(row["path"])
        row["path"] = p if p.is_absolute() else (base / p)
        t0 = time.time()
        try:
            out, log = process_subject(row, cfg, args.conn, args.label_col, args.snr)
        except Exception as e:  # keep going; report at the end
            print(f"[FAIL] {row['subject']}: {e}")
            logs.append(dict(subject=row["subject"], error=str(e)))
            continue
        np.savez_compressed(out_dir / f"{row['subject']}.npz", **out)
        log.update(subject=row["subject"], seconds=round(time.time() - t0, 1))
        logs.append(log)
        print(f"[ok] {row['subject']}: {log['n_windows']} windows, "
              f"bads={log['bad_channels']}, ICA removed={log.get('ica_excluded', [])}, "
              f"MVAR order={log['mvar_order']} ({log['seconds']} s)")
    pd.DataFrame(logs).to_csv(out_dir / "preprocessing_log.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
