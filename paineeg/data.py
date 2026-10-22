"""Load the per-subject .npz files written by 01_build_dataset.py.

Each file holds everything both models need for one participant:
  conn_pdc   (n_win, n_bands, C, C)   PDC, W[source, target]
  conn_wpli  (n_win, n_bands, C, C)   wPLI, symmetric
  bandpower  (n_win, n_bands, C)      log10 power (GAT node features)
  t_start    (n_win,)                 window start times (s)
  label, subject, ch_names, bands, sfreq, mvar_order
"""
from pathlib import Path
import numpy as np


def load_subjects(feature_dir, conn="pdc"):
    """Return a list of dicts, one per subject, with the chosen connectivity."""
    subjects = []
    for f in sorted(Path(feature_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        key = f"conn_{conn}"
        if key not in d.files:
            raise KeyError(f"{f.name} has no {key}; rebuild with --conn {conn} or both")
        if d[key].shape[0] == 0:
            print(f"[skip] {f.name}: no clean windows")
            continue
        subjects.append(dict(
            subject=str(d["subject"]),
            label=int(d["label"]),
            conn=d[key].astype(np.float32),
            bandpower=d["bandpower"].astype(np.float32),
            ch_names=[str(c) for c in d["ch_names"]],
            bands=[str(b) for b in d["bands"]],
            directed=(conn == "pdc"),
        ))
    if not subjects:
        raise FileNotFoundError(f"No subject files found in {feature_dir}")
    chs = subjects[0]["ch_names"]
    for s in subjects:
        if s["ch_names"] != chs:
            raise ValueError(f"Subject {s['subject']} has different channels; "
                             "set Config.channels so every graph has the same nodes")
    return subjects
