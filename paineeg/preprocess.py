"""Step 1 - EEG preprocessing (shared by both models).

Follows Modares-Haghighi et al. (2021) section 3.1, adapted to MNE-Python:
  * band-pass 0.5-45 Hz + notch at the line frequency
  * artifact removal: the paper used SOBI + fractal dimension (EOG) and
    spectral similarity to Fp1 (EMG). MNE has no SOBI, so we use ICA with
    ICLabel, which removes the same artifact classes automatically.
Added for robustness: bad-channel detection/interpolation and average reference.
"""
import warnings

import numpy as np
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne_icalabel")

from .config import Config


def make_montage(name):
    """standard_1020 is renamed colin27_1020 in newer MNE; use whichever exists."""
    builtin = mne.channels.get_builtin_montages()
    if name == "standard_1020" and "colin27_1020" in builtin:
        name = "colin27_1020"
    return mne.channels.make_standard_montage(name)


def load_raw(path):
    """Read any MNE-supported EEG file (.fif, .edf, .bdf, .gdf, .set, .vhdr ...)."""
    raw = mne.io.read_raw(str(path), preload=True, verbose="error")
    return raw


def _pick_channels(raw, channels):
    """Keep only `channels`, in that order, matching names case-insensitively."""
    lookup = {ch.lower().replace(" ", ""): ch for ch in raw.ch_names}
    rename, missing = {}, []
    for want in channels:
        key = want.lower().replace(" ", "")
        if key in lookup:
            rename[lookup[key]] = want
        else:
            missing.append(want)
    if missing:
        raise ValueError(f"Channels missing from recording: {missing}. "
                         f"Available: {raw.ch_names}")
    raw.rename_channels(rename)
    raw.set_channel_types({ch: "eeg" for ch in channels}, verbose="error")
    raw.pick(list(channels))
    raw.reorder_channels(list(channels))
    return raw


def _find_bad_channels(raw, z_thresh):
    """Flag flat channels and channels with extreme variance (robust z-score)."""
    data = raw.get_data()
    var = np.var(data, axis=1)
    bads = [ch for ch, v in zip(raw.ch_names, var) if v < 1e-20]
    logv = np.log(var + 1e-30)
    med = np.median(logv)
    mad = np.median(np.abs(logv - med)) * 1.4826 + 1e-12
    z = (logv - med) / mad
    bads += [ch for ch, zz in zip(raw.ch_names, z) if abs(zz) > z_thresh and ch not in bads]
    return bads


def _run_ica(raw, raw_fit, cfg: Config):
    """ICA + ICLabel. Fits on `raw_fit` (1-100 Hz, as ICLabel expects) and removes
    the artifact components from `raw`. Returns the cleaned raw and the excluded labels."""
    from mne_icalabel import label_components

    n_good = len(raw.ch_names) - len(raw.info["bads"])
    n_comp = cfg.ica_n_components or min(n_good - 1, 20)
    ica = mne.preprocessing.ICA(n_components=n_comp, method="infomax",
                                fit_params=dict(extended=True),
                                random_state=cfg.random_seed, max_iter="auto",
                                verbose="error")
    ica.fit(raw_fit, verbose="error")
    labels = label_components(raw_fit, ica, method="iclabel")
    exclude = [i for i, (lab, p) in enumerate(zip(labels["labels"], labels["y_pred_proba"]))
               if lab in cfg.iclabel_exclude and p >= cfg.iclabel_threshold]
    ica.exclude = exclude
    ica.apply(raw, verbose="error")
    return raw, [labels["labels"][i] for i in exclude]


def preprocess(raw, cfg: Config):
    """Run the full cleaning chain. Returns (clean_raw, log_dict)."""
    raw = raw.copy()
    log = {}
    if cfg.channels:
        raw = _pick_channels(raw, cfg.channels)
    else:
        raw.pick("eeg")
    if cfg.crop:
        raw.crop(*cfg.crop)

    montage = make_montage(cfg.montage)
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose="error")

    if cfg.resample_sfreq and raw.info["sfreq"] > cfg.resample_sfreq:
        raw.resample(cfg.resample_sfreq, verbose="error")

    nyq = raw.info["sfreq"] / 2
    if cfg.line_freq and cfg.line_freq < nyq:
        raw.notch_filter(np.arange(cfg.line_freq, nyq, cfg.line_freq), verbose="error")
    # ICLabel was trained on 1-100 Hz data, so ICA is fitted on a separate copy
    # taken before the analysis band-pass
    if cfg.use_ica:
        raw_fit = raw.copy().filter(cfg.ica_l_freq, min(cfg.ica_h_freq, nyq - 1), verbose="error")
    raw.filter(cfg.l_freq, min(cfg.h_freq, nyq - 1), verbose="error")

    raw.info["bads"] = _find_bad_channels(raw, cfg.bad_channel_z)
    log["bad_channels"] = list(raw.info["bads"])

    raw.set_eeg_reference("average", projection=False, verbose="error")

    if cfg.use_ica:
        raw_fit.info["bads"] = list(raw.info["bads"])
        raw_fit.set_eeg_reference("average", projection=False, verbose="error")
        raw, excluded = _run_ica(raw, raw_fit, cfg)
        log["ica_excluded"] = excluded
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
        raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw, log


def window_starts(n_times, sfreq, cfg: Config):
    """Sample indices where each window starts (1 s windows, 50 % overlap by default)."""
    win = int(round(cfg.win_sec * sfreq))
    step = max(1, int(round(win * (1 - cfg.overlap))))
    return np.arange(0, n_times - win + 1, step), win


def make_windows(raw, cfg: Config):
    """Cut the clean recording into windows and drop windows above +/- reject_uv.

    Returns data (n_win, n_ch, n_samples), start times (s), and the kept start
    indices (needed to slice band-filtered signals for wPLI).
    """
    data = raw.get_data(reject_by_annotation="NaN")
    sfreq = raw.info["sfreq"]
    starts, win = window_starts(data.shape[1], sfreq, cfg)
    thr = cfg.reject_uv * 1e-6
    keep = []
    for s in starts:
        seg = data[:, s:s + win]
        if np.isnan(seg).any():
            continue
        if np.abs(seg).max() > thr:
            continue
        keep.append(s)
    keep = np.array(keep, dtype=int)
    if keep.size == 0:
        return np.empty((0, data.shape[0], win)), keep / sfreq, keep
    windows = np.stack([data[:, s:s + win] for s in keep])
    return windows, keep / sfreq, keep
