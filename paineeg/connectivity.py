"""Step 2 - connectivity per window (shared by both models).

PDC (partial directed coherence) - the measure used by Modares-Haghighi et al.
  An MVAR model is fitted to each 1 s window (order chosen by BIC, as in the
  paper). PDC is computed from the MVAR coefficients and averaged over the
  frequencies of each band. Deviation from the paper: the MVAR is fitted on the
  broadband (0.5-45 Hz) window and PDC is read out per band, because fitting an
  MVAR to an already narrow-band-filtered signal is numerically ill-conditioned.

wPLI (weighted phase lag index) - recommended alternative (robust to volume
  conduction). Computed from the analytic (Hilbert) signal of each band, over
  the samples inside each window.

Matrix convention for BOTH measures: W[source, target]
(row = sender, column = receiver). wPLI is symmetric, PDC is directed.
"""
import numpy as np
from scipy.signal import hilbert, welch

import mne


# ------------------------------------------------------------------ MVAR / PDC
def fit_mvar(x, order, ridge=1e-6):
    """Least-squares MVAR fit. x: (n_ch, n_times). Returns lags (p, C, C), noise cov."""
    x = x - x.mean(axis=1, keepdims=True)
    C, T = x.shape
    Y = x[:, order:]
    Z = np.vstack([x[:, order - k:T - k] for k in range(1, order + 1)])  # (C*p, T-p)
    ZZ = Z @ Z.T
    ZZ += ridge * np.trace(ZZ) / ZZ.shape[0] * np.eye(ZZ.shape[0])
    A = np.linalg.solve(ZZ, Z @ Y.T).T          # (C, C*p)
    E = Y - A @ Z
    sigma = E @ E.T / E.shape[1]
    lags = np.stack([A[:, k * C:(k + 1) * C] for k in range(order)])
    return lags, sigma


def bic_order(x, max_order, ridge=1e-6):
    """Model order that minimises the Bayesian information criterion (Schwarz, 1978)."""
    C, T = x.shape
    best, best_bic = 1, np.inf
    for p in range(1, max_order + 1):
        n = T - p
        if n <= C * p:                      # not enough samples for this order
            break
        _, sigma = fit_mvar(x, p, ridge)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            continue
        bic = logdet + np.log(n) * p * C * C / n
        if bic < best_bic:
            best, best_bic = p, bic
    return best


def pdc_from_lags(lags, freqs, sfreq):
    """PDC spectrum (Baccala & Sameshima, 2001). Returns P[f, i, j] = flow j -> i."""
    p, C, _ = lags.shape
    k = np.arange(1, p + 1)
    E = np.exp(-2j * np.pi * np.outer(freqs, k) / sfreq)          # (F, p)
    Af = np.eye(C)[None] - np.einsum("fk,kij->fij", E, lags)       # (F, C, C)
    denom = np.sqrt((np.abs(Af) ** 2).sum(axis=1, keepdims=True))  # column norms
    return np.abs(Af) / denom


def choose_subject_order(windows, cfg, n_probe=30):
    """One MVAR order per subject: median BIC order over a sample of windows."""
    if isinstance(cfg.mvar_order, int):
        return cfg.mvar_order
    idx = np.linspace(0, len(windows) - 1, min(n_probe, len(windows))).astype(int)
    orders = [bic_order(windows[i], cfg.mvar_max_order, cfg.mvar_ridge) for i in idx]
    return int(np.median(orders))


def pdc_windows(windows, sfreq, bands, order, ridge=1e-6, freq_step=0.5):
    """PDC per window and band. Returns (n_win, n_bands, C, C) with W[source, target]."""
    n_win, C, _ = windows.shape
    out = np.zeros((n_win, len(bands), C, C))
    band_freqs = [np.arange(lo, hi + 1e-9, freq_step) for lo, hi in bands.values()]
    for w in range(n_win):
        lags, _ = fit_mvar(windows[w], order, ridge)
        for b, freqs in enumerate(band_freqs):
            P = pdc_from_lags(lags, freqs, sfreq).mean(axis=0)  # P[i, j] = j -> i
            W = P.T.copy()                                         # W[source, target]
            np.fill_diagonal(W, 0.0)
            out[w, b] = W
    return out


# ------------------------------------------------------------------ wPLI
def band_analytic(raw, bands):
    """Band-filter the continuous clean recording and return analytic signals.

    Filtering the continuous signal (not each 1 s window) avoids edge artefacts.
    Returns array (n_bands, C, n_times), complex.
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    out = []
    for lo, hi in bands.values():
        filt = mne.filter.filter_data(data, sfreq, lo, hi, verbose="error")
        out.append(hilbert(filt, axis=-1))
    return np.stack(out)


def wpli_windows(analytic, starts, win):
    """Time-resolved wPLI (Vinck et al., 2011) within each window.

    wPLI_ij = |mean_t Im(z_i z_j*)| / mean_t |Im(z_i z_j*)|
    Returns (n_win, n_bands, C, C), symmetric, zero diagonal.
    """
    n_b, C, _ = analytic.shape
    out = np.zeros((len(starts), n_b, C, C))
    for w, s in enumerate(starts):
        z = analytic[:, :, s:s + win]                         # (B, C, T)
        cross = np.einsum("bit,bjt->bijt", z, np.conj(z)).imag
        num = np.abs(cross.mean(axis=-1))
        den = np.abs(cross).mean(axis=-1) + 1e-20
        W = num / den
        for b in range(n_b):
            np.fill_diagonal(W[b], 0.0)
        out[w] = W
    return out


# ------------------------------------------------------------------ band power
def band_power_windows(windows, sfreq, bands):
    """log10 band power per window, band and channel -> (n_win, n_bands, C).

    Used as GAT node features and as model A (band-power baseline).
    """
    n_win, C, T = windows.shape
    f, psd = welch(windows, fs=sfreq, nperseg=min(T, int(sfreq)), axis=-1)
    out = np.zeros((n_win, len(bands), C))
    for b, (lo, hi) in enumerate(bands.values()):
        m = (f >= lo) & (f < hi)
        out[:, b] = np.log10(psd[..., m].mean(axis=-1) + 1e-30)
    return out
