"""Shared settings for the whole pipeline.

Both models (the Modares-Haghighi baseline and the GAT) read the SAME
preprocessed data and the SAME connectivity matrices, so every setting
in this file applies to both. Change values here, not inside the scripts.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class Config:
    # ---------------- Frequency bands ----------------
    # Theta is the main band for dysmenorrhea (Lee et al., 2017).
    # Alpha is the band used by Modares-Haghighi et al. (2021).
    bands: dict = field(default_factory=lambda: {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
    })

    # ---------------- Channels ----------------
    # Fix the channel list so every subject's graph has the same nodes in the
    # same order. None = use all EEG channels in the first file (all files
    # must then have the same channels).
    channels: Optional[Sequence[str]] = None
    montage: str = "standard_1020"

    # ---------------- Preprocessing ----------------
    l_freq: float = 0.5           # Modares-Haghighi: 0.5-45 Hz band-pass
    h_freq: float = 45.0
    line_freq: Optional[float] = 50.0   # 50 Hz (Australia/Europe/Iran), 60 Hz (USA)
    resample_sfreq: Optional[float] = 250.0
    crop: Optional[tuple] = None  # (tmin, tmax) in seconds, e.g. pick the eyes-closed segment
    bad_channel_z: float = 5.0    # flag channels whose robust log-variance z-score exceeds this (PREP-style)
    use_ica: bool = True          # ICA + ICLabel (replaces the paper's SOBI step)
    ica_n_components: Optional[int] = None  # None = min(n_channels - 1, 20)
    ica_l_freq: float = 1.0       # ICA/ICLabel is fitted on a 1-100 Hz copy (what ICLabel was trained on);
    ica_h_freq: float = 100.0     # the components are then removed from the 0.5-45 Hz data
    iclabel_exclude: tuple = ("eye blink", "muscle artifact", "heart beat",
                              "line noise", "channel noise")
    iclabel_threshold: float = 0.7

    # ---------------- Windows ----------------
    win_sec: float = 1.0          # Modares-Haghighi: 1 s windows ...
    overlap: float = 0.5          # ... with 50 % overlap
    reject_uv: float = 100.0      # drop windows with any sample above +/-100 uV

    # ---------------- Connectivity ----------------
    mvar_order: object = "bic"    # "bic" or an integer model order
    mvar_max_order: int = 10
    mvar_ridge: float = 1e-6      # small ridge term keeps the MVAR fit stable on 1 s windows

    # ---------------- Graph construction ----------------
    density: float = 0.9          # keep strongest 90 % of edges (best density in the paper)

    random_seed: int = 42


DEFAULT = Config()
