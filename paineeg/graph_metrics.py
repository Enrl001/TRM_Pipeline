"""Graph filtering + graph metrics per window (Modares-Haghighi et al., 2021, sec. 3.3-3.4).

Shared by the SVM baseline and (later) the GAT, so both see the same graphs.

Graph filtering (applied to every window and band separately)
  cumulative thresholding : keep the strongest d of edges, d = 10 %, 20 %, ..., 90 %
  windowed thresholding   : keep edges whose strength rank lies in a band,
                            1-10 %, 11-20 %, ..., 61-70 % (strongest first)
Both are expressed as a rank interval (lo, hi]: cumulative d = (0, d].

Metrics (bctpy), on weights normalised to [0, 1] (divided by the window's max):
  degree       in + out degree (directed PDC) / degree (wPLI), per node
  betweenness  weighted, on lengths 1/w, per node
  clustering   clustering_coef_wd (directed) / clustering_coef_wu (wPLI), per node
  local_eff    weighted local efficiency (Wang et al., 2016), per node
  global_eff   weighted global efficiency, one value
Feature vector per window = [band1: 4 x C node metrics + 1 global, band2: ...].

Matrix convention (as in connectivity.py): W[source, target].
"""
import hashlib
from pathlib import Path

import numpy as np
import bct

NODE_METRICS = ("degree", "betweenness", "clustering", "local_eff")

# The paper's 16 graph versions: 9 cumulative + 7 windowed, as (name, lo, hi)
CUMULATIVE = [(f"cum{d}", 0.0, d / 100) for d in range(10, 100, 10)]
WINDOWED = [(f"win{lo + 1}-{lo + 10}", lo / 100, (lo + 10) / 100) for lo in range(0, 70, 10)]
GRAPH_VERSIONS = CUMULATIVE + WINDOWED


def version_for_density(density):
    """The cumulative version that keeps the strongest `density` of edges."""
    return (f"cum{int(round(density * 100))}", 0.0, float(density))


def filter_graph(W, lo, hi, directed):
    """Keep edges whose strength rank (strongest first) falls in (lo, hi] of all edges.

    Undirected graphs rank each pair once (upper triangle) and stay symmetric.
    """
    C = W.shape[0]
    if directed:
        idx = np.where(~np.eye(C, dtype=bool))
    else:
        idx = np.triu_indices(C, k=1)
    w = W[idx]
    order = np.argsort(-w, kind="stable")
    m = len(w)
    keep = order[int(round(lo * m)):int(round(hi * m))]
    out = np.zeros_like(W)
    rows, cols = idx[0][keep], idx[1][keep]
    out[rows, cols] = W[rows, cols]
    if not directed:
        out[cols, rows] = W[rows, cols]
    return out


def graph_features(W, directed):
    """Metrics of one filtered graph. Returns node metrics (4, C) and global efficiency."""
    mx = W.max()
    if mx <= 0:
        return np.zeros((len(NODE_METRICS), W.shape[0])), 0.0
    W = W / mx                                  # normalise weights to [0, 1]
    L = bct.weight_conversion(W, "lengths")     # length = 1 / weight
    if directed:
        _, _, deg = bct.degrees_dir(W)          # in + out
        clu = bct.clustering_coef_wd(W)
    else:
        deg = bct.degrees_und(W)
        clu = bct.clustering_coef_wu(W)
    btw = bct.betweenness_wei(L)
    eloc = bct.efficiency_wei(W, local=True)
    eglob = bct.efficiency_wei(W)
    node = np.vstack([deg, btw, np.ravel(clu), np.ravel(eloc)]).astype(float)
    return np.nan_to_num(node), float(np.nan_to_num(eglob))


def window_features(conn, directed, lo, hi):
    """Feature matrix for one subject. conn: (n_win, n_bands, C, C) -> (n_win, n_bands * (4C + 1))."""
    n_win, n_b, C, _ = conn.shape
    X = np.zeros((n_win, n_b, len(NODE_METRICS) * C + 1))
    for w in range(n_win):
        for b in range(n_b):
            node, glob = graph_features(filter_graph(conn[w, b], lo, hi, directed), directed)
            X[w, b, :-1] = node.ravel()
            X[w, b, -1] = glob
    return X.reshape(n_win, -1)


def feature_names(bands, ch_names):
    """Names matching window_features columns, e.g. 'theta_degree_Fz', 'alpha_global_eff'."""
    names = []
    for b in bands:
        names += [f"{b}_{m}_{ch}" for m in NODE_METRICS for ch in ch_names]
        names.append(f"{b}_global_eff")
    return names


def subject_features(subjects, lo, hi, cache_dir=None, n_jobs=1, tag=""):
    """window_features for every subject (from paineeg.data.load_subjects), in parallel.

    With `cache_dir`, results are stored per subject and graph version, keyed on the
    connectivity values, so sweeps and reruns reuse them.
    """
    from joblib import Parallel, delayed

    def one(s):
        path = None
        if cache_dir is not None:
            key = hashlib.sha1(np.ascontiguousarray(s["conn"]).tobytes()).hexdigest()[:12]
            path = Path(cache_dir) / f"{s['subject']}_{tag}_{lo:.2f}-{hi:.2f}_{key}.npy"
            if path.exists():
                return np.load(path)
        X = window_features(s["conn"], s["directed"], lo, hi)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, X)
        return X

    return Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in subjects)
