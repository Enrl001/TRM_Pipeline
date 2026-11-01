"""STEP 2 (model 1): Modares-Haghighi et al. (2021) baseline.

Graph metrics of the per-window connectivity graphs -> SFFS feature selection ->
RBF SVM (binary) or the paper's hierarchical tree of binary SVMs (multi-class).

Validation is always by SUBJECT: every window of one person is in the same fold.
Scaling, feature selection and C tuning are fitted inside the training folds only.

Usage
-----
python 02_baseline_modares.py --features features/mydata --conn wpli
python 02_baseline_modares.py --features features/mydata --conn pdc \
       --manifest data/mydata/manifest.csv --label-col label_3class --tree-order 0,2,1

Main options
  --density 0.9      cumulative threshold (keep strongest 90 % of edges; paper's best)
  --sweep            run all 16 graph versions (9 cumulative, 7 windowed) and use the best
  --loso             leave-one-subject-out instead of stratified 5-fold GroupKFold
  --paper-split      hold out ~25 % of subjects as a test set; 5-fold CV on the rest for tuning
  --n-perm 100       permutation test (labels shuffled at the subject level)
  --max-features 20  cap on SFFS features
  --fine-grid        C from 0.1 to 10 in 0.05 steps (paper) instead of a coarse grid
  --manifest CSV     take labels from this CSV (column --label-col, matched on 'subject')
  --noise-dirs A,B   also evaluate feature dirs built with 01_build_dataset.py --snr
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import f_classif
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold,
                                     StratifiedShuffleSplit)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from paineeg.data import load_subjects
from paineeg import graph_metrics as gm

COARSE_C = [0.1, 0.3, 1.0, 3.0, 10.0]
FINE_C = list(np.round(np.arange(0.1, 10.0 + 1e-9, 0.05), 2))


# ------------------------------------------------------------------ data
def load_dataset(args, features_dir, version):
    """Window feature matrix, labels and subject ids for one graph version."""
    subjects = load_subjects(features_dir, conn=args.conn)
    if args.manifest:
        man = pd.read_csv(args.manifest).astype({"subject": str}).set_index("subject")
        for s in subjects:
            s["label"] = int(man.loc[s["subject"], args.label_col])
    elif args.label_col != "label":
        raise SystemExit(f"--label-col {args.label_col} needs --manifest (the .npz files "
                         "only store the label used when they were built)")
    name, lo, hi = version
    feats = gm.subject_features(subjects, lo, hi, cache_dir=Path(features_dir) / "graph_cache",
                                n_jobs=args.n_jobs, tag=args.conn)
    X = np.vstack(feats)
    groups = np.concatenate([[s["subject"]] * len(f) for s, f in zip(subjects, feats)])
    subj = pd.DataFrame(dict(subject=[s["subject"] for s in subjects],
                             label=[s["label"] for s in subjects]))
    names = gm.feature_names(subjects[0]["bands"], subjects[0]["ch_names"])
    return X, groups, subj, names


# ------------------------------------------------------------------ splits
def n_folds(labels, wanted):
    return max(2, min(wanted, np.bincount(labels).min() if len(labels) else 2))


def outer_splits(subj, args, seed):
    """Lists of (train_subjects, test_subjects)."""
    sid, lab = subj["subject"].to_numpy(), subj["label"].to_numpy()
    if args.loso:
        return [(np.delete(sid, i), sid[[i]]) for i in range(len(sid))]
    if args.paper_split:
        ss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        tr, te = next(ss.split(sid, lab))
        return [(sid[tr], sid[te])]
    k = n_folds(np.unique(lab, return_inverse=True)[1], args.folds)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(sid[tr], sid[te]) for tr, te in cv.split(sid, lab, groups=sid)]


def inner_cv(y_subj, args, seed):
    k = n_folds(y_subj, args.inner_folds)
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)


def subsample(groups, per_subject):
    """Evenly spaced windows per subject (for SFFS and C tuning only)."""
    idx = []
    for g in np.unique(groups):
        w = np.where(groups == g)[0]
        if per_subject and len(w) > per_subject:
            w = w[np.linspace(0, len(w) - 1, per_subject).astype(int)]
        idx.append(w)
    return np.concatenate(idx)


# ------------------------------------------------------------------ one binary SVM node
def svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced"))


def fit_node(X, y, groups, args, seed, n_jobs):
    """SFFS + C tuning + final fit of one binary SVM, on training windows only."""
    feat = np.where(X.std(axis=0) > 1e-12)[0]               # drop zero-variance features
    if args.prefilter and len(feat) > args.prefilter:        # univariate pre-screen (train only)
        F, _ = f_classif(X[:, feat], y)
        feat = feat[np.argsort(-np.nan_to_num(F))[:args.prefilter]]
    sub = subsample(groups, args.sel_windows)
    Xs, ys, gs = X[np.ix_(sub, feat)], y[sub], groups[sub]
    y_subj = pd.Series(ys).groupby(gs).first().to_numpy()
    cv = inner_cv(y_subj, args, seed)
    k_max = min(args.max_features, len(feat))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sffs = SFS(svm(), k_features=(1, k_max), forward=True, floating=True,
                   scoring="balanced_accuracy", cv=cv, n_jobs=n_jobs, verbose=0)
        sffs.fit(Xs, ys, groups=gs)
    chosen = feat[list(sffs.k_feature_idx_)]
    grid = GridSearchCV(svm(), {"svc__C": FINE_C if args.fine_grid else COARSE_C},
                        scoring="balanced_accuracy", cv=cv, n_jobs=n_jobs)
    grid.fit(X[np.ix_(sub, chosen)], ys, groups=gs)
    C = grid.best_params_["svc__C"]
    model = svm().set_params(svc__C=C).fit(X[:, chosen], y)
    return dict(features=chosen, C=C, model=model, sffs_score=float(sffs.k_score_))


def fit_classifier(X, y, groups, classes, order, args, seed, n_jobs):
    """Binary: one SVM (positive = highest class). Multi-class: hierarchical tree that
    peels off one class per node, in `order` (Modares-Haghighi et al., 2021)."""
    if len(classes) == 2:
        pos = classes[-1]
        return [dict(peel=pos, **fit_node(X, (y == pos).astype(int), groups, args, seed, n_jobs))]
    nodes, remaining = [], list(order)
    for c in order[:-1]:
        m = np.isin(y, remaining)
        nodes.append(dict(peel=c, **fit_node(X[m], (y[m] == c).astype(int), groups[m],
                                             args, seed, n_jobs)))
        remaining.remove(c)
    return nodes


def predict(nodes, X, classes, order):
    """Window predictions and (binary only) decision scores for the positive class."""
    if len(classes) == 2:
        n = nodes[0]
        score = n["model"].decision_function(X[:, n["features"]])
        return np.where(score > 0, classes[-1], classes[0]), score
    pred = np.full(len(X), order[-1])
    undecided = np.ones(len(X), bool)
    for n in nodes:
        hit = undecided & (n["model"].decision_function(X[:, n["features"]]) > 0)
        pred[hit] = n["peel"]
        undecided &= ~hit
    return pred, None


# ------------------------------------------------------------------ metrics
def scores(y, pred, score, classes):
    out = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred))
    cm = confusion_matrix(y, pred, labels=classes)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, spec = tp / (tp + fn), tn / (tn + fp)
    if len(classes) == 2:                       # positive class = classes[-1]
        out.update(sensitivity=sens[-1], specificity=sens[0])
        out["auc"] = roc_auc_score(y == classes[-1], score) if len(np.unique(y)) == 2 else np.nan
    else:
        out.update(sensitivity=np.nanmean(sens), specificity=np.nanmean(spec))
    return {k: float(v) for k, v in out.items()}


def subject_level(groups, y, pred, score, classes, rule):
    """Subject prediction = majority vote (or mean decision score) over its windows."""
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        counts = np.array([(pred[m] == c).sum() for c in classes])
        row = dict(subject=g, true=int(y[m][0]), n_windows=int(m.sum()))
        if score is not None:
            row["mean_score"] = float(score[m].mean())
        if rule == "mean" and score is not None:
            row["pred"] = int(classes[-1] if row["mean_score"] > 0 else classes[0])
        elif score is not None and counts[0] == counts[-1]:   # binary tie -> mean score
            row["pred"] = int(classes[-1] if row["mean_score"] > 0 else classes[0])
        else:
            row["pred"] = int(classes[np.argmax(counts)])
        for c, n in zip(classes, counts):
            row[f"frac_{c}"] = n / m.sum()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ one full evaluation
def evaluate(X, groups, subj, args, seed=0, n_jobs=1, keep_details=True):
    """Run the whole subject-wise CV. Returns a dict of results."""
    classes = [int(c) for c in sorted(subj["label"].unique())]
    order = [int(c) for c in args.tree_order.split(",")] if args.tree_order else classes
    if sorted(order) != classes:
        raise SystemExit(f"--tree-order {order} must list every class {classes}")
    lab = dict(zip(subj["subject"], subj["label"]))
    y = np.array([lab[g] for g in groups])
    folds, oof, subj_rows, selected = [], [], [], []
    for k, (tr_s, te_s) in enumerate(outer_splits(subj, args, seed)):
        tr, te = np.isin(groups, tr_s), np.isin(groups, te_s)
        nodes = fit_classifier(X[tr], y[tr], groups[tr], classes, order, args, seed, n_jobs)
        pred, score = predict(nodes, X[te], classes, order)
        sdf = subject_level(groups[te], y[te], pred, score, classes, args.subject_rule)
        sdf["fold"] = k
        w = scores(y[te], pred, score, classes)
        s = scores(sdf["true"].to_numpy(), sdf["pred"].to_numpy(),
                   sdf["mean_score"].to_numpy() if score is not None else None, classes)
        folds.append(dict(fold=k, n_test_subjects=len(te_s),
                          **{f"window_{m}": v for m, v in w.items()},
                          **{f"subject_{m}": v for m, v in s.items()},
                          C=";".join(str(n["C"]) for n in nodes)))
        oof.append(pd.DataFrame(dict(group=groups[te], y=y[te], pred=pred,
                                     score=score if score is not None else np.nan)))
        subj_rows.append(sdf)
        for i, n in enumerate(nodes):
            selected += [dict(fold=k, node=i, peel=n["peel"], feature=int(f)) for f in n["features"]]
    oof = pd.concat(oof)
    sdf = pd.concat(subj_rows, ignore_index=True)
    binary = len(classes) == 2
    pooled_w = scores(oof["y"].to_numpy(), oof["pred"].to_numpy(),
                      oof["score"].to_numpy() if binary else None, classes)
    pooled_s = scores(sdf["true"].to_numpy(), sdf["pred"].to_numpy(),
                      sdf["mean_score"].to_numpy() if binary else None, classes)
    res = dict(classes=classes, order=order, pooled_window=pooled_w, pooled_subject=pooled_s,
               per_fold=pd.DataFrame(folds))
    if keep_details:
        res.update(oof=oof, subjects=sdf, selected=pd.DataFrame(selected))
    return res


def permuted_score(X, groups, subj, args, i):
    rng = np.random.default_rng(10_000 + i)
    perm = subj.copy()
    perm["label"] = rng.permutation(perm["label"].to_numpy())
    r = evaluate(X, groups, perm, args, seed=args.seed, n_jobs=1, keep_details=False)
    return [r["pooled_subject"]["balanced_accuracy"], r["pooled_window"]["balanced_accuracy"],
            r["pooled_subject"].get("auc", np.nan)]


# ------------------------------------------------------------------ outputs
def summarise(per_fold):
    num = per_fold.drop(columns=["fold", "n_test_subjects", "C"])
    return {c: dict(mean=float(num[c].mean()), sd=float(num[c].std(ddof=1)) if len(num) > 1 else 0.0)
            for c in num.columns}


def plot_confusion(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = res["classes"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (title, yt, yp) in zip(axes, [
            ("Windows", res["oof"]["y"], res["oof"]["pred"]),
            ("Subjects", res["subjects"]["true"], res["subjects"]["pred"])]):
        cm = confusion_matrix(yt, yp, labels=classes)
        ax.imshow(cm, cmap="Blues")
        lo, hi = cm.min(), cm.max()
        for i in range(len(classes)):
            for j in range(len(classes)):
                # imshow scales colours from min to max, so pick the text colour the same way
                dark = hi > lo and (cm[i, j] - lo) / (hi - lo) > 0.5
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if dark else "black")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} (pooled over folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(res, names, out, args, version, perm=None, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    res["per_fold"].to_csv(out / "per_fold.csv", index=False)
    res["subjects"].to_csv(out / "subject_predictions.csv", index=False)
    sel = res["selected"]
    n_folds_run = res["per_fold"].shape[0]
    if len(sel):
        sel = (sel.assign(feature=[names[i] for i in sel["feature"]])
               .groupby(["node", "peel", "feature"]).size().rename("count").reset_index())
        sel["fraction_of_folds"] = sel["count"] / n_folds_run
        sel = sel.sort_values(["node", "count", "feature"], ascending=[True, False, True])
    sel.to_csv(out / "selected_features.csv", index=False)
    plot_confusion(res, out / "confusion_matrix.png")
    metrics = dict(conn=args.conn, label_col=args.label_col, classes=res["classes"],
                   tree_order=res["order"], graph_version=version[0],
                   rank_interval=[version[1], version[2]],
                   validation="loso" if args.loso else "paper-split" if args.paper_split
                   else f"stratified group {res['per_fold'].shape[0]}-fold",
                   pooled_window=res["pooled_window"], pooled_subject=res["pooled_subject"],
                   per_fold_mean_sd=summarise(res["per_fold"]))
    if perm is not None:
        metrics["permutation"] = perm
    if extra:
        metrics.update(extra)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update(graph_version=version[0], n_features_total=len(names))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2, default=str))


def fmt(d):
    return " ".join(f"{k}={v:.3f}" for k, v in d.items())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="folder of .npz files from 01_build_dataset.py")
    ap.add_argument("--conn", choices=["pdc", "wpli"], default="pdc")
    ap.add_argument("--manifest", default=None, help="CSV with a 'subject' column and the label column")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--tree-order", default=None, help="class order for the multi-class tree, e.g. 0,2,1")
    ap.add_argument("--density", type=float, default=0.9)
    ap.add_argument("--sweep", action="store_true", help="evaluate all 16 graph versions")
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--paper-split", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=None,
                    help="CV folds inside training data for SFFS and C (default 3; 5 with --paper-split)")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--max-features", type=int, default=20)
    ap.add_argument("--prefilter", type=int, default=30,
                    help="keep the N best features by ANOVA F (inside each training fold) before SFFS; 0 = off")
    ap.add_argument("--sel-windows", type=int, default=20,
                    help="windows per subject used by SFFS and C tuning (final model uses all); 0 = all")
    ap.add_argument("--fine-grid", action="store_true")
    ap.add_argument("--subject-rule", choices=["vote", "mean"], default="vote")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="leakage check: shuffle subject labels once before everything")
    ap.add_argument("--noise-dirs", default=None,
                    help="comma-separated feature dirs built with --snr; evaluated with the same settings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None, help="default results/baseline_<conn>/")
    args = ap.parse_args()
    if args.inner_folds is None:
        args.inner_folds = 5 if args.paper_split else 3
    if args.loso and args.paper_split:
        raise SystemExit("choose --loso or --paper-split, not both")
    out = Path(args.out or f"results/baseline_{args.conn}")
    t0 = time.time()

    def prepare(features_dir, version):
        X, groups, subj, names = load_dataset(args, features_dir, version)
        if args.shuffle_labels:
            subj["label"] = np.random.default_rng(args.seed).permutation(subj["label"].to_numpy())
        return X, groups, subj, names

    version = gm.version_for_density(args.density)
    sweep_rows = None
    if args.sweep:
        sweep_rows = []
        for v in gm.GRAPH_VERSIONS:
            X, groups, subj, names = prepare(args.features, v)
            r = evaluate(X, groups, subj, args, seed=args.seed, n_jobs=args.n_jobs, keep_details=False)
            sweep_rows.append(dict(version=v[0], lo=v[1], hi=v[2],
                                   **{f"window_{k}": x for k, x in r["pooled_window"].items()},
                                   **{f"subject_{k}": x for k, x in r["pooled_subject"].items()}))
            print(f"[sweep] {v[0]:>8}: subject bAcc={r['pooled_subject']['balanced_accuracy']:.3f} "
                  f"window bAcc={r['pooled_window']['balanced_accuracy']:.3f}")
        sw = pd.DataFrame(sweep_rows).sort_values(
            ["subject_balanced_accuracy", "window_balanced_accuracy"], ascending=False)
        out.mkdir(parents=True, exist_ok=True)
        sw.to_csv(out / "sweep.csv", index=False)
        best = sw.iloc[0]
        version = (best["version"], float(best["lo"]), float(best["hi"]))
        print(f"[sweep] best graph version: {version[0]} (chosen on the same CV -> optimistic)")

    X, groups, subj, names = prepare(args.features, version)
    print(f"{len(subj)} subjects, {len(X)} windows, {X.shape[1]} features, "
          f"classes {[int(c) for c in sorted(subj['label'].unique())]}, graph version {version[0]}")
    res = evaluate(X, groups, subj, args, seed=args.seed, n_jobs=args.n_jobs)
    print(f"[window ] {fmt(res['pooled_window'])}")
    print(f"[subject] {fmt(res['pooled_subject'])}")

    perm = None
    if args.n_perm > 0:
        null = Parallel(n_jobs=args.n_jobs, verbose=5)(          # prints progress
            delayed(permuted_score)(X, groups, subj, args, i) for i in range(args.n_perm))
        null = np.array(null)
        perm = dict(n_perm=args.n_perm, labels_shuffled_at="subject level")
        stats = [("subject_balanced_accuracy", res["pooled_subject"]["balanced_accuracy"]),
                 ("window_balanced_accuracy", res["pooled_window"]["balanced_accuracy"])]
        if "auc" in res["pooled_subject"]:          # binary: threshold-free subject statistic
            stats.append(("subject_auc", res["pooled_subject"]["auc"]))
        for i, (name, obs) in enumerate(stats):
            perm[name] = dict(observed=obs, null_mean=float(np.nanmean(null[:, i])),
                              p=float((1 + (null[:, i] >= obs).sum()) / (args.n_perm + 1)))
            print(f"[perm   ] {name}: observed {obs:.3f}, null mean {perm[name]['null_mean']:.3f}, "
                  f"p={perm[name]['p']:.3f}")

    extra = {}
    if args.noise_dirs:
        rows = [dict(features=args.features, snr="clean",
                     **{f"window_{k}": x for k, x in res["pooled_window"].items()},
                     **{f"subject_{k}": x for k, x in res["pooled_subject"].items()})]
        for d in args.noise_dirs.split(","):
            Xn, gn, sn, _ = prepare(d.strip(), version)
            r = evaluate(Xn, gn, sn, args, seed=args.seed, n_jobs=args.n_jobs, keep_details=False)
            rows.append(dict(features=d.strip(), snr=d.strip(),
                             **{f"window_{k}": x for k, x in r["pooled_window"].items()},
                             **{f"subject_{k}": x for k, x in r["pooled_subject"].items()}))
            print(f"[noise  ] {d}: subject bAcc={r['pooled_subject']['balanced_accuracy']:.3f} "
                  f"window bAcc={r['pooled_window']['balanced_accuracy']:.3f}")
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "noise_robustness.csv", index=False)

    extra["runtime_s"] = round(time.time() - t0, 1)
    if args.shuffle_labels:
        extra["labels_shuffled"] = True
    write_outputs(res, names, out, args, version, perm, extra)
    print(f"Saved results to {out}/ ({extra['runtime_s']} s)")


if __name__ == "__main__":
    main()
